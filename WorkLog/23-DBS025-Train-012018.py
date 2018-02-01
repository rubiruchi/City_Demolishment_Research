import geopandas as gpd
import pandas as pd
import numpy as np

import shapely
#from matplotlib import pyplot as plt
from descartes import PolygonPatch
from shapely.geometry import Polygon, mapping
from shapely.geometry.multipolygon import MultiPolygon
from shapely import wkt
from shapely.geometry import Point
from geopandas import GeoSeries, GeoDataFrame
from gurobipy import *
from geopy.distance import great_circle

from multiprocessing import Process
from joblib import Parallel, delayed
import time
import sys


def affect(x1,x2):
    dis = great_circle((x1.x,x1.y),(x2.x,x2.y)).meters
    return 1.0/dis if dis <= 240 else 0


if  __name__ == '__main__':

    # get gdf
    df = pd.read_csv('OriginalDataBaseFile/RealPropertiesExtraClean.csv')
    df.rename(columns={"Parcel":"geometry"},inplace=True)
    df.geometry = df.geometry.apply(lambda var: MultiPolygon([wkt.loads(var)]))
    gdf = gpd.GeoDataFrame(df)

    # filter gdf by edge
    df2 = pd.read_csv('OriginalDataBaseFile/RealGoodNeighbors.csv',low_memory=False,skiprows = lambda x: x == 137240 )
    idsettotal =  set(df2.aID.unique()).union(set(df2.bID.unique()))
    gdf = gdf[gdf.ID.apply(lambda var: var in idsettotal)]
    # filter gdf row by houses size
    gdf = gdf[(gdf.geometry.area> 3e-09) & (gdf.geometry.area< 5e-08) ]

    # get house set
    ogdf = gdf[gdf.IsVacant == 0]
    vgdf = gdf[gdf.IsVacant == 1]
    print "Total Object:", len(gdf)
    print "Residents:" ,len(ogdf)
    print "Vacant:", len(vgdf)

    # get edge
    amerge = pd.merge(df2,gdf,left_on=["aID"],right_on=["ID"])
    bmerge = pd.merge(df2,gdf,left_on=["bID"],right_on=["ID"])
    edge = pd.merge(amerge,bmerge,left_on=["aID","bID"],right_on=["aID","bID"]).iloc[:,:4]


    # init budget
    Budget = 100000
    demolish_2_story, demolish_3_story = 13000, 22000
    r_relocate, o_relocate = 85000, 170000
    wall_2_story, wall_3_story = 14000, 25000
    cost_reduction = 2000

    # cost
    Cost = [ 13000 + (85000 if gdf.loc[i,"IsVacant"]== 0 else 0)
             for i in gdf.index
            ]
    Benefit = [ cost_reduction for i in xrange(len(edge))]
    wi = [ 14000 for i in xrange(len(edge))]
    wj = [ 14000 for i in xrange(len(edge))]


    # set variable
    test0 = Model()

    x = test0.addVars(gdf.ID,vtype = GRB.BINARY,name = "x")

    edgetuple = zip(edge.aID,edge.bID)
    y = test0.addVars(edgetuple,vtype = GRB.BINARY,name = "y")

    bigM = test0.addVars(ogdf.ID,vtype = GRB.CONTINUOUS,name = "bigM",lb = -GRB.INFINITY, ub = 0.0)


    # set constraint
    Budget_Constraint = test0.addConstr((quicksum(Cost[i]*x[gdf.ID.iloc[i]] for i in xrange(len(gdf))) -
                                         quicksum((Benefit[i]+wi[i]+wj[i])*y[edgetuple[i]] for i in xrange(len(edgetuple))) +
                                         quicksum(wi[i]*x[edgetuple[i][1]] for i in xrange(len(edgetuple))) +
                                         quicksum(wj[i]*x[edgetuple[i][0]] for i in xrange(len(edgetuple)))
                                         <= Budget   )
                                        , name = "Budget_Constraint")

    CD1 = test0.addConstrs((x[edgetuple[i][1]] + x[edgetuple[i][0]] - y[edgetuple[i]] <= 1
                              for i in xrange(len(edgetuple))
                                                   ),name = "CD1")

    CD2 = test0.addConstrs((-x[edgetuple[i][1]]  + y[edgetuple[i]] <= 0
                              for i in xrange(len(edgetuple))
                                                   ),name = "CD2")

    CD3 = test0.addConstrs((-x[edgetuple[i][0]]  + y[edgetuple[i]] <= 0
                              for i in xrange(len(edgetuple))
                                                   ),name = "CD3")


    s = time.time()
    def f(i):
        test0.addConstr(( bigM[ogdf.ID.iloc[i]]
                          <=
                          quicksum( affect(ogdf.geometry.iloc[i].centroid,vgdf.geometry.iloc[v].centroid)*(x[vgdf.ID.iloc[v]]-1)
                                    for v in xrange(len(vgdf)))
                          + 200000.0*x[ogdf.ID.iloc[i]] )
                        , name = "for each occupied")

        if (i+1)%10 == 0:
            sys.stdout.write(str(i+1)+"/139585 files , time: "+str(time.time()-s)+'\r')
            sys.stdout.flush()

    Parallel(n_jobs=4)(delayed(f)(i) for i in xrange(len(ogdf)))
    print "total time:", (time.time()-s)

    # optimize
    test0.setObjective( quicksum(bigM[i] for i in ogdf.ID), GRB.MAXIMIZE)
    test0.optimize()

    # write result
    result = pd.DataFrame([ 1.0 if x[i].X == 1.0 or abs(x[i].X - 1.0) < 0.000001
                                else 0.0 for i in gdf.ID ], index = gdf.ID)
    result.to_csv('result_color-01-20-18.csv')



import csv
import shapefile
import time
import sys

from math import radians
from math import sin, cos, sqrt, atan2 # approximate radius of earth in meters
from gurobipy import *
import numpy as np
import pandas as pd
import geopandas as gpd
#import shapely
#from descartes import PolygonPatch
#from shapely.geometry import Polygon, mapping
from shapely.geometry.multipolygon import MultiPolygon
from shapely import wkt


#################################################################
#
#     Read/ Preprocess gdf
#     ---------
#     read_geo_file(name)  - return geodataframe
#     preprocess_data(gdf,edf)
#
#################################################################

def read_geo_file(file_name = 'OriginalDataBaseFile/RealPropertiesExtraClean.csv'):
    """
    read file from csv and return geodataframe
    """
    df = pd.read_csv(file_name)
    df.rename(columns={"Parcel":"geometry"},inplace=True)
    df.geometry = df.geometry.apply(lambda var: MultiPolygon([wkt.loads(var)])) # string to multipolygon

    gdf = gpd.GeoDataFrame(df)
    gdf["Block"] = gdf.BlockLot.apply(lambda num: num[:4])
    gdf.crs = {'init' :'epsg:4326'}

    gdf['x'] = gdf.geometry.centroid.x # Longitude
    gdf['xradians'] = gdf.x.apply(radians) # x coords to radians
    gdf['y'] = gdf.geometry.centroid.y # Latitude
    gdf['yradians'] = gdf.y.apply(radians) # y coords to radians
    gdf['yx'] = zip(gdf.y,gdf.x) # (lat, lon)

    # for optimization computation
    gdf['xyradians'] = zip(gdf.xradians,gdf.yradians)
    gdf['xyrID'] = zip(gdf.xyradians,gdf.ID)

    return gdf

class preprocess_data(object):
    def __init__(self,gdf,edf):

        self.gdf = gdf
        self.edf = edf
        self.target_building_size()
        self.get_house_subset()
        self.ogdf, self.vgdf = self.get_house_subset()
        self.info()

    def target_building_size(self,lower = 3e-09, upper = 5e-08, inplace = True):
        """
        return sub-gdf with gdf.geometry.area in (lower, upper)
        if inplace = True, update edf also
        """
        if not inplace:
            return self.gdf[(self.gdf.geometry.area> lower) & (self.gdf.geometry.area< upper) ]

        self.gdf = self.gdf[(self.gdf.geometry.area> lower) & (self.gdf.geometry.area< upper) ]
        self.edf = self.edf[(self.edf.aID.isin(set(self.gdf.ID))) & (self.edf.bID.isin(set(self.gdf.ID)))]

    def get_house_subset(self):
        ogdf = self.gdf[self.gdf.IsVacant == 0]
        vgdf = self.gdf[self.gdf.IsVacant == 1]
        return ogdf, vgdf

    def info(self):
        print "total:", len(self.gdf)
        print "residents:" ,len(self.ogdf)
        print "vacant:", len(self.vgdf)
        print "edges:", len(self.edf)


"""
Basic dataframe
"""
data = preprocess_data(read_geo_file(), pd.read_csv('OriginalDataBaseFile/RealGoodNeighbors.csv',low_memory=False,
                                   skiprows = lambda x: x == 137240 ) )

gdf, ogdf, vgdf, edf = data.gdf, data.ogdf, data.vgdf, data.edf
edgetuple = zip(edf.aID,edf.bID)


#################################################################
#
#     Read/ run Gurobi Model
#     read_model(name) - return model,x,y,bigM
#     LPSolve(model,x,y,bigM,early= True)
#        > budget_constraint > set_objective > solve
#
#################################################################

def read_model(model_name = 'model-02-04-18-d500'):

    model = read(model_name + '.mps')
    model.read(model_name + '.prm')
    # bigM method
    bigM = tupledict()
    # neighbors
    y = tupledict()
    # houese
    x = tupledict()

    for var in model.getVars():
        if var.VarName[:1] == "x":
            x[int(var.VarName[2:-1])] = var
        elif var.VarName[:1] == "y":
            y[tuple(map(int,var.VarName[2:-1].split(",")))] = var
        if var.VarName[:4] == "bigM":
            bigM[int(var.VarName[5:-1])] = var

    return model, x, y, bigM



class LPSolve(object):
    def __init__(self, model = None, x = None, y = None, bigM = None, early = True):
        if model != None:
            self.model = model
            self.x = x
            self.y = y
            self.bigM = bigM
        else:
            self.model, self.x, self.y, self.bigM = read_model()
        self.solcounts = 0
        self.status = []
        self.difference = []

    def budget_constraint(self,
                          Budget = 5000000,
                          demolish_2_story = 13000, demolish_3_story = 22000,
                          r_relocate = 85000, o_relocate = 170000,
                          wall_2_story = 14000, wall_3_story = 25000,
                          cost_reduction = 0):

        # cost list
        Cost = [ 13000 + (85000 if gdf.loc[i,"IsVacant"]== 0 else 0) for i in gdf.index ]
        Benefit = [ cost_reduction for i in xrange(len(edf))]
        wi = [ 14000 for i in xrange(len(edf))] # 01-10-18
        wj = [ 14000 for i in xrange(len(edf))]


        Budget_Constraint = self.model.addConstr((quicksum(Cost[i]*self.x[gdf.ID.iloc[i]] for i in xrange(len(gdf))) -
                     quicksum((Benefit[i]+wi[i]+wj[i])*self.y[edgetuple[i]] for i in xrange(len(edgetuple))) +
                     quicksum(wi[i]*self.x[edgetuple[i][1]] for i in xrange(len(edgetuple))) +
                     quicksum(wj[i]*self.x[edgetuple[i][0]] for i in xrange(len(edgetuple)))
                     <=
                     Budget
                    )
                     , name = "Budget_Constraint")

        # save into clase which we can check later
        self.Budget = Budget
        self.demolish_2_story, self.demolish_3_story = demolish_2_story, demolish_3_story
        self.r_relocate, self.o_relocate = r_relocate, o_relocate
        self.wall_2_story, self.wall_3_story = wall_2_story, wall_3_story
        self.cost_reduction = cost_reduction

        # for update status
        self.Cost = [ 13000 + (85000 if gdf.loc[i,"IsVacant"]== 0 else 0) for i in gdf.index ]
        self.Benefit = [ cost_reduction for i in xrange(len(edf))]
        self.wi = [ 14000 for i in xrange(len(edf))] # 01-10-18
        self.wj = [ 14000 for i in xrange(len(edf))]


    def set_objective(self, impact = True):
        if impact:
            self.model.setObjective( quicksum(self.bigM[i] for i in vgdf.ID), GRB.MAXIMIZE)
        else:
            self.model.setObjective( quicksum(self.x[i] for i in gdf.ID), GRB.MAXIMIZE)

    def solve(self,difference = False):

        if self.solcounts == 0:
            pass

        elif sum(self.x[gdf.ID.iloc[i]].X for i in xrange(len(gdf))) != 0:
            self.model.addConstr(
                (
                quicksum(self.x[gdf.ID.iloc[i]]
                         if self.x[gdf.ID.iloc[i]].X == 1 or abs(self.x[gdf.ID.iloc[i]].X - 1.0) < 0.000001
                         else 0 for i in xrange(len(gdf)))
                <= sum( 1
                       if self.x[gdf.ID.iloc[i]].X == 1 or abs(self.x[gdf.ID.iloc[i]].X - 1.0) < 0.000001
                       else 0 for i in xrange(len(gdf)))-1
                ),name = 'temp'
            )
            if difference:
                self.difference += [ gdf.ID.iloc[i] for i in xrange(len(gdf)) if self.x[gdf.ID.iloc[i]].X == 1 or abs(self.x[gdf.ID.iloc[i]].X - 1.0) < 0.000001 ]
                lambda_ = 0.1
                self.model.setObjective( quicksum(self.bigM[i] for i in vgdf.ID) - lambda_*quicksum(self.x[i]  for i in self.difference ) , GRB.MAXIMIZE)


        self.model.optimize()
        self.solcounts += 1

        self.__update_status(difference)

    def __update_status(self,difference,lambda_=0.1):
        spent = (sum(self.Cost[i]*self.x[gdf.ID.iloc[i]].X for i in xrange(len(gdf))) -
                 sum((self.Benefit[i]+self.wi[i]+self.wj[i])*self.y[edgetuple[i]].X for i in xrange(len(edgetuple))) +
                 sum(self.wi[i]*self.x[edgetuple[i][1]].X for i in xrange(len(edgetuple))) +
                 sum(self.wj[i]*self.x[edgetuple[i][0]].X for i in xrange(len(edgetuple))) )


        num_houses = sum(self.x[i].X  for i in gdf.ID)

        #print "Budget : %s   number of houses : %s" %(spent, num_houses)
        print("Budget : %s   number of houses : %s   ObjVal : %s   Running Time : %s" %
               (spent,num_houses,self.model.ObjVal,self.model.Runtime))
        if not difference:
            self.status.append((spent,num_houses,self.model.ObjVal,self.model.Runtime))
        else:
            self.status.append((spent,num_houses,self.model.ObjVal + lambda_*sum(1  for i in self.difference if self.x[i].X == 1 or abs(self.x[i].X - 1.0) < 0.000001 ),
                self.model.Runtime))
    def get_status(self):
        try:
            return ("Budget : %s   number of houses : %s   ObjVal : %s   Running Time : %s" %
                        self.status[-1])
        except:
            print "please calling self.solve() before calling self.get_status"







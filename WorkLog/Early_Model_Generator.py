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
from shapely.geometry.multipolygon import MultiPolygon
from shapely import wkt
#import shapely
#from descartes import PolygonPatch
#from shapely.geometry import Polygon, mapping


######################################################
#
#    
#      Read / Preprocess
#
#
######################################################
def read_geo_file(file_name = 'OriginalDataBaseFile/RealPropertiesExtraClean.csv'):
    """ 
    read file from csv 
    return geodataframe
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
    def __init__(self,gdf,edf,
                 lower=3e-09, upper=5e-08, auto = True):
        """
        input---
        gdf = Geodataframe, 
        edf = edgeset dataframe
        
        variable---
        gdf,edf,ogdf,vgdf 
        
        function---
        target_building_size()
        get_house_subset()
        info()
        """
        self.gdf = gdf
        self.edf = edf
        if auto:
            self.target_building_size(lower,upper)
            self.ogdf, self.vgdf = self.get_house_subset()
            self.info()
        else:
            self.ogdf, self.vgdf = None, None
        
    def target_building_size(self,lower = 3e-09, upper = 5e-08, inplace = True):
        """ 
        return sub-gdf with gdf.geometry.area in (lower, upper)
        if inplace = True, update edf also
        """
        if not inplace:
            return self.gdf[(self.gdf.geometry.area> lower) & (self.gdf.geometry.area< upper) ] 
        
        self.gdf = self.gdf[(self.gdf.geometry.area> lower) & (self.gdf.geometry.area< upper) ] 
        self.edf = self.edf[(self.edf.aID.isin(set(self.gdf.ID))) 
                          & (self.edf.bID.isin(set(self.gdf.ID)))]
    
    def get_house_subset(self, inplace = True):      
        ogdf = self.gdf[self.gdf.IsVacant == 0]
        vgdf = self.gdf[self.gdf.IsVacant == 1]
        if not inplace:
            return ogdf, vgdf
        self.ogdf, self.vgdf = self.ogdf, self.vgef
        
    def info(self):
        print "total:", len(self.gdf)
        print "residents:" ,len(self.ogdf)
        print "vacant:", len(self.vgdf)
        print "edges:", len(self.edf)
        
    
    
######################################################
#
#    
#      Grurobi early Stage Setup
#
#
######################################################
def real_dis_affect(col,col2,power = 1, upper = 500):
    lon1, lat1 = col
    lon2, lat2 = col2
    R = 6373.0
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    dis = R * c*1000 
    
    return 1/(dis**power) if dis < upper else 0
        
"""
set up early stage linear constraints (Gurobi)
before budget constraint
"""
def mps_generater(uppers, powers):
    for upper in uppers:
        for power in powers:
            model = Model() 
            model.Params.Threads = 4 # parallel

            x = model.addVars(gdf.ID,vtype = GRB.BINARY,name = "x")
            y = model.addVars(edgetuple,vtype = GRB.BINARY,name = "y")
            bigM = model.addVars(vgdf.ID,vtype = GRB.CONTINUOUS,name = "bigM",lb = -GRB.INFINITY, ub = 0.0)
            model.update()


            # y_{ij} = x_i and x_j
            and1 = model.addConstrs((x[edgetuple[i][1]] + x[edgetuple[i][0]] - y[edgetuple[i]] <= 1 
                                  for i in xrange(len(edgetuple))
                                 ),name = "and1")

            and2 = model.addConstrs((-x[edgetuple[i][1]]  + y[edgetuple[i]] <= 0 
                                  for i in xrange(len(edgetuple))
                                 ),name = "and2")

            and3 = model.addConstrs((-x[edgetuple[i][0]]  + y[edgetuple[i]] <= 0 
                                  for i in xrange(len(edgetuple))
                                 ),name = "and3")


            # big M Constraint
            s = time.time()
            for h in xrange(len(vgdf)):
                model.addConstr(
                    ( bigM[vgdf.ID.iloc[h]] <= 
                      quicksum( ogdf[((ogdf.xradians <= vgdf.xradians.iloc[h] + 0.00012) & 
            (ogdf.xradians >= vgdf.xradians.iloc[h] - 0.00012) &
            (ogdf.yradians <= vgdf.yradians.iloc[h] + 0.0001) & 
            (ogdf.yradians >= vgdf.yradians.iloc[h] - 0.0001) )].xyrID.apply(lambda var:
                 real_dis_affect(var[0],vgdf.xyradians.iloc[h],power,upper)*(x[var[1]]-1)) )
                      + 200000.0*x[vgdf.ID.iloc[h]] 
                    ), name = "occupied"+str(h) ) 

                if (h+1)%10 == 0:
                    sys.stdout.write(str(h+1)+"/" +str(len(vgdf))+ " files , time: "+str(time.time()-s)+'\r')
                    sys.stdout.flush()
            print "total time: ", round(time.time()-s,3) , " sec"


            model.write('model/model-d%s-p%s.lp' %(upper,power))
            model.write('model/model-d%s-p%s.mps'%(upper,power))
            model.write('model/model-d%s-p%s.prm'%(upper,power))
        
if __name__ == '__main__':
    
    ######################################################
    # Read and Preprocess data
    # Note: skip the last row in edge csv 
    ######################################################
    data = preprocess_data(read_geo_file(), pd.read_csv('OriginalDataBaseFile/RealGoodNeighbors.csv',low_memory=False,
                                       skiprows = lambda x: x == 137240 ) )

    gdf, ogdf, vgdf, edf = data.gdf, data.ogdf, data.vgdf, data.edf
    edgetuple = zip(edf.aID,edf.bID)
    
    ######################################################
    # Generate mps,lp,prm file
    ######################################################
    mps_generater()
    sys.exit("finished generating model.mps**************")
    
    
    
    
    
    
    
    




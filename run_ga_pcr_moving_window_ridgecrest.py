# -*- coding: utf-8 -*-
"""
Created on Sun Jun 11 19:34:19 2023

@author: nekhtari
"""
import matplotlib.pyplot as plt
import numpy as np
import pdal
import json
import GA_real
from numpy.random import seed
import compare_with_icp


import geopandas as gpd
from shapely.geometry import Point



WS = 100
SS = 50
BOUNDS = [445750, 448550, 3953600, 3957000]


shapefile_path  = r'D:\Working\Ridgecrest\Processing 2024\Polygons_utm.shp'
polygons = gpd.read_file(shapefile_path)
poly_index = 0      # 0 for westr, 1 for east



fixed_file =  r'D:\Working\Ridgecrest\Processing 2024\ICP\roi\ncalm12_roi_west.las' #r'./data/fixed3.laz'
moving_file = r'D:\Working\Ridgecrest\Processing 2024\ICP\roi\sgm_wgs84_heights12_roi_west.las' #r'./data/moving3.laz'
output_file = r'D:\Working\Ridgecrest\Processing 2024\ICP\ga\west_{}_{}.txt'.format(WS, SS) #r'./data/output.txt'




reader1 = [
    {
        "type":"readers.las",
        "filename":fixed_file
    },
    {
        "type":"filters.normal",
        "knn":5
    }
]

pipeline1 = pdal.Pipeline(json.dumps(reader1))
pipeline1.execute()
arrays1 = pipeline1.arrays
view1 = arrays1[0]
X1 = view1['X']
Y1 = view1['Y']
Z1 = view1['Z']
NX = view1['NormalX']
NY = view1['NormalY']
NZ = view1['NormalZ']
Normal = np.vstack([NX, NY, NZ]).T



# Read in The Moving Dataset - Smaller, Don't Need Normals
reader2 = [
    {
        "type":"readers.las",
        "filename":moving_file
    }
]

pipeline2 = pdal.Pipeline(json.dumps(reader2))
pipeline2.execute()
arrays2 = pipeline2.arrays
view2 = arrays2[0]
X2 = view2['X']
Y2 = view2['Y']
Z2 = view2['Z']



''' ----------------------------------------------------------------------- '''
''' ----------------------------------------------------------------------- '''
''' ----------------------------------------------------------------------- '''










''' ************************ Run GA registration ************************ '''

# bounds6 = np.array([[-0.1, 0.1], [-0.1, 0.1], [-0.1, 0.1], [-1, 1], [-1, 1], [-1, 1]]) * 3
bounds3 = np.array([[-1, 1], [-1, 1], [-1, 1]]) * 2

config = dict([("plot_rmse", False),
               ("population_size", 80),
                ("num_params", 3),
                ("num_bits", 10),
                ("bounds", bounds3),
                ("selection", "roulette wheel"),
                ("selection_rate", 1),
                ("cross_over", "two_point"),
                ("mutation_rate", 0.1),
                ("max_generations", 80),
                ("epsilon", 1e-9)],)


# g = GA_real.GA(fixed, Normal, moving, output_file, config)
# g.run_ga()


# # fig = plt.figure()
# # plt.plot(g.score)

# ind_best = np.argmin(np.array(g.score))
# print("\n the best solution after {} generations has a fitness score of {} and is as follows:".format(ind_best + 1, np.round(np.min(g.score), 4)))
# print(g.best)






# ''' ************************ Run ICP registration ************************ '''
configs_icp = {
'bounds' : BOUNDS,
'method' : 'translation_only',
'prop_errors' : True,
'threshold' : 0,
'window_size' : WS,
'step_size' : SS,
'margin': 2,
'min_points' : 200,
'converge' : 0.000001,
'max_iter' : 40,
'outlier_multiplier' : 5,
'outlier_percent' : 0.95,
'output_basename' : 'trans_icp_results'
}

config_icp = compare_with_icp.icp_configs(configs_icp)

# # res1, RMSE, Max = compare_with_icp.transicp(moving, fixed, Normal, config_icp)











window_size = WS
step_size = SS
margin = 2
min_points = 200

# Variables to hold the ICP vector origins (X, Y) and displacements (dx, dy)
X = []
Y = []
dxi, dxg = [], []
dyi, dyg = [], []
dzi, dzg = [], []
DX, DY, DZ = [], [], []
RMSEi, RMSEg = [], []
prop_err = []

B = BOUNDS # [446650, 446750, 3955800, 3955900]
# B = [273100, 273600, 3289300, 3289800]
iii = 0
for y in range(B[2], B[3], step_size):
    dxr, dyr, dzr = [], [], []
    for x in range(B[0], B[1], step_size):
        point = Point(x, y)
        poly = polygons.iloc[poly_index].geometry
        if poly.contains(point):
            indx = (x <= X1) & (X1 <= (x + window_size))
            indy = (y <= Y1) & (Y1 <= (y + window_size))
            ind = indx & indy
            Xa = np.stack((X1[ind], Y1[ind], Z1[ind]), axis = 1)
            Na = Normal[ind, :]
            
            indx = (x <= X2) & (X2 <= (x + window_size))
            indy = (y <= Y2) & (Y2 <= (y + window_size))
            ind = indx & indy
            Xb = np.stack((X2[ind], Y2[ind], Z2[ind]), axis = 1)
            
            
            mean_x, mean_y, mean_z = np.mean(X1), np.mean(Y1), np.mean(Z1)
            
            Xa = Xa - np.stack((mean_x, mean_y, mean_z))
            Xb = Xb - np.stack((mean_x, mean_y, mean_z))
            
            
            if ((len(Xa) < min_points) | (len(Xb) < min_points)):
                dxr.append(0)
                dyr.append(0)
                dzr.append(0)
                continue
            
            
            ''' ---- for icp ----- '''
            res, rmse, max_residual = compare_with_icp.transicp(Xb, Xa, Na, config_icp)
            dxi.append(res[0])
            dyi.append(res[1])
            dzi.append(res[2])
            RMSEi.append(rmse)
            
            
            ''' ---- for ga ----- '''
            g = GA_real.GA(Xa, Na, Xb, output_file, config)
            g.run_ga()
            dxg.append(g.best[0])
            dyg.append(g.best[1])
            dzg.append(g.best[2])
            RMSEg.append(g.score[-1])
    
    
            X.append(x + step_size/2)
            Y.append(y + step_size/2)

    

ga_roi1 = np.stack((dxg, dyg, dzg, RMSEg), axis = 1)
print(np.mean(ga_roi1, axis = 0))
print(np.std(ga_roi1, axis = 0))

icp_roi1 = np.stack((dxi, dyi, dzi, RMSEi), axis = 1)
print(np.mean(icp_roi1, axis = 0))
print(np.std(icp_roi1, axis = 0))

np.savetxt(r'D:\Working\Ridgecrest\Processing 2024\ICP\ga\ga_west_{}_{}.txt'.format(WS, SS), ga_roi1, fmt='%3.6e', delimiter='\t')
np.savetxt(r'D:\Working\Ridgecrest\Processing 2024\ICP\ga\icp_west_{}_{}.txt'.format(WS, SS), icp_roi1, fmt='%3.6e', delimiter='\t')




















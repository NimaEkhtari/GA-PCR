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
# seed(40)



output_file = 'test'
inlas = './data/test.laz'

p1 = [
    {
    'type':'readers.las',
    'filename':inlas,
    "use_eb_vlr":True,
    "nosrs":True
    },
    {
    "limits": "Classification[2:6]",
    "type": "filters.range"
    },
    {
        "type":"filters.normal",
        "knn":5
    }
]


p1 = pdal.Pipeline(json.dumps(p1))
p1.execute()




A = p1.arrays[0]
inds = np.random.permutation(len(A))
br = int(np.round(len(A) * 0.4))
ind1 = inds[0:br]
ind2 = inds[br:]
view1  = A[ind1]
view2 = A[ind2]

X1 = view1['X']
Y1 = view1['Y']
Z1 = view1['Z']
NX = view1['NormalX']
NY = view1['NormalY']
NZ = view1['NormalZ']
Na = np.stack((NX, NY, NZ), axis = 1)


X2 = view2['X']
Y2 = view2['Y']
Z2 = view2['Z']

''' ----------------------------------------------------------------------- '''
''' ----------------------------------------------------------------------- '''
''' ----------------------------------------------------------------------- '''




''' ************************ Run GA registration ************************ '''

# bounds6 = np.array([[-0.1, 0.1], [-0.1, 0.1], [-0.1, 0.1], [-1, 1], [-1, 1], [-1, 1]]) * 3
bounds3 = np.array([[-1, 1], [-1, 1], [-1, 1]]) * 5

config = dict([("population_size", 80),
                ("num_params", 3),
                ("num_bits", 10),
                ("bounds", bounds3),
                ("selection", "roulette wheel"),
                ("selection_rate", 1),
                ("cross_over", "two_point"),
                ("mutation_rate", 0.1),
                ("max_generations", 80),
                ("epsilon", 1e-9)],)




''' ************************ Run ICP registration ************************ '''
configs_icp = {
'bounds' :[273100, 273600, 3289300, 3289800],
'method' : 'translation_only',
'prop_errors' : True,
'threshold' : 0,
'window_size' :25,
'step_size' : 25,
'margin': 2,
'min_points' : 50,
'converge' : 0.000001,
'max_iter' : 40,
'outlier_multiplier' : 5,
'outlier_percent' : 0.95,
'output_basename' : 'trans_icp_results'
}

config_icp = compare_with_icp.icp_configs(configs_icp)

''' ----------------------------------------------------------------------- '''
''' ----------------------------------------------------------------------- '''


# Variables to hold the ICP vector origins (X, Y) and displacements (dx, dy)

dxi, dxg = [], []
dyi, dyg = [], []
dzi, dzg = [], []
RMSEi, RMSEg = [], []


sh = np.random.uniform(-5, 5, (30, 3))
for i in range(30):
    
    ShiftX, ShiftY, ShiftZ = sh[i, 0], sh[i, 1], sh[i, 2]
    
    X1 = view1['X']
    Y1 = view1['Y']
    Z1 = view1['Z']
    X2 = view2['X'] + ShiftX
    Y2 = view2['Y'] + ShiftY
    Z2 = view2['Z'] + ShiftZ


    # Removing centroid
    mean_x = np.mean(np.hstack((X1, X2)))
    mean_y = np.mean(np.hstack((Y1, Y2)))
    mean_z = np.mean(np.hstack((Z1, Z2)))
    X1 = X1 - mean_x
    Y1 = Y1 - mean_y
    Z1 = Z1 - mean_z
    X2 = X2 - mean_x
    Y2 = Y2 - mean_y
    Z2 = Z2 - mean_z
    

    Xa = np.stack((X1, Y1, Z1), axis = 1)
    Xb = np.stack((X2, Y2, Z2), axis = 1)
    
   
    ''' ---- for icp ----- '''
    res, rmse, max_residual = compare_with_icp.transicp(Xb, Xa, Na, config_icp)
    dxi.append(res[0])
    dyi.append(res[1])
    dzi.append(res[2])
    RMSEi.append(rmse)
    
    
    ''' ---- for ga ----- '''
    g = GA_real.GA(Xb, Na, Xa, output_file, config)
    g.run_ga()
    dxg.append(-g.best[0])
    dyg.append(-g.best[1])
    dzg.append(-g.best[2])
    RMSEg.append(g.score[-1])

    print(i)   
    

ga_1 = np.stack((dxg, dyg, dzg), axis = 1)
ga = ga_1 + sh
print(np.mean(ga, axis = 0))
print(np.std(ga, axis = 0))

icp_1 = np.stack((dxi, dyi, dzi), axis = 1)
icp = icp_1 + sh
print(np.mean(icp, axis = 0))
print(np.std(icp, axis = 0))

np.savetxt('ga_init.txt', ga_1, fmt='%3.6e', delimiter='\t')
np.savetxt('icp_init.txt', icp_1, fmt='%3.6e', delimiter='\t')




















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

# fixed_file = r'D:\Working\BAA\Task 6\6.3\From Craig\cloudb.ply'
# moving_file = r'D:\Working\BAA\Task 6\6.3\From Craig\clouda.ply'
# output_file = r'D:\Working\BAA\Task 6\6.3\From Craig\output.txt'

fixed_file =  r'./data/fixed.laz'
moving_file = r'./data/moving.laz'
output_file = r'./data/output.txt'



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



# Compute X1 Point Cloud Centroid To Remove It






''' ************************ Run GA registration ************************ '''

# bounds6 = np.array([[-0.1, 0.1], [-0.1, 0.1], [-0.1, 0.1], [-1, 1], [-1, 1], [-1, 1]]) * 3
bounds3 = np.array([[-1, 1], [-1, 1], [-1, 1]]) * 1

config = dict([("population_size", 40),
                ("num_params", 3),
                ("num_bits", 10),
                ("bounds", bounds3),
                ("selection", "roulette wheel"),
                ("selection_rate", 1),
                ("cross_over", "two_point"),
                ("mutation_rate", 0.05),
                ("max_generations", 60),
                ("epsilon", 1e-9)],)


# g = GA_real.GA(fixed, Normal, moving, output_file, config)
# g.run_ga()


# # fig = plt.figure()
# # plt.plot(g.score)

# ind_best = np.argmin(np.array(g.score))
# print("\n the best solution after {} generations has a fitness score of {} and is as follows:".format(ind_best + 1, np.round(np.min(g.score), 4)))
# print(g.best)






# ''' ************************ Run ICP registration ************************ '''
# configs_icp = {
# 'bounds' :[273100, 273600, 3289300, 3289800],
# 'method' : 'translation_only',
# 'prop_errors' : True,
# 'threshold' : 0,
# 'window_size' :50,
# 'step_size' : 50,
# 'margin': 2,
# 'min_points' : 50,
# 'converge' : 0.000001,
# 'max_iter' : 40,
# 'outlier_multiplier' : 5,
# 'outlier_percent' : 0.95,
# 'output_basename' : 'trans_icp_results'
# }

# config_icp = compare_with_icp.icp_configs(configs_icp)

# # res1, RMSE, Max = compare_with_icp.transicp(moving, fixed, Normal, config_icp)











window_size = 100
step_size = 50
margin = 2
min_points = 50

# Variables to hold the ICP vector origins (X, Y) and displacements (dx, dy)
X = []
Y = []
dx = []
dy = []
dz = []
DX, DY, DZ = [], [], []
RMSE, fitn = [], []
prop_err = []

B = [273100, 273600, 3289300, 3289800]
iii = 0
for y in range(B[2], B[3], step_size):
    dxr, dyr, dzr = [], [], []
    for x in range(B[0], B[1], step_size):
        iii += 1
        if iii > 25:
            print('stop')
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
        
        # res, rmse, max_residual = compare_with_icp.transicp(Xb, Xa, Na, config_icp)
        
        g = GA_real.GA(Xb, Na, Xa, output_file, config)
        g.run_ga()
         
         

        dx.append(-g.best[0])
        dy.append(-g.best[1])
        dz.append(-g.best[2])
        # dxr.append(res[0])
        # dyr.append(res[1])
        # dzr.append(res[2])
        RMSE.append(g.score[-1])
        
        # rmse.append(registration_icp.inlier_rmse)
        # fitn.append(registration_icp.fitness)

        X.append(x + step_size/2)
        Y.append(y + step_size/2)
        
        

        
    # DX.append(dxr)
    # DY.append(dyr)
    # DZ.append(dzr)
    print('{}% done'.format(np.ceil(y / (B[3] - B[2]) * 100)))



# res = np.stack([X, Y, dx, dy, dz], axis = 1)


# res[:, 0] = res[:, 0] + 698000
# res[:, 1] = res[:, 1] + 4005200
# sname = '{0}_{1}_{2}.txt'.format(config_icp.output_basename, window_size, step_size)
# np.savetxt(sname, res, delimiter=' ', fmt='%.3f')
























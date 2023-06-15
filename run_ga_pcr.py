# -*- coding: utf-8 -*-
"""
Created on Sun Jun 11 19:34:19 2023

@author: nekhtari
"""
import matplotlib.pyplot as plt
import numpy as np
import pdal
import json
import GA


# fixed_file = r'D:\Working\BAA\Task 6\6.3\From Craig\cloudb.ply'
# moving_file = r'D:\Working\BAA\Task 6\6.3\From Craig\clouda.ply'
# output_file = r'D:\Working\BAA\Task 6\6.3\From Craig\output.txt'

fixed_file =  r'.\data\fixed.laz'
moving_file = r'.\data\moving.laz'
output_file = r'.\data\output.txt'



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
mean_x, mean_y, mean_z = np.mean(X1), np.mean(Y1), np.mean(Z1)



fixed  = np.array([X1 - mean_x, Y1 - mean_y, Z1 - mean_z]).T
moving = np.array([X2 - mean_x, Y2 - mean_y, Z2 - mean_z]).T


bounds = np.array([[-1, 1], [-1, 1], [-1, 1], [-1, 1], [-1, 1], [-1, 1]])


config = dict([("population_size", 100),
               ("num_params", 6),
               ("bounds", bounds),
               ("selection", "random"),
               ("selection_rate", 1),
               ("cross_over", "one_point"),
               ("mutation_rate", 0.20),
               ("max_generations", 200),
               ("epsilon", 1e-6)])


g = GA.GA(fixed, Normal, moving, output_file, config)
g.run_ga()


fig = plt.figure()
plt.plot(g.score)

ind_best = np.argmin(np.array(g.score))
print("the best solution was reached in generation {} and equals:".format(ind_best))
print(g.best[ind_best])






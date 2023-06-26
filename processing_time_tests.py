#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 25 16:49:15 2023

@author: nima
"""

import matplotlib.pyplot as plt
import numpy as np
import pdal
import json
import faiss
import time


fixed_file =  r'./data/fixed.las'
moving_file = r'./data/moving.las'
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


mean_x, mean_y, mean_z = np.mean(X1), np.mean(Y1), np.mean(Z1)
fixed  = np.array([X1 - mean_x, Y1 - mean_y, Z1 - mean_z]).T
moving = np.array([X2 - mean_x, Y2 - mean_y, Z2 - mean_z]).T

''' ----------------------------------------------------------------------- '''
''' ----------------------------------------------------------------------- '''
''' ----------------------------------------------------------------------- '''
f = arrays1[0]
m = arrays2[0]

f['X'], f['Y'], f['Z'] = fixed[:,0], fixed[:,1], fixed[:,2]
m['X'], m['Y'], m['Z'] = moving[:,0], moving[:,1], moving[:,2]

pipeline = [
    {
        'type':'filters.icp'
    }
]

start = time.time()
p = pdal.Pipeline(json.dumps(pipeline), arrays = [f, m])
p.execute()

met = p.metadata
t = met.get('metadata').get('filters.icp').get('transform')
t = [float(val) for val in t.split()]
print('icp results = {}, {}, {}'.format(t[3], t[7], t[11]))
print('computation time is {}'.format(np.round(time.time() - start, 3)))

''' ----------------------------------------------------------------------- '''
''' ----------------------------------------------------------------------- '''
''' ----------------------------------------------------------------------- '''

d = 3               # Dimension of the point cloud
k = 1               # Num closest points to search for


''' test 1 - time it takes to create index '''
start = time.time()
for i in range(100):
    
    index = faiss.IndexFlatL2(d)
    index.add(fixed)
end = time.time()
avg = (end - start) / 100
print('Creating index on fixed points takes {} seconds on average.'.format(np.round(avg, 3)))


start = time.time()
for i in range(100):
    D, I = index.search(moving, k)
    I = I.reshape(len(I), )
end = time.time()
avg = (end - start) / 100
print('Searching for closest fixed points for moving points takes {} seconds on average.'.format(np.round(avg, 3)))









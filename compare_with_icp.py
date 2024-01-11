# -*- coding: utf-8 -*-
"""
Created on Fri Jan  5 10:32:35 2024

@author: ncalm
"""


import numpy as np
import faiss
import pdal
import json
import argparse
import math
import utilities
from scipy.spatial import KDTree




class icp_configs:
    def __init__(self, conf):
        self.bounds             = conf.get('bounds') 
        self.window_size        = conf.get('window_size')
        self.step_size          = conf.get('step_size')
        self.threshold          = conf.get('threshold')
        self.margin             = conf.get('margin')
        self.min_points         = conf.get('min_points')
        self.converge           = conf.get('converge')
        self.max_iter           = conf.get('max_iter')
        self.outlier_multiplier = conf.get('outlier_multiplier') #points larger than this many times the RMSE removed
        self.outlier_percent    = conf.get('outlier_percent') #However, we will keep 95% of the points.
        self.method             = conf.get('method')
        self.propagate          = conf.get('prop_errors')
        self.output_basename    = conf.get('output_basename')
        
        


def transicp(moving, fixed, fixed_normal, conf):
    loop = True
    icp_trans = [0,0,0]
    count = 0
    wsquared = 0
    k = 1
    
    means = np.round(np.mean(fixed, axis = 0), 2)
    X1 = moving - means
    X2 = fixed - means
       
    index = faiss.IndexFlatL2(3)
    index.add(X2)
    
    while loop == True:
        #Set Up The Point Indexing - Needs to Be Updated Every Iteration
        search_pts = X1 + icp_trans
        
        D, I = index.search(search_pts, k)
        misc = np.zeros((len(X1),1)).astype('float32')
        A = np.zeros((len(X1),3)).astype('float32')
        Normal = np.zeros((3,1)).astype('float32')
        for i in range(len(X1)):
            Normal = fixed_normal[I[i,0], :]
            
            Vec_Diff = (search_pts[i, :] - X2[I[i,0], :])
            Vec_DiffT = Vec_Diff.transpose()
            misc[i, :] = Vec_DiffT.dot(Normal)
            A[i, 0:3] = fixed_normal[I[i,0], :]
            

        AT = A.transpose()
        ATA = AT.dot(A)
        N = np.linalg.inv(ATA)
        U = AT.dot(misc)
        delcap = -N.dot(U)
        # offset = np.linalg.norm(delcap)
        icp_trans[0] = icp_trans[0] + delcap[0,0]
        icp_trans[1] = icp_trans[1] + delcap[1,0]
        icp_trans[2] = icp_trans[2] + delcap[2,0]
        
        # print(icp_trans)
        count = count + 1
        if np.all(np.abs(delcap) < conf.converge) or count > conf.max_iter:
            loop = False
    
    # Compute Final RMSE of Misclosure
    outlier = 0
    wsquared = 0
    resi = 0
    
    XX2 = np.squeeze(X2[I, :])
    Vec_Diff2 = (search_pts - XX2)  # I think we already added it no need to add again + delcap.transpose()
    NNormal = np.squeeze(fixed_normal[I, :])
    resi = np.sum(Vec_Diff2 * NNormal, axis = 1)

    # outlier = np.sum(np.abs(resi) > 0.02)
    wsquared = np.sum(resi ** 2)
    RMSE = math.sqrt(wsquared / len(X1))
    Max = np.amax(np.fabs(misc))

        
    
    return (icp_trans, RMSE, Max)













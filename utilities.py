# -*- coding: utf-8 -*-
"""
Created on Wed Oct 11 09:28:34 2023
This file houses all the common utilities that are shared by all different gcd
components.

@author: nekhtari
"""


import pdal
import numpy as np


def get_points(las, operation):
    p = pdal.Reader(las, use_eb_vlr=True).pipeline()
    if operation == 'icp':
        p |= pdal.Filter.normal(knn = 8)
    p.execute()
    las = p.get_dataframe(0)

    points = np.transpose(np.vstack([las['X'], las['Y'], las['Z']]))
    cov = np.transpose(np.vstack([las['VarianceX'], las['VarianceY'],
                                  las['VarianceZ'], las['CovarianceXY'],
                                  las['CovarianceXZ'], las['CovarianceYZ']]))
    
    if operation == 'icp':
        normals = np.transpose(np.vstack([las['NormalX'], las['NormalY'], las['NormalZ']]))
    else:
        normals = None
    minx = p.metadata['metadata']['readers.las']['minx']
    maxx = p.metadata['metadata']['readers.las']['maxx']
    miny = p.metadata['metadata']['readers.las']['miny']
    maxy = p.metadata['metadata']['readers.las']['maxy']
    bounds = [minx, maxx, miny, maxy]
    return (points, cov, normals, bounds)




def form_cov_matrix(tpu):
    n = len(tpu)
    C = np.zeros((3, 3))                       # Cov matrix of a point (3 x 3)
    cov = []
    for i in range(n):
        
        C[0, 0] = tpu[i, 0]                        # var_x
        C[1, 1] = tpu[i, 1]                        # var_y
        C[2, 2] = tpu[i, 2]                        # var_z
        C[0, 1] = C[1, 0] = tpu[i, 3]           # cov_xy
        C[0, 2] = C[2, 0] = tpu[i, 4]           #_cov_xz
        C[1, 2] = C[2, 1] = tpu[i, 5]           # cov_yz
        cov.append(C)
        C = np.zeros((3, 3))                    # Without this all elements of cov are re-written
    return cov
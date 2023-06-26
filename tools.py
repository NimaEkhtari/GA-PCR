#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 22 21:19:38 2023

@author: nima
"""
import numpy as np

def to_binary(array, nb):

    scale = (2 ** nb) -1
    ar = ((array + 3) / 6) * scale
    arr = np.int16(np.round(ar))
    
    b = []
    for a in arr:
        c = []
        for A in a:
            B = format(A, 'b')
            bit_length = len(B)
            leading_zeros = (nb - bit_length) * '0'
            c.append(leading_zeros + B)
        b.append(''.join(c))

    return b


def to_array(bits, nb):
    # ensure bits is a list
    if type(bits) is not list:
        bits = [bits]
    Arr = []
    for b in bits:
        ch = [b[i : i + nb] for i in range(0, len(b), nb)]
    
        a = [int(c, 2) for c in ch]
        Arr.append(a)

    scale = (2 ** nb) - 1
    A = ((np.array(Arr) / scale) * 6) - 3
    
    return A
        

# num_bits = 10
# e = np.array([[0, -0.5, 0.5, -1], [1, 0.01, 0.03, -0.06]])
# r = to_binary(e, num_bits)
# for R in r:
#     print(R)

# A = to_array(r, num_bits)
# for a in A:
#     print(np.round(a, 2))
    

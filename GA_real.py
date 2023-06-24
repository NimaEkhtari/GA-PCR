#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 22 22:48:02 2023
This code is the original GA code that works with real number
@author: nima
"""


import numpy as np
import faiss
from tqdm import tqdm
from time import sleep
import matplotlib.pyplot as plt


class GA:
        
    def __init__(self, fixed, normal, moving, output, config):
        self.fixed  = fixed
        self.normal = normal
        self.moving = moving
        self.output = output
        
        self.config = config
        if config.get("num_params") == 6:
            self.scales = np.array([np.pi / 6, np.pi / 6, np.pi / 6, 1, 1, 1])
        elif config.get("num_params") == 3:
            self.scales = np.array([1, 1, 1])
        
        self.population = None
        self.ind = None
        self.transforms = []
        self.score = []
        self.best = []
        
    
    def run_ga(self):
        self.initialize_population()
        
        max_gen = self.config.get("max_generations")
        generation = 0
        condition = True
        with tqdm(total = max_gen) as pbar:
            while condition:
                
                if generation >= 1:
                    self.selection()
                    self.cross_over()
                    self.mutation()
                
                self.fitness()
                generation += 1
                
                if generation >= max_gen:
                    condition = False
                if generation >= 2:
                    b2 = np.array(self.best[-2])
                    b1 = np.array(self.best[-1])
                    if (np.abs(b1 - b2) < self.config.get("epsilon")).all():
                        condition = False
                        print('early stopping initiated.')
                
                
                if generation == 1:
                    plt.axis([1, self.config.get("max_generations"), 0, 0.5])
                    plt.title("RMSE over generations")
                    plt.xlabel("Generations")
                    plt.ylabel("RMSE (m)")
                elif generation > 1:
                    plt.plot([generation - 1, generation], 
                             [self.score[-2], self.score[-1]], color = 'blue')
                    plt.pause(0.05)
                    
                sleep(0.05)
                pbar.update(1)
        
        plt.show()
        self.best = [item  * self.scales for item in self.best]


    def initialize_population(self):
        ''' 
            function to randomly create the initial population using config
            values. Population is of the size [n, m], each paramter is one
            column in this array, and its values are within specified bounds.
            First 3 genes are rotations, next 3 are translations
        '''
        n = self.config.get("population_size")
        m = self.config.get("num_params")
        b = self.config.get("bounds")
        self.population = np.random.uniform(b[:, 0], b[:, 1], (n, m))
        
    
    def selection(self):
        if self.config.get("selection") == "random":
            ns = int((self.config.get("population_size") * self.config.get("selection_rate")) / 2)
            self.ind = np.random.permutation(ns)
    
    
    def cross_over(self):
        if self.config.get("cross_over") == "one_point":
            k = len(self.ind)
            n = self.config.get("population_size")
            m = self.config.get("num_params")
            idx = np.arange(k, n)
            
            ch = self.population.copy()
            p1 = ch[self.ind].tolist()
            p2 = ch[idx].tolist()
            
            r = np.random.randint(1, m-1, k)        # Ensuring first and last genne won't be the cross over point
            os1, os2 = np.zeros((k, m)), np.zeros((k, m))
            for i in range(k):
                os1[i] = p1[i][:r[i]] + p2[i][r[i]:]
                os2[i] = p2[i][:r[i]] + p1[i][r[i]:]
            self.population = np.vstack((os1, os2))
    
    
    def mutation(self):
        ''' 
            function to perform mutation. Which is done by randomly creating 
            new chromosomes (similar to initialize_population routine). The
            number of mutated chromosomes is 10% of population size
        '''
        n = self.config.get("population_size")
        m = self.config.get("num_params")
        b = self.config.get("bounds")
        nn = int(np.floor(n * 0.1))
        new_ch = np.random.uniform(b[:, 0], b[:, 1], (nn, m))
        
        inds = np.random.permutation(n)
        self.population[inds.tolist()[:nn]] = new_ch
    
    
    def get_transforms(self):
        self.transforms = []
        for i in range(self.config.get("population_size")):
            x = self.population[i, :] * self.scales
            R = np.eye(3)
            T = np.zeros(3)
            R[0, 0] =  np.cos(x[2]) * np.cos(x[1])
            R[0, 1] = -np.sin(x[2]) * np.cos(x[0]) + np.cos(x[2]) * np.sin(x[1]) * np.sin(x[0])
            R[0, 2] =  np.sin(x[2]) * np.sin(x[0]) + np.cos(x[2]) * np.sin(x[1]) * np.cos(x[0])
            R[1, 0] =  np.sin(x[2]) * np.cos(x[1])
            R[1, 1] =  np.cos(x[2]) * np.cos(x[0]) + np.sin(x[2]) * np.sin(x[1]) * np.sin(x[0])
            R[1, 2] = -np.cos(x[2]) * np.sin(x[0]) + np.sin(x[2]) * np.sin(x[1]) * np.cos(x[0])
            R[2, 0] = -np.sin(x[1])
            R[2, 1] =  np.cos(x[1]) * np.sin(x[0])
            R[2, 2] =  np.cos(x[1]) * np.cos(x[0])
            T[0] = x[3]
            T[1] = x[4]
            T[2] = x[5]

            transform = np.eye(4)
            transform[:3, :3] = R
            transform[:3, 3] = T
            self.transforms.append(transform)
    
    
    
    def fitness(self):
        if self.config.get("num_params") == 6:
            self.rigid_body_unscaled()
        elif self.config.get("num_params") == 3:
            self.translation_only()
        else:
            print("incorrect number of parameters in config")
            
                
    
    
    def translation_only(self):
        # USE FAISS to Build a tree index for fast correspondence search
        d = 3               # Dimension of the point cloud
        k = 1               # Num closest points to search for
        
        index = faiss.IndexFlatL2(d)
        index.add(self.fixed)
        
        D, I = index.search(self.moving, k)
        I = I.reshape(len(I), )

        rmse = []
        for i in range(self.config.get("population_size")):
            transform = self.population[i, :] * self.scales
            temp_moving = self.moving + np.asarray(transform)
            
            res = np.sum((temp_moving - self.fixed[I]) * self.normal[I], axis = 1)
            
            a=np.where((res > (np.mean(res) - (3 * np.std(res)))) & (res < (np.mean(res) + (3 * np.std(res)))))
            rmse.append(np.sqrt(np.sum(res[a[0]]**2) / len(I[a[0]])))
            # rmse.append(np.sqrt(np.sum(res**2) / len(I)))
        
        rmse = np.asarray(rmse)
        self.score.append(np.min(rmse))      # Not needed anymore
        self.best.append(self.population[np.argmin(rmse)])
        


    def rigid_body_unscaled(self):
        # USE FAISS to Build a tree index for fast correspondence search
        d = 3               # Dimension of the point cloud
        k = 1               # Num closest points to search for
        
        index = faiss.IndexFlatL2(d)
        index.add(self.fixed)
        
        D, I = index.search(self.moving, k)
        I = I.reshape(len(I), )


        self.get_transforms()
        rmse = []
        for i in range(self.config.get("population_size")):
            
            P = np.hstack((self.moving, np.ones((self.moving.shape[0], 1))))
            transform = self.transforms[i]
            temp_moving = np.transpose(transform @ P.T)[:, 0:3]
            
            # D, I = index.search(temp_moving, k)
            # I = I.reshape(len(I), )
            
            res = np.sum((temp_moving - self.fixed[I]) * self.normal[I], axis = 1)
            rmse.append(np.sqrt(np.sum(res**2) / len(I)))
        
        rmse = np.asarray(rmse)
        self.score.append(np.min(rmse))      # Not needed anymore
        self.best.append(self.population[np.argmin(rmse)])
        
        
        
        # b1 = np.sum(self.fixed * self.normals, axis=1)
        # b2 = np.sum(self.moving * self.normals, axis=1)
        # b = np.expand_dims(b1 - b2, axis=1)
        # A1 = np.cross(self.moving, self.normals)
        # A2 = normals
        # A = np.hstack((A1, A2))

        # # If weighted
        # # x = np.linalg.inv(A.T @ weights @ A) @ A.T @ weights @ b

        # # If not weighted
        # # x = np.linalg.inv(A.T @ A) @ A.T @ b
        
        
        # misc = np.zeros(D.shape)
        # for i in range(len(D)):
        #     Normal = self.normal[:, I[i]]

        #     Vec_Diff = (self.moving[i,:] - self.fixed[I[i,0],:])
        #     Vec_DiffT = Vec_Diff.transpose()
        #     misc[i,0] = Vec_DiffT.dot(Normal)

        # A = self.fixed[I]

            

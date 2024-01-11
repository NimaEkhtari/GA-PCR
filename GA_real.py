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
from sklearn.neighbors import KDTree
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
        
        self.Pop = None
        self.Fit = None
        self.fitness = None
        self.population = None
        self.p1_ind = None
        self.p2_ind = None
        self.transforms = []
        self.score = []
        self.best = None
        self.generation = 0
        self.bests = []
        
    
    def run_ga(self):
        
        self.initialize_population()

        max_gen = self.config.get("max_generations")
        condition = True
        with tqdm(total = max_gen) as pbar:
            while condition:
                
                if self.generation >= 1:
                    # ax = plt.figure().add_subplot(projection='3d')
                    # x = self.population[:, 0]
                    # y = self.population[:, 1]
                    # z = self.population[:, 2]
                    # ax.scatter(x, y, z, c =  self.config.get("population_size") * ['b'])

                    self.selection()
                    # for i in range(round(self.config.get("population_size") / 2)):
                            
                        # x = [self.population[self.p1_ind[i], 0], self.population[self.p2_ind[i], 0]]
                        # y = [self.population[self.p1_ind[i], 1], self.population[self.p2_ind[i], 1]]
                        # z = [self.population[self.p1_ind[i], 2], self.population[self.p2_ind[i], 2]]
                        # ax.plot(x, y, zs=z)
                    
                    self.cross_over()
                    # x = self.population[:, 0]
                    # y = self.population[:, 1]
                    # z = self.population[:, 2]
                    # ax.scatter(x, y, z, c =  self.config.get("population_size") * ['r'])
                    # plt.show()
                    
                    self.mutation()
                    # x = self.population[:, 0]
                    # y = self.population[:, 1]
                    # z = self.population[:, 2]
                    # ax.scatter(x, y, z, c =  self.config.get("population_size") * ['g'])
                    # plt.show()
                    
                    self.carry_best()
                
                self.calc_fitness()
                self.bests.append(self.best)
                self.generation += 1
                
                # if (np.mod(self.generation + 1, 10) == 0) | (self.generation == 0):
                    # ax = plt.figure().add_subplot(projection='3d')
                    # x = self.population[:, 0]
                    # y = self.population[:, 1]
                    # z = self.population[:, 2]
                    # ax.scatter(x, y, z, c =  self.config.get("population_size") * ['b'])
                    # plt.show()
                
                
                
                if self.generation >= max_gen:
                    condition = False
                if self.generation >= 2:
                    b2 = np.array(self.best[-2])
                    b1 = np.array(self.best[-1])
                    if (np.abs(b1 - b2) < self.config.get("epsilon")).all():
                        condition = False
                        print('early stopping initiated.')
                
                
                if self.generation == 1:
                    plt.axis([1, self.config.get("max_generations"), 0, 0.5])
                    plt.title("RMSE over generations")
                    plt.xlabel("Generations")
                    plt.ylabel("RMSE (m)")
                elif self.generation > 1:
                    plt.plot([self.generation - 1, self.generation], 
                              [self.score[-2], self.score[-1]], color = 'blue')
                    plt.pause(0.05)
                    
                sleep(0.05)
                pbar.update(1)
        
        plt.show()
        self.best = self.best * self.scales
        # self.best = [item  * self.scales for item in self.best]
        


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
            
        if self.config.get("selection") == "roulette wheel":
            n = self.config.get("population_size")
            N = int(np.floor(n / 2))
            Fit = 1 / (self.fitness + self.config.get("epsilon"))
            probs = Fit / np.sum(Fit)
            inds = np.random.choice(n, (N, 2), p = probs)

            self.p1_ind = inds[:, 0]
            self.p2_ind = inds[:, 1]
    



    
    def cross_over(self):
        n = self.config.get("population_size")
        m = self.config.get("num_params")
        mg = self.config.get("max_generations")
        g = self.generation
        M = 0.4 * np.exp(-4 * (g / mg))
        
        t1 = np.random.normal(0, M, (n * 2, m))
        t2 = np.random.permutation(t1)
        t3 = t2[0 : int(n/2), :]
        t4 = t2[int(n/2) : n, :]
        
        P1 = self.population[self.p1_ind, :]
        P2 = self.population[self.p2_ind, :]
        O1 = P1 + (t3 * (P2 - P1))
        O2 = P2 + (t4 * (P2 - P1))
        # O1 = P1 + (t3[:, np.newaxis] * (P2 - P1))
        # O2 = P2 + (t4[:, np.newaxis] * (P2 - P1))
        
        self.population = np.vstack((O1, O2))

    
    
    def mutation(self):
        ''' 
            function to perform mutation. Which is done by randomly creating 
            new chromosomes (similar to initialize_population routine). The
            number of mutated chromosomes is 10% of population size
        '''
        n = self.config.get("population_size")
        m = self.config.get("num_params")
        r = self.config.get("mutation_rate")
        b = self.config.get("bounds")
        nn = int(np.floor(n * m * r))
        
        
        inds = np.random.permutation(int(n * m))
        ind = inds[:nn]
        
        # Adaptive mutation
        mg = self.config.get("max_generations")
        g = self.generation
        M = 0.5 * np.exp(-1.5 * (g / mg))
        mute = np.random.normal(0, M, int(n * m))
        
        q = self.population.reshape(int(n * m), )
        q[ind] = q[ind] + mute[ind]
        
        self.population = q.reshape((n, m))

    
    
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
    
    
    
    def calc_fitness(self):
        if self.config.get("num_params") == 6:
            self.rigid_body_unscaled()
        elif self.config.get("num_params") == 3:
            self.translation_only()
        else:
            print("incorrect number of parameters in config")
            
                

    def carry_best(self):
        n = self.config.get("population_size")
        
        ind = np.random.randint(0, n, 1)[0]
        self.population[ind] = self.best_ch
        

    
    def translation_only(self):
        # USE FAISS to Build a tree index for fast correspondence search
        d = 3               # Dimension of the point cloud
        k = 1               # Num closest points to search for
        
        # kdtree = KDTree(self.fixed)
        translations = self.population
        
        rmse = []
        for i in range(self.config.get("population_size")):
            transform = translations[i, :] * self.scales
            temp_moving = self.moving + np.asarray(transform)
            
            kdtree = KDTree(temp_moving)
            D, I = kdtree.query(self.fixed, k = k)
            I = np.ravel(I)
            # I = I.reshape(len(I), )
            
            
            res = np.sum((temp_moving[I] - self.fixed) * self.normal[I], axis = 1)
            rmse.append(np.sqrt(np.sum(res**2) / len(I)))
        
        rmse = np.asarray(rmse)
        self.fitness = rmse
        self.score.append(np.min(rmse))      # Not needed anymore
        self.ind_best = np.argmin(rmse)
        self.best = translations[self.ind_best]
        self.best_ch = self.population[self.ind_best]


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

            


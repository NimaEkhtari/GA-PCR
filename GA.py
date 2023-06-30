# -*- coding: utf-8 -*-
"""
Created on Sun Jun 11 14:59:26 2023

@author: nekhtari
"""
import numpy as np
import faiss
import random
from tqdm import tqdm
from time import sleep
import matplotlib.pyplot as plt
import tools


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
        self.best = None
        self.best_ch = None
        self.ind_best = 0
        self.best_transform = None
        
    
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
                    self.carry_best()
                
                self.fitness()
                generation += 1
                
                if generation >= max_gen:
                    condition = False
                
                
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
        self.best = self.best * self.scales




    def initialize_population(self):
        ''' 
            function to randomly create the initial population using config
            values. Population is a list of the size [n, ], each element is one
            binary string of size m * b , and its values are concatenated
            binary representations of parameters of the solution in population.
            When config.num_params is 6, the first 3 genes are rotations, 
            next 3 are translations.
            When config.num_params is 3, the 3 parameters are Tx, Ty, Tz
        '''
        n = self.config.get("population_size")
        m = self.config.get("num_params")
        b = self.config.get("bounds")
        P = np.random.uniform(b[:, 0], b[:, 1], (n, m))
        b = self.config.get("num_bits")
        self.population = tools.to_binary(P, b)
        
    
    def selection(self):
        if self.config.get("selection") == "random":
            ns = int((self.config.get("population_size") * self.config.get("selection_rate")) / 2)
            self.ind = np.random.permutation(ns)
    


    
    def cross_over(self):
        if self.config.get("cross_over") == "one_point":
            k = len(self.ind)
            n = self.config.get("population_size")
            m = self.config.get("num_params")
            b = self.config.get("num_bits")
            l = int(m * b)
            idx = np.arange(k, n)
            
            ch = self.population.copy()
            p1 = [ch[i] for i in self.ind]
            p2 = [ch[i] for i in idx]
            

            r = np.random.randint(1, l-1, k)        # Ensuring first and last genne won't be the cross over point
            os1, os2 = [], []
            for i in range(k):
                os1.append(p1[i][:r[i]] + p2[i][r[i]:])
                os2.append(p2[i][:r[i]] + p1[i][r[i]:])
            next_gen = os1 + os2
            self.population = next_gen
    



    
    def mutation(self):
        ''' 
            fThis function performs the bit flip mutation in which 1/m*b 
            chromosomes are randomly selected to have a bit flipped.
        '''
        n = self.config.get("population_size")
        m = self.config.get("num_params")
        b = self.config.get("num_bits")
        l = int(m * b)
        mp = (1 / l) * 20    # Mutation probability set to 1 over length of chromosome
        
        num_mutate = int(np.ceil(mp * n))
        ind_ch = np.random.randint(0, n, num_mutate)
        ind_ge = np.random.randint(0, l, num_mutate)
        c = 0
        for i in ind_ch:
            s = self.population[i]
            if s[ind_ge[c]] == '0':
                self.population[i] = s[: ind_ge[c]] + '1' + s[ind_ge[c] + 1 :]
            else:
                self.population[i] = s[: ind_ge[c]] + '0' + s[ind_ge[c] + 1 :]
            c += 1

    

    def carry_best(self):
        n = self.config.get("population_size")
        
        ind = np.random.randint(0, n, 1)[0]
        self.population[ind] = self.best_ch

 

   
    def get_transforms(self):
        
        population = tools.to_array(self.population, self.config.get("num_bits"))
        self.transforms = []
        for i in range(self.config.get("population_size")):
            x = population[i, :] * self.scales
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
        
        # Starting from the second generation, move the point cloud according to last best
        if self.best is None:
            temp_moving = self.moving
        else:
            temp_moving = self.moving + np.asarray(self.best)
        
        # D, I = index.search(temp_moving, k)
        # I = I.reshape(len(I), )
        
        translations = tools.to_array(self.population, self.config.get("num_bits"))
        
        rmse = []
        for i in range(self.config.get("population_size")):
            transform = translations[i, :] * self.scales
            temp_moving = self.moving + np.asarray(transform)
            
            D, I = index.search(temp_moving, k)
            I = I.reshape(len(I), )
            
            
            res = np.sum((temp_moving - self.fixed[I]) * self.normal[I], axis = 1)
            # a = np.where((res > (np.mean(res) - (3 * np.std(res)))) & (res < (np.mean(res) + (3 * np.std(res)))))
            # rmse.append(np.sqrt(np.sum(res[a[0]] ** 2) / len(I[a[0]])))
            
            a = np.where((D > np.percentile(D, 90)) | (D < np.percentile(D, 10)))
            rmse.append(np.sqrt(np.sum(res[a[0]] ** 2) / len(I[a[0]])))
            # rmse.append(np.sqrt(np.sum(res**2) / len(I)))
        
        rmse = np.asarray(rmse)
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

        if self.best is None:
            temp_moving = self.moving
        else:
            P = np.hstack((self.moving, np.ones((self.moving.shape[0], 1))))
            temp_moving = np.transpose(self.best_transform @ P.T)[:, 0:3]
        
        # D, I = index.search(temp_moving, k)
        # I = I.reshape(len(I), )


        self.get_transforms()
        rmse = []
        for i in range(self.config.get("population_size")):
            
            P = np.hstack((self.moving, np.ones((self.moving.shape[0], 1))))
            transform = self.transforms[i]
            temp_moving = np.transpose(transform @ P.T)[:, 0:3]
            
            D, I = index.search(temp_moving, k)
            I = I.reshape(len(I), )
            
            res = np.sum((temp_moving - self.fixed[I]) * self.normal[I], axis = 1)
            
            a = np.where((D > np.percentile(D, 90)) | (D < np.percentile(D, 10)))
            rmse.append(np.sqrt(np.sum(res[a[0]] ** 2) / len(I[a[0]])))
            
            # a = np.where((res > (np.mean(res) - (3 * np.std(res)))) & (res < (np.mean(res) + (3 * np.std(res)))))
            # rmse.append(np.sqrt(np.sum(res[a[0]] ** 2) / len(I[a[0]])))
        
        rmse = np.asarray(rmse)
        self.score.append(np.min(rmse))      # Not needed anymore
        self.ind_best = np.argmin(rmse)
        
        self.best = tools.to_array(self.population[self.ind_best], self.config.get("num_bits"))
        
        self.best_ch = self.population[self.ind_best]
        self.best_transform = self.transforms[self.ind_best]
        
        
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

            



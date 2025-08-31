import numpy as np
import pandas as pd
from time import time, sleep
import datetime
from collections.abc import Iterable
import math
from typing import Dict, Tuple, List, Optional, Any
from tqdm import trange

import Population as p
import Constants as c


class AcorContinuousDomain(object):
    """
    Ant Colony Optimization for Continuous Domains (ACOR) in Python.

    This code is used to implement the meta-heuristic algorithm know as Ant Colony Optimization 
    to solve the optimization (minimization/minimization) of bounded continous-domain functions

    This code is a python port of the Matlab implementation provided Kalami Heris and further details can be found in this reference:

        Mostapha Kalami Heris, ACO for Continuous Domains in MATLAB (URL: https://yarpiz.com/67/ypea104-acor), Yarpiz, 2015.

    Also further reference was made to the Indra Chandra Joseph Riadi thesis that can be found here:

        http://usir.salford.ac.uk/30721/1/Riadi_Thesis_2014_jan.pdf

    """

    def __init__(self, 
            n_pop: int, 
            n_vars: int,
            cost_func: object, 
            domain_bounds: Dict):
        """
        Constructor
        :param: n_pop: population size
        :param: n_vars: number of variables
        :param: cost_func_ cost function
        :param: domain_bounds: Continuous domain lower/upper bounds
        """
        self.__domain_bounds = domain_bounds
        self.__l_bound = domain_bounds[0]
        self.__u_bound = domain_bounds[1]
        self.__cost_func = cost_func
        self.__n_pop = n_pop
        self.__pops_sorted = None
        self.__n_vars = n_vars
        self.__n_ants = c.AcoConstants.N_ANTS
        self.__max_terations = c.AcoConstants.MAX_ITERATIONS
        self.__q = c.AcoConstants.Q
        self.__zeta = c.AcoConstants.ZETA
        self.__random = np.random.RandomState(c.HelperConstants.RANDOM_SEED)
        self.__final_best_solution = None 
        self.__best_solutions = [None]*self.__max_terations
        self.__probs = None
        self.__new_pops = None   

    def __initialization(self):
        """
        Initialization of the ACO algorithm
        """
        pops = p.Populations(self.__n_pop, self.__n_vars, self.__cost_func, self.__domain_bounds)
        self.__pops_sorted = pops.ant_populations
        self.__final_best_solution = pops.best_population
        self.__w = self.__computePdf()
        self.__probs = self.__w/np.sum(self.__w)
        self.__means = np.zeros((self.__n_pop, self.__n_vars))
        self.__sigmas = np.zeros((self.__n_pop, self.__n_vars))

    def initialization(self):
        """
        Wrapper method (for unit testing) of Initialization of the ACO algorithm
        """
        self.__initialization()

    def __computePdf(self) -> object:
        """
        Computes the PDF values
        """
        points = np.array(range(self.__n_pop), dtype=float)
        # Solution Weights
        w = 1/(np.sqrt(2*np.pi)*self.__q*float(self.__n_pop))*np.square(np.exp(-0.5*((points-1)/(self.__q*float(self.__n_pop)))))
        return w

    def computePdf(self) -> object:
        """
        Wrapper method (for unit testing) of computation of the PDF values
        """
        self.__computePdf()
        

    def __createMeans(self):
        """
        Creates the ACO algorithm means 
        """
        for i in range(self.__n_pop):
            self.__means[i, :] = np.reshape(self.__pops_sorted[i].position, (1, -1))  

    def createMeans(self):
        """
        Wrapper method (for unit testing) of the creation of the ACO algorithm means 
        """ 
        self.__createMeans()    

    def __createStandardDeviation(self):
        """
        Creates the ACO algorithm Satndard Deviation (sigmas)
        """
        for l_i in range(self.__n_pop):
            d = 0.0
            for r_i in range(self.__n_pop):
                d += np.abs(self.__means[l_i, :] - self.__means[r_i, :])
            self.__sigmas[l_i, :] = (self.__zeta * d) / (self.__n_pop - 1) 

    def createStandardDeviation(self):
        """
        Wrapper method (for unit testing) of the creation of the ACO algorithm Satndard Deviation (sigmas)
        """
        self.__createStandardDeviation()

    def __rouletteWheelSelection(self):
        """
        Roulette wheel selection strategy for selecting the optimal Guassain Kernel
        """
        r = self.__random.rand()
        c = np.cumsum(np.reshape(self.__probs, (-1)))
        j = np.argwhere(r <= c)[0,0]
        return j

    def rouletteWheelSelection(self):
        """
        Wrapper method (for unit testing) of the creation of Roulette wheel selection strategy for selecting the optimal Guassain Kernel
        """
        return self.__rouletteWheelSelection()

    def __constructNewPopulationSolution(self):
        """
        Constructs the new ACO Population solution
        """        
        self.__createMeans()
        self.__createStandardDeviation()
        self.__new_pops = p.Populations.createEmptyNewPopulations(self.__n_ants, self.__n_vars)
        for i in range(self.__n_ants):
            for j in range(self.__n_vars):
                # Select Gaussian Kernel
                k = self.__rouletteWheelSelection()
                # Generate Gaussian Random Variable
                self.__new_pops[i].position[j] = self.__means[k, j] + self.__sigmas[k, j] * self.__random.randn()
            # Apply Variable Bounds
            self.__new_pops[i].position = np.maximum(self.__new_pops[i].position, self.__l_bound)
            self.__new_pops[i].position = np.minimum(self.__new_pops[i].position, self.__u_bound)
            # Evaluation
            self.__new_pops[i].cost_function = self.__cost_func(self.__new_pops[i].position)

    def constructNewPopulationSolution(self):
        """
        Wrapper method (for unit testing) of the creation of the new ACO Population solution
        """
        self.__constructNewPopulationSolution()      
                
    def runMainLoop(self):
        """
        Executes ACO algorithm main loop
        """
        self.__initialization()
        for i in trange(self.__max_terations):
          self.__constructNewPopulationSolution()
          # Merge Main Population (Archive) and New Population (Samples)
          self.__pops_sorted = self.__pops_sorted + self.__new_pops
          # Sort Population
          self.__pops_sorted = sorted(self.__pops_sorted, key=lambda x: x.cost_function, reverse=False)
          # Delete Extra Members
          self.__pops_sorted = self.__pops_sorted[:self.__n_pop]
          # Update Best Solution Ever Found
          self.__final_best_solution = self.__pops_sorted[0]
          # Store Best Cost
          self.__best_solutions[i] = self.__final_best_solution
    
    @property
    def pops(self):
        """
        Getter property of the 'self.__pops' attribute
        """
        return self.__pops_sorted

    @property
    def new_pops(self):
        """
        Getter property of the 'self.__new_pops' attribute
        """
        return self.__new_pops

    @property
    def final_best_solution(self):
        """
        Getter property of the 'self.__best_solution' attribute
        """
        return self.__final_best_solution

    @property
    def best_solutions(self):
        """
        Getter property of the 'self.__best_solutions' attribute
        """
        return self.__best_solutions

    @property
    def probs(self):
        """
        Getter property of the 'self.__probs' attribute
        """
        return self.__probs

    @property
    def means(self):
        """
        Getter property of the 'self.__meanss' attribute
        """
        return self.__means

    @property
    def sigmas(self):
        """
        Getter property of the 'self.__sigmas' attribute
        """
        return self.__sigmas


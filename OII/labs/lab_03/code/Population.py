from typing import Dict, Tuple, List, Optional, Any
import numpy as np

import Constants as c

class Population(object):
    """
    Specifies an ACO Population
    """
    def __init__(self,
        position, 
        cost_function) -> None:
        """
        Constructor
        """
        self.position = position
        self.cost_function = cost_function


class Populations(object):
    """
    Specifies Populations of the Ant Colony i.e. the Archive Size
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
        :param: domain_bounds: continious domain lower/upper bounds
        """
        self.__l_bound = domain_bounds[0]
        self.__u_bound = domain_bounds[1]
        self.__cost_func = cost_func
        self.__n_pop = n_pop
        self.__pops = [None]*n_pop
        self.__pops_sorted = None
        self.__best_pop = None
        self.__n_vars = n_vars
        self.__random = np.random.RandomState(c.HelperConstants.RANDOM_SEED)        

    def __initializePopulation(self):
        """
        Initializes the ant colony populations
        """
        positions = np.random.uniform()
        for i in range(self.__n_pop):
            position = self.__random.uniform(self.__l_bound, self.__u_bound, (self.__n_vars, 1))
            cost = self.__cost_func(position)
            self.__pops[i] = Population(position=position, cost_function=cost)

    @property
    def ant_populations(self):
        """
        Getter property to retrieve the 'self.__pops' collection
        """
        self.__initializePopulation()        
        self.__pops_sorted = sorted(self.__pops, key=lambda x: x.cost_function, reverse=False)
        return self.__pops_sorted

    @property
    def best_population(self):
        """
        Getter property to retrieve the best 'self.__pops'
        """
        return self.__pops_sorted[0]

    @staticmethod
    def createEmptyNewPopulations(n_ants: int, n_vars: int) -> object:
        """
        Creates an empty collection of ant populations
        :params: n_ants: number of ants
        :params: n_vars: number of variables
        """
        position = np.zeros((n_vars, 1))
        empty_populations = [Population(position=position, cost_function=None) for _ in range(n_ants)]
        return empty_populations


    


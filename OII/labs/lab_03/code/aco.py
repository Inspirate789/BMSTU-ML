from math import sin, cos, pow, pi
import numpy as np

import Population, Acor, Constants
from time import process_time

def f(x: float) -> float:
    return sin(x)*(sin(x)+cos(x))

def g(x: float) -> float:
    return x**3 - x + 3

polynomial_power = 5
func = f
x_start, x_end = -2*pi, 2*pi
x_step = 0.001

x = [i for i in np.arange(x_start, x_end + x_step/2, x_step)]
function_inputs = [[pow(arg, polynomial_power - i) for i in range(polynomial_power + 1)] for arg in x]

def error(solution):
    deviation = 0
    for i in range(len(x)):
        desired_output = func(x[i])
        actual_output = np.sum(solution*function_inputs[i])
        deviation += (desired_output - actual_output)**2
    return (deviation/len(x))**0.5

def cost(function_input):
    return -func(function_input[0])


start_time = process_time()
acor = Acor.AcorContinuousDomain(n_pop=Constants.AcoConstants.N_POP, 
                                 n_vars=polynomial_power+1,
                                 cost_func=error,
                                 domain_bounds=[-20, 20])
acor.runMainLoop()
print("Time: {diff} seconds".format(diff = process_time() - start_time))
print("Parameters of the best solution : {solution}".format(solution=acor.final_best_solution.position))
print("Cost value of the best solution = {solution_cost}".format(solution_cost=acor.final_best_solution.cost_function))

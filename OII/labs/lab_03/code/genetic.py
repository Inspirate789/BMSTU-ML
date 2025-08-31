from math import sin, cos, pow, pi
import numpy as np
import pygad
from time import process_time

def f(x: float) -> float:
    return sin(x)*(sin(x)+cos(x))

def g(x: float) -> float:
    return x**3 - x + 3

polynomial_power = 3
func = g
x_start, x_end = -5, 5
x_step = 0.001

x = [i for i in np.arange(x_start, x_end + x_step/2, x_step)]
function_inputs = [[pow(arg, polynomial_power - i) for i in range(polynomial_power + 1)] for arg in x]

def fitness(_, solution, solution_idx):
    deviation = 0
    for i in range(len(x)):
        desired_output = func(x[i])
        actual_output = np.sum(solution*function_inputs[i])
        deviation += (desired_output - actual_output)**2
    return 1.0 / ((deviation/len(x))**0.5)

start_time = process_time()
ga_instance = pygad.GA(num_generations=1000,
                       num_parents_mating=4,
                       fitness_func=fitness,
                       sol_per_pop=8,
                       num_genes=polynomial_power+1,
                       init_range_low=-2,
                       init_range_high=5,
                       parent_selection_type="sss",
                       keep_parents=1,
                       crossover_type="single_point",
                       mutation_type="random",
                       mutation_percent_genes=10)
ga_instance.run()
solution, solution_fitness, _ = ga_instance.best_solution()
print("Time: {diff} seconds".format(diff = process_time() - start_time))
print("Parameters of the best solution : {solution}".format(solution=solution))
print("Fitness value of the best solution = {solution_fitness}".format(solution_fitness=solution_fitness))


ga_instance.plot_fitness()

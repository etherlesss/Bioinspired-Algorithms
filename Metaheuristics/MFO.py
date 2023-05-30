import random as r
import numpy as np
import math

#
#   Moth-flame optimizer module.
#   With guidance taken from https://github.com/7ossam81/EvoloPy/blob/master/optimizers/MFO.py, adapted to fit our college subject and to keep consistency with the paper.
#   We also turning the original MFO to B-MFO since the Feature Selection is a discrete problem, meanwhile MFO is continuous, but it's done outside in the solver.
#

def MFO(iter, max_iter, dim, N, initial_population, bestSolutions, fitness, BestFitnessArray):
    flame_no = round(N - iter * ((N - 1) / max_iter))

    M = initial_population.copy()
    OM = fitness.copy()

    # If iteration is equal to 0, otherwise is not the first gen
    if (iter == 0):
        # Sort the first moth population
        sorted_fitness = np.sort(OM)
        I = np.argsort(OM)

        sorted_population = M[I, :]

        # Update the flames
        F = sorted_population
        OF = sorted_fitness
    else:
        # Initalize flames
        F = bestSolutions[:flame_no, :]
        OF = BestFitnessArray[:flame_no]

        # Sort Flames
        double_population = np.concatenate((F, M), axis=0)
        double_fitness = np.concatenate((OF, OM), axis=0)
        double_sorted_fitness = np.argsort(double_fitness, axis=0)

        double_population = np.take_along_axis(double_population, double_sorted_fitness, axis=0)
        double_fitness = np.take_along_axis(double_fitness, double_sorted_fitness, axis=0)

        # Update the flames
        F = np.split(double_population, [flame_no], axis=0)[0]
        OF = np.split(double_fitness, [flame_no], axis=0)[0]

    # a linearly decreases from -1 to -2 to calculate t in equation [3.12]
    a = -1 + iter * ((-1) / max_iter)

    # For each moth
    for i in range(0, N):
        # And for each dimension
        for j in range(0, dim):
            if (i <= flame_no):
                # Update D, using equation [3.13], D is the distance of the moth to the flame
                D = abs(sorted_population[i, j] - M[i, j])
                b = 1
                t = (a - 1) * r.random() + 1

                # Update M[i,j] with equation [3.12], with respect to the corresponding moth
                M[i, j] = (D * math.exp(b * t) * math.cos(t * 2 * math.pi) + sorted_population[i, j])

            if (i > flame_no):
                # Update the position of each moth with respect to ONE flame, still with Equation [3.13]
                D = abs(sorted_population[i, j] - M[i, j])
                b = 1
                t = (a - 1) * r.random() + 1

                #  Update M[i,j] with equation [3.12], with respect to the corresponding moth
                M[i, j] = (D * math.exp(b * t) * math.cos(t * 2 * math.pi) + sorted_population[flame_no, j])
    
    return M, F
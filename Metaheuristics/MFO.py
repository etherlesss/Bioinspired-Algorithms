import random
import numpy as np
import math

#
#   Moth-flame optimizer module.
#   With guidance taken from https://github.com/7ossam81/EvoloPy/blob/master/optimizers/MFO.py, adapted to fit our college subject and to keep consistency with the paper.
#   We also turning the original MFO to B-MFO since the Feature Selection is a discrete problem, meanwhile MFO is continuous, but it's done outside in the solver.
#

def MFO(iter, max_iter, dim, N, initial_population, bestSolutions, fitness, BestFitnessArray):
    # Update the number of flames with equation [3.14] from the paper
    flame_no = int(np.ceil(N - (iter + 1) * ((N - 1) / max_iter)))

    # Initialize population of moths and their fitness array
    M = initial_population.copy()
    OM = fitness.copy()

    # If the iteration is equal to 0
    if (iter == 0):
        # Sort the first moth population
        order = OM.argsort(axis=0)
        M = M[order, :]
        OM = OM[order]

        # Update flames
        F = np.copy(M)
        OF = np.copy(OM)
    else:
        # Initialize flames
        F = bestSolutions[:flame_no, :]
        OF = BestFitnessArray[:flame_no]

        # Sort the moths
        double_population = np.vstack((F, M))
        double_fitness = np.hstack((OF, OM))

        order = double_fitness.argsort(axis=0)
        double_population = double_population[order, :]
        double_fitness = double_fitness[order]

        # Update flames
        F = double_population[:flame_no, :]
        OF = double_fitness[:flame_no]

    # r linearly decreases from -1 to -2 to calculate t in equation [3.12]
    r = -1 + iter * ((-1) / max_iter)

    # Prepare population for the loop
    double_population = np.vstack((F, M))

    # For each moth
    for i in range(0, N):
        # And for each dimension
        for j in range(0, dim):
            if (i <= flame_no):
                # Update D, using equation [3.13], D is the distance of the moth to the flame
                D = abs(double_population[i, j] - M[i, j])
                b = 1
                t = (r - 1) * random.random() + 1

                # Update M[i,j] with equation [3.12], with respect to the corresponding moth
                M[i, j] = (D * math.exp(b * t) * math.cos(t * 2 * math.pi) + double_population[i, j])

            if (i > flame_no):
                # Update the position of each moth with respect to ONE flame, still with Equation [3.13]
                D = abs(double_population[i, j] - M[i, j])
                b = 1
                t = (r - 1) * random.random() + 1

                #  Update M[i,j] with equation [3.12], with respect to the corresponding moth
                M[i, j] = (D * math.exp(b * t) * math.cos(t * 2 * math.pi) + double_population[flame_no, j])

    return M, F
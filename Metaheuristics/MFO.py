import random
import numpy as np
import math
import time

from Operations.read import readDataset
from Problem.FS import FeatureSelection
from solution import solution

#
#   Moth-flame optimizer module.
#   With guidance taken from https://github.com/7ossam81/EvoloPy/blob/master/optimizers/MFO.py, adapted to fit our college subject.
#   We also turning the original MFO to B-MFO since the Feature Selection is a discrete problem, meanwhile MFO is continuous.
#

def MFO(path:str):
    max_iter = 0 # Max iterations
    lb = -100 # Lower bounds
    ub = 100 # Upper bounds
    dim = len(readDataset.getInstance.columns) # The number of dimensions is based on the population (which are the columns)
    N = 50 # Number of moths (search agents)

    if not isinstance(lb, list):
        lb = [lb] * dim
    if not isinstance(ub, list):
        ub = [ub] * dim

    # Initialize the position of moths (I function on the paper) M is the moth matrix, OM is the array of fitness values of each moth
    M = np.zeros((N, dim))
    for i in range(dim):
        M[:, i] = np.random.uniform(0, 1 , N) * (ub[i] - lb[i] + lb[i])
    OM = np.full(N, float("inf"))

    # S is the convergence curve
    S = np.zeros(max_iter)

    sorted_population = np.copy(M)
    sorted_fitness = np.zeros(N)

    # F are the flames, OF is the array of fitness values of each flame
    F = np.copy(M)
    OF = np.zeros(N)

    double_population = np.zeros((2 * N, dim))
    double_fitness = np.zeros((2 * N))

    double_sorted_population = np.zeros((N, dim))
    double_sorted_fitness = np.zeros((N, dim))

    previous_population = np.zeros((N, dim))
    previous_fitness = np.zeros((N, dim))

    s = solution()

    print(f"MFO is optimizing {path}")

    # START TIMER
    start_timer = time.time()
    start_time = time.strftime("%Y-%m-%d-%H-%M-%S")

    # Main loop (P Function)
    for i in range(1, max_iter):
        # Update number of flames with equation [3.14] from the paper
        flame_no = round(N - i * ((N - 1) / max_iter))

        for i in range(0, N):
            # Check if moths go out of bounds from the search space
            for j in range(dim):
                M[i,j] = np.clip(M[i, j], lb[j], ub[j])
            
            # Evaluate moths
            OM[i] = FeatureSelection.fitness(M[i, :])

        if (i == 1): # If iteration is equal to 1
            # Sort the first moth population
            sorted_fitness = np.sort(OM)
            I = np.argsort(OM)

            sorted_population = M[I, :]

            # Update the flames
            F = sorted_population
            OF = sorted_fitness
        else:
            # Sort the moths
            double_population = np.concatenate((previous_population, F), axis = 0)
            double_fitness = np.concatenate((previous_fitness, OF), axis = 0)
            double_sorted_fitness = np.sort(double_fitness)
            I2= np.argsort(double_fitness)

            for newindex in range(0, 2 * N):
                double_sorted_population[newindex, :] = np.array(double_population[I2[newindex], :])

            sorted_fitness = double_sorted_fitness
            sorted_population = double_sorted_population

            # Update the flames
            F = sorted_population
            OF = sorted_fitness

        # Update the position the best flame obtained so far
        best_flame_score = sorted_fitness[0]
        best_flame_pos = sorted_population[0, :]

        previous_population = M
        previous_fitness = OM

        # a linearly dicreases from -1 to -2 to calculate t in equation [3.12]
        a = -1 + i * ((-1) / max_iter)


        # For each moth
        for i in range(0, N):
            # And for each dimension
            for j in range(0, dim):
                if (i <= flame_no):
                    # Update D, using equation [3.13], D is the distance of the moth to the flame
                    D = abs(sorted_population[i, j] - M[i, j])
                    b = 1
                    t = (a - 1) * random.random() + 1

                    # Update M[i,j] with equation [3.12], with respect to the corresponding moth
                    M[i, j] = (D * math.exp(b * t) * math.cos(t * 2 * math.pi) + sorted_population[i, j])

                if (i > flame_no):
                    # Update the position of each moth with respect to ONE flame, still with Equation [3.13]
                    D = abs(sorted_population[i, j] - M[i, j])
                    b = 1
                    t = (a - 1) * random.random() + 1

                    #  Update M[i,j] with equation [3.12], with respect to the corresponding moth
                    M[i, j] = (D * math.exp(b * t) * math.cos(t * 2 * math.pi) + sorted_population[flame_no, j])

        # BINARIZATION GOES HERE
        
        S[i] = best_flame_score

        # Display best fitness with the iteration
        if (i % 1 == 0):
            print(f"#{str(i)} the fitness is: {str(best_flame_score)}")
        
        i += 1

    # END TIMER
    end_timer = time.time()
    s.endTime = time.strftime("%Y-%m-%d-%H-%M-%S")
    s.executionTime = end_timer - start_timer
    s.convergence = S
    s.optimizer = "MFO"
    s.objfname = path

    return s
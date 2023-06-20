import numpy as np
import time
import matplotlib.pyplot as plt

from Problem.FS import FeatureSelection as FS
from Util.solverPrint import initialPrint, iterationPrint, finalPrint
from Discretization.discretization import applyBinarization
from Metaheuristics.MFO import MFO

def solver(max_iter, population, d_path, discretization, classifier, Cparams):
    # Read dataset
    instance = FS(d_path)

    # Start timer
    start_timer = time.time()

    # Cycle timer
    time_cycle1 = time.time()

    # Generate population
    initial_population = np.random.randint(low=0, high=2, size=(population, instance.getTotalFeature()))

    # Generate fitness array
    fitness                 = np.zeros(population)
    accuracy                = np.zeros(population)
    f1Score                 = np.zeros(population)
    precision               = np.zeros(population)
    recall                  = np.zeros(population)
    mcc                     = np.zeros(population)
    errorRate               = np.zeros(population)
    totalFeatureSelected    = np.zeros(population)

    # Array of ranked solutions
    ranked_solutions = np.zeros(population)

    # Calculate how factible is each individual and calculate initial fitness
    for i in range(initial_population.__len__()):

        if not instance.factibility(initial_population[i]): # Without selected features
            initial_population[i] = instance.newSolution()

        selection = np.where(initial_population[i] == 1)[0]
        fitness[i], accuracy[i], f1Score[i], precision[i], recall[i], mcc[i], errorRate[i], totalFeatureSelected[i] = instance.fitness(selection, classifier, Cparams)

    # Sort ranked solutions
    ranked_solutions = np.argsort(fitness)

    # Save the best one
    bestRS = ranked_solutions[0]

    # Save best solutions
    Best = initial_population[bestRS].copy()
    BestFitness = fitness[bestRS]
    BestAccuracy = accuracy[bestRS]
    BestF1Score = f1Score[bestRS]
    BestPrecision = precision[bestRS]
    BestRecall = recall[bestRS]
    BestMcc = mcc[bestRS]
    BestErrorRate = errorRate[bestRS]
    BestTFS = totalFeatureSelected[bestRS]

    # Save best solutions for our metaheuristic
    BestFitnessArray = fitness[ranked_solutions] 
    accuracyArray                = np.zeros(population)
    f1ScoreArray                 = np.zeros(population)
    precisionArray               = np.zeros(population)
    recallArray                  = np.zeros(population)
    mccArray                     = np.zeros(population)
    errorRateArray               = np.zeros(population)
    totalFeatureSelectedArray    = np.zeros(population)
    bestSolutions = initial_population[ranked_solutions]

    # Generate binary population
    b_matrix = initial_population.copy()

    # Stop cycle timer
    time_cycle2 = time.time()

    # Show initial fitness
    initialPrint(fitness, BestFitness, time_cycle1, time_cycle2, BestAccuracy, BestF1Score, BestPrecision, BestRecall, BestMcc, BestErrorRate, BestTFS)

    # Main Loop
    for iter in range(0, max_iter):
        # Metaheuristic timer
        start_mh_timer = time.time()
        for i in range(bestSolutions.__len__()):
            selection = np.where(bestSolutions[i] == 1)[0]
            BestFitnessArray[i], accuracyArray[i], f1ScoreArray[i], precisionArray[i], recallArray[i], mccArray[i], errorRateArray[i], totalFeatureSelectedArray[i] = instance.fitness(selection, classifier, Cparams)

        # Disturb population
        population = MFO(iter, max_iter, instance.getTotalFeature(), len(initial_population), initial_population, bestSolutions, fitness, BestFitnessArray)

        # Binarization
        for i in range(initial_population.__len__()):
            initial_population[i] = applyBinarization(initial_population[i].tolist(), discretization[0], discretization[1], Best, b_matrix[i].tolist())
        
            # If there's no features selected we make a new one
            if not instance.factibility(initial_population[i]):
                initial_population[i] = instance.newSolution()

            selection = np.where(initial_population[i] == 1)[0]
            fitness[i], accuracy[i], f1Score[i], precision[i], recall[i], mcc[i], errorRate[i], totalFeatureSelected[i] = instance.fitness(selection, classifier, Cparams)

        # Rank solutions once again
        ranked_solutions = np.argsort(fitness)

        # Save best
        if fitness[ranked_solutions[0]] < BestFitness:
            bestIdx = ranked_solutions[0]
            BestFitness = fitness[ranked_solutions[0]]
            Best = initial_population[ranked_solutions[0]]
            BestAccuracy = accuracy[bestIdx]
            BestF1Score = f1Score[bestIdx]
            BestPrecision = precision[bestIdx]
            BestRecall = recall[bestIdx]
            BestMcc = mcc[bestIdx]
            BestErrorRate = errorRate[bestIdx]
            BestTFS = totalFeatureSelected[bestIdx]
        b_matrix = initial_population.copy()

        end_mh_timer = time.time()

        # Print iteration results
        iterationPrint(iter, BestFitness, start_mh_timer, end_mh_timer, BestAccuracy, BestF1Score, BestPrecision, BestRecall, BestMcc, BestErrorRate, BestTFS)

    finalPrint(BestFitness, BestTFS)

    end_timer = time.time()
    exec_time = end_timer - start_timer
    exec_time = round(exec_time, 3)
    print(f"Execution time: {str(exec_time)}s")
    print(f"Result: {str(Best.tolist())}")
    
    moths = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30]
    reverseMoths = np.sort(moths)[::-1]
    sortedAccuracy = np.sort(accuracyArray)[::-1]
    sortedFitness = np.sort(BestFitnessArray)[::-1]


    #accuracy values 
    plt.plot(reverseMoths,sortedAccuracy)
    plt.xlabel("moth n°")
    plt.ylabel("accuracy")
    plt.title("accuracy values")
    plt.show()
    
    #fitness values
    plt.plot(moths,sortedFitness)
    plt.xlabel("moth n°")
    plt.ylabel("fitness")
    plt.title("fitness values")
    plt.show()
    
    plt.plot(sortedAccuracy,sortedFitness)
    plt.xlabel("accuracy")
    plt.ylabel("fitness")
    plt.title("accuracy v/s fitness")
    plt.show()
    # print(f"Result length is {len(Best)}")
import numpy as np

def initialPrint(fitness, bestFitness, time_cycle1, time_cycle2, bestAccuracy, bestF1Score, bestPrecision, bestRecall, bestMcc, bestErrorRate, bestTFS):
    print("------------------------------------------------------------------------------------------------------")
    print(
        f"Initial fitness: {str(fitness)}\n" +
        f"Best initial fitness: {str(np.min(fitness))}"
    )
    print("------------------------------------------------------------------------------------------------------\n")

    print(
        f"Best: {str(bestFitness)} " +
        f"Time: {str(round(time_cycle2 - time_cycle1, 3))}s " +
        f"Acc: {str(bestAccuracy)} " +
        f"Fs: {str(bestF1Score)} " +
        f"Precision: {str(bestPrecision)} " +
        f"Recall: {str(bestRecall)} " +
        f"MCC: {str(bestMcc)} " +
        f"eR: {str(bestErrorRate)} " +
        f"TFS: {str(bestTFS)}\n"
    )

def iterationPrint(iter, bestFitness, mh_time1, mh_time2, bestAccuracy, bestF1Score, bestPrecision, bestRecall, bestMcc, bestErrorRate, bestTFS):
    print(
        f"i: {str(iter + 1)} "+
        f"Best: {str(bestFitness)} " +
        f"Time: {str(round(mh_time2 - mh_time1, 3))}s " +
        f"Acc: {str(bestAccuracy)} " +
        f"Fs: {str(bestF1Score)} " +
        f"Precision: {str(bestPrecision)} " +
        f"Recall: {str(bestRecall)} " +
        f"MCC: {str(bestMcc)} " +
        f"eR: {str(bestErrorRate)} " +
        f"TFS: {str(bestTFS)} "
    )

def finalPrint(bestFitness, bestTFS):
    print("------------------------------------------------------------------------------------------------------")
    print(
        f"Best fitness: {str(bestFitness)}\n" +
        f"Selected features: {str(bestTFS)}"
    )
    print("------------------------------------------------------------------------------------------------------")
from readData import *
import time
import numpy as np
from ML import KNN

def FeatureSelection():
    inicio = time.time()

    def fitness(self, individuo, clasificador, parametrosC):
        accuracy = 0 
        f1Score = 0
        presicion = 0
        recall = 0
        mcc = 0
        trainingData, testingData, trainingClass, testingClass = self.selection(individuo)
        # cm, accuracy, f1Score, presicion, recall, mcc = self.KNN(trainingData, testingData, trainingClass, testingClass)

        accuracy, f1Score, presicion, recall, mcc = KNN(trainingData, testingData, trainingClass, testingClass, int(parametrosC.split(":")[1]))
            
        errorRate = np.round((1 - accuracy), decimals=3)

        fitness = np.round(( self.getGamma() * errorRate ) + ( ( 1 - self.getGamma() ) * ( len(individuo) / self.getTotalFeature() ) ), decimals=3)

        # return fitness, cm, accuracy, f1Score, presicion, recall, mcc, errorRate
        return fitness, accuracy, f1Score, presicion, recall, mcc, errorRate, len(individuo)


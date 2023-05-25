from Operations import read
from ML import KNN, RandomForest, Xgboost

import numpy as np

class FeatureSelection:
    # ATRIBUTES
    def __init__(self) -> None:
        self.__data = None
        self.__classes = None
        self.__trainingData = None
        self.__trainingClass = None
        self.__testingData = None
        self.__testingClass = None
        self.__gamma = 0.99
        self.__totalFeature = None

    # SETTERS
    def setData(self, data):
        self.__data = data
    def setClasses(self, classes):
        self.__classes = classes
    def setTrainingData(self, trainingData):
        self.__trainingData = trainingData
    def setTrainingClass(self, trainingClass):
        self.__trainingData = trainingClass
    def setTestingData(self, testingData):
        self.__testingData = testingData
    def setTestingClass(self, testingClass):
        self.__testingClass = testingClass
    def setGamma(self, gamma):
        self.__gamma = gamma
    def setTotalFeature(self, totalFeature):
        self.__totalFeature = totalFeature

    # GETTERS
    def getData(self):
        return self.__data
    def getClasses(self):
        return self.__classes
    def getTrainingData(self):
        return self.__trainingData
    def getTrainingClass(self):
        return self.__trainingClass
    def getTestingData(self):
        return self.__testingData
    def getTestingClass(self):
        return self.__testingClass
    def getGamma(self):
        return self.__gamma
    def getTotalFeature(self):
        return self.__totalFeature
    
    def readInstance(self):
        self.setClasses = read.readDataset.getSavedClass()
        self.setData = read.readDataset.getInstance()
        self.setTotalFeature(len(self.__data))

    def fitness(self, individual, classificator, parametrosC):
        accuracy = 0 
        f1Score = 0
        presicion = 0
        recall = 0
        mcc = 0
        trainingData, testingData, trainingClass, testingClass = self.selection(individual)
        # cm, accuracy, f1Score, presicion, recall, mcc = self.KNN(trainingData, testingData, trainingClass, testingClass)

        if classificator == 'KNN':
            accuracy, f1Score, presicion, recall, mcc = KNN(trainingData, testingData, trainingClass, testingClass, int(parametrosC.split(":")[1]))
        if classificator == 'RandomForest':
            accuracy, f1Score, presicion, recall, mcc = RandomForest(trainingData, testingData, trainingClass, testingClass)
        
        if classificator == 'Xgboost':
            accuracy, f1Score, presicion, recall, mcc = Xgboost(trainingData, testingData, trainingClass, testingClass)
            
        errorRate = np.round((1 - accuracy), decimals=3)

        fitness = np.round(( self.getGamma() * errorRate ) + ( ( 1 - self.getGamma() ) * ( len(individual) / self.getTotalFeature() ) ), decimals=3)

        # return fitness, cm, accuracy, f1Score, presicion, recall, mcc, errorRate
        return fitness, accuracy, f1Score, presicion, recall, mcc, errorRate, len(individual)
from Util.read import readDataset
from ML.KNN import KNN
from ML.RandomForest import RandomForest
from ML.Xgboost import Xgboost
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

import numpy as np

class FeatureSelection:
    # ATRIBUTES
    def __init__(self, d_path) -> None:
        self.__data = None
        self.__classes = None
        self.__trainingData = None
        self.__trainingClass = None
        self.__testingData = None
        self.__testingClass = None
        self.__gamma = 0.99
        self.__totalFeature = None
        self.readInstance(d_path)

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
    
    def readInstance(self, d_path):
        dataset_reader = readDataset(d_path)
        
        # Class for verification
        self.setClasses(dataset_reader.getSavedClass())
        # print(self.__classes)

        # Data without the class and non-relevant information
        self.setData(dataset_reader.getInstance())
        # print(self.__data)

        # The number of features is equal to the number of columns
        self.setTotalFeature(len(dataset_reader.instance.columns))
        # print(self.__totalFeature)

    def selection(self, selection):
        data = self.getData().iloc[:, selection]

        scaler = preprocessing.MinMaxScaler()

        train_ratio = 0.8
        test_ratio = 0.2
        SEED = 12
        
        trainingData, testingData, trainingClass, testingClass  = train_test_split(
            data,
            self.getClasses(),
            test_size= 1 - train_ratio,
            random_state=SEED,
            stratify=self.getClasses()
        )

        trainingData = scaler.fit_transform(trainingData)
        testingData = scaler.fit_transform(testingData)

        return trainingData, testingData, trainingClass, testingClass

    def fitness(self, individual, classifier, Cparams):
        accuracy = 0 
        f1Score = 0
        presicion = 0
        recall = 0
        mcc = 0
        trainingData, testingData, trainingClass, testingClass = self.selection(individual)

        if classifier == 'KNN':
            accuracy, f1Score, presicion, recall, mcc = KNN(trainingData, testingData, trainingClass, testingClass, int(Cparams.split(":")[1]))
        if classifier == 'RandomForest':
            accuracy, f1Score, presicion, recall, mcc = RandomForest(trainingData, testingData, trainingClass, testingClass)
        
        if classifier == 'Xgboost':
            accuracy, f1Score, presicion, recall, mcc = Xgboost(trainingData, testingData, trainingClass, testingClass)
            
        errorRate = np.round((1 - accuracy), decimals=3)

        fitness = np.round(( self.getGamma() * errorRate ) + ( ( 1 - self.getGamma() ) * ( len(individual) / self.getTotalFeature() ) ), decimals=3)

        return fitness, accuracy, f1Score, presicion, recall, mcc, errorRate, len(individual)
    
    def factibility(self, individual):
        suma = np.sum(individual)
        if suma > 0:
            return True
        else:
            return False
    
    def newSolution(self):
        return np.random.randint(low=0, high=2, size = self.getTotalFeature())
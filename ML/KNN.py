import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import matthews_corrcoef

#
# KNN module
# Taken directly from: https://github.com/FelipeCisternasCaneo/OII450 
# Translated and edited to keep consistency
#

def KNN(trainingData, testingData, trainingClass, testingClass, k):
        # Classifier training
        # With metric = 'minkowski' y p = 1 manhattan distance is being used
        # With metric = 'minkowski' y p = 2 euclidan distance is being used

        classifier = KNeighborsClassifier(
            n_neighbors = k,
            metric      = 'minkowski',
            p           = 2
        )
        classifier.fit( trainingData , trainingClass )

        # Classifier prediction
        predictionClass = classifier.predict(testingData)

        accuracy    = np.round(accuracy_score(testingClass, predictionClass), decimals=3)
        f1Score     = np.round(f1_score(testingClass, predictionClass), decimals=3)
        precision   = np.round(precision_score(testingClass, predictionClass, zero_division=1), decimals=3)
        recall      = np.round(recall_score(testingClass, predictionClass), decimals=3)
        mcc         = np.round(matthews_corrcoef(testingClass, predictionClass), decimals=3)

        return accuracy, f1Score, precision, recall, mcc
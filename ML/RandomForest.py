import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score, recall_score, matthews_corrcoef

#
# RandomForest module
# Taken directly from: https://github.com/FelipeCisternasCaneo/OII450 
# Translated and edited to keep consistency
#

def RandomForest(trainingData, testingData, trainingClass, testingClass):
    SEED = 12
    rf = RandomForestClassifier(
        criterion = 'gini', 
        n_estimators = 40, 
        max_depth = 40, 
        max_features = "log2", 
        n_jobs=-1, 
        random_state = SEED)
    
    rf.fit(trainingData, trainingClass)
    
    # Predict the Test set results
    predictionClass = rf.predict(testingData)
    
    accuracy    = np.round(accuracy_score(testingClass, predictionClass), decimals=3)
    f1Score     = np.round(f1_score(testingClass, predictionClass), decimals=3)
    presicion   = np.round(precision_score(testingClass, predictionClass, zero_division=1), decimals=3)
    recall      = np.round(recall_score(testingClass, predictionClass), decimals=3)
    mcc         = np.round(matthews_corrcoef(testingClass, predictionClass), decimals=3)

    return accuracy, f1Score, presicion, recall, mcc
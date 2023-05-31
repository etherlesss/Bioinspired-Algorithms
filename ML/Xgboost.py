import numpy as np
from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score, recall_score, matthews_corrcoef
import xgboost 

#
# Xgboost module
# Taken directly from: https://github.com/FelipeCisternasCaneo/OII450 
# Translated and edited to keep consistency
#

def Xgboost(trainingData, testingData, trainingClass, testingClass):
    
    SEED = 12
    
    xgb = xgboost.XGBClassifier(
        tree_method='approx',  # 'approx', 'auto', 'exact', 'gpu_hist', 'hist'
        eval_metric="error", 
        n_estimators=20, max_depth=10, subsample=0.77, learning_rate=0.15,
        seed=SEED)
    
    xgb.fit(trainingData, trainingClass)
    
    # Predict the Test set results
    predictionClass = xgb.predict(testingData)
    
    accuracy    = np.round(accuracy_score(testingClass, predictionClass), decimals=3)
    f1Score     = np.round(f1_score(testingClass, predictionClass), decimals=3)
    presicion   = np.round(precision_score(testingClass, predictionClass, zero_division=1), decimals=3)
    recall      = np.round(recall_score(testingClass, predictionClass), decimals=3)
    mcc         = np.round(matthews_corrcoef(testingClass, predictionClass), decimals=3)

    return accuracy, f1Score, presicion, recall, mcc
    
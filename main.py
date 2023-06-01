from Util.selection import selection as exec_installer
from Solver.solverFS import solver

# Dependency Installer
exec_installer()

# Dataset options
"""
    echocardiogram.data
    sonar.all-data
"""

# Dataset path
d_path = "./Datasets/sonar.all-data"

# Discretization options
"""
    'V1-STD', 'V1-COM', 'V1-PS', 'V1-ELIT',
    'V2-STD', 'V2-COM', 'V2-PS', 'V2-ELIT',
    'V3-STD', 'V3-COM', 'V3-PS', 'V3-ELIT',
    'V4-STD', 'V4-COM', 'V4-PS', 'V4-ELIT',
    'S1-STD', 'S1-COM', 'S1-PS', 'S1-ELIT',
    'S2-STD', 'S2-COM', 'S2-PS', 'S2-ELIT',
    'S3-STD', 'S3-COM', 'S3-PS', 'S3-ELIT',
    'S4-STD', 'S4-COM', 'S4-PS', 'S4-ELIT',
"""

# Classifier options
"""
    KNN, RandomForest, Xgboost
"""

# Parameters
max_iter = 50
population = 20
discretization = ["V1", "STD"]
classifier = "KNN"
# Classifier params for KNN module, k is the number of neighbors, meanwhile 5 indicates that the number of neighbors is five
Cparams = f'k:5'

# Solver call
solver(max_iter, population, d_path, discretization, classifier, Cparams)
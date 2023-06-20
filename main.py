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
    'V1-STD', 'V1-COM', 'V1-PS', 'V1-ELIT', 'X1-STD', 'X1-COM', 'X1-PS', 'X1-ELIT',
    'V2-STD', 'V2-COM', 'V2-PS', 'V2-ELIT', 'X2-STD', 'X2-COM', 'X2-PS', 'X2-ELIT',
    'V3-STD', 'V3-COM', 'V3-PS', 'V3-ELIT', 'X3-STD', 'X3-COM', 'X3-PS', 'X3-ELIT',
    'V4-STD', 'V4-COM', 'V4-PS', 'V4-ELIT', 'X4-STD', 'X4-COM', 'X4-PS', 'X4-ELIT',
    'S1-STD', 'S1-COM', 'S1-PS', 'S1-ELIT', 'Z1-STD', 'Z1-COM', 'Z1-PS', 'Z1-ELIT',
    'S2-STD', 'S2-COM', 'S2-PS', 'S2-ELIT', 'Z2-STD', 'Z2-COM', 'Z2-PS', 'Z2-ELIT',
    'S3-STD', 'S3-COM', 'S3-PS', 'S3-ELIT', 'Z3-STD', 'Z3-COM', 'Z3-PS', 'Z3-ELIT',
    'S4-STD', 'S4-COM', 'S4-PS', 'S4-ELIT', 'Z4-STD', 'Z4-COM', 'Z4-PS', 'Z4-ELIT',
"""

# Classifier options
"""
    KNN, RandomForest, Xgboost
"""

# Parameters
max_iter = 100
population = 30
discretization = ["X4", "COM"]
classifier = "KNN"
# Classifier params for KNN module, k is the number of neighbors, meanwhile 5 indicates that the number of neighbors is five
Cparams = f'k:5'

# Solver call
solver(max_iter, population, d_path, discretization, classifier, Cparams)
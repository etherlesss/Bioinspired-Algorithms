from Operations import read, selection

# TO DO: Solver call + Metaheuristic binarization

# Dependency installer
selection.selection()

# Dataset path
path = "./Datasets/echocardiogram.data"

# This may or may not be removable in the future? serves purpose for testing for now.
instance = read.readDataset(path)

# 2 lines of test
n_columns = len(instance.instance.columns)
print(n_columns)

# Solver call
# (SOLVER)
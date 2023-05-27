from Operations import read, selection
from Metaheuristics import MFO

# NOT USED YET: Discretization
# TO DO: Metaheuristic implementation (MFO) + Metaheuristic Binarization, which goes in hand with previous point.
# WHAT CAN PROBABLY BE DONE BETTER: FS (Feature Selection)

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

# Start metaheuristic
#MFO.MFO(n_columns, path)
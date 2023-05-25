from Operations import read, selection

# NOT USED YET: Discretization
# TO DO: Metaheuristic implementation (MFO) + Metaheuristic Binarization, which goes in hand with previous point.
# WHAT CAN PROBABLY BE DONE BETTER: FS (Feature Selection)

# Dependency installer
selection.selection()

# This may or may not be removable in the future? serves purpose for testing for now.
instance = read.readDataset("./Datasets/echocardiogram.data")
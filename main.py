from Operations import read, selection

# NOT USED YET: Discretization
# TO DO: Metaheuristic implementation (MFO)
# WHAT CAN PROBABLY BE DONE BETTER: FS

# Instalador de dependencias
selection.selection()

# This may or may not be removable in the future? serves purpose for testing for now.
instance = read.readDataset("./Datasets/echocardiogram.data")
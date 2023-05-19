import pandas as pd

# READ DATASET
dataset = pd.read_csv('Dataset/echocardiogram.data', on_bad_lines='skip')

# Guardar los datos en un vector (col 2)
still_alive = dataset.iloc[:, 1]

dataset = dataset.drop(dataset.columns[[1]], axis=1)
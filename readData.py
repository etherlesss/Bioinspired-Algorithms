import pandas as pd

# READ DATASET
dataset = pd.read_csv('Dataset/echocardiogram.data', on_bad_lines='skip')

# PARSE NON-RELEVANT DATA
dataset = dataset.drop(dataset.columns[[10,11,12]], axis=1)
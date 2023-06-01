import pandas as pd
import time
import os
import statistics

class readDataset:
    
    def __init__(self, d_path:str) -> None:
        self.instance = None
        self.dataset = None
        self.saved_class = None

        print("Executing reading module...")
        time.sleep(2)
        os.system("cls")

        print(f"Reading {d_path}\n")

        file_name = d_path.split("/")[-1]  # Get the file name from the path
        name_parts = file_name.split(".")
        instance_name = name_parts[0] # Get the name of the instance

        if (instance_name == "echocardiogram"):
            self.readEchocardiogram(d_path)
        if (instance_name == "sonar"):
            self.readSonar(d_path)

    def readEchocardiogram(self, d_path):
        # Read Dataset
        self.dataset = pd.read_csv(d_path, on_bad_lines='skip')

        # Extract class for result confirmation (2nd dataset column)
        self.saved_class = self.dataset.iloc[:, 1]

        # Extract non-calculable columns
        self.dataset = self.dataset.drop(self.dataset.columns[[10, 11, 12]], axis=1)
        
        # Delete saved column
        self.instance = self.dataset.drop(self.dataset.columns[[1]], axis=1)

        # Get median and average
        median = []
        promedio = []
        guardar = []
        suma = 0
        
        df = pd.DataFrame(self.instance)
        
        # Columns
        for i in range(df.shape[1]):
            # Rows
            for j in range(df.shape[0]):
                if(df.iloc[j,i] == "?"):
                    continue 
                guardar.append(float(df.iloc[j,i]))
                suma += float(df.iloc[j,i])
            median.append(statistics.median(guardar))
            promedio.append(round(suma/df.shape[0],3))
            suma = 0
            guardar = []
        
        # print(median)
        # print(promedio)

        # Fill "?" values with median if the column is categorical or average if is real

        # columns
        for i in range(df.shape[1]):
            # rows
            for j in range(df.shape[0]):
                if df.iloc[j,i] == "?":
                    '''
                    verificar la posición de las columnas para ver si la variable
                    es categórica o real
                    
                    1 survival - real
                    2 age-at-heart-attack - real
                    3 pericardial-effusion - categorico
                    4 fractional-shortening - real
                    5 epss - real
                    6 lvdd - real
                    7 wall-motion-score - real
                    8 wall-motion-index - real
                    9 mult - real 
                    
                    '''
                    if i == 3:
                        df.iloc[j,i] = median[i]
                    else:
                        df.iloc[j,i] = promedio[i]
                    
        # Print the whole table
        '''            
        for i in range(0,df.shape[0]):
                    print(i+1,end=' ')
                    for j in range(0,df.shape[1]):
                        print(df.iloc[i,j],end=' ')
                    print()
        '''
        # Print table
        # print(f"Instancia:\n{self.instance}\n")
        
        # Print class
        # print(f"Clase:\n{self.saved_class}")
    
    def readSonar(self, d_path):
        # Read Dataset
        self.dataset = pd.read_csv(d_path, on_bad_lines='skip')

        # Extract class for result confirmation (60th dataset column)
        self.saved_class = self.dataset.iloc[:, 60]

        self.saved_class = self.saved_class.replace({
            'R':0,
            'M':1
        })
        self.saved_class = self.saved_class.values
        
        # Delete saved column
        self.instance = self.dataset.drop(self.dataset.columns[[60]], axis=1)

    def getInstance(self):
        return self.instance
    def getSavedClass(self):
        return self.saved_class
import pandas as pd
import time
import os

class readDataset:
    def __init__(self, path:str) -> None:
        print("Executing reading module...")
        time.sleep(2)
        os.system("cls")

        print(f"Reading {path}")

        # Leer dataset
        self.dataset = pd.read_csv(path, on_bad_lines='skip')

        # Extraer clases para confirmacion de resultados (2da columna del dataset)
        self.saved_class = self.dataset.iloc[:, 1]

        # Extraer columnas que no sean calculables
        self.dataset = self.dataset.drop(self.dataset.columns[[10, 11, 12]], axis=1)

        # Obtener mediana


        # Obtener promedio


        # Rellenar valores "?" con mediana si la columna es categorica o promedio si es real


        # Eliminar columna que se guardo
        self.instance = self.dataset.drop(self.dataset.columns[[1]], axis=1)

        print(f"Instancia:\n{self.instance}\n")

        print(f"Clase:\n{self.saved_class}")

    def getInstance(self):
        return self.instance
    def getSavedClass(self):
        return self.saved_class
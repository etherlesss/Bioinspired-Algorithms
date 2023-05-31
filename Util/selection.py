from Dependencies.installer import d_installer

import time
import os

class selection:
    def __init__(self) -> None:
        if (self.selection()):
            d_installer('./Dependencies/list.txt')
            time.sleep(1.5)
            os.system("cls")
        else:
            print("You chose to not install dependencies.\n!! If the program fails to execute, it might be due to missing dependencies. !!")
            time.sleep(2.5)
            os.system("cls")
    
    def selection(self):
        os.system("cls")
        try:
            selection = input("Wish to run the dependency installer? (y/n): ")
            if (selection.lower() == 'y'):
                return True
            elif (selection.lower() == 'n'):
                return False
            else:
                return self.selection()
        except ValueError:
            print("Invalid input: Please choose yes or no (y/n).")
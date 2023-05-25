from Dependencies import installer

import time
import os

class selection:
    def __init__(self) -> None:
        if (self.selection()):
            installer.d_installer('./Dependencies/list.txt')
            time.sleep(2.5)
            os.system("cls")
        else:
            print("You chose to not install dependencies.")
            time.sleep(2.5)
            os.system("cls")
    
    def selection(self):
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
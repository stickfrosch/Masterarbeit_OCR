# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.
import pandas as pd
import numpy as np
import cv2
import ocrtryout



pd.set_option('display.max_rows', 50)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

dataset = pd.read_csv("Unfallorte2020_LinRef.csv", delimiter=';', header=None, skiprows=1, names=['OBJECTID','UIDENTSTLAE','ULAND','UREGBEZ','UKREIS','UGEMEINDE','UJAHR','UMONAT','USTUNDE','UWOCHENTAG','UKATEGORIE','UART','UTYP1','ULICHTVERH','IstRad','IstPKW','IstFuss','IstKrad','IstGkfz','IstSonstige','LINREFX','LINREFY','XGCSWGS84','YGCSWGS84','STRZUSTAND'])
#print(dataset.head())


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press ⌘F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/


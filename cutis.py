import matplotlib.pyplot as plt
import numpy as np
import math as m
from pathlib import Path
import pandas as pd

here = Path('.')
for file in here.rglob('*.Spe'):
    with open(file, 'r') as spectrum_file:
        spectrum_data = spectrum_file.readlines()







    
    flux_list = []
    is_data_reached = False
    for line in spectrum_data:
        line = line.strip()        #It'll remove next line jump in readline
        lines = [line.replace(' ', '') for line in line]
        if line == "$DATA:":
            is_data_reached = True   #this will tell the loop do operations after Data line is reached 
            continue
        if line == "$ROI:":
            break
        if is_data_reached:
            if line == "0 1023":      
                continue
            flux_list.append(float(line.split(",")[0]))

        

        for data in flux_list:
         print(data) 

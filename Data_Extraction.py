import matplotlib.pyplot as plt
import numpy as np
import math as m

filename = "BGO_Ba_0_10.Spe"
with open(filename, 'r') as file:
    spectrum_data = file.readlines()


counts = []
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
        counts.append(float(line.split(",")[0]))

      


for channels, photons in enumerate(counts):
    print(channels)
    print(photons)
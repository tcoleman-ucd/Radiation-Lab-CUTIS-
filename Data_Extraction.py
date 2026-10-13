
import matplotlib.pyplot as plt
import numpy as np
import math as m
import glob


def extraction(filename):
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
    return counts

spe_files = glob.glob("*.Spe")

spectra = {}
channels = []
for file in spe_files:
    spectra[file] = extraction(file)
for i,j in enumerate(spectra[file]):
    channels.append(i)
    





for i, header in enumerate(spe_files):
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(channels, spectra[header])
    ax.set_title(f"Spectrum Data: {header}")
    ax.set_xlabel("x")
    ax.set_ylabel(f"{header}  (units)")
plt.show()


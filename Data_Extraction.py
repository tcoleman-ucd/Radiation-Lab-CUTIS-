
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
           #ndex =np.where(spectrum_data == "$MEAS_TIM:")
           #print(spectrum_data)
        index = spectrum_data.index("$MEAS_TIM:\n")
        time = spectrum_data[index+1]
        time = float(time.split(" ")[0])

        if line == "$DATA:":
            is_data_reached = True   #this will tell the loop do operations after Data line is reached 
            continue
        if line == "$ROI:":
            break
        if is_data_reached:
            if line == "0 1023":      
                continue
            counts.append(float(line.split(",")[0]))
        channels = list(range(len(counts)))
    return counts, channels, time




def counts_per_sec(filename):
    """
    Time needs to be in seconds
    """
    counts,channels,time = extraction(filename)

    count_sec = np.array(counts)/time
    return count_sec, channels


spe_files = glob.glob("*.Spe")

spectra = {}
#channels = []
for file in spe_files:
    spectra[file], channels,time = extraction(file)      #Th edictionary just gets the counts


def remove_background(filenameb, files):
    background_counts_per_sec,_= counts_per_sec(filenameb)
    no_b_spectra = {}
    for file in files:
        file_count_per_sec,_ = counts_per_sec(file)
    
        background_removed = np.array(file_count_per_sec)- np.array(background_counts_per_sec)
        no_b_spectra[file] = background_removed
    return no_b_spectra

#practice = remove_background("BGO_Background_180s.Spe", spe_files)
#print(practice)
def amplitude(channels): 
    noback = remove_background("BGO_Background_180s.Spe", spe_files)
    calibration = {}
    for keys, values in noback.items():
        if "0_10cm" in keys:
            amplitude_index = np.argmax(values.ravel())
            channel = channels[amplitude_index]
            
           
    
            
            calibration[keys] = channel(keys)
        return calibration


for i, header in enumerate(spe_files):
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(channels, practice[header])
    ax.set_title(f"Spectrum Data: {header}")
    ax.set_xlabel("x")
    ax.set_ylabel(f"{header}  (units)")
plt.show()

plt.close("all")


#We to convery to counts per second the background and hen minus that from each pof the files
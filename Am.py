import matplotlib.pyplot as plt
import numpy as np
import math as m
import glob
import Data_Extraction
from pathlib import Path
from scipy.optimize import curve_fit

here = Path(".")

Am = here/"BGO_Ba_0_10cm_180s.Spe"
g = "BGO_Background_180s.Spe"


count_sec, channels = Data_Extraction.counts_per_sec(Am)
count_sec1, channels1 = Data_Extraction.counts_per_sec(g)

def remove_background(filenameb, files):

   background_removed = filenameb- files
    
   return background_removed

background_removed = remove_background(count_sec, count_sec1)


def gaussian(x, A1, A2, sigma1, sigma2, mu1, mu2):
    g1 = (A1/(sigma1*np.sqrt(2*np.pi))*np.exp(-(x-mu1)**2/(2*sigma1**2)))
    g2 = (A2/(sigma2*np.sqrt(2*np.pi))*np.exp(-(x-mu2)**2/(2*sigma2**2)))
    return g1+g2 
par1,par2 = curve_fit(gaussian,channels, count_sec, p0=[35,8 ,10, 20, 5,5])

curvex = np.linspace(min(channels), max(channels))
curvey = gaussian(curvex,*par1)

plt.close("all")

# Plot raw spectrum vs background-subtracted data with fit overlay
fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(channels, background_removed, label="Background Subtracted Data", color="black", alpha=0.7)
ax.plot(curvex, curvey, label="Gaussian Fit", color="red", linewidth=2)
ax.set_xlabel("Channel")
ax.set_ylabel("Counts / sec")
ax.set_title("BGO Cs-137 Spectrum Peak Fit")
ax.legend()
plt.grid(True)

plt.show()
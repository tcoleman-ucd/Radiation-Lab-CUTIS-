"This code is in charge of calculating the resolution of each detector"
""

#import libraries
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
#import gaussian and calibration to obtain necessary parameters
from gaussian import fit_peak
from calibration import (calibration_peaks, get_centroids, calibrate,
                         channel_to_energy)

#resolution model - resolution varies with energy
def resolution_model(E, a, b, c):
    """
    Compute resolution model:
    R2 = (δE/E)^2= αE^-2 + bE^-1 + c 

    """
    return a + b * E + c * (E ** 2)

def resolution():

    channels, channel_errs, energies = get_centroids(calibration_peaks)
    cal = calibrate(channels, channel_errs, energies)

    c1 = cal["params"][0] #slope

    energies = []
    delta_E_list = []
    delta_E_err_list = []
    R_values = []
    R_errs = []
    
    for background, filename, low, high, guesses, peak_energies in calibration_peaks:
            
            results = fit_peak(background, filename, low, high, guesses, show_plot=False)

            for res, Energyknown in zip(results, peak_energies):
                 fwhm_ch = res["fwhm"] #fwhm in channels
                 fwhm_ch_err = res["fwhm_err"] #fwhm error in channels

                 delta_E = fwhm_ch * c1 #convert to energy
                 delta_E_err = fwhm_ch_err * c1 #convert to energy

                 R = delta_E/Energyknown
                 R_err = delta_E_err/Energyknown

                 energies.append(Energyknown)
                 delta_E_list.append(delta_E)
                 delta_E_err_list.append(delta_E_err)
                 R_values.append(R)
                 R_errs.append(R_err)

    energies = np.array(energies)
    delta_E_list = np.array(delta_E_list)
    delta_E_err_list = np.array(delta_E_err_list)
    R_values = np.array(R_values)
    R_errs = np.array(R_errs)

    y_sq = delta_E_list ** 2
    y_sq_errs = 2 * delta_E_list * delta_E_err_list

    popt, pcov = curve_fit(resolution_model, energies, y_sq, sigma=y_sq_errs, absolute_sigma=True)
    popt_err = np.sqrt(np.diag(pcov))

     #hard to see and interpret on a linearlt scaled graph - plot energy-resolution
    #on a logarithmic axes.
    E_line = np.logspace(np.log10(min(energies)*0.8), np.log10(max(energies)*1.2), 200)    

    delta_E_fit = resolution_model(E_line, *popt)
    #print(f"this is WHAT I WANT TO PRINT{popt}")
    R_fit = np.sqrt(np.maximum(delta_E_fit, 0)) / E_line

    plt.figure(figsize = (10, 6))
    plt.errorbar(energies, R_values, yerr = R_errs, fmt="o",label = "Errors")
    plt.plot(E_line, R_fit, "r-", linewidth=2, label=r"Fit: $R = \sqrt{aE^{-2} + bE^{-1} + c}$")
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Energy (keV)")
    plt.ylabel("Resolution")
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.legend()
    plt.show()
     
     


if __name__ == "__main__":
    
    
    resolution()
    

            
        
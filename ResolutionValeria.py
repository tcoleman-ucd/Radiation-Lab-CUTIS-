import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

from gaussian import fit_peak
from calibration import (calibration_peaks, get_centroids, calibrate,
                         channel_to_energy)

def resolution_sq_model(E, a, b, c):
    """
    Equivalent to Eq. 17 in the lab handout:
    (delta E)^2 = a + b*E + c*E^2
    """
    return a + b * E + c * (E ** 2)

if __name__ == "__main__":
    
    ch, ch_err, energy_peak = get_centroids(calibration_peaks)
    cal = calibrate(ch, ch_err, energy_peak)
    print(f"THIS IS THE OUTPUT SO FAR{ch}, {ch_err},{energy_peak}")
    print(f"cal {cal}")
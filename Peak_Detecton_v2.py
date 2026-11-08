"""
peak_fitting.py  (simple version)

Fits a Gaussian peak (or two peaks) sitting on a straight-line background.
Uses extraction() from data_extraction.py to get the counts and channels.

Steps:
  1. plot_spectrum("BGO_Cs_0_10cm_420s.Spe")   -> look at the plot, pick a window
  2. fit_peak(filename, low, high, [peak channel guesses])
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

from Practice import extraction
from pathlib import Path

# ---------------------------------------------------------------------------
# Peak Finding
# ---------------------------------------------------------------------------
from scipy.signal import find_peaks, peak_widths

from scipy.signal import find_peaks, peak_widths
import matplotlib.pyplot as plt


from scipy.signal import find_peaks, peak_widths
from scipy.ndimage import gaussian_filter1d
import numpy as np
import matplotlib.pyplot as plt



import numpy as np
import matplotlib.pyplot as plt

from scipy.signal import find_peaks, peak_widths
from scipy.ndimage import gaussian_filter1d



def detect_peaks(
        channels,
        counts,
        significance_threshold=3.5,
        min_width=3,
        min_distance=8,
        smooth_sigma=1.5,
        show_plot=False):

    channels = np.asarray(channels)
    counts = np.asarray(counts, dtype=float)

    # --------------------------------------------------
    # 1. Smooth the original spectrum
    # --------------------------------------------------

    counts_smooth = gaussian_filter1d(
        counts,
        sigma=smooth_sigma
    )

    # --------------------------------------------------
    # 2. Find potential peaks
    # --------------------------------------------------

    peaks, properties = find_peaks(
        counts_smooth,
        prominence=0,
        width=min_width,
        distance=min_distance
    )

    # --------------------------------------------------
    # 3. Calculate local significance
    # --------------------------------------------------

    widths = peak_widths(
        counts_smooth,
        peaks,
        rel_height=0.5
    )[0]

    peak_info = []

    for i, peak in enumerate(peaks):

        prominence = properties["prominences"][i]

        # Estimate the local background from the
        # prominence contour of the detected peak.

        local_background = max(
            counts_smooth[peak] - prominence,
            1
        )

        # Approximate Poisson significance
        significance = (
            prominence / np.sqrt(local_background)
        )

        if significance < significance_threshold:
            continue

        peak_info.append({
            "channel": channels[peak],
            "height": counts[peak],
            "prominence": prominence,
            "width": widths[i],
            "significance": significance
        })

    # --------------------------------------------------
    # 4. Plot
    # --------------------------------------------------

    if show_plot:

        plt.figure(figsize=(12, 5))

        plt.plot(
            channels,
            counts,
            color="black",
            label="Spectrum"
        )

        plt.plot(
            channels,
            counts_smooth,
            color="royalblue",
            alpha=0.7,
            label="Smoothed"
        )

        for peak in peak_info:

            plt.axvline(
                peak["channel"],
                color="red",
                linestyle="--",
                alpha=0.7
            )

        plt.xlabel("Channel")
        plt.ylabel("Counts")
        plt.title("Detected Peaks")
        plt.legend()
        plt.show()

    return peak_info




# ---------------------------------------------------------------------------
# The models
# ---------------------------------------------------------------------------
def gaussian(x, A, mu, sigma):
    """A = area of the peak, mu = centre (channel), sigma = width."""
    return A / (sigma * np.sqrt(2 * np.pi)) * np.exp(-(x - mu) ** 2 / (2 * sigma ** 2))


def one_peak(x, A1, mu1, sigma1, m, b):
    """One Gaussian + a straight line (m*x + b) for the background."""
    return gaussian(x, A1, mu1, sigma1) + m * x + b


def two_peaks(x, A1, mu1, sigma1, A2, mu2, sigma2, m, b):
    """Two Gaussians + a straight line for the background."""
    return gaussian(x, A1, mu1, sigma1) + gaussian(x, A2, mu2, sigma2) + m * x + b


# (need a third peak? copy two_peaks, add A3, mu3, sigma3, and add it to this dictionary)
models = {1: one_peak, 2: two_peaks}


# ---------------------------------------------------------------------------
# Step 1: look at the whole spectrum
# ---------------------------------------------------------------------------
def plot_spectrum(filename):
    counts, channels, time = extraction(filename)
    plt.plot(channels, counts)
    plt.title(filename)
    plt.xlabel("Channel")
    plt.ylabel("Counts")
    plt.show()


# ---------------------------------------------------------------------------
# Step 2: fit the peak(s) inside a window of channels
# ---------------------------------------------------------------------------
def fit_peak(filename, low, high, guess_channels, show_plot=True):
    """
    filename       : the .Spe file
    low, high      : channel window around the peak(s), e.g. 400 and 600
    guess_channels : rough channel of each peak, e.g. [500] or [195, 230]
    show_plot      : set to False to skip the plot (used by the calibration loop)
    """
    # 1. get the data and cut out the window
    counts, channels, time = extraction(filename)
    x = np.array(channels[low:high])
    y = np.array(counts[low:high])

    # 2. starting guesses for the fit: for each peak -> [area, centre, width]
    start = []
    for guess in guess_channels:
        height = y[guess - low]  # counts at the guessed channel
        width = (high - low) / 10  # rough starting width: a tenth of your window
        area = height * width * np.sqrt(2 * np.pi)
        start += [area, guess, width]
    start += [0, min(y)]  # background line: slope 0, height = lowest count

    # 3. do the fit  (error on each point = sqrt(counts), because counting is Poisson)
    model = models[len(guess_channels)]
    errors = np.sqrt(np.maximum(y, 1))
    popt, pcov = curve_fit(model, x, y, p0=start, sigma=errors, absolute_sigma=True)
    popt_err = np.sqrt(np.diag(pcov))  # uncertainty on every fitted number

    # 4. print the answers   (the last two numbers are the background line)
    peaks = popt[:-2].reshape(-1, 3)  # one row per peak: A, mu, sigma
    peaks_err = popt_err[:-2].reshape(-1, 3)
    results = []
    for i in range(len(peaks)):
        A, mu, sigma = peaks[i]
        A_err, mu_err, sigma_err = peaks_err[i]
        fwhm = 2.355 * sigma  # FWHM = 2.355 * sigma
        fwhm_err = 2.355 * sigma_err
        print(f"Peak {i + 1}: centre = {mu:.2f} +/- {mu_err:.2f} channels, "
              f"FWHM = {fwhm:.2f} +/- {fwhm_err:.2f} channels, area = {A:.0f} +/- {A_err:.0f}")
        results.append({"mu": mu, "mu_err": mu_err, "fwhm": fwhm, "fwhm_err": fwhm_err,
                        "A": A, "A_err": A_err})

    # 5. plot the WHOLE spectrum and draw the fit right on top of the data curve
    if show_plot:
        x_smooth = np.linspace(low, high, 500)  # smooth x values just inside the fit window
        m, b = popt[-2], popt[-1]
        plt.figure(figsize=(9, 5))
        plt.plot(channels, counts, color="black", label="data")
        plt.plot(x_smooth, model(x_smooth, *popt), "r", linewidth=2, label="Gaussian fit")
        if len(peaks) > 1:  # with 2+ peaks, also show each one on its own
            for i in range(len(peaks)):
                plt.plot(x_smooth, gaussian(x_smooth, *peaks[i]) + m * x_smooth + b, "--",
                         label=f"peak {i + 1}")
        # plt.xlim(low - 50, high + 50)            # <- un-comment to zoom in around the peak
        plt.title(f"{filename}")
        plt.xlabel("Channel")
        plt.ylabel("Counts")
        plt.legend()
        plt.show()

    return results


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # filename = "BGO_Cs_0_10cm_180s.Spe"
    # filename2 = "BGO_Ba_0_10cm_300s.Spe"
    # filename3 = "BGO_Am_0_10cm_180s.Spe"
    # filename4= "BGO_Co_0_10cm_180s.Spe"
    #
    # filename = "BGO_Cs_0_10cm_180s.Spe"
    data_folder = Path("CUITS")
    files = [data_folder/"BGO_Ba_0_10cm_300s.Spe", data_folder/"BGO_Co_0_10cm_420s.Spe", data_folder/"BGO_Cs_0_10cm_180s.Spe", data_folder/"BGO_Am_0_10cm_180s.Spe"]
    for file in files:
        counts, channels, time = extraction(file)
        background = counts[counts < np.percentile(counts, 75)]
        noise = np.std(background)

        peaks = detect_peaks(
            channels,
            counts,
            significance_threshold=3.5,
            min_width=3,
            min_distance=8,
            smooth_sigma=1.5,
            show_plot=True
        )

        for peak in peaks:
            print(peak)

    # plot_spectrum(filename)                              # 1. look first, then pick numbers
    # fit_peak(filename, low=260, high=360, guess_channels=[280])   # 2. fit (edit these numbers)
    #
    #
    # plot_spectrum(filename2)                              # 1. look first, then pick numbers
    # fit_peak(filename2, low=115, high=200, guess_channels=[140, 167])
    # fit_peak(filename2, low=55, high=85, guess_channels=[77])

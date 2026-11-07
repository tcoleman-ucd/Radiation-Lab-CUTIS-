"""
calibration.py  

Turns channels into energy.

How it works:
  1. For each source file, fit_peak() finds the centre (channel) of each photopeak.
  2. We pair each centre with the KNOWN energy of that line (Table 2 in the handout).
  3. We fit a straight line   E = c1 * channel + c0              (Eq. 15 in the handout)
  4. We TEST the line: also fit  E = c2 * channel^2 + c1 * channel + c0
     and check whether c2 is consistent with zero.
        - c2 within 3 sigma of zero  -> the line is fine, use it
        - c2 NOT consistent with zero -> the detector is non-linear, use the quadratic
  5. channel_to_energy() converts any channel to an energy WITH its uncertainty.

Do this once per detector (BGO, NaI, ...): only list files from that detector.

THOMAS YOU ARE DOING THE CALIBRATION FOR THE OTHER DETECTOR AND POSSIBLY AUTOMATING FINDING THE PEAKS THE REGION OF THE PEAK


VALERIA IS DOING THE RESOLUTION/UNDERSTAND WHATEVER THAT MEANS


CODY IMPLEMNET THE .MCA AND MACTH

ME AND CODY CAN TAKE A LOOK AT THE EFFICIENCIES
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

from gaussian import fit_peak


# ---------------------------------------------------------------------------
# Step 1: list the peaks you want to use for the calibration
#   (filename, low, high, [guess channels], [energies in keV, SAME ORDER as the guesses])
#   Energies are from Table 2:  Cs 661.657 | Co 1173.228, 1332.492 | Ba 356.0129, 302.8508
# ---------------------------------------------------------------------------
Background =  "BGO_Background_180s.Spe"
calibration_peaks = [
    (Background, "BGO_Cs_0_10cm_180s.Spe", 260, 360, [280], [661.657]),
    (Background, "BGO_Co_0_10cm_420s.Spe", 506, 693, [547, 632], [1173.228, 1332.492]),
    (Background, "BGO_Ba_0_10cm_300s.Spe", 115, 200, [143, 167], [302.8508, 356.0129]),
]


# ---------------------------------------------------------------------------
# Step 2: fit every peak and collect (centre channel, its error, known energy)
# ---------------------------------------------------------------------------
def get_centroids(peak_list):
    channels, channel_errs, energies = [], [], []
    for background, filename, low, high, guesses, peak_energies in peak_list:
        if len(guesses) != len(peak_energies):
            raise ValueError(f"{filename}: need one energy for each guessed peak")

        results = fit_peak(background, filename, low, high, guesses, show_plot=False)
        for result, energy in zip(results, peak_energies):
            if not np.isfinite(result["mu_err"]):      # the fit failed -> don't use this peak
                print(f"WARNING: fit failed for {filename} (energy {energy} keV), skipping it. "
                      f"Try a different window or guess.")
                continue
            channels.append(result["mu"])
            channel_errs.append(result["mu_err"])
            energies.append(energy)
    return np.array(channels), np.array(channel_errs), np.array(energies)


# ---------------------------------------------------------------------------
# Step 3: fit a polynomial   order 1 = straight line,  order 2 = quadratic
#   params come out highest power first:  order 1 -> [c1, c0]   order 2 -> [c2, c1, c0]
# ---------------------------------------------------------------------------
def fit_polynomial(channels, channel_errs, energies, order):
    def model(x, *p):
        return np.polyval(p, x)

    # curve_fit only takes errors on y (the energy), but our uncertainty is on the channel.
    # Fix: multiply each channel error by the local slope (keV per channel).
    start = np.polyfit(channels, energies, order)                 # rough first fit
    slope = np.polyval(np.polyder(start), channels)
    energy_errs = slope * channel_errs

    params, cov = curve_fit(model, channels, energies, p0=start,
                            sigma=energy_errs, absolute_sigma=True)
    errs = np.sqrt(np.diag(cov))

    dof = len(channels) - (order + 1)                              # degrees of freedom
    residuals = energies - np.polyval(params, channels)
    chi2_red = np.sum((residuals / energy_errs) ** 2) / dof if dof > 0 else np.nan
    return {"params": params, "errs": errs, "cov": cov, "chi2_red": chi2_red,
            "order": order, "residuals": residuals}


def print_fit(name, fit, channels, energies):
    print(f"\n{name} calibration")
    labels = ["c2", "c1", "c0"][-(fit["order"] + 1):]
    for label, value, err in zip(labels, fit["params"], fit["errs"]):
        print(f"  {label} = {value:.6g} +/- {err:.2g}")
    print(f"  reduced chi-squared = {fit['chi2_red']:.1f}   (near 1 is good; much bigger than 1 "
          f"means the line/curve misses the points by more than the error bars)")
    print("\n  channel    known E (keV)   calibrated E (keV)   difference (keV)")
    for ch, E in zip(channels, energies):
        E_cal = np.polyval(fit["params"], ch)
        print(f"  {ch:7.2f}    {E:11.3f}    {E_cal:15.3f}    {E_cal - E:13.3f}")


# ---------------------------------------------------------------------------
# Step 4: linear fit, linearity test, and pick the calibration to use
# ---------------------------------------------------------------------------
def calibrate(channels, channel_errs, energies):
    if len(channels) < 2:
        raise ValueError("Need at least 2 calibration peaks (add more to calibration_peaks)")

    linear_fit = fit_polynomial(channels, channel_errs, energies, order=1)
    print_fit("LINEAR   E = c1*channel + c0", linear_fit, channels, energies)

    if len(channels) < 4:
        print("\n(Need at least 4 calibration peaks to test linearity with a c2 term.)")
        linear_fit["kind"] = "linear"
        return linear_fit

    quad_fit = fit_polynomial(channels, channel_errs, energies, order=2)
    print_fit("QUADRATIC   E = c2*channel^2 + c1*channel + c0", quad_fit, channels, energies)

    c2, c2_err = quad_fit["params"][0], quad_fit["errs"][0]
    sigma_from_zero = abs(c2) / c2_err
    print(f"\nLINEARITY TEST:  c2 = {c2:.3e} +/- {c2_err:.1e}   ({sigma_from_zero:.1f} sigma from zero)")

    if sigma_from_zero < 3:
        print("  -> consistent with zero: a straight line is fine. Using LINEAR.")
        linear_fit["kind"] = "linear"
        chosen = linear_fit
    else:
        print("  -> NOT consistent with zero: the detector is non-linear. Using QUADRATIC.")
        quad_fit["kind"] = "quadratic"
        chosen = quad_fit
    chosen["linear_fit"] = linear_fit          # keep the line too, so the plot can compare
    return chosen


# ---------------------------------------------------------------------------
# Convert any channel to energy, WITH its uncertainty (you need this for resolution later)
# Works for both the linear and the quadratic calibration.
# ---------------------------------------------------------------------------
def channel_to_energy(channel, channel_err, cal):
    params, cov = cal["params"], cal["cov"]
    energy = np.polyval(params, channel)

    slope = np.polyval(np.polyder(params), channel)      # dE/dchannel (keV per channel)
    powers = np.array([channel ** k for k in range(len(params) - 1, -1, -1)])
    #   linear: [channel, 1]      quadratic: [channel^2, channel, 1]
    variance = ((slope * channel_err) ** 2               # error from the channel itself
                + powers @ cov @ powers)                 # error from c's (incl. correlations)
    return energy, np.sqrt(variance)


# ---------------------------------------------------------------------------
# Step 5: the calibration plot (top: curve, bottom: residuals)
# ---------------------------------------------------------------------------
def plot_calibration(channels, channel_errs, energies, cal, title="Calibration"):
    x = np.linspace(0, channels.max() * 1.1, 200)
    fig, (ax, ax_res) = plt.subplots(2, 1, figsize=(8, 7), sharex=True,
                                     gridspec_kw={"height_ratios": [3, 1]})

    ax.errorbar(channels, energies, xerr=channel_errs, fmt="o", color="black",
                label="fitted photopeaks")
    if "linear_fit" in cal and cal["kind"] == "quadratic":      # show the line for comparison
        ax.plot(x, np.polyval(cal["linear_fit"]["params"], x), "b--", label="linear fit")
    ax.plot(x, np.polyval(cal["params"], x), "r", label=f"{cal['kind']} fit")
    ax.set_ylabel("Energy (keV)")
    ax.set_title(title)
    ax.legend()

    ax_res.axhline(0, color="red")
    if "linear_fit" in cal:
        ax_res.plot(channels, -cal["linear_fit"]["residuals"], "bs", label="linear")
    ax_res.plot(channels, -cal["residuals"], "ko", label=cal["kind"])
    ax_res.set_xlabel("Channel")
    ax_res.set_ylabel("calibrated - known (keV)")
    ax_res.legend()
    plt.show()


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    channels, channel_errs, energies = get_centroids(calibration_peaks)
    cal = calibrate(channels, channel_errs, energies)
    plot_calibration(channels, channel_errs, energies, cal, "BGO calibration")

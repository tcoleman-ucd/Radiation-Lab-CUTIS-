"""
resolution.py
-------------
Energy resolution of the detector.

    R(E) = FWHM_E / E          (FWHM_E in keV, E = photopeak energy in keV)

Steps:
  1. Fit every calibration photopeak -> centre (channel) and FWHM (channels).
  2. Calibrate -> energy E and FWHM in keV, with uncertainties.
        FWHM_E = (dE/dchannel at the peak) * FWHM_channels
  3. R = FWHM_E / E, with propagated uncertainty.
  4. Fit the trend of R with energy:
        (a) power law      R = k * E^n        (scintillators: n close to -0.5)
        (b) R^2 = a + b/E                      (statistical + constant term)
  5. Plot, and quote R at 661.657 keV (Cs-137), the usual benchmark.

Run once per detector (this uses the BGO list from calibration.py).
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

from gaussian import fit_peak
from calibration import (calibration_peaks, get_centroids, calibrate,
                         channel_to_energy)


# ---------------------------------------------------------------------------
# Step 1: fit every peak, keep centre AND FWHM
# ---------------------------------------------------------------------------
def get_peak_data(peak_list):
    """Returns arrays: mu, mu_err, fwhm, fwhm_err (channels), known energies."""
    mu, mu_err, fwhm, fwhm_err, known_E = [], [], [], [], []
    for filename, low, high, guesses, peak_energies in peak_list:
        results = fit_peak(filename, low, high, guesses, show_plot=False)
        for r, E in zip(results, peak_energies):
            if not (np.isfinite(r["mu_err"]) and np.isfinite(r["fwhm_err"])):
                print(f"WARNING: fit failed for {filename} ({E} keV), skipping.")
                continue
            mu.append(r["mu"]);      mu_err.append(r["mu_err"])
            fwhm.append(r["fwhm"]);  fwhm_err.append(r["fwhm_err"])
            known_E.append(E)
    return (np.array(mu), np.array(mu_err), np.array(fwhm),
            np.array(fwhm_err), np.array(known_E))


# ---------------------------------------------------------------------------
# Step 2: channel-width -> energy-width (keV), with uncertainty
# ---------------------------------------------------------------------------
def slope_and_error(channel, cal):
    """dE/dchannel at `channel` and its uncertainty from the calibration params."""
    params, cov = cal["params"], cal["cov"]
    n = len(params)
    slope = np.polyval(np.polyder(params), channel)
    # d(slope)/d(param_i): param i multiplies channel^k with k = n-1-i,
    # so slope contains k * param_i * channel^(k-1)
    grad = np.array([(n - 1 - i) * channel ** (n - 2 - i) if (n - 1 - i) > 0 else 0.0
                     for i in range(n)])
    return slope, np.sqrt(grad @ cov @ grad)


def energy_resolution(mu, mu_err, fwhm, fwhm_err, cal):
    # channel_to_energy handles one channel at a time, so loop over the peaks
    E, E_err, slope, slope_err = [], [], [], []
    for m, me in zip(mu, mu_err):
        e, ee = channel_to_energy(m, me, cal)
        s, se = slope_and_error(m, cal)
        E.append(e); E_err.append(ee); slope.append(s); slope_err.append(se)
    E, E_err = np.array(E), np.array(E_err)
    slope, slope_err = np.array(slope), np.array(slope_err)

    fwhm_E = slope * fwhm
    fwhm_E_err = np.sqrt((slope * fwhm_err) ** 2 + (fwhm * slope_err) ** 2)

    R = fwhm_E / E
    # relative errors add in quadrature (E and FWHM_E share the calibration,
    # so this slightly overestimates the error: fine for a lab report)
    R_err = R * np.sqrt((fwhm_E_err / fwhm_E) ** 2 + (E_err / E) ** 2)
    return E, E_err, fwhm_E, fwhm_E_err, R, R_err


# ---------------------------------------------------------------------------
# Step 3: fit the models of R vs E
# ---------------------------------------------------------------------------
def power_law(E, k, n):
    return k * E ** n


def stat_model(E, a, b):
    """R^2 = a + b/E   ->   R = sqrt(a + b/E)"""
    return np.sqrt(np.maximum(a + b / E, 0))


def fit_model(model, E, R, R_err, p0):
    popt, pcov = curve_fit(model, E, R, p0=p0, sigma=R_err, absolute_sigma=True,
                           maxfev=20000)
    perr = np.sqrt(np.diag(pcov))
    dof = len(E) - len(popt)
    chi2 = np.sum(((R - model(E, *popt)) / R_err) ** 2)
    chi2_red = chi2 / dof if dof > 0 else np.nan
    return popt, perr, chi2_red


# ---------------------------------------------------------------------------
# Step 4: plot
# ---------------------------------------------------------------------------
def plot_resolution(E, E_err, R, R_err, fits, title="Energy resolution"):
    Ex = np.linspace(E.min() * 0.8, E.max() * 1.1, 300)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.errorbar(E, R * 100, xerr=E_err, yerr=R_err * 100, fmt="o", color="black",
                label="measured")
    for label, model, popt in fits:
        ax.plot(Ex, model(Ex, *popt) * 100, label=label)
    ax.set_xlabel("Energy (keV)")
    ax.set_ylabel("Resolution FWHM / E  (%)")
    ax.set_title(title)
    ax.legend()

    # log-log view: a power law is a straight line, slope = n
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    ax2.errorbar(E, R, yerr=R_err, fmt="o", color="black", label="measured")
    for label, model, popt in fits:
        ax2.plot(Ex, model(Ex, *popt), label=label)
    ax2.set_xscale("log"); ax2.set_yscale("log")
    ax2.set_xlabel("Energy (keV)"); ax2.set_ylabel("Resolution FWHM / E")
    ax2.set_title(title + " (log-log)")
    ax2.legend()
    plt.show()


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # --- calibration (same as calibration.py) ---
    ch, ch_err, known = get_centroids(calibration_peaks)
    cal = calibrate(ch, ch_err, known)

    # --- resolution of each photopeak ---
    mu, mu_err, fwhm, fwhm_err, _ = get_peak_data(calibration_peaks)
    E, E_err, fwhm_E, fwhm_E_err, R, R_err = energy_resolution(
        mu, mu_err, fwhm, fwhm_err, cal)

    print("\n  E (keV)        FWHM (keV)      R = FWHM/E (%)")
    for e, ee, f, fe, r, re in zip(E, E_err, fwhm_E, fwhm_E_err, R, R_err):
        print(f"  {e:7.1f} +/- {ee:3.1f}   {f:6.1f} +/- {fe:4.1f}   {100*r:6.2f} +/- {100*re:4.2f}")

    # --- model fits ---
    fits = []
    p, pe, c = fit_model(power_law, E, R, R_err, p0=[1.0, -0.5])
    print(f"\nPower law  R = k*E^n:  k = {p[0]:.4g} +/- {pe[0]:.2g},  "
          f"n = {p[1]:.3f} +/- {pe[1]:.3f}   (reduced chi2 = {c:.2f})")
    print("  (n = -0.5 is the ideal photon-statistics expectation)")
    fits.append((f"R = k E^n  (n = {p[1]:.2f})", power_law, p))

    p2, pe2, c2 = fit_model(stat_model, E, R, R_err, p0=[1e-3, 1.0])
    print(f"R^2 = a + b/E:  a = {p2[0]:.3g} +/- {pe2[0]:.2g},  "
          f"b = {p2[1]:.3g} +/- {pe2[1]:.2g} keV   (reduced chi2 = {c2:.2f})")
    fits.append(("R^2 = a + b/E", stat_model, p2))

    # --- benchmark at Cs-137 ---
    print(f"\nResolution at 661.657 keV (power-law fit): {100*power_law(661.657, *p):.2f} %")

    plot_resolution(E, E_err, R, R_err, fits, "BGO energy resolution")
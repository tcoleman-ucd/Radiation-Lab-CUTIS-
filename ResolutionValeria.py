import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


def resolution_model(E, a, b, c):
    """FWHM^2 = a + b*E + c*E^2   (so R^2 = a/E^2 + b/E + c)."""
    return a + b * E + c * E ** 2


def resolution(cal, title="Resolution", show=True):
    """Energy resolution from a Calibration object (uses the peaks stored in it)."""
    p, p_err = cal.fit["params"], cal.fit["errs"]
    
    
    if cal.fit["kind"] == "linear":
        slope = np.full_like(cal.channels, p[0])
    else:
        slope = 2 * p[0] * cal.channels + p[1]

    E = cal.energies                       # known energies (keV)
    
    dE = cal.fwhm * slope                  # FWHM in keV
    
    dE_err = cal.fwhm_err * slope
    R, R_err = dE / E, dE_err / E

    if len(E) < 4:
        print("Warning: fewer than 4 peaks, the 3-parameter fit is poorly constrained.")

    # fit FWHM^2 vs E
    y2, y2_err = dE ** 2, 2 * dE * dE_err
    p0 = [1.0, 1.0, np.mean(R) ** 2]
    popt, pcov = curve_fit(resolution_model, E, y2, p0=p0, sigma=y2_err,
                           absolute_sigma=True, bounds=(0, np.inf), maxfev=20000)
    perr = np.sqrt(np.diag(pcov))

    R_fit = np.sqrt(resolution_model(E, *popt)) / E
    residuals = R - R_fit
    chi2_red = np.sum((residuals / R_err) ** 2) / (len(E) - 3) if len(E) > 3 else np.nan

    print(f"\n{title}: R^2 = a/E^2 + b/E + c")
    for n, v, e in zip("abc", popt, perr):
        print(f"  {n} = {v:.4g} +/- {e:.2g}")
    print(f"  reduced chi2 = {chi2_red:.2f}")
    print(f"{'E (keV)':>10} {'FWHM (keV)':>12} {'R':>10} {'R err':>10} {'R fit':>10}")
    for row in zip(E, dE, R, R_err, R_fit):
        print("{:10.2f} {:12.3f} {:10.4f} {:10.4f} {:10.4f}".format(*row))


    E_line = np.logspace(np.log10(E.min() * 0.8), np.log10(E.max() * 1.2), 200)
    R_line = np.sqrt(resolution_model(E_line, *popt)) / E_line

    plt.figure(figsize=(10, 6))
    plt.errorbar(E, R*100, yerr=R_err*100, fmt="o", color="black", capsize=3, label="Data")
    plt.plot(E_line, R_line*100, "r-", lw=2, label=r"Fit: $R=\sqrt{aE^{-2}+bE^{-1}+c}$")
    plt.xscale("log"); plt.yscale("log")
    plt.xlabel("Energy (keV)"); plt.ylabel("Resolution  ΔE/E")
    plt.title(title); plt.grid(True, which="both", ls="--", alpha=0.5); plt.legend()
        

    plt.figure(figsize=(7, 3.5))
    plt.errorbar(E, residuals, yerr=R_err, fmt="o", color="black", capsize=3)
    plt.axhline(0, color="red", ls="--")
    plt.xlabel("Energy (keV)"); plt.ylabel(r"$R_{data}-R_{fit}$")
    plt.title(f"{title} residuals"); plt.grid(True, ls="--", alpha=0.5)
    plt.show()

    return {"params": popt, "errs": perr, "E": E, "R": R, "R_err": R_err,
            "chi2_red": chi2_red}
    
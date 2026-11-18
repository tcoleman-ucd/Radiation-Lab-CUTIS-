"""Calculate and plot energy resolution from identified photopeak fits."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit


PLOT_DIR = Path("Plots")


def resolution_model(E, a, b, c):
    """FWHM(E)^2 = a + b E + c E^2, with energies in keV."""
    return a + b * E + c * E**2


def with_error(value, error, sig=2):
    
    if not np.isfinite(value):
        return "n/a"
    if not np.isfinite(error) or error <= 0:
        return f"{value:.4g} ± n/a"

  
    if value != 0 and (abs(value) >= 1e5 or abs(value) < 1e-3):
        power = int(np.floor(np.log10(abs(value))))
        scale = 10.0**power
        return f"({with_error(value / scale, error / scale, sig)})e{power:+d}"

    decimals = max(sig - 1 - int(np.floor(np.log10(error))), 0)
    return f"{value:.{decimals}f} ± {error:.{decimals}f}"


def resolution(detector, title=None):
    """Calculate R=FWHM/E, fit the manual's model and save a labelled plot."""
    if detector.calibration is None:
        raise ValueError("Calibrate the detector first.")

    peaks = detector.calibration_peak_records
    if not peaks:
        raise ValueError("No identified photopeaks are available.")

    E = np.array([peak["energy"] for peak in peaks], dtype=float)
    fwhm = np.array([peak["fwhm"] for peak in peaks], dtype=float)
    fwhm_error = np.array([peak["fwhm_error"] for peak in peaks], dtype=float)

    # The linear calibration converts widths from channels to keV.
    m = abs(float(detector.calibration["gradient"]))
    # The calibration slope uncertainty also contributes to the width error.
    slope_variance = float(detector.calibration["covariance"][0, 0])
    if np.isfinite(slope_variance) and slope_variance >= 0:
        m_error = np.sqrt(slope_variance)
    else:
        # Two-point calibrations may not have an estimable slope covariance.
        print(f"Warning: {detector.name}: calibration slope uncertainty "
              "is unavailable; resolution errors exclude this contribution.")
        m_error = 0.0
    dE = m * fwhm
    dE_error = np.hypot(m * fwhm_error, fwhm * m_error)
    R = dE / E
    R_error = dE_error / E  # tabulated photopeak energies are treated as fixed

    valid = (np.isfinite(E) & np.isfinite(R) & np.isfinite(R_error)
             & (E > 0) & (dE > 0) & (dE_error > 0))
    if not valid.all():
        print(f"Warning: {detector.name}: ignoring {sum(~valid)} invalid resolution points.")

    E, R, R_error, dE, dE_error = (v[valid] for v in (E, R, R_error, dE, dE_error))
    sources = np.array([p["isotope"] for p in peaks])[valid]
    if len(E) == 0:
        raise ValueError("No usable resolution measurements remain.")

    # Fit FWHM^2 rather than R^2, as advised by the laboratory manual.
    params = None
    errors = None
    covariance = None
    if len(E) >= 4:
        squared_fwhm = dE**2
        squared_error = 2 * dE * dE_error
        try:
            params, covariance = curve_fit(
                resolution_model, E, squared_fwhm,
                sigma=squared_error, absolute_sigma=True,
                bounds=(0, np.inf), maxfev=20000,
            )
            errors = np.sqrt(np.diag(covariance))
        except (RuntimeError, ValueError) as error:
            print(f"{detector.name}: resolution model fit failed ({error}).")
    else:
        print(f"{detector.name}: need at least four valid photopeaks for a "
              "three-parameter resolution fit with a residual check.")

    if params is not None:
        print(f"\n{detector.name} resolution model (FWHM^2 = a + bE + cE^2):")
        for name, unit, value, uncertainty in zip(
                "abc", ("keV^2", "keV", ""), params, errors):
            print(f"  {name} = {with_error(value, uncertainty)} {unit}".rstrip())

    print(f"\n{detector.name} photopeak resolutions:")
    print(f"{'Source':8}{'E (keV)':>12}{'FWHM (keV)':>20}{'R = FWHM/E (%)':>22}")
    for source, energy, width, width_err, r, r_err in zip(
            sources, E, dE, dE_error, R, R_error):
        print(f"{source:8}{energy:12.3f}"
              f"{with_error(width, width_err):>20}"
              f"{with_error(100 * r, 100 * r_err):>22}")

    fig, ax = plt.subplots(figsize=(8, 5.5))
    for source in dict.fromkeys(sources):
        mask = sources == source
        ax.errorbar(E[mask], 100 * R[mask], yerr=100 * R_error[mask],
                    fmt="o", capsize=3, label=source)

    # Restore the original isotope/energy labels for each individual photopeak.
    offsets = [(-10, 28), (10, -32)]
    for i, (source, energy, r) in enumerate(zip(sources, E, R)):
        ax.annotate(
            f"{source} {energy:.0f} keV", (energy, 100*r),
            textcoords="offset points", xytext=offsets[i % 2],
            ha="center", fontsize=9,
            arrowprops=dict(arrowstyle="-", color="gray", lw=0.8)
        )

    if params is not None:
        E_line = np.logspace(
            np.log10(E.min() * 0.8), np.log10(E.max() * 1.2), 200
        )
        # Multiply by 100 so the curve is in % like the data points.
        R_line = 100 * np.sqrt(resolution_model(E_line, *params)) / E_line
        ax.plot(
            E_line, R_line, "r-", lw=2,
            label=r"Fit: $R=\sqrt{aE^{-2}+bE^{-1}+c}$"
        )

    ax.set(xscale="log", yscale="log", xlabel="Gamma-ray energy (keV)",
           ylabel="Energy resolution FWHM / E (%)",
           title=title or f"{detector.name} energy resolution")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    PLOT_DIR.mkdir(exist_ok=True)
    for extension in ("pdf", "jpg"):
        fig.savefig(PLOT_DIR / f"Resolution_{detector.name}.{extension}",
                    dpi=300, bbox_inches="tight")
    plt.close(fig)

    return {"params": params, "errs": errors, "covariance": covariance,
            "E": E, "R": R, "R_err": R_error,
            "fwhm_keV": dE, "fwhm_keV_err": dE_error,
            "source": sources}

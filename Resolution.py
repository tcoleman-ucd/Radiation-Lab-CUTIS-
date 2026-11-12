
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from pathlib import Path


def resolution_model(E, a, b, c):
    """FWHM^2 = a + b*E + c*E^2 (so R^2 = a/E^2 + b/E + c)."""

    return a + b * E + c * E ** 2


def resolution(detector, title="Resolution", show=True):
    """Energy resolution using calibration peaks stored in a Detector."""

    if detector.calibration is None:
        raise ValueError("Calibrate the detector first.")

    cal = detector.calibration
    peaks = detector.calibration_peak_records

    # Retrieve calibration data.
    channels = np.asarray(cal["channels"], dtype=float)
    channel_errs = np.asarray(
        cal["channel_errors"],
        dtype=float
    )

    E = np.asarray(cal["energies"], dtype=float)

    fwhm = np.array([
        peak["fwhm"]
        for peak in peaks
    ])

    fwhm_err = np.array([
        peak["fwhm_error"]
        for peak in peaks
    ])

    if len(peaks) != len(E):
        raise ValueError(
            "Number of calibration peaks and energies does not match."
        )

    if len(E) < 3:
        raise ValueError(
            "At least three calibration peaks are required "
            "for the resolution model."
        )

    if np.any(E <= 0):
        raise ValueError(
            "Resolution requires positive reference energies."
        )

    # Linear energy calibration: E = m*C + c
    # Convert Gaussian widths from channels to keV.
    slope = abs(cal["gradient"])

    dE = fwhm * slope
    dE_err = fwhm_err * slope

    if np.any(dE <= 0) or np.any(dE_err <= 0):
        raise ValueError(
            "FWHM and its uncertainty must be positive."
        )

    R = dE / E
    R_err = dE_err / E

    if len(E) < 4:
        print(
            "Warning: fewer than 4 peaks, "
            "the 3-parameter fit is poorly constrained."
        )

    # Fit FWHM^2 vs E.
    y2 = dE ** 2
    y2_err = 2 * dE * dE_err

    p0 = [1.0, 1.0, np.mean(R) ** 2]

    popt, pcov = curve_fit(
        resolution_model,
        E,
        y2,
        p0=p0,
        sigma=y2_err,
        absolute_sigma=True,
        bounds=(0, np.inf),
        maxfev=20000
    )

    perr = np.sqrt(np.diag(pcov))

    R_fit = np.sqrt(
        resolution_model(E, *popt)
    ) / E

    residuals = R - R_fit

    chi2_red = (
        np.sum((residuals / R_err) ** 2) / (len(E) - 3)
        if len(E) > 3
        else np.nan
    )

    print(f"\n{title}: R^2 = a/E^2 + b/E + c")

    for n, v, e in zip("abc", popt, perr):
        print(f"  {n} = {v:.4g} +/- {e:.2g}")

    print(f"  reduced chi2 = {chi2_red:.2f}")

    # Identify detector and isotope for each fitted peak.
    detectors, isotopes = [], []

    for pk in peaks:
        tokens = Path(pk["source"]).stem.split("_")

        detectors.append(detector.name)
        isotopes.append(
            pk.get(
                "isotope",
                tokens[1] if len(tokens) > 1 else "Unknown"
            )
        )

    # Reconstruct fitted energies from the calibration.
    E_fit = detector.channel_to_energy(channels)

    # Energy uncertainty from centroid uncertainty.
    E_fit_err = slope * channel_errs

    print(
        "\nDetector  Isotope  E known  E fit   E fit err  "
        "FWHM (keV)  FWHM err      R    R err   R fit"
    )

    for i, pk in enumerate(peaks):

        det = detectors[i]
        iso = isotopes[i]

        print(
            f"{det:8}  {iso:7}  "
            f"{E[i]:7.1f}  "
            f"{E_fit[i]:6.1f}  "
            f"{E_fit_err[i]:9.2f}  "
            f"{dE[i]:10.2f}  "
            f"{dE_err[i]:8.2f}  "
            f"{R[i]:6.3f}  "
            f"{R_err[i]:6.3f}  "
            f"{R_fit[i]:6.3f}"
        )

    E_line = np.logspace(
        np.log10(E.min() * 0.8),
        np.log10(E.max() * 1.2),
        200
    )

    R_line = np.sqrt(
        resolution_model(E_line, *popt)
    ) / E_line

    if show:

        plt.figure(figsize=(10, 6))

        plt.errorbar(
            E,
            R * 100,
            yerr=R_err * 100,
            fmt="o",
            color="black",
            capsize=3,
            label="Data"
        )

        offsets = [
            (-10, 28),
            (10, -32)
        ]

        for i, pk in enumerate(peaks):

            iso = isotopes[i]

            plt.annotate(
                f"{iso} {E[i]:.0f} keV",
                (E[i], R[i] * 100),
                textcoords="offset points",
                xytext=offsets[i % 2],
                ha="center",
                fontsize=9,
                arrowprops=dict(
                    arrowstyle="-",
                    color="gray",
                    lw=0.8
                )
            )

        plt.plot(
            E_line,
            R_line * 100,
            "r-",
            lw=2,
            label=r"Fit: $R=\sqrt{aE^{-2}+bE^{-1}+c}$"
        )

        plt.xscale("log")
        plt.yscale("log")

        plt.xticks(
            [10, 20, 50, 100, 200, 500, 1000, 2000],
            ["10", "20", "50", "100", "200", "500", "1000", "2000"]
        )

        plt.yticks(
            [1, 2, 5, 10, 20, 50, 60, 70, 80, 90, 100],
            ["1", "2", "5", "10", "20", "50", "60", "70", "80", "90", "100"]
        )

        plt.xlabel("Energy (keV)")
        plt.ylabel("Resolution  ΔE/E (%)")
        plt.title(title)

        plt.grid(
            True,
            which="both",
            ls="--",
            alpha=0.5
        )

        plt.legend()

        # Residual plot.
        plt.figure(figsize=(7, 3.5))

        plt.errorbar(
            E,
            residuals,
            yerr=R_err,
            fmt="o",
            color="black",
            capsize=3
        )

        plt.axhline(
            0,
            color="red",
            ls="--"
        )

        plt.xlabel("Energy (keV)")
        plt.ylabel(r"$R_{data}-R_{fit}$")
        plt.title(f"{title} residuals")

        plt.grid(
            True,
            ls="--",
            alpha=0.5
        )

        plt.show()

    return {
        "params": popt,
        "errs": perr,
        "E": E,
        "R": R,
        "R_err": R_err,
        "chi2_red": chi2_red
    }

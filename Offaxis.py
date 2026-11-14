import numpy as np
import matplotlib.pyplot as plt

from fitting import fit_roi, gaussian


def get_rois(detector, source):
    """One job per calibration ROI of this source (single- or two-peak).

    Each job: roi=(lower, upper), n=peaks in ROI, guesses=calibration
    centroids (channels), energies=assigned energies (keV), lowest peak first.
    """
    fit_results = detector.spectra[source].fit_results
    if not fit_results:
        raise RuntimeError(f"No {source} peaks were fitted during calibration.")

    records = [r for r in detector.calibration_peak_records
               if r["isotope"] == source]

    jobs = []
    for fit in fit_results:
        centroids = [c["centroid"] for c in fit["components"]]
        energies = [min(records, key=lambda r: abs(r["centroid"] - c))["energy"]
                    for c in centroids]
        jobs.append(dict(roi=fit["roi"], n=fit["n_peaks"],
                         guesses=centroids, energies=energies))
    return jobs


def fit_all_angles(spectra, job):
    """Fit one ROI at every angle. Returns {angle: fit_dict}."""
    lo, hi = job["roi"]
    guesses = job["guesses"] if job["n"] == 2 else None

    fits = {}
    for a in sorted(spectra):
        try:
            fits[a] = fit_roi(spectra[a], lo, hi, n=job["n"],
                              peak_guesses=guesses, corrected=True)
        except (ValueError, RuntimeError) as err:
            print(f"{a} deg: fit failed ({err}), skipping.")
    return fits


def plot_fit_checks(spectra, fits, roi, name=""):
    """One panel per angle, to check each fit by eye."""
    lo, hi = roi
    pad = (hi - lo) / 2
    x_fit = np.linspace(lo, hi, 300)

    fig, axes = plt.subplots(1, len(fits), figsize=(5 * len(fits), 4),
                             squeeze=False)
    for ax, angle in zip(axes[0], sorted(fits)):
        spec = spectra[angle]
        view = (spec.channels > lo - pad) & (spec.channels < hi + pad)
        ax.plot(spec.channels[view], spec.corrected_cps()[view], "k.", ms=3)
        ax.plot(x_fit, gaussian(x_fit, *fits[angle]["parameters"]), "r-")
        ax.axvspan(lo, hi, color="blue", alpha=0.1)
        ax.set_title(f"{angle} deg")
        ax.set_xlabel("Channel")

    axes[0][0].set_ylabel("Corrected CPS")
    fig.suptitle(name)
    plt.tight_layout()


def normalise(x, x_err,i0):
    """Divide by the 0 deg value (first entry) and propagate the error."""
    r = x / x[i0]
    r_err = r * np.sqrt((x_err / x) ** 2 + (x_err[i0] / x[i0]) ** 2)
    r_err[i0] = 0.0          # 0 deg divided by itself
    return r, r_err


def analyse(peaks, spectra, calibration):
    """peaks: {angle: peak_dict} for one peak. Returns a result dict."""
    angles = np.array(sorted(peaks))
    if 0 not  in angles:
        raise ValueError("No 0 deg measurement, cannot normalise.")
    i0 = int(np.where(angles == 0)[0][0])
    def col(key):
        return np.array([peaks[a][key] for a in angles])

    area, area_err = col("area"), col("area_error")
    mu, mu_err = col("centroid"), col("centroid_error")
    fwhm, fwhm_err = col("fwhm"), col("fwhm_error")

    # Resolution dE/E in %, same definition as Resolution.py
    m, c = calibration["gradient"], calibration["intercept"]
    E = m * mu + c
    res = 100 * m * fwhm / E
    res_err = res * np.sqrt((fwhm_err / fwhm) ** 2 + (m * mu_err / E) ** 2)

    # Peak-to-total (only the peak error is propagated)
    total = np.array([spectra[a].corrected_cps().sum() for a in angles])
    ptt, ptt_err = area / total, area_err / total

    r = {"angles": angles, "energy": E, "centroid": mu,
         "res": res, "res_err": res_err}
    for key, (x, err) in {"eff": (area, area_err), "ptt": (ptt, ptt_err),
                          "res": (res, res_err), "mu": (mu, mu_err)}.items():
        r["rel_" + key], r["rel_" + key + "_err"] = normalise(x, err,i0)
    return r


TITLES = {
    "rel_eff": "Photopeak count rate",
    "rel_ptt": "Peak-to-total ratio",
    "rel_res": "Energy resolution (ΔE/E)",
    "rel_mu":  "Peak centroid (control)",
}


def plot_comparison(all_r):
    """all_r: {(detector, source, energy): result dict}.

    One figure per (source, energy), with the detectors overlaid.
    """
    for source, energy in sorted({(s, e) for _, s, e in all_r}):
        fig, axes = plt.subplots(2, 2, figsize=(11, 8))

        for ax, (key, title) in zip(axes.flat, TITLES.items()):
            for (det, s, e), r in all_r.items():
                if (s, e) == (source, energy):
                    ax.errorbar(r["angles"], r[key], yerr=r[key + "_err"],
                                fmt="o-", capsize=4, label=det)
            ax.axhline(1, color="k", ls="--", alpha=0.4)
            ax.set_title(title)
            ax.set_xlabel("Off-axis angle θ (deg)")
            ax.set_ylabel("Relative to 0°")
            ax.grid(alpha=0.3)
            ax.legend()

        fig.suptitle(f"{source} {energy:.1f} keV off-axis response")
        plt.tight_layout()
    plt.show()


def run_off_axis(detector, spectra, source):
    """spectra: {angle: Spectrum} with background attached.

    Returns {energy_keV: result dict} for every calibration peak of the source.
    """
    out = {}
    for job in get_rois(detector, source):
        fits = fit_all_angles(spectra, job)
        plot_fit_checks(spectra, fits, job["roi"], f"{detector.name} {source}")

        for j, energy in enumerate(job["energies"]):
            peaks = {a: f["components"][j] for a, f in fits.items()}
            r = analyse(peaks, spectra, detector.calibration)
            out[energy] = r

            print(f"\n{detector.name} {source} {energy:.1f} keV")
            print("Angle  RelEff   RelP/T   R (%)   Centroid")
            for i, a in enumerate(r["angles"]):
                print(f"{a:5.0f}  {r['rel_eff'][i]:6.3f}  {r['rel_ptt'][i]:6.3f}  "
                      f"{r['res'][i]:6.2f}  {r['centroid'][i]:8.2f}")
    plt.show()      # show the fit checks for this source
    return out
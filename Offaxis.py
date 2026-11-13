import numpy as np
import matplotlib.pyplot as plt
from spectra import Spectrum
from detector import Detector
from Efficiency import absolute_efficiency, plot_absolute_efficiency
from pathlib import Path
from interactive_fitting import interactive_fit, fit_roi

background = "BGO_Background_180s.Spe"
measurements = {
    0:  "BGO_Cs_0_10cm_180s.Spe",
    45: "BGO_Cs_45(1)_10cm_180s.Spe",
    90: "BGO_Cs_90(1)_10cm_180s.Spe"}


REUSE_ROI = True

def find_file(name):
    path = next(Path(".").rglob(name), None)
    if path is None:
        raise FileNotFoundError(f"Could not find {name}")
    return path

    background = Spectrum.from_file(find_file(background_name))  # load once


def load_off_axis(background, measurements):
    spectra = {}
    for angle, filename in measurements.items():
        spec = Spectrum.from_file(find_file(filename))
        spec.background = background
        spectra[angle] = spec
    return spectra

def main_peak(peaks):
    """Get the tallest peak."""
    return max(peaks, key=lambda p: p["amplitude"])

def fit_all_angles(spectra, reuse_roi=True):
    results = {}
    roi = None

    for angle in sorted(spectra):
        spec = spectra[angle]

        if reuse_roi and roi is not None:
            # Same window as the first angle, no clicking
            try:
                fit = fit_roi(spec, roi[0], roi[1], corrected=True, n=1)
            except (ValueError, RuntimeError) as err:
                print(f"{angle} deg: fit failed ({err}), skipping.")
                continue
            peak = main_peak(fit["components"])

        else:
            print(f"\n=== {angle} deg: press 1, drag over the Cs-137 photopeak, Enter ===")
            peaks = interactive_fit(spec, corrected=True)
            if len(peaks) == 0:
                print(f"{angle} deg: no peak fitted, skipping.")
                continue
            peak = main_peak(peaks)
            if reuse_roi:
                roi = spec.rois[0]   # remember the window for the other angles

        results[angle] = peak

    return results

def analyse(results):
    angles = np.array(sorted(results))

    area      = np.array([results[a]["area"] for a in angles])
    area_err  = np.array([results[a]["area_error"] for a in angles])
    mu        = np.array([results[a]["centroid"] for a in angles])
    mu_err    = np.array([results[a]["centroid_error"] for a in angles])
    fwhm      = np.array([results[a]["fwhm"] for a in angles])
    fwhm_err  = np.array([results[a]["fwhm_error"] for a in angles])

    # Resolution R = FWHM / centroid (%)
    res = 100 * fwhm / mu
    res_err = res * np.sqrt((fwhm_err / fwhm)**2 + (mu_err / mu)**2)

    # Normalize to the 0 deg measurement
    if 0 not in angles:
        raise ValueError("No 0 deg measurement, cannot normalize.")
    i0 = int(np.where(angles == 0)[0][0])

    rel_eff = area / area[i0]
    rel_eff_err = rel_eff * np.sqrt((area_err / area)**2 + (area_err[i0] / area[i0])**2)

    rel_res = res / res[i0]
    rel_res_err = rel_res * np.sqrt((res_err / res)**2 + (res_err[i0] / res[i0])**2)

    # The 0 deg point is divided by itself, so its error is really zero
    # (the formula above double counts it). Set it explicitly.
    rel_eff_err[i0] = 0
    rel_res_err[i0] = 0

    return angles, rel_eff, rel_eff_err, rel_res, rel_res_err, mu

def plot_results(angles, rel_eff, rel_eff_err, rel_res, rel_res_err):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8), sharex=True)

    ax1.errorbar(angles, rel_eff, yerr=rel_eff_err, fmt="o-", capsize=4,
                 label="Measured BGO response")
    ax1.plot(angles, np.cos(np.radians(angles)), "k--", alpha=0.5,
             label=r"$\cos(\theta)$ reference")
    ax1.set_ylabel("Relative photopeak count rate\n"
                   r"$\mathrm{Area}(\theta)/\mathrm{Area}(0^\circ)$")
    ax1.set_title(r"Off-axis response of BGO (normalized to $0^\circ$)")
    ax1.grid(alpha=0.3)
    ax1.legend()

    ax2.errorbar(angles, rel_res, yerr=rel_res_err, fmt="s-", color="tab:orange",
                 capsize=4, label="Normalized resolution")
    ax2.axhline(1.0, color="k", linestyle="--", alpha=0.5)
    ax2.set_xlabel(r"Source off-axis angle $\theta$ (degrees)")
    ax2.set_ylabel(r"$R(\theta)/R(0^\circ)$")
    ax2.set_ylim(0.7, 1.3)
    ax2.grid(alpha=0.3)
    ax2.legend()

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    spectra = load_off_axis(background, measurements)
    results = fit_all_angles(spectra, reuse_roi=REUSE_ROI)

    angles, rel_eff, rel_eff_err, rel_res, rel_res_err, mu = analyse(results)

    print("\nAngle  RelEff   +/-      Centroid")
    for a, e, de, m in zip(angles, rel_eff, rel_eff_err, mu):
        print(f"{a:5.0f}  {e:6.3f}  {de:6.3f}  {m:8.2f}")

    plot_results(angles, rel_eff, rel_eff_err, rel_res, rel_res_err)
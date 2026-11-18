import numpy as np
import matplotlib.pyplot as plt
import os
from fitting import fit_roi, gaussian
from geometry import cylinder_geometric_factor, square_geometric_factor
from Efficiency import (get_emission_fraction, get_emission_fraction_error,
                        extract_measurement_date, source_activity, SOURCE_PARAMETERS)
### We should also consider FWHM and peak height as functions of angle. - THomas
plot_folder = "Plots"


def with_error(value, error, sig=2):
    """
    Format 'value ± error' with the error rounded to `sig` significant
    figures and the value rounded to the same decimal place, e.g.
    with_error(12.3456, 0.0432) -> '12.346 ± 0.043'.
    Very large or very small numbers use a shared power of ten.
    A zero error (e.g. the 0 deg reference point) prints the value alone.
    """
    if not np.isfinite(value):
        return "n/a"
    if error == 0:
        return f"{value:.4g}"
    if not np.isfinite(error):
        return f"{value:.4g} ± n/a"

    if value != 0 and (abs(value) >= 1e5 or abs(value) < 1e-3):
        power = int(np.floor(np.log10(abs(value))))
        scale = 10.0**power
        return f"({with_error(value / scale, error / scale, sig)})e{power:+d}"

    decimals = max(sig - 1 - int(np.floor(np.log10(abs(error)))), 0)
    return f"{value:.{decimals}f} ± {error:.{decimals}f}"


def get_rois(detector, source):

    fit_results = detector.spectra[source].fit_results
    
    records = [r for r in detector.calibration_peak_records
               if r["isotope"] == source]
    if not fit_results or not records:
        print(f"{detector.name} {source}: no calibration peaks, skipping.")
        return []

    jobs = []
    for fit in fit_results:
        centroids = [c["centroid"] for c in fit["components"]]
        energies = [min(records, key=lambda r: abs(r["centroid"] - c))["energy"]
                    for c in centroids]
        jobs.append(dict(roi=fit["roi"], n=fit["n_peaks"],
                         guesses=centroids, energies=energies))
    return jobs


def fit_all_angles(spectra, job):
    
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
    os.makedirs(plot_folder, exist_ok=True)
    filename = f"{name} ROI.jpg"
    fig.savefig(os.path.join(plot_folder, filename), dpi=300, bbox_inches="tight")
    plt.close(fig)


def normalise(x, x_err,i0):

    r = x / x[i0]
    r_err = r * np.sqrt((x_err / x) ** 2 + (x_err[i0] / x[i0]) ** 2)
    r_err[i0] = 0.0          # 0 deg divided by itself
    return r, r_err


def analyse(peaks, spectra, calibration):
  
    angles = np.array(sorted(peaks))
    if 0 not  in angles:
        raise ValueError("No 0 deg measurement, cannot normalise.")
    i0 = int(np.where(angles == 0)[0][0])
    def col(key):
        return np.array([peaks[a][key] for a in angles])

    area, area_err = col("area"), col("area_error")
    mu, mu_err = col("centroid"), col("centroid_error")
    fwhm, fwhm_err = col("fwhm"), col("fwhm_error")
    amp, amp_err =col("amplitude"), col("amplitude_error")

    # Resolution dE/E in %, same definition as Resolution.py
    m, c = calibration["gradient"], calibration["intercept"]
    E = m * mu + c
    res = 100 * m * fwhm / E
    res_err = res * np.sqrt((fwhm_err / fwhm) ** 2 + (m * mu_err / E) ** 2)
    fwhm_keV = m*fwhm
    fwhm_keV_err = m*fwhm_err

    # Peak-to-total (only the peak error is propagated)
    total = np.array([spectra[a].corrected_cps().sum() for a in angles])
    ptt, ptt_err = area / total, area_err / total

    r = {"angles": angles, "energy": E, "centroid": mu,
         "centroid_err": mu_err,
         "area": area, "area_err": area_err,
         "res": res, "res_err": res_err, "fwhm_keV": fwhm_keV, 
         "fwhm_keV_err": fwhm_keV_err,
         "height" : amp, "height_err": amp_err
         }
    for key, (x, err) in {"area": (area, area_err), "ptt": (ptt, ptt_err),
                          "res": (res, res_err), "mu": (mu, mu_err),
                          "height": (amp, amp_err), "fwhm":(fwhm_keV,fwhm_keV_err)}.items():
        r["rel_" + key], r["rel_" + key + "_err"] = normalise(x, err,i0)
    return r


TITLES = {
    "rel_area": "Photopeak count rate",
    "rel_ptt": "Peak-to-total ratio",
    "rel_res": "Energy resolution (ΔE/E)",
    "rel_mu":  "Peak centroid (control)",
    "rel_fwhm": "FWHM",
    "rel_height": "Peak height"
}


def plot_comparison(all_r):
    
    for source, energy in sorted({(s, e) for _, s, e in all_r}):
        fig, axes = plt.subplots(2, 3, figsize=(16, 8))
        detector_names = "_".join(sorted({det for det, _, _ in all_r}))
        for ax, (key, title) in zip(axes.flat, TITLES.items()):
            for (det, s, e), r in all_r.items():
                if (s, e) == (source, energy):
                    ax.errorbar(r["angles"], r[key], yerr=r[key + "_err"],
                                fmt="o-", capsize=4, label=det)
            ax.axhline(1, color="k", ls="--", alpha=0.4)
            ax.set_title(f"{detector_names} {title}")
            ax.set_xlabel("Off-axis angle θ (deg)")
            ax.set_ylabel("Relative to 0°")
            ax.grid(alpha=0.3)
            ax.legend()

        fig.suptitle(f"{source} {energy:.1f} keV off-axis response")
        plt.tight_layout()
        
        os.makedirs(plot_folder, exist_ok=True)
        filename = f"{detector_names}_{source}_{energy:.1f}keV.jpg"
        fig.savefig(os.path.join(plot_folder, filename), dpi=300, bbox_inches="tight")
        plt.close(fig)
    


def angular_geometric_factor(detector, distance, angle_degrees):
    """Fraction of isotropically emitted photons incident on a detector face."""
    theta = np.deg2rad(angle_degrees)
    if detector.name in ("BGO", "NaI"):
        return cylinder_geometric_factor(detector.radius, 0.0508,
                                         distance, theta)
    if detector.name == "CdTe":
        return square_geometric_factor(0.003, 0.001, distance, theta)
    raise ValueError(f"Unknown detector geometry: {detector.name}")


def add_angular_efficiencies(detector, source, result, spectra, geometric_factors):
    """From fitted peak count rates, compute absolute and intrinsic efficiencies."""
    emission = get_emission_fraction(source, result["line_energy"])
    emission_error = get_emission_fraction_error(source, result["line_energy"])
    angles = result["angles"]
    activity = np.array([source_activity(
        extract_measurement_date(spectra[int(angle)].filename), source)
        for angle in angles])
    activity_error = activity * SOURCE_PARAMETERS[source]["accuracy"] / 100
    G = np.array([geometric_factors[int(a)] for a in angles])

    rates = result["area"]
    rate_error = result["area_err"]
    if np.any(rates <= 0) or np.any(G <= 0):
        print(f"Warning: {detector.name} {source} {result['line_energy']:.1f}: "
              "nonpositive peak rate or geometric factor; angular efficiencies omitted.")
        return

    abs_eff = rates / (activity * emission)
    abs_err = abs_eff * np.sqrt((rate_error / rates)**2 +
                                (activity_error / activity)**2 +
                                (emission_error / emission)**2)
    intrinsic = abs_eff / G
    intrinsic_err = abs_err / G  

   
    i0 = int(np.where(angles == 0)[0][0])

    
    stat_abs_err = abs_eff * rate_error / rates
    stat_int_err = intrinsic * rate_error / rates

    rel_G, rel_G_err = normalise(G, np.zeros_like(G), i0)
    rel_abs, rel_abs_err = normalise(abs_eff, stat_abs_err, i0)
    rel_int, rel_int_err = normalise(intrinsic, stat_int_err, i0)

    result.update(geometry_factor=G,
                  absolute_efficiency=abs_eff,
                  absolute_efficiency_error=abs_err,
                  intrinsic_efficiency=intrinsic,
                  intrinsic_efficiency_error=intrinsic_err,
                  rel_geometry_factor=rel_G,
                  rel_geometry_factor_err=rel_G_err,
                  rel_absolute_efficiency=rel_abs,
                  rel_absolute_efficiency_err=rel_abs_err,
                  rel_intrinsic_efficiency=rel_int,
                  rel_intrinsic_efficiency_err=rel_int_err)


def plot_angular_efficiencies(detector, source, energy, result):
    """Plot G, absolute and intrinsic efficiency against angle, relative to 0 deg."""
    if "rel_geometry_factor" not in result:
        return

    angles = result["angles"]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    columns = [
        ("rel_geometry_factor", "rel_geometry_factor_err",
         "Geometric factor G"),
        ("rel_absolute_efficiency", "rel_absolute_efficiency_err",
         "Absolute photopeak efficiency"),
        ("rel_intrinsic_efficiency", "rel_intrinsic_efficiency_err",
         "Intrinsic photopeak efficiency"),
    ]
    for ax, (key, err_key, title) in zip(axes, columns):
        ax.errorbar(angles, result[key], yerr=result[err_key],
                    fmt="o-", capsize=3)
        ax.axhline(1, color="k", ls="--", alpha=0.4)
        ax.set(title=title, xlabel="Off-axis angle (degrees)",
               ylabel="Relative to 0°")
        ax.grid(alpha=0.3)
    fig.suptitle(f"{detector.name} {source}: {energy:.1f} keV "
                 "normalised angular efficiencies")
    fig.tight_layout()
    os.makedirs(plot_folder, exist_ok=True)
    for extension in ("pdf", "jpg"):
        fig.savefig(os.path.join(
            plot_folder,
            f"Offaxis_efficiencies_{detector.name}_{source}_{energy:.1f}keV.{extension}"
        ), dpi=300, bbox_inches="tight")
    plt.close(fig)


def run_off_axis(detector, source):
 

    spectra = detector.off_axis_spectra[source]

    geometric_factors = {
        angle: angular_geometric_factor(detector, spectrum.distance, angle)
        for angle, spectrum in spectra.items()
    }

    out = {}
    for job in get_rois(detector, source):
        fits = fit_all_angles(spectra, job)
        if 0 not in fits:
            print(f"{detector.name} {source}: no on-axis fit; skipping this ROI.")
            continue
        plot_fit_checks(spectra, fits, job["roi"], f"{detector.name} {source}")

        for j, energy in enumerate(job["energies"]):
            peaks = {a: f["components"][j] for a, f in fits.items()}
            r = analyse(peaks, spectra, detector.calibration)
            r["line_energy"] = energy
            # Keep the existing get_rois() identification method as requested.
            add_angular_efficiencies(detector, source, r, spectra, geometric_factors)
            plot_angular_efficiencies(detector, source, energy, r)
            out[energy] = r

            print(f"\n{detector.name} {source} {energy:.1f} keV")
            print(f"{'Angle':>5}  {'RelArea':>14}  {'RelP/T':>14}  {'R (%)':>14}  "
                  f"{'Centroid':>16}  {'Amplitude':>16}  {'FWHM (keV)':>14}  "
                  f"{'G':>10}  {'AbsEff':>20}  {'IntEff':>20}")
            for i, a in enumerate(r["angles"]):
                print(f"{a:5.0f}  "
                      f"{with_error(r['rel_area'][i], r['rel_area_err'][i]):>14}  "
                      f"{with_error(r['rel_ptt'][i], r['rel_ptt_err'][i]):>14}  "
                      f"{with_error(r['res'][i], r['res_err'][i]):>14}  "
                      f"{with_error(r['centroid'][i], r['centroid_err'][i]):>16}  "
                      f"{with_error(r['height'][i], r['height_err'][i]):>16}  "
                      f"{with_error(r['fwhm_keV'][i], r['fwhm_keV_err'][i]):>14}  "
                      f"{r['geometry_factor'][i]:10.3e}  "
                      f"{with_error(r['absolute_efficiency'][i], r['absolute_efficiency_error'][i]):>20}  "
                      f"{with_error(r['intrinsic_efficiency'][i], r['intrinsic_efficiency_error'][i]):>20}")
    plt.show()      # show the fit checks for this source
    return out

"""
Efficiency.py
--------------

Calculate absolute detector efficiency using fitted Gaussian peaks.

Works with:
    Detector.py
    fitting.py
    interactive_fitting.py
    Data_Extraction.py

Usage:
    detector.calibrate()

    from Efficiency import absolute_efficiency

    result = absolute_efficiency(detector)

    print(result)
"""

from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
from geometry import cylinder_geometric_factor, square_geometric_factor
from fitting import fit_roi
from scipy.optimize import curve_fit


# ============================================================
# SOURCE INFORMATION
# ============================================================

SOURCE_PARAMETERS = {
    "Am": {
        "initial_activity": 11.92,
        "accuracy": 5.0,
        "half_life": 432.2
    },

    "Ba": {
        "initial_activity": 10.82,
        "accuracy": 4.8,
        "half_life": 10.537
    },

    "Co": {
        "initial_activity": 11.32,
        "accuracy": 1.9,
        "half_life": 5.2714
    },

    "Cs": {
        "initial_activity": 12.41,
        "accuracy": 3.7,
        "half_life": 30.09
    }
}


# Initial activities are assumed to be in microcuries.
# Verify this against your source certificates.

UCI_TO_BQ = 37000

CALIBRATION_DATE = "1979/02/01 12:00:00"


# Gamma emission probabilities (photons per decay).
# Add the remaining values from your laboratory manual.

# Photon emission fractions and their 1-sigma uncertainties from the
# course manual, Table 2, p. 57. Keys are photopeak energies in keV.
EMISSION_FRACTIONS = {
    "Co": {1173.228: 0.9985, 1332.492: 0.999826},
    "Ba": {53.1622: 0.0214, 79.6142: 0.0265, 80.9979: 0.329,
           276.3989: 0.0716, 302.8508: 0.1834,
           356.0129: 0.6205, 383.8485: 0.0894},
    "Cs": {661.657: 0.8499},
    "Am": {26.3446: 0.024, 33.1963: 0.00121, 59.5409: 0.3578},
}

EMISSION_FRACTION_ERRORS = {
    "Co": {1173.228: 0.0003, 1332.492: 0.000006},
    "Ba": {53.1622: 0.0003, 79.6142: 0.0005, 80.9979: 0.003,
           276.3989: 0.0005, 302.8508: 0.0013,
           356.0129: 0.0019, 383.8485: 0.0006},
    "Cs": {661.657: 0.0020},
    "Am": {26.3446: 0.003, 33.1963: 0.00003, 59.5409: 0.0009},
}



# ============================================================
# DATE INFORMATION
# ============================================================

def source_age(measurement_date):
    """
    Calculate the age of the radioactive source in seconds.
    """

    calibration_time = datetime.strptime(
        CALIBRATION_DATE,
        "%Y/%m/%d %H:%M:%S"
    )

    measurement_time = datetime.strptime(
        measurement_date,
        "%m/%d/%Y %H:%M:%S"
    )

    age = (
        measurement_time - calibration_time
    ).total_seconds()

    if age < 0:
        raise ValueError(
            "Measurement date is earlier than the source "
            "calibration date."
        )

    return age


def source_activity(measurement_date, source):
    """
    Calculate the remaining source activity in Bq.

    Uses radioactive decay:

        A(t) = A0 * 0.5 ** (t / T_half)

    Initial activity is assumed to be in microcuries.
    """

    if source not in SOURCE_PARAMETERS:
        raise ValueError(
            f"Unknown radioactive source: {source}"
        )

    parameters = SOURCE_PARAMETERS[source]

    initial_activity = parameters["initial_activity"]

    half_life_years = parameters["half_life"]

    # Convert half-life from years to seconds.
    half_life_seconds = (
        half_life_years * 365.25 * 24 * 60 * 60
    )

    age = source_age(measurement_date)

    remaining_activity_uci = (
        initial_activity *
        0.5 ** (age / half_life_seconds)
    )

    # Convert microcuries to Becquerels.
    activity_bq = remaining_activity_uci * UCI_TO_BQ

    return activity_bq


# ============================================================
# EXTRACT MEASUREMENT DATE
# ============================================================

def extract_measurement_date(filename):
    """
    Extract measurement date from an SPE or MCA file.

    Supports the formats used in the spectroscopy files.
    """

    path = Path(filename)

    if not path.exists():
        raise FileNotFoundError(
            f"Could not find spectrum file: {filename}"
        )

    suffix = path.suffix.lower()

    with open(path, "r") as file:
        lines = [line.strip() for line in file]

    if suffix == ".spe":

        try:
            index = lines.index("$DATE_MEA:")
            measurement_date = lines[index + 1]

        except (ValueError, IndexError):
            raise ValueError(
                "Could not find $DATE_MEA in SPE file."
            )

    elif suffix == ".mca":

        measurement_date = None

        for line in lines:
            if line.startswith("START_TIME - "):
                measurement_date = line.replace(
                    "START_TIME - ", ""
                ).strip()
                break

        if measurement_date is None:
            raise ValueError(
                "Could not find START_TIME in MCA file."
            )

    else:
        raise ValueError(
            f"Unsupported file type: {suffix}"
        )

    # Validate date format.
    datetime.strptime(
        measurement_date,
        "%m/%d/%Y %H:%M:%S"
    )

    return measurement_date


# ============================================================
# SELECT A PEAK
# ============================================================

def select_peak(detector, source):
    """
    Display fitted peaks for a source and ask the user
    which peak should be used for efficiency.
    """

    records = [
        peak
        for peak in detector.calibration_peak_records
        if peak["isotope"] == source
    ]

    if not records:
        raise ValueError(
            f"No fitted calibration peaks found for {source}."
        )

    records = sorted(
        records,
        key=lambda peak: peak["energy"]
    )

    print("\nAvailable fitted peaks")
    print("-" * 65)

    for i, peak in enumerate(records, 1):

        print(
            f"{i}. Energy: {peak['energy']:.3f} keV | "
            f"Centroid: {peak['centroid']:.2f} channels | "
            f"Count rate: {peak['area']:.3f} CPS"
        )

    print("-" * 65)

    while True:

        answer = input(
            "\nSelect the peak number for efficiency calculation: "
        ).strip()

        try:
            choice = int(answer)

            if 1 <= choice <= len(records):
                return records[choice - 1]

            print("Please select one of the listed peaks.")

        except ValueError:
            print("Please enter a valid peak number.")


# ============================================================
# EMISSION FRACTION
# ============================================================

def get_emission_fraction(source, energy):
    """
    Obtain the gamma emission fraction.

    Uses the stored emission fractions when available.
    Otherwise, asks the user to enter the value.
    """

    source_lines = EMISSION_FRACTIONS.get(source, {})

    for known_energy, fraction in source_lines.items():

        if np.isclose(
            energy,
            known_energy,
            atol=0.01
        ):
            print(
                f"Using emission fraction {fraction} "
                f"for {energy:.3f} keV."
            )

            return fraction

    print(
        f"\nNo stored emission fraction for "
        f"{source} at {energy:.3f} keV."
    )

    while True:

        answer = input(
            "Enter emission fraction (0 to 1): "
        ).strip()

        try:
            fraction = float(answer)

            if 0 < fraction <= 1:
                return fraction

            print("Emission fraction must be greater than 0 and at most 1.")

        except ValueError:
            print("Please enter a valid number.")


def get_emission_fraction_error(source, energy):
    """1-sigma uncertainty in a Table 2 photon emission probability."""
    for known_energy, uncertainty in EMISSION_FRACTION_ERRORS.get(source, {}).items():
        if np.isclose(energy, known_energy, rtol=0, atol=0.01):
            return uncertainty
    print(f"Warning: no emission-fraction uncertainty for {source} {energy:.3f} keV.")
    return 0.0


# ============================================================
# FIT ADDITIONAL DISTANCE SPECTRA
# ============================================================

def fit_distance_peaks(detector, distance):
    """Fit additional distance spectra using the original calibration ROIs."""

    records = []

    for source, spectrum in detector.distance_spectra[distance].items():

        # Reuse ROIs selected during calibration
        for original_fit in detector.spectra[source].fit_results:

            guesses = [
                peak["centroid"]
                for peak in original_fit["components"]
            ]

            try:
                fit = fit_roi(
                    spectrum,
                    *original_fit["roi"],
                    n=original_fit["n_peaks"],
                    corrected=True,
                    peak_guesses=guesses if original_fit["n_peaks"] == 2 else None
                )

            except (ValueError, RuntimeError) as error:
                print(f"{source}: peak fitting failed ({error})")
                continue

            # Retain the original photopeak identifications
            for original, fitted in zip(
                original_fit["components"], fit["components"]
            ):

                if original.get("accepted"):

                    records.append({
                        **fitted,
                        "isotope": source,
                        "energy": original["energy"],
                        "source": spectrum.filename,
                        "accepted": True
                    })

    return records


# ============================================================
# ABSOLUTE EFFICIENCY
# ============================================================

def absolute_efficiency(detector, spectra=None, peak_records=None):
    """
    Calculate absolute photopeak efficiencies.

    By default, use the calibration spectra and their fitted peaks.
    Alternative spectra and peak records can be supplied for
    measurements at other distances.
    """

    if spectra is None:
        spectra = detector.spectra

    if peak_records is None:
        peak_records = detector.calibration_peak_records

    results = {}

    if detector.calibration is None:
        raise ValueError(
            "Calibrate the detector before calculating efficiency."
        )

    if not peak_records:
        raise ValueError(
            "No fitted peak records are available."
        )

    # --------------------------------------------------------
    # Loop through sources
    # --------------------------------------------------------

    for source, spectrum in spectra.items():

        if source not in SOURCE_PARAMETERS:
            print(f"\nSkipping unknown source: {source}")
            continue

        # Retrieve fitted peaks belonging to this source
        peaks = [
            peak for peak in peak_records
            if peak["isotope"] == source
        ]

        if not peaks:
            print(f"\nNo fitted peaks available for {source}.")
            continue

        # Obtain measurement date and remaining activity
        measurement_date = extract_measurement_date(
            spectrum.filename
        )

        activity_bq = source_activity(
            measurement_date,
            source
        )

        activity_error = (
            activity_bq *
            SOURCE_PARAMETERS[source]["accuracy"] / 100
        )

        results[source] = []

        print(f"\nSource: {source}")
        print(f"Measurement date: {measurement_date}")
        print(f"Activity: {activity_bq:.3f} +/- {activity_error:.3f} Bq")

        # Calculate efficiency for every fitted peak
        for peak in peaks:

            energy = peak["energy"]
            count_rate = peak["area"]
            count_rate_error = peak["area_error"]

            emission_fraction = get_emission_fraction(
                source,
                energy
            )

            emission_fraction_error = get_emission_fraction_error(source, energy)

            absolute_efficiency = (
                count_rate /
                (activity_bq * emission_fraction)
            )

            # Propagate peak-area and source-activity uncertainties
            relative_variance = (
                (count_rate_error / count_rate) ** 2
                + (activity_error / activity_bq) ** 2
                + (emission_fraction_error / emission_fraction) ** 2
            ) if count_rate > 0 else np.nan

            efficiency_error = (
                absolute_efficiency * np.sqrt(relative_variance)
            )

            result = {
                "source": source,
                "energy": energy,
                "count_rate": count_rate,
                "count_rate_error": count_rate_error,
                "activity_bq": activity_bq,
                "activity_error": activity_error,
                "emission_fraction": emission_fraction,
                "emission_fraction_error": emission_fraction_error,
                "absolute_efficiency": absolute_efficiency,
                "efficiency_error": efficiency_error,
            }

            results[source].append(result)

            print(
                f"\n  Energy: {energy:.3f} keV"
                f"\n  Count rate: {count_rate:.4f} +/- "
                f"{count_rate_error:.4f} CPS"
                f"\n  Emission fraction: {emission_fraction:.6f}"
                f"\n  Absolute efficiency: "
                f"{absolute_efficiency:.6e} +/- "
                f"{efficiency_error:.2e}"
            )

    print("\n" + "=" * 65)
    print("EFFICIENCY CALCULATION COMPLETE")
    print("=" * 65)

    return results



def intrinsic_eff(detector, efficiency_results=None, spectra=None):
    """
    Calculate intrinsic efficiency using the detector geometry.
    Supports measurements at different distances.
    """

    if efficiency_results is None:
        efficiency_results = detector.efficiency

    if spectra is None:
        spectra = detector.spectra

    if not efficiency_results:
        raise ValueError("Calculate absolute efficiency first.")

    results = {}

    for source, peaks in efficiency_results.items():

        dist = spectra[source].distance

        # Detector dimensions from Sheila:
        # BGO and NaI: 2" diameter x 2" length
        # CdTe: 3 x 3 x 1 mm
        # On-axis measurements: theta = 0 radians

        if detector.name in ("BGO", "NaI"):
            G = cylinder_geometric_factor(
                detector.radius, 0.0508, dist, 0
            )

        elif detector.name == "CdTe":
            G = square_geometric_factor(
                0.003, 0.001, dist, 0
            )

        else:
            raise ValueError(f"Unknown detector: {detector.name}")

        results[source] = []

        for peak in peaks:

            abs_eff = peak["absolute_efficiency"]
            abs_err = peak["efficiency_error"]

            updated_peak = peak.copy()

            updated_peak["geometry_factor"] = G
            updated_peak["intrinsic_efficiency"] = abs_eff / G
            updated_peak["intrinsic_efficiency_error"] = abs_err / G

            results[source].append(updated_peak)
        print(f"\nSource: {source}")
        print(
                        
                        f"\n  Intrinsic efficiency: "
                        f"{ updated_peak["intrinsic_efficiency"]:.6e} +/- "
                        f"{updated_peak["intrinsic_efficiency_error"]:.2e}"
                    )
        
        print("\n" + "=" * 65)
        print("INTRINSIC EFFICIENCY CALCULATION COMPLETE")
        print("=" * 65)

    return results



def calculate_distance_efficiencies(detector):
    """Calculate efficiencies for all additional distances."""

    results = {}

    for distance, spectra in detector.distance_spectra.items():

        # Fit peaks using the original calibration ROIs
        peaks = fit_distance_peaks(detector, distance)

        # Calculate absolute efficiency
        absolute = absolute_efficiency(
            detector,
            spectra=spectra,
            peak_records=peaks
        )

        # Calculate intrinsic efficiency
        results[distance] = intrinsic_eff(
            detector,
            efficiency_results=absolute,
            spectra=spectra
        )

    return results



def efficiency_model(x, a, b, c):
    """Quadratic model for ln(efficiency) as a function of ln(energy)."""
    return a + b*x + c*x**2


def fit_intrinsic_efficiency(detector):
    """Fit ln(eps) = a + b ln(E) + c ln(E)^2 (course manual, Eq. 20)."""
    if not detector.efficiency:
        raise ValueError("Calculate intrinsic efficiency first.")

    peaks = [p for source_peaks in detector.efficiency.values() for p in source_peaks]
    E = np.array([p["energy"] for p in peaks], dtype=float)
    eff = np.array([p["intrinsic_efficiency"] for p in peaks], dtype=float)
    err = np.array([p["intrinsic_efficiency_error"] for p in peaks], dtype=float)

    valid = (np.isfinite(E) & np.isfinite(eff) & np.isfinite(err)
             & (E > 0) & (eff > 0) & (eff <= 1) & (err > 0))
    if not np.all(valid):
        print(f"Warning: {detector.name}: {sum(~valid)} intrinsic-efficiency "
              "point(s) are nonphysical or invalid and will not be fitted.")

    if valid.sum() < 4:
        print(f"{detector.name}: need at least four valid efficiency points "
              "to fit a quadratic in log-energy and assess its residuals.")
        return None

    x = np.log(E[valid])
    y = np.log(eff[valid])
    sigma_y = err[valid] / eff[valid]  # first-order uncertainty in ln(eff)
    try:
        coefficients, covariance = curve_fit(
            efficiency_model, x, y,
            sigma=sigma_y, absolute_sigma=True
        )
    except (RuntimeError, ValueError) as error:
        print(f"{detector.name}: intrinsic-efficiency fit failed ({error}).")
        return None

    a, b, c = coefficients
    print(f"\n{detector.name} intrinsic efficiency: ln(eps) = a + b ln(E) + c ln(E)^2")
    for label, value, uncertainty in zip(
        "abc", (a, b, c), np.sqrt(np.diag(covariance))
    ):
        print(f"  {label} = {value:.5g} +/- {uncertainty:.2g}")

    return {"coefficients": np.array([a, b, c]), "covariance": covariance,
            "energy_min": float(E[valid].min()),
            "energy_max": float(E[valid].max())}


def plot_efficiency(detector, kind, show=False):
    if not detector.efficiency:
        raise ValueError("Calculate efficiency first.")

    if kind == "absolute":
        value_key, error_key = "absolute_efficiency", "efficiency_error"
    elif kind == "intrinsic":
        value_key, error_key = "intrinsic_efficiency", "intrinsic_efficiency_error"
    else:
        raise ValueError(f"Unknown efficiency type: {kind}")

    fig, ax = plt.subplots(figsize=(8, 5.5))
    for source, peaks in detector.efficiency.items():
        E = np.array([p["energy"] for p in peaks], dtype=float)
        values = np.array([p[value_key] for p in peaks], dtype=float)
        errors = np.array([p[error_key] for p in peaks], dtype=float)
        valid = (np.isfinite(E) & np.isfinite(values) & np.isfinite(errors)
                 & (E > 0) & (values > 0) & (errors >= 0))
        if valid.any():
            ax.errorbar(E[valid], values[valid], yerr=errors[valid],
                        fmt="o", capsize=3, label=source)
        if (~valid).any():
            print(f"Warning: {detector.name} {source}: {sum(~valid)} points "
                  "could not be shown on logarithmic axes.")

    if kind == "intrinsic":
        fit = getattr(detector, "efficiency_fit", None)
        if fit is not None:
            # Display the model beyond the fitted points, as requested.
            # Outside measured energies it is an extrapolation.
            E_line = np.logspace(np.log10(10), np.log10(2000), 500)
            curve = np.exp(efficiency_model(np.log(E_line), *fit["coefficients"]))
            ax.plot(
                E_line, curve, "r-", lw=2,
                label=r"Fit: $\ln\varepsilon_{\mathrm{int}} = a + b\ln E + c(\ln E)^2$"
            )
        ax.axhline(1, ls="--", lw=1, color="gray", alpha=0.6)

    ax.set(xscale="log", yscale="log", xlabel="Gamma-ray energy (keV)",
           ylabel=f"{kind.capitalize()} photopeak efficiency",
           title=f"{detector.name} {kind} photopeak efficiency")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    folder = Path("Plots")
    folder.mkdir(exist_ok=True)
    for extension in ("pdf", "jpg"):
        fig.savefig(folder / f"Efficiency_{kind}_{detector.name}.{extension}",
                    dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


def plot_absolute_efficiency(detector, show=False):
    """Plot measured absolute photopeak efficiency against energy."""
    plot_efficiency(detector, "absolute", show=show)


def plot_intrinsic_efficiency(detector, show=False):
    """Plot intrinsic efficiency plus the fitted log-polynomial model."""
    plot_efficiency(detector, "intrinsic", show=show)

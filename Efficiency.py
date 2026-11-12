
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

import numpy as np


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

EMISSION_FRACTIONS = {

    "Co": {
        1173.228: 0.9985,
        1332.492: 0.999826
    }

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


# ============================================================
# ABSOLUTE EFFICIENCY
# ============================================================

def absolute_efficiency(detector, source=None):
    """
    Calculate absolute efficiency at a user-selected photopeak.

    Parameters
    ----------
    detector:
        A calibrated Detector object.

    source:
        Optional isotope name, e.g. "Co".
        If omitted, the user selects a source.

    Returns
    -------
    Dictionary containing:
        source
        energy
        count_rate
        count_rate_error
        activity_bq
        emission_fraction
        absolute_efficiency
        efficiency_error
    """

    if detector.calibration is None:
        raise ValueError(
            "Calibrate the detector before calculating efficiency."
        )

    if not detector.calibration_peak_records:
        raise ValueError(
            "No calibration peak records are available."
        )

    # --------------------------------------------------------
    # Select source
    # --------------------------------------------------------

    available_sources = list(dict.fromkeys(
        peak["isotope"]
        for peak in detector.calibration_peak_records
    ))

    if source is None:

        print("\nAvailable radioactive sources:")

        for i, name in enumerate(available_sources, 1):
            print(f"{i}. {name}")

        while True:

            answer = input(
                "\nSelect source number: "
            ).strip()

            try:
                choice = int(answer)

                if 1 <= choice <= len(available_sources):
                    source = available_sources[choice - 1]
                    break

                print("Invalid selection.")

            except ValueError:
                print("Please enter a number.")

    elif source not in available_sources:

        raise ValueError(
            f"No fitted peaks available for {source}."
        )

    # --------------------------------------------------------
    # Select fitted peak
    # --------------------------------------------------------

    peak = select_peak(detector, source)

    energy = peak["energy"]

    count_rate = peak["area"]
    count_rate_error = peak["area_error"]

    # --------------------------------------------------------
    # Determine measurement date and source activity
    # --------------------------------------------------------

    spectrum_filename = next(
        spectrum.filename
        for isotope, spectrum in detector.spectra.items()
        if isotope == source
    )

    # Spectrum.filename may only contain the basename.
    # Look in the CUITS folder if it is not in the current directory.

    path = Path(spectrum_filename)

    if not path.exists():
        path = Path(__file__).parent / "CUITS" / spectrum_filename

    measurement_date = extract_measurement_date(path)

    activity_bq = source_activity(
        measurement_date,
        source
    )

    # --------------------------------------------------------
    # Obtain emission fraction
    # --------------------------------------------------------

    emission_fraction = get_emission_fraction(
        source,
        energy
    )

    # --------------------------------------------------------
    # Calculate absolute efficiency
    # --------------------------------------------------------

    denominator = activity_bq * emission_fraction

    efficiency = count_rate / denominator

    # Propagate the peak-area uncertainty.
    efficiency_error = count_rate_error / denominator

    # --------------------------------------------------------
    # Display results
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print(f"{detector.name} ABSOLUTE EFFICIENCY")
    print("=" * 60)

    print(f"Source:              {source}")
    print(f"Gamma energy:        {energy:.3f} keV")

    print(
        f"Count rate:          "
        f"{count_rate:.4f} +/- {count_rate_error:.4f} CPS"
    )

    print(f"Source activity:     {activity_bq:.3f} Bq")
    print(f"Emission fraction:   {emission_fraction:.6f}")

    print(
        f"Absolute efficiency: "
        f"{efficiency:.6e} +/- {efficiency_error:.6e}"
    )

    print("=" * 60)

    return {
        "source": source,
        "energy": energy,
        "count_rate": count_rate,
        "count_rate_error": count_rate_error,
        "activity_bq": activity_bq,
        "emission_fraction": emission_fraction,
        "absolute_efficiency": efficiency,
        "efficiency_error": efficiency_error
    }

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


def calibrate(peaks, energies):

    if len(peaks) != len(energies):
        raise ValueError(
            "Need one energy for each fitted peak."
        )

    channels = np.array([
        peak["centroid"]
        for peak in peaks
    ])

    channel_errors = np.array([
        peak["centroid_error"]
        for peak in peaks
    ])

    energies = np.asarray(
        energies,
        dtype=float
    )

    def linear(channel, m, c):
        return m * channel + c

    popt, pcov = curve_fit(
        linear,
        channels,
        energies,
        sigma=np.maximum(channel_errors, 1e-6),
        absolute_sigma=True
    )

    gradient, intercept = popt

    fitted_energies = linear(
        channels,
        gradient,
        intercept
    )

    residuals = energies - fitted_energies

    return {
        "gradient": gradient,
        "intercept": intercept,
        "covariance": pcov,
        "channels": channels,
        "channel_errors": channel_errors,
        "energies": energies,
        "residuals": residuals
    }


def channel_to_energy(channel, calibration):

    return (
        calibration["gradient"] * channel
        + calibration["intercept"]
    )


def energy_to_channel(energy, calibration):

    return (
        energy - calibration["intercept"]
    ) / calibration["gradient"]


def plot_calibration(calibration):

    channels = calibration["channels"]
    errors = calibration["channel_errors"]
    energies = calibration["energies"]

    m = calibration["gradient"]
    c = calibration["intercept"]

    x = np.linspace(
        0,
        channels.max() * 1.1,
        500
    )

    fig, (ax1, ax2) = plt.subplots(
        2,
        1,
        figsize=(8, 8),
        gridspec_kw={"height_ratios": [3, 1]},
        sharex=True
    )

    ax1.errorbar(
        channels,
        energies,
        xerr=errors,
        fmt="o",
        color="black",
        label="Calibration Points"
    )

    ax1.plot(
        x,
        m * x + c,
        color="red",
        label=f"E = {m:.6f}C + {c:.3f}"
    )

    ax1.set_ylabel("Energy (keV)")
    ax1.set_title("Energy Calibration")
    ax1.grid(alpha=0.3)
    ax1.legend()

    ax2.axhline(
        0,
        color="black",
        linewidth=1
    )

    ax2.plot(
        channels,
        calibration["residuals"],
        "o"
    )

    ax2.set_xlabel("Channel")
    ax2.set_ylabel("Residual\n(keV)")
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.show()

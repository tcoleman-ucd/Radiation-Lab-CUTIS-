"""Fit and compare linear and quadratic energy calibrations."""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import os

def linear(channel, m, c):
    return m * channel + c


def quadratic(channel, a, m, c):
    return a * channel**2 + m * channel + c


def calibrate(peaks, energies):

    if len(peaks) != len(energies):
        raise ValueError("Need one energy for each fitted peak.")

    if len(peaks) < 2:
        raise ValueError("At least two calibration peaks are needed.")

    channels = np.array([peak["centroid"] for peak in peaks])
    channel_errors = np.array([peak["centroid_error"] for peak in peaks])
    energies = np.asarray(energies, dtype=float)

    # Fit a straight line: E = m*C + c.
    linear_parameters, linear_covariance = curve_fit(
        linear,
        channels,
        energies
    )

    gradient, intercept = linear_parameters
    linear_residuals = energies - linear(channels, gradient, intercept)
    linear_rms = np.sqrt(np.mean(linear_residuals**2))

    # Try a quadratic too, to see whether the calibration is really linear.
    # Four points give at least one residual degree of freedom for this fit.
    quadratic_result = None

    if len(channels) >= 4:
        quadratic_parameters, quadratic_covariance = curve_fit(
            quadratic,
            channels,
            energies,
            p0=[0, gradient, intercept]
        )

        quadratic_residuals = energies - quadratic(channels, *quadratic_parameters)
        quadratic_rms = np.sqrt(np.mean(quadratic_residuals**2))

        quadratic_result = {
            "coefficients": quadratic_parameters,
            "covariance": quadratic_covariance,
            "residuals": quadratic_residuals,
            "rms": quadratic_rms
        }

    # Keep the linear calibration as the default for the rest of the code.
    return {
        "gradient": gradient,
        "intercept": intercept,
        "covariance": linear_covariance,
        "channels": channels,
        "channel_errors": channel_errors,
        "energies": energies,
        "residuals": linear_residuals,
        "linear_rms": linear_rms,
        "quadratic": quadratic_result
    }


def channel_to_energy(channel, calibration):
    return calibration["gradient"] * channel + calibration["intercept"]


def energy_to_channel(energy, calibration):
    return (energy - calibration["intercept"]) / calibration["gradient"]


def plot_calibration(calibration):
    plot_folder = "Plots"
    channels = calibration["channels"]
    channel_errors = calibration["channel_errors"]
    energies = calibration["energies"]

    gradient = calibration["gradient"]
    intercept = calibration["intercept"]
    quadratic_fit = calibration["quadratic"]

    # Only compare the curves over the range covered by our calibration points.
    x = np.linspace(channels.min(), channels.max(), 500)

    fig, (ax1, ax2) = plt.subplots(
        2, 1,
        figsize=(8, 8),
        gridspec_kw={"height_ratios": [3, 1]},
        sharex=True
    )

    ax1.errorbar(
        channels, energies,
        xerr=channel_errors,
        fmt="o", color="black",
        label="Calibration points"
    )

    ax1.plot(x, linear(x, gradient, intercept),
             color="red", label="Linear fit")

    ax2.axhline(0, color="black", linewidth=1)
    ax2.plot(channels, calibration["residuals"], "o",
             color="red", label="Linear residuals")

    if quadratic_fit is not None:
        a, m, c = quadratic_fit["coefficients"]

        ax1.plot(x, quadratic(x, a, m, c),
                 color="blue", linestyle="--", label="Quadratic fit")

        ax2.plot(channels, quadratic_fit["residuals"], "s",
                 color="blue", label="Quadratic residuals")

        # Check whether the quadratic coefficient differs significantly from zero.
        a_error = np.sqrt(quadratic_fit["covariance"][0, 0])
        print(f"Quadratic coefficient: {a:.6g} +/- {a_error:.6g} keV/channel^2")
        print(f"Linear RMS residual: {calibration['linear_rms']:.3f} keV")
        print(f"Quadratic RMS residual: {quadratic_fit['rms']:.3f} keV")
    else:
        print("At least four calibration points are needed to compare a quadratic fit.")

    ax1.set_ylabel("Energy (keV)")
    ax1.set_title("Linear and Quadratic Energy Calibration")
    ax1.grid(alpha=0.3)
    ax1.legend()

    ax2.set_xlabel("Channel")
    ax2.set_ylabel("Residual (keV)")
    ax2.grid(alpha=0.3)
    ax2.legend()

    file_path = os.path.join(plot_folder, "Calibration.jpg")
    plt.savefig(file_path, format = "jpg", dpi=300, bbox_inches = "tight")
    plt.close()

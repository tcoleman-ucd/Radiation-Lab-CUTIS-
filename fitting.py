"""Gaussian peak fitting functions that accept a Spectrum object."""

import numpy as np
from scipy.optimize import curve_fit


# True: identify energies in the original plot. False: identify in the terminal.
IDENTIFY_IN_PLOT = True

# Keeping this incase we want to return to it
# def continuum(x, a, b, c, d):
#     """Third-order polynomial continuum, as in the previous assignment."""
#     return a * x**3 + b * x**2 + c * x + d

# Linear continuum
def continuum(x, m, c):
    """Linear background beneath a photopeak."""
    return m * x + c


def gaussian(x, A1, mu1, sigma1, m, c,
             A2=None, mu2=None, sigma2=None):

    """One or two area-normalised Gaussians on a linear continuum."""

    peak1 = (A1 / (sigma1 * np.sqrt(2 * np.pi))) * np.exp(
        -((x - mu1)**2) / (2 * sigma1**2)
    )

    fitted_continuum = continuum(x, m, c)
    fit = peak1 + fitted_continuum

    # Optional second Gaussian
    if all(p is not None for p in (A2, mu2, sigma2)):

        peak2 = (A2 / (sigma2 * np.sqrt(2 * np.pi))) * np.exp(
            -((x - mu2)**2) / (2 * sigma2**2)
        )

        fit = fit + peak2

    elif any(p is not None for p in (A2, mu2, sigma2)):

        raise ValueError(
            "The second Gaussian requires A2, mu2 and sigma2."
        )

    return fit


def fit_roi(spectrum, lower, upper, n=1, corrected=True, peak_guesses=None):
    """
    Fit one or two Gaussian peaks plus a cubic continuum.

    n = 1 : Single Gaussian
    n = 2 : Two simultaneous Gaussians

    For n=2, provide peak_guesses=[mu1, mu2] in channels.
    Returns fitted parameters and uncertainties.
    """

    if upper <= lower:
        raise ValueError(
            "Upper ROI boundary must exceed lower boundary."
        )

    if n not in (1, 2):
        raise ValueError("Only one or two peaks are supported.")

    if corrected:
        count_rates = spectrum.corrected_cps()
    else:
        count_rates = spectrum.cps()

    # Extract ROI

    mask = ((spectrum.channels >= lower) & (spectrum.channels <= upper))

    channels_roi = spectrum.channels[mask]
    counts_roi = count_rates[mask]

    n_parameters = 3 * n + 2
    # We need more data points than fitting parameters
    if len(channels_roi) <= n_parameters:
        raise ValueError("Not enough channels to fit the Gaussian model.")

    # Ok we're just gonna start with zero continuum coefficients
    # and let curve_fit() find it.
    # a_initial = 0.0
    # b_initial = 0.0
    # c_initial = 0.0
    # d_initial = 0.0

    # Ok it doesnt work very well.
    # Below is an AI suggestion it seems to work much better - Thomas

    # Estimate a straight-line continuum from the ends of the ROI.
    # This is only an initial guess, curve_fit() will still improve this
    # Estimate the continuum from the edges of the ROI
    n_channels = len(channels_roi)
    edge_n = max(2, n_channels // 5)

    # Average the first and last 20% of the ROI
    x1 = np.mean(channels_roi[:edge_n])
    y1 = np.mean(counts_roi[:edge_n])

    x2 = np.mean(channels_roi[-edge_n:])
    y2 = np.mean(counts_roi[-edge_n:])

    # Initial gradient and intercept
    m_initial = (y2 - y1) / (x2 - x1)
    c_initial = y1 - m_initial * x1

    # Subtract the estimated continuum to find the peak
    continuum_initial = continuum(
        channels_roi, m_initial, c_initial
    )

    residual = counts_roi - continuum_initial

    # Estimate one peak automatically, or use two manually chosen positions.
    if n == 1:
        mu1_initial = channels_roi[np.argmax(residual)]  # The highest count rate above the continuum
    else:
        if peak_guesses is None or len(peak_guesses) != 2:
            raise ValueError("Provide two approximate peak positions in channels.")

        mu1_initial, mu2_initial = sorted(peak_guesses)

    # Estimate initial peak positions
    if n == 1:
        mu1_initial = channels_roi[np.argmax(residual)]

    else:
        if peak_guesses is None or len(peak_guesses) != 2:
            raise ValueError(
                "Provide two approximate peak positions in channels."
            )

        mu1_initial, mu2_initial = sorted(peak_guesses)

    # Estimate the initial Gaussian amplitude and width
    if n == 1:

        amplitude1_initial = np.max(residual) # the highest count above the continuum

        if amplitude1_initial <= 0:
            raise ValueError("No positive peak found in the ROI.")

        half_max_channels = channels_roi[
            residual >= amplitude1_initial / 2
            ]

        fwhm_initial = half_max_channels[-1] - half_max_channels[0]

        if fwhm_initial <= 0:
            raise ValueError("Could not estimate a positive FWHM.")

        sigma_initial = fwhm_initial / 2.355

    else:

        # Initial width estimate for two overlapping peaks
        sigma_initial = (upper - lower) / 10

        amplitude1_initial = residual[
            np.argmin(np.abs(channels_roi - mu1_initial))
        ]

        if amplitude1_initial <= 0:
            raise ValueError("No positive peak at the selected position.")

    # Convert the estimated amplitude to integrated Gaussian area
    A1_initial = amplitude1_initial * sigma_initial * np.sqrt(2 * np.pi)

    # Initial guesses for Gaussian area, centroid, width and continuum
    initial_guess = [
        A1_initial, mu1_initial, sigma_initial,
        m_initial, c_initial
    ]

    # Bounds for one Gaussian and a linear continuum
    lower_bounds = [0, lower, 0, -np.inf, -np.inf]
    upper_bounds = [np.inf, upper, np.inf, np.inf, np.inf]

    # Optional second Gaussian
    if n == 2:

        # Estimate the amplitude of the second peak
        amplitude2_initial = residual[
            np.argmin(np.abs(channels_roi - mu2_initial))
        ]

        if amplitude2_initial <= 0:
            raise ValueError("No positive second peak at the selected position.")

        # Convert amplitude to integrated Gaussian area
        A2_initial = amplitude2_initial * sigma_initial * np.sqrt(2 * np.pi)

        # Add the second Gaussian's initial parameters
        initial_guess.extend([
            A2_initial, mu2_initial, sigma_initial
        ])

        # Add bounds for the second Gaussian (area, centroid, width)
        lower_bounds.extend([0, lower, 0])
        upper_bounds.extend([np.inf, upper, np.inf])

    parameters, covariance = curve_fit(
        gaussian,
        channels_roi,
        counts_roi,
        p0=initial_guess,
        bounds=(lower_bounds, upper_bounds),
        maxfev=20000
    )
    # Weighted fitting with Poisson uncertainties would be good here but I don't have time to implement it -Thomas

    if not np.all(np.isfinite(covariance)):
        raise RuntimeError(
            "Fit converged but uncertainties could not be estimated."
        )

    parameter_errors = np.sqrt(np.diag(covariance))

    # Calculate the properties of each fitted Gaussian
    components = []

    # Parameter positions for the first Gaussian, and second if present
    # Order: A1, mu1, sigma1, a, b, c, d, A2, mu2, sigma2
    # Calculate the properties of each fitted Gaussian
    components = []

    # Parameter positions: A1, mu1, sigma1, m, c, A2, mu2, sigma2
    peak_indices = [(0, 1, 2)]

    if n == 2:
        peak_indices.append((5, 6, 7))

    for A_index, mu_index, sigma_index in peak_indices:
        # Extract fitted parameters
        A = parameters[A_index]
        mu = parameters[mu_index]
        sigma = parameters[sigma_index]

        # Extract parameter uncertainties
        A_error = parameter_errors[A_index]
        mu_error = parameter_errors[mu_index]
        sigma_error = parameter_errors[sigma_index]

        # Calculate FWHM and its uncertainty
        fwhm = 2.355 * sigma
        fwhm_error = 2.355 * sigma_error

        # A is already the integrated Gaussian area
        area = A
        area_error = A_error

        # Calculate peak height from the fitted area
        amplitude = A / (sigma * np.sqrt(2 * np.pi))

        # we have to propagate the uncertainties in A and sigma
        # including their covariance

        # Propagate the area and width uncertainties into the height
        dH_dA = 1 / (sigma * np.sqrt(2 * np.pi))
        dH_dsigma = -A / (sigma ** 2 * np.sqrt(2 * np.pi))

        variance_amplitude = (
                dH_dA ** 2 * covariance[A_index, A_index]
                + dH_dsigma ** 2 * covariance[sigma_index, sigma_index]
                + 2 * dH_dA * dH_dsigma * covariance[A_index, sigma_index]
        )


        amplitude_error = np.sqrt(max(variance_amplitude, 0))

        components.append({
            "amplitude": amplitude,
            "amplitude_error": amplitude_error,
            "centroid": mu,
            "centroid_error": mu_error,
            "sigma": sigma,
            "sigma_error": sigma_error,
            "fwhm": fwhm,
            "fwhm_error": fwhm_error,
            "area": area,
            "area_error": area_error
        })

    # Put the lower-channel peak first
    if n == 2:
        if components[0]["centroid"] > components[1]["centroid"]:
            components.reverse()

    # Store the fitting results
    result = {
        "roi": (lower, upper),
        "n_peaks": n,
        "components": components,
        "continuum_coefficients": parameters[3:5],
        "parameters": parameters,
        "covariance": covariance
    }

    # Allow direct access to single Gaussian parameters
    if n == 1:
        result.update(components[0])


    return result


def interactive_fit(spectrum, corrected=True, known_energies=None,
                    identify_in_plot=None):
    """Open the original ROI selector, optionally identifying peaks in the plot."""
    from interactive_fitting import interactive_fit as run_interactive_fit

    if identify_in_plot is None:
        identify_in_plot = IDENTIFY_IN_PLOT

    return run_interactive_fit(
        spectrum,
        corrected=corrected,
        reference_energies= known_energies if identify_in_plot else None,
    )

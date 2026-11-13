"""Gaussian peak fitting functions that accept a Spectrum object."""

import numpy as np
from scipy.optimize import curve_fit


def continuum(x, a, b, c, d):
    """Third-order polynomial continuum, as in the previous assignment."""
    return a * x**3 + b * x**2 + c * x + d


def gaussian(x, A1, mu1, sigma1, a, b, c, d,
             A2=None, mu2=None, sigma2=None):
    """One or two Gaussian peaks on a shared cubic continuum."""

    peak1 = A1 * np.exp(
        -((x - mu1)**2) / (2 * sigma1**2)
    )

    fitted_continuum = continuum(x, a, b, c, d)
    fit = peak1 + fitted_continuum

    # Add second Gaussian if all parameters are provided
    if all(p is not None for p in (A2, mu2, sigma2)):

        peak2 = A2 * np.exp(
            -((x - mu2)**2) / (2 * sigma2**2)
        )

        fit = fit + peak2

    # Raise an error if only some parameters were provided
    elif any(p is not None for p in (A2, mu2, sigma2)):

        raise ValueError("The second Gaussian requires A2, mu2 and sigma2.")

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

    n_parameters = 3 * n + 4
    # We need more data points than fitting parameters,
    # but this is only for the maths, doesn't guarantee a good fit. - Thomas
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
    # This is only an initial guess; the final fit still uses a cubic.
    n_channels = len(channels_roi)
    edge_n = max(2, n_channels // 5)

    edge_channels = np.concatenate([
        channels_roi[:edge_n],
        channels_roi[-edge_n:]
    ])
    edge_counts = np.concatenate([
        counts_roi[:edge_n],
        counts_roi[-edge_n:]
    ])

    m_initial, intercept_initial = np.polyfit(
        edge_channels, edge_counts, 1
    )

    a_initial = 0.0
    b_initial = 0.0
    c_initial = m_initial
    d_initial = intercept_initial

    continuum_initial = continuum(
        channels_roi,
        a_initial, b_initial, c_initial, d_initial
    )
    residual = counts_roi - continuum_initial

    # Estimate one peak automatically, or use two manually chosen positions.
    if n == 1:
        mu1_initial = channels_roi[np.argmax(residual)]

    else:
        if peak_guesses is None or len(peak_guesses) != 2:
            raise ValueError("Provide two approximate peak positions in channels.")

        mu1_initial, mu2_initial = sorted(peak_guesses)

    # Estimate the peak amplitude and width before fitting.
    if n == 1:
        # Estimate the amplitude and width from the continuum-subtracted peak
        A1_initial = np.max(residual)

        half_max_channels = channels_roi[residual >= A1_initial / 2]

        fwhm_initial = half_max_channels[-1] - half_max_channels[0]

        sigma_initial = fwhm_initial / 2.355


    else:
        # Rough initial width for both overlapping peaks
        sigma_initial = (upper - lower) / 10

        # Estimate amplitude at the selected peak position
        A1_initial = residual[
            np.argmin(np.abs(channels_roi - mu1_initial))
        ]

    initial_guess = [
        A1_initial, mu1_initial, sigma_initial,
        a_initial, b_initial, c_initial, d_initial
    ]

    lower_bounds = [0, lower, 0, -np.inf, -np.inf, -np.inf, -np.inf]
    upper_bounds = [np.inf, upper, np.inf, np.inf, np.inf, np.inf, np.inf]
    # These are AI suggested bounds for fitting -Thomas

    # Optional second Gaussian
    if n == 2:
        A2_initial = residual[
            np.argmin(np.abs(channels_roi - mu2_initial))
        ]

        initial_guess.extend([A2_initial, mu2_initial, sigma_initial])

        # These are AI suggested -Thomas
        lower_bounds.extend([0, lower, 0])
        upper_bounds.extend([np.inf, upper, np.inf])

        # NOTE: Currently if the peaks are picked too close together it kinda messes up - Thomas

    parameters, covariance = curve_fit(
        gaussian,
        channels_roi,
        counts_roi,
        p0=initial_guess,
        bounds=(lower_bounds, upper_bounds),
        maxfev=20000
    )

    if not np.all(np.isfinite(covariance)):
        raise RuntimeError(
            "Fit converged but uncertainties could not be estimated."
        )

    parameter_errors = np.sqrt(np.diag(covariance))

    # Calculate the properties of each fitted Gaussian
    components = []

    # Parameter positions for the first Gaussian, and second if present
    # Order: A1, mu1, sigma1, a, b, c, d, A2, mu2, sigma2
    peak_indices = [(0, 1, 2)]

    if n == 2:
        peak_indices.append((7, 8, 9))

    for A_index, mu_index, sigma_index in peak_indices:

        # Extract fitted parameters and their uncertainties
        A = parameters[A_index]
        mu = parameters[mu_index]
        sigma = parameters[sigma_index]

        A_error = parameter_errors[A_index]
        mu_error = parameter_errors[mu_index]
        sigma_error = parameter_errors[sigma_index]

        # Convert Gaussian standard deviation to FWHM
        fwhm = 2.355 * sigma
        fwhm_error = 2.355 * sigma_error

        # Integrated Gaussian area, used later for detector efficiency and off-axis
        area = A * sigma * np.sqrt(2 * np.pi)

        # The area depends on two fitted parameters, A and sigma.
        # Both have uncertainties, so both contribute to the uncertainty in the area.

        # Here are the uncertainties in amplitude and width,
        # including their covariance from the fit
        variance_area = (
            sigma**2 * covariance[A_index, A_index]
            + A**2 * covariance[sigma_index, sigma_index]
            + 2 * A * sigma * covariance[A_index, sigma_index]
        )

        # Alternatively, we could fit the Gaussian area directly like in the manual.
        # Then its uncertainty can be read from the covariance matrix instead.
        # Which I'll probably do - Thomas

        area_error = np.sqrt(max(variance_area, 0)) * np.sqrt(2 * np.pi)

        components.append({
            "amplitude": A,
            "amplitude_error": A_error,
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


    result = {
        "roi": (lower, upper),
        "n_peaks": n,
        "components": components,
        "continuum_coefficients": parameters[3:7],
        "parameters": parameters,
        "covariance": covariance
    }

    # Allow single-peak results to be accessed directly, e.g. result["centroid"]
    if n == 1:
        result.update(components[0])

    return result


def interactive_fit(spectrum, corrected=True):
    """Open the ROI selection interface from its separate module."""
    from interactive_fitting import interactive_fit as run_interactive_fit
    return run_interactive_fit(spectrum, corrected=corrected)

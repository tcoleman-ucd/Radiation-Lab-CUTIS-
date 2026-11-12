"""Interactive ROI selection and plotting (separate from the fitting calculations)."""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import SpanSelector

from fitting import fit_roi, gaussian, continuum

# AI suggested.

def interactive_fit(spectrum, corrected=True):
    """
    Interactively select ROIs by clicking and dragging.

    Controls:

    Left click + drag : Select ROI
    For two peaks     : Click roughly on each peak after choosing the ROI
    1                 : Fit one Gaussian in next ROI
    2                 : Fit two Gaussians in next ROI
    Enter             : Finish analysis
    Backspace         : Remove most recent ROI
    Escape            : Finish analysis
    """

    if corrected:
        count_rates = spectrum.corrected_cps()
        title = "Background Subtracted Spectrum"

    else:
        count_rates = spectrum.cps()
        title = "Raw Spectrum"

    spectrum.rois = []
    spectrum.peaks = []
    spectrum.fit_results = []

    n_peaks = 1

    fig, ax = plt.subplots(figsize=(12, 6))

    fig.subplots_adjust(bottom=0.15)

    # Plot spectrum

    ax.plot(
        spectrum.channels,
        count_rates,
        color="black",
        linewidth=0.8,
        label="Spectrum"
    )

    ax.set_xlabel("Channel")
    ax.set_ylabel("Counts per Second")

    ax.set_title(
        f"{Path(spectrum.filename).name}\n{title}"
    )

    ax.grid(alpha=0.3)

    # Instructions

    instruction = (
        "Click and drag to select an ROI\n"
        "1: Single Gaussian | 2: Double Gaussian\n"
        "Double Gaussian: select ROI, then click each peak\n"
        "ENTER: Finish | BACKSPACE: Undo | ESC: Cancel/Finish"
    )

    instruction_text = ax.text(
        0.02,
        0.96,
        instruction,
        transform=ax.transAxes,
        verticalalignment="top",
        fontsize=9,
        bbox=dict(
            facecolor="white",
            alpha=0.8
        )
    )

    # Store plot artists for each ROI

    roi_artists = []

    # While a double-peak ROI is selected, collect two centroid clicks.
    pending_roi = None
    peak_guesses = []
    guess_artists = []

    def clear_pending():
        nonlocal pending_roi

        pending_roi = None
        peak_guesses.clear()

        for artist in guess_artists:
            artist.remove()
        guess_artists.clear()

        selector.set_active(True)
        instruction_text.set_text(instruction)
        fig.canvas.draw_idle()

    def on_select(xmin, xmax):
        nonlocal pending_roi

        lower = min(xmin, xmax)
        upper = max(xmin, xmax)

        if upper <= lower:
            return

        if n_peaks == 1:
            perform_fit(lower, upper, 1)
            return

        # Pause ROI selection while the user clicks two peak positions.
        pending_roi = (lower, upper)
        peak_guesses.clear()
        selector.set_active(False)

        guess_artists.append(
            ax.axvspan(lower, upper, color="orange", alpha=0.10)
        )
        instruction_text.set_text(
            "Double Gaussian: click the approximate centre of peak 1, "
            "then peak 2\nESC/BACKSPACE: Cancel selection"
        )
        fig.canvas.draw_idle()
        print("\nClick the approximate centre of each of the two peaks.")

    def on_click(event):
        if pending_roi is None or event.inaxes is not ax:
            return
        if event.button != 1 or event.xdata is None:
            return

        lower, upper = pending_roi
        if not (lower < event.xdata < upper):
            print("Please click inside the selected ROI.")
            return

        peak_guesses.append(event.xdata)
        guess_artists.append(
            ax.axvline(event.xdata, color="orange", linestyle=":", linewidth=1.5)
        )
        fig.canvas.draw_idle()

        if len(peak_guesses) == 2:
            perform_fit(lower, upper, 2, peak_guesses=peak_guesses)
            clear_pending()

    def perform_fit(lower, upper, fit_n, peak_guesses=None):
        try:
            result = fit_roi(
                spectrum,
                lower,
                upper,
                corrected=corrected,
                n=fit_n,
                peak_guesses=peak_guesses
            )
        except (ValueError, RuntimeError) as error:
            print(f"\nROI fitting failed: {error}")
            return

        roi_number = len(spectrum.fit_results) + 1

        result["name"] = f"ROI {roi_number}"

        spectrum.fit_results.append(result)

        spectrum.rois.append((lower, upper))

        # Store individual peaks for calibration

        spectrum.peaks.extend(result["components"])

        # Plot selected region

        roi_patch = ax.axvspan(
            lower,
            upper,
            color="blue",
            alpha=0.12
        )

        # Generate smooth fitted curve

        x_fit = np.linspace(
            lower,
            upper,
            300
        )

        parameters = result["parameters"]

        a, b, c, d = parameters[3:7]
        fitted_continuum = continuum(x_fit, a, b, c, d)

        # Plot complete fit

        total_fit = gaussian(
            x_fit,
            *parameters
        )

        total_line, = ax.plot(
            x_fit,
            total_fit,
            color="red",
            linewidth=2,
            label="Gaussian + Continuum"
        )

        artists = [
            roi_patch,
            total_line
        ]

        # Plot individual Gaussian components

        for peak in result["components"]:

            A = peak["amplitude"]
            mu = peak["centroid"]
            sigma = peak["sigma"]

            gaussian_fit = (
                A * np.exp(
                    -((x_fit - mu)**2)
                    / (2 * sigma**2)
                )
                + fitted_continuum
            )

            gaussian_line, = ax.plot(
                x_fit,
                gaussian_fit,
                color="green",
                linestyle="--",
                linewidth=1.5
            )

            centroid_line = ax.axvline(
                mu,
                color="purple",
                linestyle=":",
                alpha=0.8
            )

            artists.extend([
                gaussian_line,
                centroid_line
            ])

        roi_artists.append(artists)

        fig.canvas.draw_idle()

        print(
            f"\nROI {roi_number} fitted successfully "
            f"({fit_n} peak(s))."
        )

        for i, peak in enumerate(result["components"], 1):

            print(f"\nPeak {i}")

            print(
                f"Centroid: {peak['centroid']:.3f} "
                f"+/- {peak['centroid_error']:.3f} channels"
            )

            print(
                f"FWHM: {peak['fwhm']:.3f} "
                f"+/- {peak['fwhm_error']:.3f} channels"
            )

            print(
                f"Area: {peak['area']:.3f} "
                f"+/- {peak['area_error']:.3f} CPS channels"
            )

    # ----------------------------------------------
    # INTERACTIVE SPAN SELECTOR
    # ----------------------------------------------

    selector = SpanSelector(
        ax,
        on_select,
        "horizontal",
        useblit=True,
        props=dict(
            alpha=0.25,
            facecolor="blue"
        ),
        interactive=False,
        drag_from_anywhere=False
    )

    # ----------------------------------------------
    # KEYBOARD CONTROLS
    # ----------------------------------------------

    def on_key(event):
        nonlocal n_peaks

        if event.key in ("1", "2"):
            if pending_roi is not None:
                print("Finish or cancel the current ROI before changing peak count.")
                return

            n_peaks = int(event.key)
            print(f"\nNext ROI will fit {n_peaks} Gaussian peak(s).")

        elif event.key == "escape" and pending_roi is not None:
            clear_pending()
            print("\nDouble-peak selection cancelled.")

        elif event.key in ("enter", "escape", "q"):
            if pending_roi is not None:
                clear_pending()
            print("\nFinishing ROI selection...")
            plt.close(fig)

        elif event.key == "backspace":
            if pending_roi is not None:
                clear_pending()
                print("\nDouble-peak selection cancelled.")
                return

            if len(spectrum.fit_results) == 0:
                return

            # Remove last fitted ROI
            last_fit = spectrum.fit_results.pop()
            spectrum.rois.pop()
            n_removed = last_fit["n_peaks"]
            del spectrum.peaks[-n_removed:]

            # Remove associated plot artists
            artists = roi_artists.pop()
            for artist in artists:
                artist.remove()

            print("\nLast ROI removed.")
            fig.canvas.draw_idle()

    fig.canvas.mpl_connect("key_press_event", on_key)
    fig.canvas.mpl_connect("button_press_event", on_click)

    # Keep selector alive during interaction

    spectrum.selector = selector

    plt.tight_layout()

    plt.show()

    # ----------------------------------------------
    # PRINT FINAL RESULTS
    # ----------------------------------------------

    print("\n" + "=" * 55)
    print("SPECTRUM ANALYSIS RESULTS")
    print("=" * 55)

    if len(spectrum.peaks) == 0:

        print("No peaks fitted.")
        print("This spectrum will be skipped during calibration")

    else:

        for i, peak in enumerate(spectrum.peaks, 1):

            print(f"\nPeak {i}")

            print(
                f"Centroid: {peak['centroid']:.3f}"
                f" +/- {peak['centroid_error']:.3f}"
            )

            print(
                f"Amplitude: {peak['amplitude']:.3f}"
                f" +/- {peak['amplitude_error']:.3f}"
            )

            print(
                f"FWHM: {peak['fwhm']:.3f}"
                f" +/- {peak['fwhm_error']:.3f}"
            )

            print(
                f"Area: {peak['area']:.3f}"
                f" +/- {peak['area_error']:.3f}"
            )

    print("\n" + "=" * 55)

    return spectrum.peaks

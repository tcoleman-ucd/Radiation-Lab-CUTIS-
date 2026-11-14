"""Interactive ROI selection, peak fitting and optional energy identification."""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import SpanSelector, RadioButtons, Button

from fitting import fit_roi, gaussian, continuum


def interactive_fit(spectrum, corrected=True, known_energies=None):
    """
    Click and drag to select ROIs, fitting one or two Gaussians per ROI.

    If known_energies is provided, identify every fitted peak in the plot.
    Otherwise, return the fitted peaks without identifying their energies.

    Controls:
        Drag       Select ROI
        1 / 2      Fit one / two Gaussians in the next ROI
        Enter      Finish after identifying the current peak
        Backspace  Undo the most recent ROI
        Escape     Cancel pending selection, or finish
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

    identify_in_plot = known_energies is not None
    energies = list(known_energies) if identify_in_plot else []
    used_energies = set()
    n_peaks = 1

    fig, ax = plt.subplots(figsize=(13, 7))
    if identify_in_plot:
        fig.subplots_adjust(left=0.08, right=0.75, top=0.86, bottom=0.12)
    else:
        fig.subplots_adjust(bottom=0.15)

    ax.plot(spectrum.channels, count_rates, color="black", linewidth=0.8)
    ax.set_xlabel("Channel")
    ax.set_ylabel("Counts per Second")
    ax.set_title(f"{Path(spectrum.filename).name}\n{title}")
    ax.grid(alpha=0.3)

    instruction = (
        "Drag: Select ROI | 1: One Gaussian | 2: Two Gaussians\n"
        "Double Gaussian: click two approximate peak positions\n"
        "ENTER: Finish | BACKSPACE: Undo | ESC: Cancel/Finish"
    )
    instruction_text = ax.text(
        0.02, 0.97, instruction, transform=ax.transAxes,
        va="top", fontsize=9,
        bbox=dict(facecolor="white", alpha=0.85),
    )

    roi_artists = []
    pending_roi = None
    peak_guesses = []
    guess_artists = []

    # Each item contains (fitted peak dictionary, centroid line artist).
    # The queue is filled after each ROI fit and emptied by confirmation.
    pending_identifications = []

    if identify_in_plot:
        # Keep the panel compact. Its height depends on the number of
        # reference energies rather than filling most of the window.
        panel_x, panel_width = 0.79, 0.19
        fig.text(panel_x, 0.93, "Peak identification", fontsize=11, weight="bold")
        status_text = fig.text(
            panel_x, 0.87, "Select an ROI first.",
            fontsize=9, va="top",
        )
        fig.text(panel_x, 0.755, "Reference energy", fontsize=9)

        options = ["Unknown / skip"] + [f"{e:.4f} keV" for e in energies]
        radio_height = 0.045 * len(options) + 0.025
        radio_bottom = 0.72 - radio_height
        radio_ax = fig.add_axes([panel_x, radio_bottom, panel_width, radio_height])
        radio_ax.set_facecolor(fig.get_facecolor())
        for spine in radio_ax.spines.values():
            spine.set_visible(False)
        radio = RadioButtons(radio_ax, options, active=0)
        for label in radio.labels:
            label.set_fontsize(9)

        # "Unknown / skip" is an option in the list, so a separate
        # Skip button would be redundant.
        button_bottom = radio_bottom - 0.08
        assign_ax = fig.add_axes([panel_x, button_bottom, panel_width, 0.055])
        assign_button = Button(assign_ax, "Confirm / next")

        fig.text(
            panel_x, button_bottom - 0.05,
            "Green = identified\nGrey = unidentified",
            fontsize=8,
            va="top",
        )

    def clear_pending_roi():
        nonlocal pending_roi
        pending_roi = None
        peak_guesses.clear()
        for artist in guess_artists:
            artist.remove()
        guess_artists.clear()
        selector.set_active(not pending_identifications)
        instruction_text.set_text(instruction)
        fig.canvas.draw_idle()

    def show_identification_prompt():
        if not identify_in_plot:
            return

        if pending_identifications:
            peak, _ = pending_identifications[0]
            selector.set_active(False)
            radio.set_active(0)  # Never preselect an energy by accident.
            status_text.set_text(
                f"Channel: {peak['centroid']:.2f}\n"
                "Choose energy, then confirm."
            )
            instruction_text.set_text(
                "Identify or skip this fitted peak before choosing another ROI.\n"
                "BACKSPACE: Undo this ROI"
            )
        else:
            selector.set_active(True)
            status_text.set_text("Drag to fit another ROI.")
            instruction_text.set_text(instruction)

        fig.canvas.draw_idle()

    def accept_identification(energy):
        if not pending_identifications:
            return

        if energy is not None and energy in used_energies:
            status_text.set_text(
                f"{energy:.4f} keV already assigned.\n"
                "Choose another or Unknown / skip."
            )
            fig.canvas.draw_idle()
            return

        peak, centroid_line = pending_identifications.pop(0)
        peak["energy"] = energy
        peak["accepted"] = energy is not None

        if energy is None:
            centroid_line.set_color("gray")
            print(f"Channel {peak['centroid']:.3f}: unidentified.")
        else:
            used_energies.add(energy)
            centroid_line.set_color("green")
            print(f"Channel {peak['centroid']:.3f}: assigned {energy:.4f} keV.")

        show_identification_prompt()

    if identify_in_plot:
        def on_assign(event):
            if not pending_identifications:
                return
            index = options.index(radio.value_selected)
            energy = None if index == 0 else energies[index - 1]
            accept_identification(energy)

        assign_button.on_clicked(on_assign)

    def perform_fit(lower, upper, fit_n, guesses=None):
        try:
            result = fit_roi(
                spectrum, lower, upper, n=fit_n,
                corrected=corrected, peak_guesses=guesses,
            )
        except (ValueError, RuntimeError) as error:
            print(f"\nROI fitting failed: {error}")
            return

        roi_number = len(spectrum.fit_results) + 1
        result["name"] = f"ROI {roi_number}"
        spectrum.fit_results.append(result)
        spectrum.rois.append((lower, upper))
        spectrum.peaks.extend(result["components"])

        artists = [ax.axvspan(lower, upper, color="blue", alpha=0.12)]
        x_fit = np.linspace(lower, upper, 300)
        parameters = result["parameters"]
        a, b, c, d = parameters[3:7]
        fitted_continuum = continuum(x_fit, a, b, c, d)
        total_line, = ax.plot(
            x_fit, gaussian(x_fit, *parameters),
            color="red", linewidth=2,
        )
        artists.append(total_line)

        centroid_lines = []
        for peak in result["components"]:
            A, mu, sigma = peak["amplitude"], peak["centroid"], peak["sigma"]
            single = A * np.exp(-((x_fit - mu)**2) / (2 * sigma**2))
            gaussian_line, = ax.plot(
                x_fit, single + fitted_continuum,
                color="green", linestyle="--", linewidth=1.5,
            )
            centroid_line = ax.axvline(
                mu, color="purple", linestyle=":", alpha=0.8,
            )
            artists.extend([gaussian_line, centroid_line])
            centroid_lines.append(centroid_line)

        roi_artists.append(artists)
        print(f"\nROI {roi_number} fitted successfully ({fit_n} peak(s)).")
        for i, peak in enumerate(result["components"], 1):
            print(
                f"Peak {i}: centroid = {peak['centroid']:.3f} +/- "
                f"{peak['centroid_error']:.3f} channels, "
                f"FWHM = {peak['fwhm']:.3f} channels"
            )

        if identify_in_plot:
            for peak, line in zip(result["components"], centroid_lines):
                peak["energy"] = None
                peak["accepted"] = False
                pending_identifications.append((peak, line))
            show_identification_prompt()
        else:
            fig.canvas.draw_idle()

    def on_select(xmin, xmax):
        nonlocal pending_roi
        if pending_identifications:
            return
        lower, upper = min(xmin, xmax), max(xmin, xmax)
        if upper <= lower:
            return
        if n_peaks == 1:
            perform_fit(lower, upper, 1)
            return

        pending_roi = (lower, upper)
        peak_guesses.clear()
        selector.set_active(False)
        guess_artists.append(
            ax.axvspan(lower, upper, color="orange", alpha=0.10)
        )
        instruction_text.set_text(
            "Double Gaussian: click the approximate centre of each peak.\n"
            "ESC/BACKSPACE: Cancel selection"
        )
        fig.canvas.draw_idle()

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
            guesses = list(peak_guesses)
            clear_pending_roi()
            perform_fit(lower, upper, 2, guesses=guesses)

    def undo_last_roi():
        if not spectrum.fit_results:
            return
        result = spectrum.fit_results.pop()
        spectrum.rois.pop()
        del spectrum.peaks[-result["n_peaks"]:]

        for peak in result["components"]:
            if peak.get("accepted"):
                used_energies.discard(peak["energy"])

        pending_identifications.clear()
        for artist in roi_artists.pop():
            artist.remove()

        if identify_in_plot:
            show_identification_prompt()
        else:
            fig.canvas.draw_idle()
        print("\nLast ROI removed.")

    selector = SpanSelector(
        ax, on_select, "horizontal", useblit=True,
        props=dict(alpha=0.25, facecolor="blue"),
        interactive=False, drag_from_anywhere=False,
    )

    def on_key(event):
        nonlocal n_peaks
        if event.key in ("1", "2"):
            if pending_roi is not None or pending_identifications:
                print("Finish the current selection before changing peak count.")
                return
            n_peaks = int(event.key)
            print(f"\nNext ROI will fit {n_peaks} Gaussian peak(s).")

        elif event.key == "backspace":
            if pending_roi is not None:
                clear_pending_roi()
            else:
                undo_last_roi()

        elif event.key == "escape" and pending_roi is not None:
            clear_pending_roi()
            print("\nDouble-peak selection cancelled.")

        elif event.key in ("enter", "escape", "q"):
            if pending_identifications:
                print("Identify or skip every fitted peak before finishing.")
                return
            print("\nFinishing ROI selection...")
            plt.close(fig)

    fig.canvas.mpl_connect("key_press_event", on_key)
    fig.canvas.mpl_connect("button_press_event", on_click)

    # Keep widget objects alive while the figure is open.
    spectrum.selector = selector
    if identify_in_plot:
        spectrum.identification_widgets = (radio, assign_button)

    plt.show()

    print("\n" + "=" * 55)
    print("SPECTRUM ANALYSIS RESULTS")
    print("=" * 55)
    if not spectrum.peaks:
        print("No peaks fitted.")
    else:
        for i, peak in enumerate(spectrum.peaks, 1):
            label = peak.get("energy")
            text_label = f"{label:.4f} keV" if label is not None else "unidentified"
            print(
                f"Peak {i}: channel {peak['centroid']:.3f} +/- "
                f"{peak['centroid_error']:.3f}; {text_label}"
            )
    print("=" * 55)

    return spectrum.peaks

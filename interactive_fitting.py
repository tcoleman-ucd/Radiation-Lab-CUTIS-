"""Interactive ROI selection, peak fitting and optional energy identification."""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import SpanSelector, RadioButtons, Button

from fitting import fit_roi, gaussian, continuum


def interactive_fit(spectrum, corrected=True, reference_energies=None):
    """
    Click and drag to select ROIs, fitting one or two Gaussians per ROI.

    spectrum: Spectrum object to analyse.
    corrected: Whether to subtract the background.
    reference_energies: List of known photopeak energies in keV.

    Controls:
        Drag       Select ROI
        1 / 2      Fit one / two Gaussians in the next ROI
        Enter      Finish after identifying the current peak
        Backspace  Undo the most recent ROI
        Escape     Cancel pending selection, or finish
    """

    # Choose raw or background-subtracted count rates
    if corrected:
        count_rates = spectrum.corrected_cps()
        title = "Background Subtracted Spectrum"
    else:
        count_rates = spectrum.cps()
        title = "Raw Spectrum"

    # Clear previous fitting results
    spectrum.rois = []
    spectrum.peaks = []
    spectrum.fit_results = []

    # Set up peak identification
    if reference_energies is not None:
        identify_in_plot = True
        energies = list(reference_energies)
    else:
        identify_in_plot = False
        energies = []

    used_energies = set() # Using a set to remove duplicates

    # Fit one Gaussian by default
    n_peaks = 1

    # Create the figure and axes for plotting the spectrum
    fig, ax = plt.subplots(figsize=(13, 7))

    # Leave space on the right if we are identifying peaks in the plot
    if identify_in_plot:
        fig.subplots_adjust(left=0.08, right=0.75, top=0.86, bottom=0.12)
    else:
        fig.subplots_adjust(bottom=0.15)

    # Plot the spectrum as count rate against channel number
    ax.plot(spectrum.channels, count_rates, color="black", linewidth=0.8)

    # Label the axes and display the spectrum filename
    ax.set_xlabel("Channel")
    ax.set_ylabel("Counts per Second")
    ax.set_title(f"{Path(spectrum.filename).name}\n{title}")
    ax.grid(alpha=0.3)

    instruction = (
        "Press 1: Fit one Gaussian | Press 2: Fit two Gaussians\n"
        "Drag: Select ROI\n"
        "Double Gaussian: Click two approximate peak positions\n"
        "Assign reference energies to the corresponding photopeaks, working left to right\n"
        "(Please reference Table 2 (manual, p. 57), using emission fractions to estimate relative peak strengths,\n"
        "Skip peaks that cannot be reliably identified.\n"
        "ENTER: Finish | BACKSPACE: Undo | ESC: Cancel/Finish"
    )
    # Place the instructions in the upper-left corner of the plot
    instruction_text = ax.text(
        0.02, 0.97, instruction, transform=ax.transAxes,
        va="top", fontsize=9,
        bbox=dict(facecolor="white", alpha=0.85),
    )

    # Store plots for completed ROI fits
    roi_plots = []

    # Store an unfinished ROI while selecting two Gaussian peaks
    pending_roi = None

    # Store approximate peak positions selected by clicking
    peak_guesses = []

    # Store temporary plots during peak selection
    guess_plots = []

    # Store fitted peaks waiting for energy identification
    pending_identifications = []

    # Create the reference energy selection panel
    if identify_in_plot:

        panel_x = 0.79
        panel_width = 0.19

        # Panel title and status message
        fig.text(
            panel_x, 0.93, "Peak Identification",
            fontsize=11, weight="bold"
        )

        status_text = fig.text(
            panel_x, 0.87, "Select an ROI first.",
            fontsize=9, va="top"
        )

        fig.text(panel_x, 0.755, "Reference Energy", fontsize=9)

        # List the available reference energies
        options = ["Unknown / skip"]

        for energy in energies:
            options.append(f"{energy:.4f} keV")

        # Adjust the list size based on the number of energies
        energy_list_height = 0.045 * len(options) + 0.025
        energy_list_bottom = 0.72 - energy_list_height

        # Create the selectable energy list
        energy_ax = fig.add_axes([panel_x, energy_list_bottom, panel_width, energy_list_height])

        # Match the background colour and remove the border
        energy_ax.set_facecolor(fig.get_facecolor())

        for spine in energy_ax.spines.values():
            spine.set_visible(False)

        energy_selector = RadioButtons(energy_ax, options, active=0)

        for label in energy_selector.labels:
            label.set_fontsize(9)

        # Create the confirmation button below the energy list
        button_bottom = energy_list_bottom - 0.08

        assign_ax = fig.add_axes([panel_x, button_bottom, panel_width, 0.055])

        confirm_button = Button(assign_ax, "Confirm / next")

        fig.text(
            panel_x, button_bottom - 0.05,
            "Green = identified\nGrey = unidentified",
            fontsize=8, va="top"
        )

    def clear_pending_roi():
        """Cancel an unfinished ROI selection."""
        nonlocal pending_roi

        # Reset the selected ROI and approximate peak positions
        pending_roi = None
        peak_guesses.clear()

        # Remove temporary lines and shading from the plot
        for plot in guess_plots:
            plot.remove()
        guess_plots.clear()

        # Allow another ROI to be selected if no peaks need identification
        if pending_identifications:
            selector.set_active(False)
        else:
            selector.set_active(True)

        # Restore the original instructions and update the figure
        instruction_text.set_text(instruction)
        fig.canvas.draw_idle()

    def show_identification_prompt():
        """
        Function to update the list of reference energies and don't allow
        the user to select another ROI
        """
        if not identify_in_plot:
            return

        # Check if any fitted peaks still need an energy assigned
        if pending_identifications:

            peak, _ = pending_identifications[0]

            # Disable ROI selection until the peak is identified
            selector.set_active(False)

            # Reset the energy selection to Unknown / skip
            energy_selector.set_active(0)

            # Display the channel of the peak being identified
            status_text.set_text(
                f"Channel: {peak['centroid']:.2f}\n"
                "Choose energy, then confirm."
            )

            instruction_text.set_text(
                "Identify or skip this fitted peak before choosing another ROI.\n"
                "BACKSPACE: Undo this ROI"
            )

        else:

            # Allow the user to select another ROI
            selector.set_active(True)
            status_text.set_text("Drag to fit another ROI.")
            instruction_text.set_text(instruction)

        fig.canvas.draw_idle()

    def accept_identification(energy):
        """
        Function to store the selected reference energy in a dictionary
        """
        if not pending_identifications:
            return

        # Prevent assigning the same reference energy twice
        if energy is not None and energy in used_energies:
            status_text.set_text(
                f"{energy:.4f} keV already assigned.\n"
                "Choose another or Unknown / skip."
            )
            fig.canvas.draw_idle()
            return

        # Get the next peak waiting for identification
        peak, centroid_line = pending_identifications.pop(0)

        # Store the selected energy
        peak["energy"] = energy
        peak["accepted"] = energy is not None

        # Change the centroid line colour depending on identification
        if energy is None:
            centroid_line.set_color("gray")
            print(f"Channel {peak['centroid']:.3f}: unidentified.")

        else:
            used_energies.add(energy)
            centroid_line.set_color("#00FF00")
            print(f"Channel {peak['centroid']:.3f}: assigned {energy:.4f} keV.")

        # Move on to the next peak
        show_identification_prompt()

    if identify_in_plot:

        def on_assign(event):

            # Do nothing if there are no peaks waiting for identification
            if not pending_identifications:
                return

            # Find which reference energy the user selected
            index = options.index(energy_selector.value_selected)

            # The first option is Unknown / skip
            if index == 0:
                energy = None
            else:
                energy = energies[index - 1]

            # Assign the selected energy to the fitted peak
            accept_identification(energy)

        # Run on_assign when the confirmation button is clicked
        assign_button.on_clicked(on_assign)

    def perform_fit(lower, upper, fit_n, guesses=None):

        # Fit the selected ROI with one or two Gaussians
        try:
            result = fit_roi(
                spectrum, lower, upper, n=fit_n,
                corrected=corrected, peak_guesses=guesses
            )

        # Report an error if fitting fails
        except (ValueError, RuntimeError) as error:
            print(f"\nROI fitting failed: {error}")
            return

        # Number and name the fitted ROI
        roi_number = len(spectrum.fit_results) + 1
        result["name"] = f"ROI {roi_number}"

        # Store the fit results, ROI boundaries, and individual peaks
        spectrum.fit_results.append(result)
        spectrum.rois.append((lower, upper))
        spectrum.peaks.extend(result["components"])

        # Shade the selected ROI
        plots = [ax.axvspan(lower, upper, color="blue", alpha=0.12)]

        # Generate 300 points for smooth plotting of the fitted curves
        x_fit = np.linspace(lower, upper, 300)

        # Get the fitted parameters
        parameters = result["parameters"]

        # Extract the linear background parameters
        m, c = parameters[3:5]
        fitted_continuum = continuum(x_fit, m, c)

        # Plot the total fitted model (Gaussians + background)
        total_line, = ax.plot(
            x_fit,
            gaussian(x_fit, *parameters),
            color="red",
            linewidth=2
        )

        # Stores everything drawn for the ROI so we can remove it when undoing
        plots.append(total_line)

        centroid_lines = []

        # Plot each fitted Gaussian separately
        for peak in result["components"]:
            # Get the fitted peak parameters
            A = peak["area"]
            mu = peak["centroid"]
            sigma = peak["sigma"]

            single = gaussian(x_fit, A, mu, sigma, m, c)

            gaussian_line, = ax.plot(
                x_fit, single,
                color="green", linestyle="--", linewidth=1.5
            )

            # Mark the fitted centroid with a vertical line
            centroid_line = ax.axvline(
                mu, color="purple", linestyle=":",
                linewidth=2.5, alpha=1.0, zorder=10
            )

            # Store the plots so they can be removed when undoing
            plots.extend([gaussian_line, centroid_line])

            # Store the centroid line for later energy identification
            centroid_lines.append(centroid_line)

        # Store all plots associated with this ROI
        roi_plots.append(plots)

        # Print the fitted ROI number and number of peaks
        print(f"\nROI {roi_number} fitted successfully ({fit_n} peak(s)).")

        # Print the fitted centroid and FWHM for each peak
        for i, peak in enumerate(result["components"], 1):
            print(
                f"Peak {i}: centroid = {peak['centroid']:.3f} +/- "
                f"{peak['centroid_error']:.3f} channels, "
                f"FWHM = {peak['fwhm']:.3f} channels"
            )

        # If identifying peaks, add them to the identification queue
        if identify_in_plot:

            for peak, line in zip(result["components"], centroid_lines):
                # Initialise the peak as unidentified
                peak["energy"] = None
                peak["accepted"] = False

                # Store the peak and its corresponding centroid line
                pending_identifications.append((peak, line))

            # Ask the user to identify the next peak
            show_identification_prompt()

        else:
            # Update the figure if energy identification is not required
            fig.canvas.draw_idle()

    def on_select(xmin, xmax):
        nonlocal pending_roi

        # Don't allow another ROI until the previous peaks are identified
        if pending_identifications:
            return

        # Get the ROI boundaries regardless of drag direction
        lower = min(xmin, xmax)
        upper = max(xmin, xmax)

        if lower == upper:
            return

        # Fit immediately if only one Gaussian is selected
        if n_peaks == 1:
            perform_fit(lower, upper, 1)
            return

        # For two Gaussians, wait for the user to click both peak positions
        pending_roi = (lower, upper)
        peak_guesses.clear()

        # Disable selecting another ROI while choosing peak positions
        selector.set_active(False)

        # Highlight the selected ROI in orange
        guess_plots.append(
            ax.axvspan(lower, upper, color="orange", alpha=0.10)
        )
        instruction_text.set_text(
            "Double Gaussian: click the approximate centre of each peak.\n"
            "ESC/BACKSPACE: Cancel selection"
        )

        fig.canvas.draw_idle()

    def on_click(event):

        # Only accept clicks when an ROI has been selected
        if pending_roi is None:
            return

        # Ignore clicks outside the graph
        if event.inaxes != ax:
            return

        # Check that the click is within the selected ROI
        lower, upper = pending_roi
        if event.xdata <= lower or event.xdata >= upper:
            print("Please click inside the selected ROI.")
            return

        # Store the approximate peak centre
        peak_guesses.append(event.xdata)

        # Draw an orange line at the selected position
        guess_plots.append(
            ax.axvline(
                event.xdata, color="orange",
                linestyle=":", linewidth=1.5
            )
        )

        fig.canvas.draw_idle()

        # Once two positions are selected, fit the double Gaussian
        if len(peak_guesses) == 2:
            guesses = list(peak_guesses)

            # Remove the temporary stuff from the graph
            clear_pending_roi()

            # Fit both Gaussians using the selected positions
            perform_fit(lower, upper, 2, guesses=guesses)

    def undo_last_roi():

        # Do nothing if there are no fitted ROIs
        if not spectrum.fit_results:
            return

        # Remove the last fitted result and its ROI boundaries
        result = spectrum.fit_results.pop()
        spectrum.rois.pop()

        # Remove the peaks belonging to the last ROI
        for i in range(result["n_peaks"]):
            spectrum.peaks.pop()

        # Make any previously assigned energies available again
        for peak in result["components"]:
            if peak.get("accepted"):
                used_energies.discard(peak["energy"])

        # Clear any unfinished energy identifications
        pending_identifications.clear()

        # Remove the lines and plotting stuff in the ROI
        for plot in roi_plots.pop():
            plot.remove()

        # Update the graph and energies panel
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

        # we need this to allow this function to change the number of Gaussians
        nonlocal n_peaks

        # Press 1 or 2 to choose the number of Gaussians
        if event.key in ("1", "2"):

            # Finish the current selection before changing the number
            if pending_roi is not None or pending_identifications:
                print("Finish the current selection before changing peak count.")
                return

            n_peaks = int(event.key)
            print(f"\nNext ROI will fit {n_peaks} Gaussian peak(s).")

        # Backspace cancels an unfinished selection or undoes the last fit
        elif event.key == "backspace":
            if pending_roi is not None:
                clear_pending_roi()
            else:
                undo_last_roi()

        # Escape cancels an unfinished double-Gaussian selection
        elif event.key == "escape" and pending_roi is not None:
            clear_pending_roi()
            print("\nDouble-peak selection cancelled.")

        # Enter, Escape or Q finishes the interactive fitting
        elif event.key in ("enter", "escape", "q"):

            # Make sure all fitted peaks have been identified first
            if pending_identifications:
                print("Identify or skip every fitted peak before finishing.")
                return

            print("\nFinishing ROI selection...")
            plt.close(fig)





    # Connect keyboard presses and mouse clicks to our functions
    fig.canvas.mpl_connect("key_press_event", on_key)
    fig.canvas.mpl_connect("button_press_event", on_click)

    # Keep the interactive controls available while the plot is open
    spectrum.selector = selector

    if identify_in_plot:
        spectrum.identification_widgets = (energy_selector, confirm_button)

    # Display the interactive spectrum
    plt.show()

    # Print the final fitting results
    print("\n" + "=" * 55)
    print("SPECTRUM ANALYSIS RESULTS")
    print("=" * 55)

    if not spectrum.peaks:
        print("No peaks fitted.")

    else:
        # Print the centroid and identified energy of each fitted peak
        for i, peak in enumerate(spectrum.peaks, 1):

            energy = peak.get("energy")

            if energy is not None:
                energy_label = f"{energy:.4f} keV"
            else:
                energy_label = "unidentified"

            print(
                f"Peak {i}: channel {peak['centroid']:.3f} +/- "
                f"{peak['centroid_error']:.3f}; {energy_label}"
            )

    print("=" * 55)

    # Return the fitted peaks for calibration and further analysis
    return spectrum.peaks
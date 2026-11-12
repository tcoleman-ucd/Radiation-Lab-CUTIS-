
"""Organise the measurements, calibration and resolution of one detector."""

from fitting import interactive_fit

from calibration import (
    calibrate as fit_calibration,
    channel_to_energy as convert_channel,
    energy_to_channel as convert_energy,
    plot_calibration as draw_calibration,
)


class Detector:

    def __init__(self, name, spectra, known_energies):
        """Store source spectra and suggested photopeak energies (keV)."""

        self.name = name
        self.spectra = dict(spectra)

        self.known_energies = {
            source: list(energies)
            for source, energies in known_energies.items()
        }

        # Background remains attached to its respective measurement.
        self.backgrounds = {
            source: spectrum.background
            for source, spectrum in self.spectra.items()
        }

        self.calibration = None
        self.calibration_peaks = []
        self.calibration_energies = []
        self.calibration_peak_records = []

        self.resolution = None
        self.efficiency = None

    # ---------------------------------------------------------
    # USER INPUT
    # ---------------------------------------------------------

    def _include_spectrum(self, source):
        """Ask whether the user wants to analyse this spectrum."""

        while True:
            answer = input(
                f"\nAnalyse {source} spectrum? [Y/n]: "
            ).strip().lower()

            if answer in ("", "y", "yes"):
                return True

            if answer in ("n", "no"):
                print(f"Skipping {source}.")
                return False

            print("Please enter y or n.")

    def _ask_peak_energies(self, source, peaks):
        """
        Ask the user to assign an energy to every fitted peak.

        Known energies are suggestions only.
        Pressing Enter accepts a suggested energy.
        """

        suggested = self.known_energies.get(source, [])

        print(f"\nAssign energies for {source}.")

        if suggested:
            print(f"Suggested energies (keV): {suggested}")

        energies = []

        for i, peak in enumerate(peaks):

            default = (
                suggested[i]
                if i < len(suggested)
                else None
            )

            print(
                f"\nPeak {i + 1}: "
                f"{peak['centroid']:.3f} +/- "
                f"{peak['centroid_error']:.3f} channels"
            )

            if default is not None:
                prompt = (
                    f"Enter energy in keV "
                    f"[{default}]: "
                )
            else:
                prompt = "Enter energy in keV: "

            while True:

                answer = input(prompt).strip()

                if answer == "" and default is not None:
                    energy = float(default)
                    break

                try:
                    energy = float(answer)

                    if energy <= 0:
                        print("Energy must be positive.")
                        continue

                    break

                except ValueError:
                    print("Please enter a valid energy in keV.")

            energies.append(energy)

        return energies

    # ---------------------------------------------------------
    # ENERGY CALIBRATION
    # ---------------------------------------------------------

    def calibrate(self, corrected=True, show_plot=True):
        """Fit selected photopeaks and calibrate the detector."""

        all_peaks = []
        all_energies = []
        peak_records = []

        # Reset previous calibration and resolution results.
        self.calibration = None
        self.calibration_peaks = []
        self.calibration_energies = []
        self.calibration_peak_records = []
        self.resolution = None

        for source, spectrum in self.spectra.items():

            print("\n" + "=" * 55)
            print(f"{self.name}: {source}")
            print("=" * 55)

            # Allow the user to skip an entire spectrum.
            if not self._include_spectrum(source):
                continue

            suggested = self.known_energies.get(source, [])

            if suggested:
                print(
                    f"\nSuggested photopeak energies: {suggested} keV"
                )

            else:
                print("\nNo suggested energies available.")

            print("\nSelect the photopeaks in the spectrum.")
            print("You can select as many ROIs as you need.")
            print("Press 1 or 2 to choose the number of peaks per ROI.")
            print("Press Enter when you have finished.")

            # Interactive peak fitting.
            peaks = interactive_fit(
                spectrum,
                corrected=corrected
            )

            # Skip spectrum if no peaks were selected.
            if not peaks:
                print(
                    f"\nNo peaks selected for {source}. "
                    "Skipping this spectrum."
                )
                continue

            # Sort fitted peaks by increasing channel.
            sorted_peaks = sorted(
                peaks,
                key=lambda peak: peak["centroid"]
            )

            print(
                f"\n{len(sorted_peaks)} peak(s) fitted "
                f"for {source}."
            )

            # Ask the user to assign energies.
            sorted_energies = self._ask_peak_energies(
                source,
                sorted_peaks
            )

            # Store the peaks and their assigned energies.
            for peak, energy in zip(
                sorted_peaks,
                sorted_energies
            ):

                all_peaks.append(peak)
                all_energies.append(energy)

                peak_records.append({
                    **peak,
                    "source": spectrum.filename,
                    "isotope": source,
                    "energy": energy
                })

        # Check that enough peaks were selected overall.
        if len(all_peaks) < 2:
            raise ValueError(
                "At least two calibration peaks are required. "
                "Please analyse more spectra or select more peaks."
            )

        # Perform energy calibration.
        self.calibration = fit_calibration(
            all_peaks,
            all_energies
        )

        self.calibration_peaks = all_peaks
        self.calibration_energies = all_energies
        self.calibration_peak_records = peak_records

        print("\n" + "=" * 55)
        print(f"{self.name} ENERGY CALIBRATION")
        print("=" * 55)

        print(
            f"Number of calibration peaks: {len(all_peaks)}"
        )

        print(
            f"Gradient = "
            f"{self.calibration['gradient']:.6f} keV/channel"
        )

        print(
            f"Intercept = "
            f"{self.calibration['intercept']:.3f} keV"
        )

        if show_plot:
            draw_calibration(self.calibration)

        return self.calibration

    # ---------------------------------------------------------
    # ENERGY CONVERSION
    # ---------------------------------------------------------

    def channel_to_energy(self, channel):

        if self.calibration is None:
            raise ValueError("Calibrate the detector first.")

        return convert_channel(
            channel,
            self.calibration
        )

    def energy_to_channel(self, energy):

        if self.calibration is None:
            raise ValueError("Calibrate the detector first.")

        return convert_energy(
            energy,
            self.calibration
        )

    # ---------------------------------------------------------
    # CALIBRATION PLOT
    # ---------------------------------------------------------

    def plot_calibration(self):

        if self.calibration is None:
            raise ValueError("Calibrate the detector first.")

        draw_calibration(self.calibration)

    # ---------------------------------------------------------
    # ENERGY RESOLUTION
    # ---------------------------------------------------------

    def calculate_resolution(self, title=None, show=True):
        """Calculate and store the detector's energy resolution."""

        if self.calibration is None:
            raise ValueError("Calibrate the detector first.")

        from Resolution import resolution as fit_resolution

        if title is None:
            title = f"{self.name} Resolution"

        self.resolution = fit_resolution(
            self,
            title=title,
            show=show
        )

        return self.resolution

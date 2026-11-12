
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
        """Store source spectra and known photopeak energies (in keV)."""

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

            if source not in self.known_energies:
                raise ValueError(
                    f"No known energies specified for {source}."
                )

            energies = self.known_energies[source]

            print(f"\n{self.name}: fitting {source}")
            print(
                f"Select photopeaks at these energies (keV): {energies}"
            )

            peaks = interactive_fit(
                spectrum,
                corrected=corrected
            )

            if len(peaks) != len(energies):
                raise ValueError(
                    f"{self.name} {source}: fitted {len(peaks)} peaks, "
                    f"but {len(energies)} known energies were supplied. "
                    "Select only the matching photopeaks and try again."
                )

            # Match peaks and energies in ascending order.
            sorted_peaks = sorted(
                peaks,
                key=lambda peak: peak["centroid"]
            )

            sorted_energies = sorted(energies)

            for peak, energy in zip(sorted_peaks, sorted_energies):

                all_peaks.append(peak)
                all_energies.append(energy)

                # Retain source, energy and fitting results together.
                peak_records.append({
                    **peak,
                    "source": spectrum.filename,
                    "isotope": source,
                    "energy": energy
                })

        if len(all_peaks) < 2:
            raise ValueError(
                "At least two calibration peaks are required."
            )

        self.calibration = fit_calibration(
            all_peaks,
            all_energies
        )

        self.calibration_peaks = all_peaks
        self.calibration_energies = all_energies
        self.calibration_peak_records = peak_records

        print(f"\n{self.name} energy calibration")
        print(
            f"Gradient  = "
            f"{self.calibration['gradient']:.6f} keV/channel"
        )
        print(
            f"Intercept = "
            f"{self.calibration['intercept']:.3f} keV"
        )

        if show_plot:
            draw_calibration(self.calibration)

        return self.calibration

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

    def plot_calibration(self):

        if self.calibration is None:
            raise ValueError("Calibrate the detector first.")

        draw_calibration(self.calibration)

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

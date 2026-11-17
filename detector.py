"""Organise the measurements, calibration and resolution of one detector."""

import fitting
from calibration import (
    calibrate as fit_calibration,
    channel_to_energy as convert_channel,
    energy_to_channel as convert_energy,
    plot_calibration as draw_calibration,
)


class Detector:

    def __init__(self, name, spectra, known_energies, radius, off_axis_spectra=None, distance_spectra=None):
        """Store source spectra and known photopeak energies (in keV)."""
        self.name = name
        self.radius = radius
        self.spectra = dict(spectra)
        self.off_axis_spectra = (off_axis_spectra if off_axis_spectra is not None else {})
        self.distance_spectra = (distance_spectra if distance_spectra is not None else {})

        self.known_energies = {
            source: list(energies)
            for source, energies in known_energies.items()
        }

        self.backgrounds = {
            source: spectrum.background
            for source, spectrum in self.spectra.items()
        }

        self.calibration = None
        self.calibration_peaks = []
        self.calibration_energies = []
        self.calibration_peak_records = []
        self.unidentified_peaks = []
        self.resolution = None
        self.efficiency = None
        self.off_axis_results = {}

    def calibrate(self, corrected=True, show_plot=True, identification=None):
        """
        Calibrate using manually identified photopeaks.

        By default, use IDENTIFY_IN_PLOT from fitting.py.
        identification='plot' or 'terminal' can override this setting.
        ROIs are always selected by clicking and dragging.
        """
        if identification is None:
            identification = "plot" if fitting.IDENTIFY_IN_PLOT else "terminal"

        if identification not in ("plot", "terminal"):
            raise ValueError("identification must be 'plot' or 'terminal'.")

        all_peaks = []
        all_energies = []
        peak_records = []
        unidentified_peaks = []

        self.calibration = None
        self.calibration_peaks = []
        self.calibration_energies = []
        self.calibration_peak_records = []
        self.unidentified_peaks = []
        self.resolution = None

        for source, spectrum in self.spectra.items():
            if source not in self.known_energies:
                raise ValueError(f"No known energies specified for {source}.")

            energies = self.known_energies[source]
            print(f"\n{self.name}: fitting {source}")
            print("Drag to select any reliable photopeaks; Enter finishes this spectrum.")

            peaks = fitting.interactive_fit(
                spectrum,
                corrected=corrected,
                known_energies=energies,
                identify_in_plot=(identification == "plot"),
            )

            if not peaks:
                print(f"No peaks selected for {source}. Skipping.")
                continue

            if identification == "terminal":
                used_energies = set()
                # Suggestions follow the ordering of the reference energy list.
                sorted_peaks = sorted(peaks, key=lambda peak: peak["centroid"])
                for i, peak in enumerate(sorted_peaks):
                    print(f"\nFitted peak at channel {peak['centroid']:.3f}")
                    for index, reference_energy in enumerate(energies, 1):
                        print(f"{index}: {reference_energy:.4f} keV")
                    print("0: Unknown / skip")

                    # Suggest the reference energy in the corresponding position.
                    # As before, only the user can verify that this is the right line.
                    default = energies[i] if i < len(energies) else None
                    if default in used_energies:
                        default = None

                    if default is not None:
                        prompt = (
                            f"Select number, alternatively enter the energy in keV "
                            f"[No Value Given = {default:.4f} keV]: "
                        )
                    else:
                        prompt = "Select number, enter energy in keV: "

                    while True:
                        choice = input(prompt).strip()
                        if choice == "0":
                            energy = None
                            break

                        if choice == "" and default is not None:
                            energy = default
                        elif choice.isdigit() and 1 <= int(choice) <= len(energies):
                            energy = energies[int(choice) - 1]
                        else:
                            try:
                                energy = float(choice)
                            except ValueError:
                                print("Enter a listed number, an energy in keV, or 0 to skip.")
                                continue

                            if not 0 < energy < float("inf"):
                                print("Energy must be a finite positive number.")
                                continue

                        if energy in used_energies:
                            print("This energy has already been assigned.")
                            continue

                        used_energies.add(energy)
                        break

                    peak["energy"] = energy
                    peak["accepted"] = energy is not None

            for peak in peaks:
                energy = peak.get("energy") if peak.get("accepted") else None
                record = {
                    **peak,
                    "source": spectrum.filename,
                    "isotope": source,
                    "energy": energy,
                    "accepted": energy is not None,
                }
                if energy is None:
                    unidentified_peaks.append(record)
                else:
                    all_peaks.append(peak)
                    all_energies.append(energy)
                    peak_records.append(record)

        # Preserve accepted and unidentified peaks even if no calibration is possible.
        self.calibration_peaks = all_peaks
        self.calibration_energies = all_energies
        self.calibration_peak_records = peak_records
        self.unidentified_peaks = unidentified_peaks

        if len(all_peaks) < 2 or len(set(all_energies)) < 2:
            raise ValueError(
                "At least two identified peaks at distinct energies "
                "are required for a linear calibration."
            )

        self.calibration = fit_calibration(all_peaks, all_energies)
        print(f"\n{self.name} energy calibration")
        print(f"Gradient  = {self.calibration['gradient']:.6f} keV/channel")
        print(f"Intercept = {self.calibration['intercept']:.3f} keV")
        print(f"Accepted peaks: {len(all_peaks)}")
        print(f"Unidentified peaks: {len(unidentified_peaks)}")

        if show_plot:
            draw_calibration(self.calibration)
        return self.calibration

    def channel_to_energy(self, channel):
        if self.calibration is None:
            raise ValueError("Calibrate the detector first.")
        return convert_channel(channel, self.calibration)

    def energy_to_channel(self, energy):
        if self.calibration is None:
            raise ValueError("Calibrate the detector first.")
        return convert_energy(energy, self.calibration)

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
        self.resolution = fit_resolution(self, title=title, show=show)
        return self.resolution

    def calculate_absolute_efficiency(self, plot=True):
        from Efficiency import absolute_efficiency, plot_absolute_efficiency

        self.efficiency = absolute_efficiency(self)

        if plot:
            plot_absolute_efficiency(self)

        return self.efficiency

    def calculate_intrinsic_efficiency(self, plot=True):
        """Calculate and store the detector's intrinsic efficiency."""

        if not self.efficiency:
            raise ValueError(
                "No absolute efficiency results stored. "
                "Run calculate_absolute_efficiency() first."
            )

        from Efficiency import intrinsic_eff, plot_intrinsic_efficiency, calculate_distance_efficiencies

        self.efficiency = intrinsic_eff(self)
        self.distance_efficiency = calculate_distance_efficiencies(self)

        if plot:
            plot_intrinsic_efficiency(self)

        return self.efficiency

    def calculate_off_axis(self, source, plot=True):
        """Calculate, store and optionally plot the off-axis response."""

        if self.calibration is None:
            raise ValueError("Calibrate the detector first.")

        from Offaxis import run_off_axis, plot_comparison

        results = run_off_axis(self, source)

        self.off_axis_results[source] = results

        if plot:
            plot_comparison({
                (self.name, source, energy): result
                for energy, result in results.items()
            })

        return results



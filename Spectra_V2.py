
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from Practice import read_spe, read_mca

from scipy.optimize import curve_fit
from matplotlib.widgets import SpanSelector


class Spectrum:

    def __init__(
        self,
        filename,
        counts,
        channels,
        live_time,
        background=None
    ):

        self.filename = str(filename)
        self.counts = np.asarray(counts, dtype=float)
        self.channels = np.asarray(channels, dtype=float)
        self.live_time = float(live_time)

        self.background = background

        self.rois = []
        self.peaks = []

        if len(self.counts) != len(self.channels):
            raise ValueError("Counts and channels must have equal lengths.")

        if self.live_time <= 0:
            raise ValueError("Live time must be positive.")

        if np.any(np.diff(self.channels) <= 0):
            raise ValueError("Channels must be strictly increasing.")

    # --------------------------------------------------
    # LOAD SPECTRUM
    # --------------------------------------------------

    @classmethod
    def from_file(cls, filename, background_file=None):

        suffix = Path(filename).suffix.lower()

        if suffix == ".spe":
            counts, channels, live_time = read_spe(filename)

        elif suffix == ".mca":
            counts, channels, live_time = read_mca(filename)

        else:
            raise ValueError(f"Unsupported file type: {suffix}")

        background = None

        if background_file is not None:
            background = cls.from_file(background_file)

        return cls(
            filename=filename,
            counts=counts,
            channels=channels,
            live_time=live_time,
            background=background
        )

    # --------------------------------------------------
    # COUNTS PER SECOND
    # --------------------------------------------------

    def cps(self):

        return self.counts / self.live_time

    def corrected_cps(self):

        if self.background is None:
            raise ValueError("No background spectrum associated with this spectrum.")

        if not np.array_equal(self.channels, self.background.channels):
            raise ValueError("Sample and background channels do not match.")

        return (
            self.cps()
            - self.background.cps()
        )

    def uncertainties(self, corrected=False):
        """
        Calculate uncertainties in counts per second.

        Assumes independent Poisson counting statistics.
        """

        sample_variance = (
            np.maximum(self.counts, 0)
            / self.live_time**2
        )

        if corrected:

            if self.background is None:
                raise ValueError("No background spectrum available.")

            if not np.array_equal(self.channels, self.background.channels):
                raise ValueError("Sample and background channels do not match.")

            background_variance = (
                np.maximum(self.background.counts, 0)
                / self.background.live_time**2
            )

            variance = sample_variance + background_variance

        else:

            variance = sample_variance

        return np.sqrt(variance)

    # --------------------------------------------------
    # PLOT SPECTRUM
    # --------------------------------------------------

    def plot(self, corrected=False):

        if corrected:
            counts = self.corrected_cps()
            title = f"{Path(self.filename).name} - Background Subtracted"

        else:
            counts = self.cps()
            title = Path(self.filename).name

        plt.figure(figsize=(10, 5))

        plt.plot(
            self.channels,
            counts,
            color="black",
            linewidth=0.8
        )

        plt.xlabel("Channel")
        plt.ylabel("Counts per Second")
        plt.title(title)

        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()

    # --------------------------------------------------
    # GAUSSIAN + LINEAR BACKGROUND MODEL
    # --------------------------------------------------

    @staticmethod
    def gaussian_linear(x, A, mu, sigma, m, c):
        """
        Gaussian peak sitting on a linear background.

        A     = Gaussian amplitude
        mu    = Gaussian centroid
        sigma = Gaussian standard deviation
        m     = Background gradient
        c     = Background intercept
        """

        gaussian = A * np.exp(
            -((x - mu)**2) / (2 * sigma**2)
        )

        background = m * x + c

        return gaussian + background

    # --------------------------------------------------
    # FIT A SINGLE ROI
    # --------------------------------------------------

    def fit_roi(self, lower, upper, corrected=True):
        """
        Fit a Gaussian plus linear background within an ROI.

        Returns fitted parameters and their uncertainties.
        """

        if upper <= lower:
            raise ValueError("Upper ROI boundary must exceed lower boundary.")

        if corrected:
            y = self.corrected_cps()
        else:
            y = self.cps()

        errors = self.uncertainties(corrected=corrected)

        # Extract ROI

        mask = (
            (self.channels >= lower)
            & (self.channels <= upper)
        )

        x_roi = self.channels[mask]
        y_roi = y[mask]
        err_roi = errors[mask]

        if len(x_roi) < 7:
            raise ValueError("ROI contains too few channels for fitting.")

        # Initial parameter estimates

        n = len(x_roi)

        edge_n = max(2, n // 5)

        edge_x = np.concatenate([
            x_roi[:edge_n],
            x_roi[-edge_n:]
        ])

        edge_y = np.concatenate([
            y_roi[:edge_n],
            y_roi[-edge_n:]
        ])

        m_initial, c_initial = np.polyfit(
            edge_x,
            edge_y,
            1
        )

        background_initial = (
            m_initial * x_roi + c_initial
        )

        residual = y_roi - background_initial

        max_index = np.argmax(residual)

        A_initial = max(
            residual[max_index],
            1e-10
        )

        mu_initial = x_roi[max_index]

        channel_spacing = np.median(np.diff(x_roi))

        sigma_initial = max(
            (upper - lower) / 10,
            channel_spacing
        )

        # Parameter bounds

        sigma_min = channel_spacing * 0.2

        sigma_max = max(
            upper - lower,
            sigma_min * 2
        )

        initial_guess = [
            A_initial,
            mu_initial,
            sigma_initial,
            m_initial,
            c_initial
        ]

        lower_bounds = [
            0,
            lower,
            sigma_min,
            -np.inf,
            -np.inf
        ]

        upper_bounds = [
            np.inf,
            upper,
            sigma_max,
            np.inf,
            np.inf
        ]

        # Weighted nonlinear least-squares fitting

        errors = np.maximum(
            err_roi,
            1e-10
        )

        parameters, covariance = curve_fit(
            self.gaussian_linear,
            x_roi,
            y_roi,
            p0=initial_guess,
            bounds=(lower_bounds, upper_bounds),
            sigma=errors,
            absolute_sigma=True,
            maxfev=20000
        )

        A, mu, sigma, m, c = parameters

        parameter_errors = np.sqrt(
            np.maximum(np.diag(covariance), 0)
        )

        # Calculate peak properties

        fwhm_factor = 2 * np.sqrt(2 * np.log(2))

        fwhm = fwhm_factor * sigma

        fwhm_error = fwhm_factor * parameter_errors[2]

        area = A * sigma * np.sqrt(2 * np.pi)

        # Error propagation for Gaussian area

        variance_area = (
            sigma**2 * covariance[0, 0]
            + A**2 * covariance[2, 2]
            + 2 * A * sigma * covariance[0, 2]
        )

        area_error = np.sqrt(
            max(variance_area, 0)
        ) * np.sqrt(2 * np.pi)

        result = {

            "roi": (lower, upper),

            "amplitude": A,
            "amplitude_error": parameter_errors[0],

            "centroid": mu,
            "centroid_error": parameter_errors[1],

            "sigma": sigma,
            "sigma_error": parameter_errors[2],

            "fwhm": fwhm,
            "fwhm_error": fwhm_error,

            "area": area,
            "area_error": area_error,

            "background_gradient": m,
            "background_intercept": c,

            "parameters": parameters,
            "covariance": covariance

        }

        return result

    # --------------------------------------------------
    # INTERACTIVE ROI SELECTION AND FITTING
    # --------------------------------------------------

    def interactive_fit(self, corrected=True):
        """
        Interactively select ROIs by clicking and dragging.

        Controls:

        Left click + drag : Select ROI
        Enter             : Finish analysis
        Backspace         : Remove most recent ROI
        Escape            : Finish analysis
        """

        if corrected:
            y = self.corrected_cps()
            title = "Background Subtracted Spectrum"

        else:
            y = self.cps()
            title = "Raw Spectrum"

        self.rois = []
        self.peaks = []

        fig, ax = plt.subplots(figsize=(12, 6))

        fig.subplots_adjust(bottom=0.15)

        # Plot spectrum

        ax.plot(
            self.channels,
            y,
            color="black",
            linewidth=0.8,
            label="Spectrum"
        )

        ax.set_xlabel("Channel")
        ax.set_ylabel("Counts per Second")

        ax.set_title(
            f"{Path(self.filename).name}\n{title}"
        )

        ax.grid(alpha=0.3)

        # Instructions

        instruction = (
            "Click and drag to select an ROI\n"
            "ENTER: Finish | BACKSPACE: Remove last ROI"
        )

        ax.text(
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

        # ----------------------------------------------
        # CALLBACK: USER SELECTS ROI
        # ----------------------------------------------

        def on_select(xmin, xmax):

            lower = min(xmin, xmax)
            upper = max(xmin, xmax)

            if upper <= lower:
                return

            try:

                result = self.fit_roi(
                    lower,
                    upper,
                    corrected=corrected
                )

            except (ValueError, RuntimeError) as error:

                print(f"\nROI fitting failed: {error}")
                return

            roi_number = len(self.peaks) + 1

            result["name"] = f"ROI {roi_number}"

            self.peaks.append(result)

            self.rois.append((lower, upper))

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

            A, mu, sigma, m, c = result["parameters"]

            total_fit = self.gaussian_linear(
                x_fit,
                A,
                mu,
                sigma,
                m,
                c
            )

            gaussian_fit = (
                A * np.exp(
                    -((x_fit - mu)**2)
                    / (2 * sigma**2)
                )
            )

            # Plot complete fit

            total_line, = ax.plot(
                x_fit,
                total_fit,
                color="red",
                linewidth=2,
                label="Gaussian + Background"
            )

            # Plot Gaussian component

            gaussian_line, = ax.plot(
                x_fit,
                gaussian_fit + m * x_fit + c,
                color="green",
                linestyle="--",
                linewidth=1.5,
                label="Gaussian Component"
            )

            # Mark centroid

            centroid_line = ax.axvline(
                mu,
                color="purple",
                linestyle=":",
                alpha=0.8
            )

            roi_artists.append([
                roi_patch,
                total_line,
                gaussian_line,
                centroid_line
            ])

            fig.canvas.draw_idle()

            print(f"\nROI {roi_number} fitted successfully.")

            print(
                f"Centroid: {mu:.3f} "
                f"+/- {result['centroid_error']:.3f} channels"
            )

            print(
                f"FWHM: {result['fwhm']:.3f} "
                f"+/- {result['fwhm_error']:.3f} channels"
            )

            print(
                f"Area: {result['area']:.3f} "
                f"+/- {result['area_error']:.3f} CPS channels"
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

            if event.key in ("enter", "escape", "q"):

                print("\nFinishing ROI selection...")

                plt.close(fig)

            elif event.key == "backspace":

                if len(self.peaks) == 0:
                    return

                # Remove last fitted ROI

                self.peaks.pop()
                self.rois.pop()

                artists = roi_artists.pop()

                for artist in artists:
                    artist.remove()

                print("\nLast ROI removed.")

                fig.canvas.draw_idle()

        fig.canvas.mpl_connect(
            "key_press_event",
            on_key
        )

        # Keep selector alive during interaction

        self.selector = selector

        plt.tight_layout()

        plt.show()

        # ----------------------------------------------
        # PRINT FINAL RESULTS
        # ----------------------------------------------

        print("\n" + "=" * 55)
        print("SPECTRUM ANALYSIS RESULTS")
        print("=" * 55)

        if len(self.peaks) == 0:

            print("No peaks fitted.")

        else:

            for i, peak in enumerate(self.peaks, 1):

                print(f"\nPeak {i}")

                print(
                    f"ROI: {peak['roi'][0]:.1f}"
                    f" - {peak['roi'][1]:.1f}"
                )

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

        return self.peaks

    # --------------------------------------------------
    # SUMMARY
    # --------------------------------------------------

    def summary(self):

        print(f"File: {Path(self.filename).name}")
        print(f"Live Time: {self.live_time:.1f} s")
        print(f"Channels: {len(self.channels)}")

        if self.background is None:
            print("Background File: None")

        else:
            print(
                f"Background File: "
                f"{Path(self.background.filename).name}"
            )

    # --------------------------------------------------
    # REPRESENTATION
    # --------------------------------------------------

    def __repr__(self):

        return (
            f"Spectrum("
            f"file='{Path(self.filename).name}', "
            f"channels={len(self.channels)}, "
            f"live_time={self.live_time:.1f}s)"
        )


# ======================================================
# MAIN
# ======================================================

def main():

    data_folder = Path("CUITS")

    spec = Spectrum.from_file(
        data_folder / "BGO_Ba_0_10cm_180s.Spe",
        background_file=data_folder / "BGO_Background_180s.Spe"
    )

    spec.summary()

    # Interactive ROI selection and Gaussian fitting

    peaks = spec.interactive_fit(
        corrected=True
    )


if __name__ == "__main__":

    main()

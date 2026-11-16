"""Store a gamma-ray spectrum and perform basic preprocessing."""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

def read_spe(filename):
    """Read a .Spe file. Returns (counts, channels, live_time_seconds)."""
    with open(filename, "r") as f:
        lines = [line.strip() for line in f]

    # $MEAS_TIM: is followed by a line like "180 185" (live time, real time)
    time = float(lines[lines.index("$MEAS_TIM:") + 1].split()[0])

    # $DATA: is followed by a header line ("0 1023" = first/last channel),
    # then one count value per line until the next "$" keyword (e.g. $ROI:)
    start = lines.index("$DATA:") + 2
    counts = []
    for line in lines[start:]:
        if line.startswith("$"):
            break
        if line:
            counts.append(float(line.replace(",", " ").split()[0]))

    counts = np.array(counts)
    channels = np.arange(len(counts))
    return counts, channels, time


def read_mca(filename):
    '''
    Function reads an .mca file (CdTe dectector) and returns lists corresponding
    to counts, channels, and time.
    ______________
    Parameters
    - .mca file: file containing gamma ray detector data.    
    ______________
    Outputs
    - Counts: List of floats values
        
    - Channels: List of float values
        
    - Time: float value    
    '''

    with open(filename, 'r') as file:
        spectrum_data = file.readlines()

    counts = []
    time = float(spectrum_data[8].replace(' ', '').replace('REAL_TIME-', ''))
    
    beginning = spectrum_data.index("<<DATA>>\n") + 1
    
    for line in spectrum_data[beginning:]:
        if line == "<<END>>\n":
            break
        else:
            counts.append(float(line))

    channels = list(range(len(counts)))
        
    return counts, channels, time


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
        self.fit_results = []

        # if len(self.counts) != len(self.channels):
        #     raise ValueError("Counts and channels must have equal lengths.")
        #
        # if self.live_time <= 0:
        #     raise ValueError("Live time must be positive.")
        #
        # if np.any(np.diff(self.channels) <= 0):
        #     raise ValueError("Channels must be strictly increasing.")


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


    def cps(self):

        return self.counts / self.live_time


    def corrected_cps(self):

        if self.background is None:
            raise ValueError(
                "No background spectrum associated with this spectrum."
            )

        if not np.array_equal(self.channels, self.background.channels):
            raise ValueError(
                "Sample and background channels do not match."
            )

        return self.cps() - self.background.cps()


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
                raise ValueError(
                    "Sample and background channels do not match."
                )

            background_variance = (
                np.maximum(self.background.counts, 0)
                / self.background.live_time**2
            )

            variance = sample_variance + background_variance

        else:

            variance = sample_variance

        return np.sqrt(variance)


    def plot(self, corrected=False):

        if corrected:
            count_rate = self.corrected_cps()
            title = f"{Path(self.filename).name} - Background Subtracted"

        else:
            count_rate = self.cps()
            title = Path(self.filename).name

        plt.figure(figsize=(10, 5))

        plt.plot(
            self.channels,
            count_rate,
            color="black",
            linewidth=0.8
        )

        plt.xlabel("Channel")
        plt.ylabel("Counts per Second")
        plt.title(title)

        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()


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


    def __repr__(self):

        return (
            f"Spectrum("
            f"file='{Path(self.filename).name}', "
            f"channels={len(self.channels)}, "
            f"live_time={self.live_time:.1f}s)"
        )

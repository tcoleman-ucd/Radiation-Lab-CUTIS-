from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from Practice import read_spe, read_mca



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
        self.counts = np.array(counts)
        self.channels = np.array(channels)
        self.live_time = float(live_time)

        # Another Spectrum object or None
        self.background = background

    @classmethod
    def from_file(cls, filename, background_file=None):
        """
        Create a Spectrum object from a .Spe or .mca file.
        """

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
        """
        Return counts per second.
        """

        return self.counts / self.live_time

    def corrected_cps(self):
        """
        Return background-subtracted counts per second.
        """

        if self.background is None:
            raise ValueError(
                "No background spectrum associated with this spectrum."
            )

        return (self.cps() - self.background.cps())

    def plot(self, corrected=False):
        """
        Plot either the raw or background-subtracted spectrum.
        """

        if corrected:

            cps = self.corrected_cps()

            title = (f"{Path(self.filename).name}"
                "\nBackground Subtracted"
            )

        else:

            cps = self.cps()

            title = Path(self.filename).name

        plt.figure(figsize=(8, 5))

        plt.plot(self.channels, cps)

        plt.title(title)
        plt.xlabel("Channel")
        plt.ylabel("Counts per Second")

        plt.grid(alpha=0.3)

        plt.show()

    def summary(self):

        print(f"File: {Path(self.filename).name}")
        print(f"Live Time: {self.live_time:.1f} s")
        print(f"Channels: {len(self.channels)}")

        if self.background is None:
            print("Background File: None")

        else:
            print("Background File: "
                f"{Path(self.background.filename).name}"
            )

    def __repr__(self):

        return (
            f"Spectrum("
            f"file='{Path(self.filename).name}', "
            f"channels={len(self.channels)}, "
            f"live_time={self.live_time:.1f}s)"
        )


def main():

    data_folder = Path("CUITS")

    spec = Spectrum.from_file(data_folder / "BGO_Co_0_10cm_420s.Spe",
        background_file=data_folder / "BGO_Background_180s.Spe"
    )

    # print(spec)

    spec.summary()

    spec.plot()

    spec.plot(corrected=True)


if __name__ == "__main__":
    main()
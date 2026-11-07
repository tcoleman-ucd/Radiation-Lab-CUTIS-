"""
data_extraction.py
------------------
Read gamma-ray spectrum files (.Spe for NaI/BGO, .mca for CdTe) and do the
first processing steps: counts -> count rate, background subtraction, and a
rough peak-channel finder.

Every function returns plain numpy arrays so the next stage of the pipeline
(peak fitting, calibration, ...) can use them directly.
"""

import re
from pathlib import Path

import numpy as np


# ----------------------------------------------------------------------------
# Reading files
# ----------------------------------------------------------------------------
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



def extraction(filename):
    """Read any supported spectrum file. Returns (counts, channels, time)."""
    suffix = Path(filename).suffix.lower()
    if suffix == ".spe":
        return read_spe(filename)
    if suffix == ".mca":
        return read_mca(filename)
    raise ValueError(f"Unsupported file type: {filename}")


# ----------------------------------------------------------------------------
# Processing
# ----------------------------------------------------------------------------
def counts_per_sec(filename):
    """Return (count rate in counts/s, channels) for one file."""
    counts, channels, time = extraction(filename)
    return counts / time, channels


def remove_background(background_file, files):
    """Subtract the background count RATE from each file's count rate.

    Returns {filename: background-subtracted count rate array}.
    (Both are converted to counts/s first, since the runs have different lengths.)
    """
    background_rate, _ = counts_per_sec(background_file)

    no_background = {}
    for file in files:
        rate, _ = counts_per_sec(file)
        if len(rate) != len(background_rate):
            raise ValueError(f"{file} and {background_file} have different channel counts")
        no_background[file] = rate - background_rate
    return no_background


def amplitude(channels, files, background_file, tag="_0_10cm"):
    """For files whose name contains `tag` (default: 0 degrees, 10 cm), find
    the channel of the highest background-subtracted point. The tag must start
    at a number boundary, so "0_10cm" does not match "30_10cm".

    Returns {filename: peak_channel}. This is only a rough first guess to seed
    the Gaussian fits, not the final centroid. It picks the tallest bin in the
    whole spectrum, so use an ROI per peak when a spectrum has several lines.
    """
    no_background = remove_background(background_file, files)

    peak_channels = {}
    for file, rate in no_background.items():
        peak_channels[file] = channels[np.argmax(rate)]
    return peak_channels


def load_spectra(folder=".", pattern="*.Spe"):
    """Load every matching file in `folder` (relative path, no hard-coding).

    Returns ({filename: counts}, channels, {filename: live_time}).
    """
    spectra, times, channels = {}, {}, None
    for path in sorted(Path(folder).glob(pattern)):
        counts, channels, time = extraction(path)
        spectra[path.name] = counts
        times[path.name] = time
    return spectra, channels, times


# ----------------------------------------------------------------------------
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    BACKGROUND = "BGO_Background_180s.Spe"
    spe_files = sorted(str(p) for p in Path(".").glob("*.Spe"))
    if not spe_files:
        raise SystemExit("No .Spe files found in the current folder.")

    _, channels, _ = extraction(spe_files[0])
    print(amplitude(channels, spe_files, BACKGROUND))

    # Quick look at every background-subtracted spectrum
    no_bg = remove_background(BACKGROUND, spe_files)
    for name, rate in no_bg.items():
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.plot(channels, rate)
        ax.set_title(f"Spectrum Data: {name}")
        ax.set_xlabel("Channel")
        ax.set_ylabel("Count rate - background (counts/s)")
    plt.show()
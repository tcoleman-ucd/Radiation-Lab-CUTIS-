# Gamma-Ray Detector Characterisation

Python tools for analysing gamma-ray spectra and comparing the performance of three detectors: **BGO**, **NaI**, and **CdTe**. This project was developed for a gamma-ray spectroscopy laboratory exercise and uses measurements from **Am-241, Ba-133, Cs-137, and Co-60**.

The analysis covers energy calibration, photopeak fitting, energy resolution, absolute and intrinsic photopeak efficiency, and detector response at different source distances and off-axis angles.

## Features

- Read `.Spe` and `.mca` spectra, including acquisition live time.
- Convert counts to counts per second (CPS) and subtract background spectra.
- Interactively select regions of interest (ROIs) and fit **one or two Gaussian peaks** on a linear continuum.
- Identify photopeaks using known gamma-ray energies.
- Fit a linear channel-to-energy calibration and compare it with a quadratic calibration where enough peaks are available.
- Calculate energy resolution (FWHM / energy) and fit its energy dependence.
- Calculate absolute and intrinsic photopeak efficiencies, including propagated uncertainties.
- Analyse angular response at **−90°, −45°, 0°, 45°, and 90°** relative to the on-axis measurement.
- Save analysis plots as PDF and/or JPG files in `Plots/`.

## Requirements

- **Python 3.12 or newer**
- NumPy
- SciPy
- Matplotlib (with an interactive graphical backend)

The fitting interface opens Matplotlib windows, so run the program in an environment that supports interactive figures (for example, a local Python terminal or IDE).

## Getting started

1. Clone or download the repository.
2. Put the measurement files somewhere **inside the repository directory**, including the necessary background spectra. The program searches recursively for filenames listed in `main.py`.
3. Install the dependencies above.
4. Run `main.py` with one or more detector names.

Analyse one detector:

```bash
python main.py BGO
```

Analyse all three:

```bash
python main.py BGO NaI CdTe
```

Skip the **off-axis calculations** for a shorter analysis run:

```bash
python main.py BGO NaI CdTe --skip-off-axis
```

Valid detector names are **`BGO`**, **`NaI`**, and **`CdTe`** (case-sensitive). The current command-line workflow performs calibration, resolution, absolute efficiency, intrinsic efficiency, and—unless skipped—off-axis analysis. Distance-dependent efficiencies are calculated as part of the intrinsic-efficiency stage.

## Interactive peak fitting

When calibration begins, the program opens each source spectrum in an interactive Matplotlib window.

| Action | Control |
| --- | --- |
| Select a fitting region | Click and drag across the spectrum |
| Fit one Gaussian in the next region | Press `1` |
| Fit two overlapping Gaussians in the next region | Press `2`, drag an ROI, then click the approximate centre of each peak |
| Undo the previous ROI | Press `Backspace` |
| Cancel an unfinished selection | Press `Escape` |
| Finish the current spectrum | Press `Enter` (or `q` when no identification is pending) |

The main script uses **plot-based peak identification**. After fitting, select a reference-energy number, hit confirm,
A terminal identification interface is also available when calling `detector.calibrate(identification="terminal")` directly from Python.
Enter a number corresponding to an energy, type an energy in keV, or enter `0` to mark a peak as unidentified. Only accepted peaks contribute to the calibration and subsequent photopeak analysis.


For overlapping features (notably some Ba-133 photopeaks), the two-Gaussian option fits both peak components simultaneously.

## Analysis workflow

### 1. Load and preprocess spectra

`spectra.py` loads the measurement and background files, converts both to CPS, and provides background-subtracted count rates. Measurements are represented by `Spectrum` objects; each detector groups its spectra in a `Detector` object (`detector.py`).

### 2. Calibrate energy

`fitting.py` and `interactive_fitting.py` determine photopeak centroids in channel space. `calibration.py` fits the default linear relationship

\[
E = mC + c,
\]

where \(C\) is channel number and \(E\) is energy in keV. A quadratic fit is also evaluated for comparison when at least four calibration peaks are available, but the **linear fit remains the default** for downstream calculations.

### 3. Calculate energy resolution

`Resolution.py` evaluates

\[
R(E) = \frac{\mathrm{FWHM}(E)}{E},
\]

with FWHM converted from channels to keV using the energy calibration. Where enough valid points are available, it fits

\[
\mathrm{FWHM}(E)^2 = a + bE + cE^2.
\]

### 4. Calculate detector efficiency

`Efficiency.py` calculates **absolute photopeak efficiency** from the fitted photopeak count rate:

\[
\varepsilon_{\mathrm{abs}} = \frac{\text{photopeak count rate}}{A(t)\,P_\gamma},
\]

where \(A(t)\) is the decay-corrected source activity and \(P_\gamma\) is the emission probability for the gamma-ray line.

The **intrinsic photopeak efficiency** is then estimated using the detector's geometric factor \(G\), calculated in `geometry.py`:

\[
\varepsilon_{\mathrm{int}} = \frac{\varepsilon_{\mathrm{abs}}}{G}.
\]

A log-polynomial model is fitted to valid intrinsic-efficiency measurements when sufficient data are available. For additional source distances, the code reuses the original fitting ROIs and stores results in `detector.distance_efficiency`.

### 5. Analyse off-axis response

`Offaxis.py` refits the photopeaks at the configured angles and compares properties including photopeak area, peak-to-total ratio, FWHM, energy resolution, and photopeak height. It also computes angular geometric factors and efficiency estimates. Relative plots are normalised to the **0°** measurement.

## Repository structure

| File | Purpose |
| --- | --- |
| `main.py` | Detector configuration, dataset filenames, and command-line entry point |
| `spectra.py` | `.Spe`/`.mca` readers, background subtraction, and `Spectrum` class |
| `detector.py` | `Detector` class and orchestration of the analysis stages |
| `fitting.py` | Single/double Gaussian fitting and peak parameter uncertainties |
| `interactive_fitting.py` | Interactive ROI selection and photopeak identification |
| `calibration.py` | Energy calibration, residuals, and calibration plots |
| `Resolution.py` | Energy-resolution calculations and model fitting |
| `Efficiency.py` | Source decay, emission fractions, and efficiency calculations |
| `geometry.py` | Circular/cylindrical and square-detector geometric factors |
| `Offaxis.py` | Angular fitting, response metrics, and comparison plots |
| `Plotting.py` | Additional/experimental plotting code; not used by the main workflow |

Python filenames are case-sensitive on some operating systems: keep the existing capitalisation for `Efficiency.py`, `Resolution.py`, and `Offaxis.py`.

## Input data and configuration

The `Detectors` dictionary in `main.py` specifies the background file and measurement filenames for every detector. It currently expects:

- On-axis spectra at **10 cm** (used for calibration).
- Additional on-axis spectra at **20 cm**.
- Off-axis spectra at **±45°** and **±90°**, with the 10 cm spectrum reused for **0°**.

The source identifiers are `Am`, `Ba`, `Cs`, and `Co`. The corresponding reference gamma energies are listed in `Known_Energies` in `main.py`.

Before using the efficiency results quantitatively, check `SOURCE_PARAMETERS`, `CALIBRATION_DATE`, and the gamma-ray emission fractions in `Efficiency.py` against the actual source certificates and laboratory documentation. **The code assumes initial activities are in microcuries (µCi)** and converts them to becquerels. Source distances and detector dimensions are expressed in **metres**.

## Output

Results are printed to the terminal and saved on the in-memory `Detector` object. Generated figures are written to the `Plots/` directory (created automatically), including files such as:

```text
Plots/
├── Calibration_BGO.pdf
├── Resolution_BGO.pdf
├── Efficiency_absolute_BGO.pdf
├── Efficiency_intrinsic_BGO.pdf
└── Offaxis_efficiencies_BGO_Cs_661.7keV.pdf
```

Some off-axis comparison and fit-check figures are written as JPG files. The exact set of plots depends on which peaks were fitted successfully and whether the off-axis analysis was enabled. **The current main script does not export the numerical results or distance-dependent plots to CSV**; those values remain in the detector objects during execution.

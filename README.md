# Gamma-ray detector analysis — approved changes

This folder contains the approved revisions to the original Python scripts, not the earlier unreviewed ZIP.

Run one detector:

```bash
python main.py BGO
```

Run several detectors:

```bash
python main.py BGO NaI CdTe
```

Skip off-axis fitting while generating calibration, resolution, and efficiency plots:

```bash
python main.py BGO --skip-off-axis
```

Plots are written to `Plots/` (JPG and PDF where implemented).

**User-requested constraints:** `get_rois()` and all original Gaussian/interactively selected fitting code remain unchanged. No `Comparison.py` module is included. The NaI radius remains **0.02921 m** in `main.py`, per your request to change it yourself to **0.0254 m**. The CdTe `.mca` reader changes *only* measurement-time selection, preferring `LIVE_TIME` and falling back to `REAL_TIME`; the original counts reader remains untouched. The intrinsic efficiency model is fit using `scipy.optimize.curve_fit`, with a fixed 10–2000 keV *display* range (parts outside the data are extrapolation). Curve equations appear in legends and resolution points retain isotope-and-energy annotations.

**Before reporting numerical efficiencies:** verify source activity units/date, detector dimensions, and the MCA header fields. No raw `.Spe`/`.mca` data were supplied, so a real end-to-end experimental analysis has not been run.

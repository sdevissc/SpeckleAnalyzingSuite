# Speckle Analyzing Suite

A Linux-native desktop application for **double star astrometry via speckle interferometry**. Measures the polar angle (position angle) and angular separation of visual binary stars from short-exposure video recordings (SER format).

This project is a modern rewrite of the [Speckle Toolbox by David Rowe](http://www.mikeoates.org/speckle/), redesigned for Linux compatibility and streamlined for batch processing of large observation campaigns.

---

## Background

Speckle interferometry is a technique that overcomes atmospheric seeing by recording thousands of short-exposure frames of a double star. Each frame freezes the instantaneous atmospheric distortion ("speckle pattern"). By computing the **bispectrum** of the averaged power spectrum, the relative position of the two components can be recovered at the diffraction limit of the telescope — far below what is achievable with conventional long-exposure imaging.

This suite implements the full reduction pipeline: from raw video calibration through to final astrometric measurements ready for submission to double star catalogs such as the Washington Double Star Catalog (WDS).

---

## Features

### 🧭 Tab 1 — Drift Alignment (Calibration)

Determines the camera orientation and plate scale by recording a star drifting across the field at sidereal speed (tracking turned off).

- Loads SER video files (single file or batch of multiple files)
- Detects stellar centroids frame by frame
- Fits the drift track using **Total Least Squares / SVD** to extract:
  - **Camera angle** (degrees) — orientation of the sensor axes relative to celestial N/E
  - **Pixel scale** (arcsec/px) — derived from sidereal drift rate and target declination
- Interactive sigma-clipping slider for outlier rejection
- Batch navigator: process multiple SER files sequentially, each with independently tuned trim and rejection parameters
- **Optics panel** — optional instrument parameters:
  - Pixel size (µm) → derives focal length
  - Aperture / diameter (mm) → derives f-ratio
  - Wavelength (nm, default 550) → computes Nyquist sampling factor
- Exports calibration to JSON (single file or batch aggregate with per-file statistics)
- Auto-fills target declination from FireCapture / SharpCap / Genika companion text files, or via Simbad name resolution

### ⚙ Tab 2 — Preprocessing

Prepares the raw SER cube for bispectrum analysis.

- Crops frames to a configurable ROI centred on the target
- Aligns frames by centroid registration
- Rejects poor-quality frames based on contrast threshold
- Outputs a cleaned `.npz` cube ready for analysis

### 🔭 Tab 3 — Analysis

Reconstructs the double star geometry from the preprocessed cube.

- Computes the **averaged power spectrum** and **averaged bispectrum**
- Iterative bispectrum phase reconstruction (configurable K_max, iterations, Wiener ε)
- Displays the autocorrelation and reconstructed image
- Derives **angular separation** (arcsec) and **position angle** (degrees) using the drift calibration
- Exports results to CSV and WDS-format text

---

## Requirements

- Python 3.10+
- PyQt6
- NumPy, SciPy
- Matplotlib / PyQtGraph
- astropy (optional, for FITS support)

Install dependencies:

```bash
pip install pyqt6 numpy scipy matplotlib pyqtgraph astropy
```

---

## Usage

```bash
python speckle_suite.py
```

### Typical workflow

1. **Drift tab**: Load one or more SER drift recordings. Enter the target declination. Run the analysis, tune the sigma slider, optionally enter pixel size / aperture / wavelength. Save the calibration JSON.
2. **Preprocess tab**: Load your science SER file. Set the ROI size and quality threshold. Run preprocessing. Save the NPZ cube.
3. **Analysis tab**: Load the NPZ cube and the calibration JSON. Run the bispectrum reconstruction. Read off the position angle and separation. Export to CSV / WDS.

---

## Calibration JSON format

The drift calibration JSON contains all parameters needed by the analysis tab:

```json
{
  "camera_angle_deg": 12.3456,
  "pixel_scale_arcsec": 0.08983,
  "sigma_angle_deg": 0.0021,
  "sigma_scale_arcsec": 0.000004,
  "focal_length_mm": 8496.0,
  "f_ratio": 21.2,
  "wavelength_nm": 550.0,
  "sampling_factor": 3.16,
  "fit_method": "TLS/SVD (Total Least Squares)",
  ...
}
```

Batch calibration files additionally contain `camera_angle_mean_deg`, `camera_angle_std_deg`, `pixel_scale_mean_arcsec`, `pixel_scale_std_arcsec`, and a `measurements` array with one entry per SER file.

---

## Acknowledgements

- Original **Speckle Toolbox** concept and algorithms: David Rowe
- Bispectrum reconstruction approach: see Weigelt (1977) and the *Journal of Double Star Observations* reduction pipeline literature
- SER file format parsing based on the SerUtils specification

---

## License

MIT License — see `LICENSE` for details.

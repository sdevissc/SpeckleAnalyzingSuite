# SpeckleAnalyzingSuite

A Linux-compatible, open-source reimplementation of the **Speckle Tool Box (STB)** workflow for double-star astrometry by speckle interferometry.

---

## Acknowledgements and Attribution

SpeckleAnalyzingSuite is an independent reimplementation inspired by the **Speckle Tool Box (STB)**, originally developed by **David Rowe** (PlainWave Instruments). The bispectrum accumulation, iterative Knox–Thompson phase retrieval, and overall reduction workflow — drift calibration, frame selection, bispectral analysis, astrometric measurement — are closely modelled on the approach demonstrated in STB. The primary motivation for this project was to provide a **Linux-compatible, open-source equivalent of STB**, reproducing its working methodology while adding support for batch processing and a modern Python/PyQt6 interface, without requiring a Windows environment.

The practical use of STB for double-star astrometry is described in:

> Harshaw, R. (2017). *The Speckle Toolbox: A Powerful Data Reduction Tool for CCD Astrometry*. Journal of Double Star Observations, 13(1), 52–67.

The algorithm descriptions in this document draw on the published literature on speckle interferometry and bispectral analysis. The description of the reduction workflow is inspired by the presentation in Harshaw (2017), which itself documents the use of STB.

No source code from STB has been reproduced. SpeckleAnalyzingSuite is an independent implementation of published algorithms, developed for personal and community observational use. The authors gratefully acknowledge the original work of David Rowe, without which this project would not exist.

---

## Overview

SpeckleAnalyzingSuite automates the full speckle interferometry pipeline, from a raw SER or FITS video sequence to a calibrated (rho, theta) measurement ready for submission to the Washington Double Star Catalog (WDS). The application is organised as five tabs:

| Tab | Name | Purpose |
|-----|------|---------|
| 1 | Drift Alignment | Camera angle and plate scale from a sidereal drift recording |
| 2 | Preprocess | Frame selection by RMS contrast, registration, ROI crop to FITS cube |
| 3 | Bispectrum | Bispectrum accumulation, phase retrieval, image reconstruction |
| 4 | Astrometry | Marker placement, calibration, measurement, aperture photometry, export |
| 5 | History | Compare your measurements against INT4 historical data and ORB6 orbits |

---

## Dependencies and Installation

### Requirements

- Linux (Ubuntu 22.04 or later recommended)
- Python 3.11 or later

### Install Python dependencies

```bash
pip install PyQt6 pyqtgraph numpy scipy astropy astroquery matplotlib
```

| Library | Min. version | Role |
|---------|-------------|------|
| PyQt6 | 6.4 | GUI (windows, widgets) |
| pyqtgraph | 0.13 | Interactive plots (drift, image display) |
| numpy | 1.24 | Numerical computing, FFT, bispectrum |
| scipy | 1.10 | Convolution, interpolation, sub-pixel shift |
| astropy | 5.3 | FITS read/write |
| astroquery | 0.4 | Simbad name resolution (History tab) |
| matplotlib | 3.7 | Polar/Cartesian plot (History tab), PNG export |

### Launch

```bash
cd /path/to/SpeckleAnalyzingSuite/v2
python -m speckle_suite
```

> Run from the directory that **contains** the `speckle_suite/` folder, not from inside it.

---

## Code Structure

The code is organised as a Python package (`speckle_suite/`) with 18 files in five layers:

```
speckle_suite/
|
|  # Shared utilities (no Qt dependency)
+-- theme.py              Colour palettes, Qt stylesheet, set_theme()
+-- settings.py           Persistent JSON preferences, working_dir()
+-- widgets.py            ResultCard, read_fits_cube(), colormaps
+-- ser_io.py             Low-level SER parser, timestamp reader
|
|  # Computation backends (numpy/scipy + QThread)
+-- preprocess_backend.py RMS contrast, centroid, registration, PreprocessWorker
+-- drift_backend.py      TLS fit, centroid streaming, DriftWorker, SimbadWorker
+-- analysis_backend.py   Bispectrum, iterative reconstruction, AnalysisWorker, NpzReconWorker
|
|  # Catalogue backends (stdlib + sqlite3, no Qt)
+-- history_catalog.py    INT4/ORB6/WDS summary access, SQLite index, CatalogWorker
+-- history_orbit.py      Kepler solver, orbital ellipse (Thiele-Innes)
|
|  # Tab UIs (PyQt6)
+-- tab_drift.py          Tab 1 -- Drift Alignment
+-- tab_preprocess.py     Tab 2 -- Preprocess
+-- tab_bispectrum.py     Tab 3 -- Bispectrum
+-- tab_astrometry.py     Tab 4 -- Astrometry & Photometry
+-- tab_history.py        Tab 5 -- History
|
|  # Application shell
+-- main_window.py        SpeckleMainWindow, SettingsDialog
+-- __main__.py           Entry point (python -m speckle_suite)
+-- __init__.py           Package metadata
```

---

## Tab 1 -- Drift Alignment

### Purpose

Determine the two quantities needed to convert pixel measurements to sky coordinates: the camera rotation angle relative to celestial north, and the plate scale (arcsec/pixel).

### Observation procedure

1. Point at a bright star near the meridian, declination |delta| < 70 degrees.
2. Switch the motor drive off.
3. Record a SER sequence of at least 30 s using capture software that writes per-frame timestamps (FireCapture or SharpCap).

### Software procedure

1. Click **Browse** and select the drift SER file. If a companion text file (FireCapture/SharpCap) is present, the declination is filled automatically.
2. Enter the target name and click **Resolve** to query Simbad.
3. Click **Run Drift Analysis**. The program streams frames one at a time, computes a centroid per frame, then fits a drift line by Total Least Squares (TLS/SVD) with iterative sigma clipping.
4. Adjust the **Start/Stop trim** and **sigma threshold** sliders -- results update live.
5. Check the result cards: camera angle, plate scale, and their 1-sigma uncertainties.
6. Click **Save Calibration (.json)** to export.

### Batch mode

Select multiple SER files at once. Each is processed sequentially and stored in memory. The navigator lets you revisit each result and fine-tune the sliders independently. The exported JSON contains individual measurements as well as the mean and standard deviation of angle and scale across the batch.

---

## Tab 2 -- Preprocess

### Purpose

Select the sharpest frames from an acquisition sequence, register them to a common centroid, crop to a square ROI, and write the result as a FITS cube ready for the Bispectrum tab.

### Procedure

1. Click **Browse** and select one or more SER or FITS files.
2. Set the **Reject worst** slider: percentage of frames to discard (lowest RMS contrast). 5-15% is recommended.
3. Choose the **ROI size**. **32x32 pixels is strongly recommended** for bispectral analysis; 64x64 is feasible but significantly slower.
4. Set the output directory and click **Run Preprocessing**.
5. Inspect the best-frame preview; use the **Frame** slider to browse all frames. Rejected frames are marked with a cross symbol.

The output FITS file carries the same stem as the source file. Its header records the frame count, quality threshold, ROI size, and maximum centroid shift.

**Quality metric.** The normalised RMS contrast Q = sigma(I)/mu(I) is computed for each frame. A high value indicates a well-resolved speckle pattern. Frames below the specified percentile are rejected.

---

## Tab 3 -- Bispectrum

### Purpose

From a preprocessed FITS cube (or a previously saved bispectrum `.npz`), accumulate the bispectrum and reconstruct a diffraction-limited image. The bispectrum is saved as a `.npz` file for use in Tab 4.

### Algorithm

1. **Bispectrum accumulation:** B(u,v) = mean[ F(u) F(v) F*(u+v) ] averaged over all frames.
2. **Iterative phase retrieval (Knox-Thompson):** the closure relation phi(u+v) ~ phi(u) + phi(v) - angle(B(u,v)) is applied iteratively, weighted by the bispectrum amplitude.
3. **Image reconstruction:** img = IFFT[ A(k) * exp(i*phi(k)) ] with soft-edge apodisation.

### Procedure

1. Click **Browse** and select one or more FITS cubes or `.npz` files. For `.npz` input, reconstruction starts immediately.
2. *(Optional)* Load a reference bispectrum (`.npz`) in the **Reference Deconvolution** section to apply Wiener deconvolution. Click **Clear** to remove it.
3. Set **Kmax**, **dKmax**, and **Iterations** (defaults 60, 9, 30 work well for a 32x32 ROI).
4. Click **Run Analysis**.
5. Click **Save Bispectrum (.npz)** to archive the result for Tab 4.

When multiple files are loaded, the navigator lets you move between them. Clicking **Run** again re-runs reconstruction on all files with the current parameters without re-browsing.

---

## Tab 4 -- Astrometry & Photometry

### Purpose

Load a reconstructed image (`.npz` from Tab 3), place markers on the two components, apply astrometric calibration, optionally measure delta-magnitude by aperture photometry, and export results.

### Procedure

1. Click **Browse** and select one or more `.npz` files. Reconstruction runs automatically.
2. Use the **Place Primary** / **Place Secondary** radio buttons below the image to select which marker to place, then click on the image. Click **Clear All** to start over.
3. Load the calibration JSON from Tab 1 via the **Browse** button in *Astrometric Calibration*. Sky coordinates (rho in arcsec, theta in degrees) appear immediately.
4. *(Optional)* Set the three aperture radii (inner star, sky inner, sky outer -- must satisfy D1 < D2 < D3) and click **Measure delta-M** to compute the magnitude difference. Tick **Show apertures** to overlay the circles on the image.
5. Export: **Save Image (.png)**, **Save Result (.json)**, **CSV Log**, or **WDS Report**.

### Multi-file navigator

When multiple `.npz` files are loaded, the navigator moves between them. Marker positions and measurements are remembered per file. The **Re-run reconstruction** button re-processes the current file with different Kmax/iterations without re-browsing.

### Uncertainty budget

Two independent contributions are computed and combined in quadrature:

- **sigma_cal**: propagated from sigma_scale and sigma_angle in the calibration file.
- **sigma_meas**: standard deviation of rho and theta across all `.npz` files where both markers have been placed (requires N >= 2).

Total: sigma_tot = sqrt(sigma_cal^2 + sigma_meas^2).

---

## Tab 5 -- History

### Purpose

Compare your measurements against published catalogue data and known orbital solutions.

> **Important:** This comparison tool must not be used to bias your determination of rho and theta. Always finalise your measurement before consulting this tab.

### Catalogues

Four data sources are downloaded from the USNO/GSU on first use:

- **INT4**: all published interferometric measurements (speckle, CCD, micrometry, Hipparcos). Indexed in a local SQLite database for fast lookup by WDS key. This is the source used for the history plot.
- **WDS summary**: pair properties -- magnitudes, spectral type, discoverer, observation span -- parsed with fixed-width column slicing following the [astrolabium WDSParser](https://github.com/TheWand3rer/astrolabium) column definitions.
- **ORB6 ephemeris**: predicted theta and rho per calendar year.
- **ORB6 orbital elements**: full Keplerian parameters (P, a, i, Omega, T, e, omega) with published uncertainties, parsed using fixed-width column slicing following the [astrolabium Orb6Parser](https://github.com/TheWand3rer/astrolabium) column definitions.

Click **Download / Update Catalogs** on first use.

### Star lookup

The *Star Lookup* field accepts three formats:

1. **WDS key directly**: e.g. `12417+0127`
2. **Discoverer designation**: e.g. `STF 1883`, `HU 628`, `STT159AB`
3. **Common name via Simbad**: e.g. `gamma Vir`, `eta Cas` (requires `astroquery`)

### Plot modes

- **Polar**: north up, east clockwise, radius = rho in arcsec (WDS convention).
- **Cartesian**: delta-RA (east right) vs delta-Dec (north up). A zoomed inset is added automatically around your measurements when error bars are available.

---

## Output Formats

| Format | Content |
|--------|---------|
| `.json` (result) | rho and theta in pixels and sky coords, sigma_cal, sigma_meas, sigma_tot, delta-mag, calibration parameters, NPZ source |
| `.csv` (log) | One row per measurement; columns include date, target, theta_sky, sigma_theta, rho, sigma_rho, delta_mag, scale, angle, cal file |
| WDS report (`.txt`) | Formatted text block for JDSO submission or WDS data centre |
| `.png` (image) | Reconstructed image with markers and aperture circles at current display levels; can be used as an overlay in the History tab |
| `.npz` (bispectrum) | Archived averaged bispectrum; reloadable in Tab 3 to skip accumulation or use as Wiener reference |
| `.json` (calibration) | Camera angle, plate scale, 1-sigma uncertainties; batch mode adds mean and std across all SER files |

---

## Practical Tips

### Acquisition
- Exposure time: 5-40 ms to freeze the wavefront.
- Number of frames: minimum 500; 1000-3000 preferred for the bispectrum.
- Use capture software that records per-frame timestamps (FireCapture, SharpCap).
- For drift: star near the meridian, |delta| < 70 degrees, sequence of at least 30 s.

### Preprocessing
- Start with ROI = 32x32 and reject = 10%.
- Check that the maximum centroid shift stays below 5 px; larger values suggest poor seeing or telescope vibration.

### Bispectrum
- Kmax approximately 0.8-1.0 x N/2 where N is the ROI size. For 32x32: Kmax = 12-14.
- dKmax = 9 is a good default; increasing to 12-15 improves phase coverage at the cost of longer computation.
- If the reconstruction shows a diffuse halo rather than two clean spots, increase the number of iterations or load a reference star bispectrum for Wiener deconvolution.

### Astrometry
- For a robust uncertainty estimate: observe the same target in 5 or more independent SER files and place markers on each in the multi-file navigator.
- The position angle theta is measured east of north (WDS convention). The 180-degree ambiguity of the autocorrelogram is resolved by the bispectrum reconstruction -- ensure the Primary marker is on the brighter component.
- Report sigma_cal and sigma_meas separately so readers can assess systematic and random contributions independently.

### Aperture photometry
- The three aperture radii must satisfy: inner (star) < sky inner < sky outer. The constraint is enforced automatically and live.
- Work on a well-reconstructed image; increase iterations in Tab 3 if the PSF cores are not clearly separated before measuring delta-M.

---

## Repository

[https://github.com/sdevissc/SpeckleAnalyzingSuite](https://github.com/sdevissc/SpeckleAnalyzingSuite)

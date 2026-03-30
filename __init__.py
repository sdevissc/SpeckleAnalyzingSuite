"""
speckle_suite
=============
Double Star Speckle Astrometry Suite — Linux Edition.

Package structure
-----------------
theme.py              Colour tokens and Qt stylesheet
settings.py           Persistent JSON preferences
widgets.py            Shared Qt widgets and helpers (ResultCard, colormaps)
ser_io.py             SER file parser (pure stdlib + numpy)

preprocess_backend.py Frame quality scoring, registration, PreprocessWorker
drift_backend.py      TLS drift fit, centroid streaming, DriftWorker, SimbadWorker
analysis_backend.py   Bispectrum accumulation, phase retrieval, AnalysisWorker

history_catalog.py    INT4 / ORB6 catalog access and SQLite index
history_orbit.py      Kepler solver, orbital ellipse computation

tab_drift.py          Tab 1 — Drift Alignment UI (DriftTab)
tab_preprocess.py     Tab 2 — Preprocess UI (PreprocessTab)
tab_analysis.py       Tab 3 — Analysis UI (AnalysisTab)
tab_history.py        Tab 4 — History UI (HistoryTab)

main_window.py        SpeckleMainWindow, SettingsDialog
__main__.py           Entry point  (python -m speckle_suite)
"""

__version__ = "2.0.0"
__author__  = "SpeckleAnalyzingSuite contributors"

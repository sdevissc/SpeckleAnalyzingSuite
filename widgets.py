"""
speckle_suite.widgets
=====================
Shared Qt widgets and UI helper functions used by multiple tabs.

Contents
--------
- ResultCard      — small metric display card
- read_fits_cube  — load a 3-D FITS file into a float32 numpy array
- primary_btn_style — accent-coloured QPushButton stylesheet fragment
- COLORMAP_NAMES / get_colormaps — pyqtgraph colormaps for the analysis tab
"""

import numpy as np
import pyqtgraph as pg

from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel
from PyQt6.QtCore import Qt

import speckle_suite.theme as theme


# ── ResultCard ─────────────────────────────────────────────────────────────

class ResultCard(QWidget):
    """Small metric display card: value (large) / unit / label."""

    def __init__(self, label: str, unit: str = "", parent=None):
        super().__init__(parent)
        self._unit = unit
        self.setMinimumWidth(110)
        self._apply_style()

        layout = QVBoxLayout(self)
        layout.setSpacing(2)
        layout.setContentsMargins(10, 8, 10, 8)

        self.value_lbl = QLabel("—")
        self.value_lbl.setObjectName("result_value")
        self.value_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.unit_lbl = QLabel(unit)
        self.unit_lbl.setObjectName("result_label")
        self.unit_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.name_lbl = QLabel(label)
        self.name_lbl.setObjectName("result_label")
        self.name_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)

        layout.addWidget(self.value_lbl)
        layout.addWidget(self.unit_lbl)
        layout.addWidget(self.name_lbl)

    def _apply_style(self) -> None:
        self.setStyleSheet(
            f"QWidget {{ background: {theme.PANEL_BG}; "
            f"border: 1px solid {theme.BORDER_COLOR}; border-radius: 8px; }}"
        )

    def set_value(self, v: str) -> None:
        self.value_lbl.setText(v)

    def refresh_style(self) -> None:
        """Re-apply inline stylesheet after a theme switch."""
        self._apply_style()


# ── FITS reader ────────────────────────────────────────────────────────────

def read_fits_cube(filepath: str) -> tuple[np.ndarray, dict]:
    """
    Read a 3-D FITS cube.

    Returns
    -------
    cube : float32 ndarray  shape (N, H, W)
    hdr  : dict of FITS header key/value pairs
    """
    from astropy.io import fits as _fits
    with _fits.open(filepath) as hdul:
        for hdu in hdul:
            if hdu.data is not None and hdu.data.ndim == 3:
                return hdu.data.astype(np.float32), dict(hdu.header)
    raise ValueError("No 3-D data array found in FITS file.")


# ── Button style helper ────────────────────────────────────────────────────

def primary_btn_style() -> str:
    """Return an accent-coloured QPushButton stylesheet fragment."""
    return (
        f"QPushButton {{"
        f"  background-color: {theme.ACCENT}; color: {theme.DARK_BG};"
        f"  border: none; border-radius: 4px;"
        f"  font-weight: bold; padding: 6px 14px;"
        f"}}"
        f"QPushButton:hover {{ background-color: {theme.ACCENT2}; }}"
        f"QPushButton:disabled {{"
        f"  background-color: {theme.BORDER_COLOR}; color: {theme.TEXT_MUTED};"
        f"}}"
    )


# ── Colormaps ──────────────────────────────────────────────────────────────

COLORMAP_NAMES: list[str] = ["Grey", "Inverted", "Hot", "Rainbow", "Viridis"]

_COLORMAPS: dict | None = None


def get_colormaps() -> dict:
    """Return (and lazily build) the dict of named pyqtgraph ColorMap objects."""
    global _COLORMAPS
    if _COLORMAPS is not None:
        return _COLORMAPS

    def _cm(stops, colors):
        return pg.ColorMap(
            np.array(stops, dtype=float),
            np.array(colors, dtype=np.uint8),
        )

    _COLORMAPS = {
        "Grey":     _cm([0, 1], [[0,0,0,255],[255,255,255,255]]),
        "Inverted": _cm([0, 1], [[255,255,255,255],[0,0,0,255]]),
        "Hot":      _cm([0, 0.33, 0.66, 1],
                        [[0,0,0,255],[200,0,0,255],[255,180,0,255],[255,255,255,255]]),
        "Rainbow":  _cm([0, 0.25, 0.5, 0.75, 1],
                        [[0,0,180,255],[0,200,255,255],[0,220,0,255],
                         [255,200,0,255],[220,0,0,255]]),
        "Viridis":  _cm([0, 0.25, 0.5, 0.75, 1],
                        [[68,1,84,255],[59,82,139,255],[33,145,140,255],
                         [94,201,98,255],[253,231,37,255]]),
    }
    return _COLORMAPS

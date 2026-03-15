"""
Double Star Speckle Astrometry Suite
=====================================
Linux Edition — Merged Application

Tabs
----
  1. Calibration — drift analysis: camera angle & pixel scale from SER drift file.
  2. Preprocess  — reads SER / FITS sequence, scores frames by RMS contrast,
                   registers & crops to ROI, writes a FITS cube.
  3. Analysis    — reads preprocessed FITS cube, accumulates bispectrum,
                   iterative phase retrieval, reconstructs diffraction-limited
                   image, measures ρ / θ astrometry with calibration.

Dependencies:
    pip install PyQt6 pyqtgraph numpy scipy astropy
"""

import sys
import json as _json
import csv
import struct
import math
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Optional

import numpy as np

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QGridLayout, QLabel, QPushButton, QLineEdit, QGroupBox,
    QFileDialog, QProgressBar, QSizePolicy, QComboBox,
    QMenuBar, QMenu, QTextEdit, QSplitter, QCheckBox, QSpinBox,
    QRadioButton, QButtonGroup, QDoubleSpinBox, QFrame, QSlider,
    QScrollArea, QTabWidget, QDialog, QFormLayout,
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt6.QtGui import QFont

import pyqtgraph as pg
from scipy.ndimage import shift as nd_shift


# ═══════════════════════════════════════════════════════════════════════════
#  Theme system  (shared)
# ═══════════════════════════════════════════════════════════════════════════

THEMES = {
    'dark': {
        'DARK_BG':      '#0d1117',
        'PANEL_BG':     '#161b22',
        'BORDER_COLOR': '#30363d',
        'ACCENT':       '#58a6ff',
        'ACCENT2':      '#3fb950',
        'TEXT_PRIMARY': '#e6edf3',
        'TEXT_MUTED':   '#8b949e',
        'WARNING':      '#d29922',
        'DANGER':       '#f85149',
    },
    'red': {
        'DARK_BG':      '#0e0500',
        'PANEL_BG':     '#1a0a00',
        'BORDER_COLOR': '#4a1500',
        'ACCENT':       '#ff6b35',
        'ACCENT2':      '#cc3300',
        'TEXT_PRIMARY': '#ffcba4',
        'TEXT_MUTED':   '#994422',
        'WARNING':      '#ff9900',
        'DANGER':       '#ff3311',
    },
    'light': {
        'DARK_BG':      '#e8e8e8',
        'PANEL_BG':     '#f3f3f3',
        'BORDER_COLOR': '#b0b8c4',
        'ACCENT':       '#0055aa',
        'ACCENT2':      '#006b2b',
        'TEXT_PRIMARY': '#1a1a1a',
        'TEXT_MUTED':   '#5a6370',
        'WARNING':      '#9a4f00',
        'DANGER':       '#bb0000',
    },
}

_theme = THEMES['dark']

def _refresh_theme_aliases():
    global DARK_BG, PANEL_BG, BORDER_COLOR, ACCENT, ACCENT2
    global TEXT_PRIMARY, TEXT_MUTED, WARNING, DANGER
    DARK_BG      = _theme['DARK_BG']
    PANEL_BG     = _theme['PANEL_BG']
    BORDER_COLOR = _theme['BORDER_COLOR']
    ACCENT       = _theme['ACCENT']
    ACCENT2      = _theme['ACCENT2']
    TEXT_PRIMARY = _theme['TEXT_PRIMARY']
    TEXT_MUTED   = _theme['TEXT_MUTED']
    WARNING      = _theme['WARNING']
    DANGER       = _theme['DANGER']

_refresh_theme_aliases()


def build_stylesheet(t: dict) -> str:
    return f"""
QMainWindow, QWidget {{
    background-color: {t['DARK_BG']};
    color: {t['TEXT_PRIMARY']};
    font-family: 'JetBrains Mono', 'Fira Code', 'Courier New', monospace;
    font-size: 12px;
}}
QTabWidget::pane {{
    border: 1px solid {t['BORDER_COLOR']};
    border-radius: 4px;
}}
QTabBar::tab {{
    background: {t['PANEL_BG']};
    border: 1px solid {t['BORDER_COLOR']};
    border-bottom: none;
    border-radius: 4px 4px 0 0;
    padding: 6px 24px;
    color: {t['TEXT_MUTED']};
    font-weight: bold;
    letter-spacing: 1px;
}}
QTabBar::tab:selected {{
    background: {t['DARK_BG']};
    color: {t['ACCENT']};
    border-bottom: 2px solid {t['ACCENT']};
}}
QTabBar::tab:hover:!selected {{
    color: {t['TEXT_PRIMARY']};
}}
QGroupBox {{
    border: 1px solid {t['BORDER_COLOR']};
    border-radius: 6px;
    margin-top: 10px;
    padding: 8px;
    font-weight: bold;
    color: {t['ACCENT']};
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    left: 10px;
    padding: 0 4px;
    color: {t['ACCENT']};
    font-size: 11px;
    letter-spacing: 1px;
    text-transform: uppercase;
}}
QLineEdit, QSpinBox, QDoubleSpinBox {{
    background-color: {t['PANEL_BG']};
    border: 1px solid {t['BORDER_COLOR']};
    border-radius: 4px;
    padding: 4px 8px;
    color: {t['TEXT_PRIMARY']};
    selection-background-color: {t['ACCENT']};
}}
QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus {{
    border-color: {t['ACCENT']};
}}
QLineEdit:read-only {{ color: {t['TEXT_MUTED']}; }}
QPushButton {{
    background-color: {t['PANEL_BG']};
    border: 1px solid {t['BORDER_COLOR']};
    border-radius: 4px;
    padding: 6px 14px;
    color: {t['TEXT_PRIMARY']};
    font-weight: bold;
}}
QPushButton:hover {{ border-color: {t['ACCENT']}; color: {t['ACCENT']}; }}
QPushButton:disabled {{ background-color: {t['BORDER_COLOR']}; color: {t['TEXT_MUTED']}; }}
QProgressBar {{
    border: 1px solid {t['BORDER_COLOR']};
    border-radius: 4px;
    background-color: {t['PANEL_BG']};
    text-align: center;
    color: {t['TEXT_PRIMARY']};
    height: 14px;
}}
QProgressBar::chunk {{
    background-color: {t['ACCENT']};
    border-radius: 3px;
}}
QTextEdit {{
    background-color: {t['PANEL_BG']};
    border: 1px solid {t['BORDER_COLOR']};
    border-radius: 4px;
    color: {t['TEXT_MUTED']};
    font-size: 11px;
}}
QComboBox {{
    background-color: {t['PANEL_BG']};
    border: 1px solid {t['BORDER_COLOR']};
    border-radius: 4px;
    padding: 4px 8px;
    color: {t['TEXT_PRIMARY']};
}}
QComboBox:hover {{ border-color: {t['ACCENT']}; }}
QComboBox::drop-down {{ border: none; }}
QComboBox QAbstractItemView {{
    background-color: {t['PANEL_BG']};
    border: 1px solid {t['BORDER_COLOR']};
    color: {t['TEXT_PRIMARY']};
    selection-background-color: {t['ACCENT']};
}}
QMenuBar {{
    background-color: {t['PANEL_BG']};
    color: {t['TEXT_PRIMARY']};
    border-bottom: 1px solid {t['BORDER_COLOR']};
}}
QMenuBar::item:selected {{ background-color: {t['BORDER_COLOR']}; }}
QMenu {{
    background-color: {t['PANEL_BG']};
    border: 1px solid {t['BORDER_COLOR']};
    color: {t['TEXT_PRIMARY']};
}}
QMenu::item:selected {{ background-color: {t['ACCENT']}; color: {t['DARK_BG']}; }}
QSlider::groove:horizontal {{
    height: 4px;
    background: {t['BORDER_COLOR']};
    border-radius: 2px;
}}
QSlider::handle:horizontal {{
    background: {t['ACCENT']};
    border: none;
    width: 14px; height: 14px;
    margin: -5px 0;
    border-radius: 7px;
}}
QSlider::sub-page:horizontal {{
    background: {t['ACCENT']};
    border-radius: 2px;
}}
QFrame#separator {{
    background-color: {t['BORDER_COLOR']};
    max-height: 1px;
}}
QLabel#result_value {{
    color: {t['ACCENT2']};
    font-size: 20px;
    font-weight: bold;
}}
QLabel#result_label {{
    color: {t['TEXT_MUTED']};
    font-size: 10px;
}}
QCheckBox {{ color: {t['TEXT_MUTED']}; spacing: 6px; }}
QCheckBox::indicator {{
    width: 14px; height: 14px;
    border: 1px solid {t['BORDER_COLOR']};
    border-radius: 3px;
    background: {t['PANEL_BG']};
}}
QCheckBox::indicator:checked {{ background: {t['ACCENT']}; border-color: {t['ACCENT']}; }}
"""


# ═══════════════════════════════════════════════════════════════════════════
#  Shared result card widget
# ═══════════════════════════════════════════════════════════════════════════

class ResultCard(QWidget):
    def __init__(self, label: str, unit: str = "", parent=None):
        super().__init__(parent)
        self._unit = unit
        self.setMinimumWidth(110)
        self.setStyleSheet(f"""
            QWidget {{ background: {PANEL_BG}; border: 1px solid {BORDER_COLOR};
                       border-radius: 8px; }}
        """)
        layout = QVBoxLayout(self)
        layout.setSpacing(2)
        layout.setContentsMargins(10, 8, 10, 8)
        self.value_lbl = QLabel("—")
        self.value_lbl.setObjectName("result_value")
        self.value_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.unit_lbl  = QLabel(unit)
        self.unit_lbl.setObjectName("result_label")
        self.unit_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.name_lbl  = QLabel(label)
        self.name_lbl.setObjectName("result_label")
        self.name_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.value_lbl)
        layout.addWidget(self.unit_lbl)
        layout.addWidget(self.name_lbl)

    def set_value(self, v: str):
        self.value_lbl.setText(v)

    def refresh_style(self):
        self.setStyleSheet(f"""
            QWidget {{ background: {PANEL_BG}; border: 1px solid {BORDER_COLOR};
                       border-radius: 8px; }}
        """)


# ═══════════════════════════════════════════════════════════════════════════
#  Shared FITS reader
# ═══════════════════════════════════════════════════════════════════════════

def read_fits_cube(filepath: str):
    """Read a 3-D FITS cube. Returns (cube_float32, header_dict)."""
    from astropy.io import fits as _fits
    with _fits.open(filepath) as hdul:
        for hdu in hdul:
            if hdu.data is not None and hdu.data.ndim == 3:
                cube = hdu.data.astype(np.float32)
                hdr  = dict(hdu.header)
                return cube, hdr
        raise ValueError("No 3-D data array found in FITS file.")


# ═══════════════════════════════════════════════════════════════════════════
#  Primary-button style helper  (shared)
# ═══════════════════════════════════════════════════════════════════════════

def _primary_btn_style() -> str:
    return f"""
    QPushButton {{
        background-color: {ACCENT}; color: {DARK_BG};
        border: none; border-radius: 4px;
        font-weight: bold; padding: 6px 14px;
    }}
    QPushButton:hover {{ background-color: {ACCENT2}; }}
    QPushButton:disabled {{ background-color: {BORDER_COLOR}; color: {TEXT_MUTED}; }}
    """



# ═══════════════════════════════════════════════════════════════════════════
#  ── PREPROCESS BACKEND ──────────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════

# ── SER parser ─────────────────────────────────────────────────────────────

@dataclass
class SERHeader:
    file_id:      str
    lu_id:        int
    color_id:     int
    little_endian:int
    image_width:  int
    image_height: int
    pixel_depth:  int
    frame_count:  int
    observer:     str
    instrument:   str
    telescope:    str
    date_time:    int
    date_time_utc:int


def parse_ser_header(data: bytes) -> SERHeader:
    return SERHeader(
        file_id       = data[0:14].decode('ascii', errors='ignore').rstrip('\x00'),
        lu_id         = struct.unpack_from('<i', data, 14)[0],
        color_id      = struct.unpack_from('<i', data, 18)[0],
        little_endian = struct.unpack_from('<i', data, 22)[0],
        image_width   = struct.unpack_from('<i', data, 26)[0],
        image_height  = struct.unpack_from('<i', data, 30)[0],
        pixel_depth   = struct.unpack_from('<i', data, 34)[0],
        frame_count   = struct.unpack_from('<i', data, 38)[0],
        observer      = data[42:82].decode('ascii',  errors='ignore').rstrip('\x00'),
        instrument    = data[82:122].decode('ascii', errors='ignore').rstrip('\x00'),
        telescope     = data[122:162].decode('ascii',errors='ignore').rstrip('\x00'),
        date_time     = struct.unpack_from('<q', data, 162)[0],
        date_time_utc = struct.unpack_from('<q', data, 170)[0],
    )


COLOR_MONO       = 0
COLOR_BAYER_RGGB = 8
COLOR_BGR        = 100
COLOR_RGB        = 101


def _ser_frame_iter(filepath: str, header: SERHeader):
    """Yield frames from a SER file as float32 arrays."""
    bytes_per_pixel = 2 if header.pixel_depth > 8 else 1
    frame_bytes = header.image_width * header.image_height * bytes_per_pixel
    dtype = np.uint16 if bytes_per_pixel == 2 else np.uint8
    with open(filepath, 'rb') as f:
        f.seek(178)
        for _ in range(header.frame_count):
            raw = f.read(frame_bytes)
            if len(raw) < frame_bytes:
                break
            frame = np.frombuffer(raw, dtype=dtype).reshape(
                header.image_height, header.image_width).astype(np.float32)
            yield frame


# ── Quality metric ─────────────────────────────────────────────────────────

def rms_contrast(frame: np.ndarray) -> float:
    """Normalized RMS contrast = std / mean. Higher = sharper."""
    m = float(frame.mean())
    if m <= 0:
        return 0.0
    return float(frame.std()) / m


# ── Centroid & registration ────────────────────────────────────────────────

def find_centroid(frame: np.ndarray) -> tuple:
    from scipy.ndimage import uniform_filter
    smoothed = uniform_filter(frame.astype(np.float32), size=5)
    peak_rc  = np.unravel_index(np.argmax(smoothed), smoothed.shape)
    pr, pc   = peak_rc
    win = 16
    h, w = frame.shape
    r0 = max(pr - win, 0);  r1 = min(pr + win, h)
    c0 = max(pc - win, 0);  c1 = min(pc + win, w)
    patch = frame[r0:r1, c0:c1].astype(np.float64)
    patch = np.clip(patch - patch.min(), 0, None)
    total = patch.sum()
    if total <= 0:
        return float(pr), float(pc)
    rows_idx = np.arange(r0, r1)
    cols_idx = np.arange(c0, c1)
    cr = float((patch.sum(axis=1) @ rows_idx) / total)
    cc = float((patch.sum(axis=0) @ cols_idx) / total)
    return cr, cc


def register_and_crop(frame: np.ndarray,
                      centroid_rc: tuple,
                      roi_size: int) -> np.ndarray:
    h, w   = frame.shape
    cr, cc = centroid_rc
    dy = cr - h / 2.0
    dx = cc - w / 2.0
    shifted = nd_shift(frame.astype(np.float32),
                       shift=(-dy, -dx), order=3, mode='reflect')
    half = roi_size // 2
    mr, mc = h // 2, w // 2
    return shifted[mr - half : mr + half,
                   mc - half : mc + half].copy()


# ── Preprocess worker ──────────────────────────────────────────────────────

class PreprocessWorker(QThread):
    progress = pyqtSignal(int)
    status   = pyqtSignal(str)
    preview  = pyqtSignal(object)
    quality  = pyqtSignal(object)
    finished = pyqtSignal(object)
    error    = pyqtSignal(str)

    def __init__(self, filepath: str, file_type: str,
                 best_pct: float, roi_size: int, output_path: str):
        super().__init__()
        self.filepath    = filepath
        self.file_type   = file_type
        self.best_pct    = best_pct
        self.roi_size    = roi_size
        self.output_path = output_path
        self._stop       = False

    def stop(self):
        self._stop = True

    def run(self):
        try:
            self._process()
        except Exception as e:
            import traceback
            self.error.emit(f"{e}\n{traceback.format_exc()}")

    def _process(self):
        self.status.emit("Reading frames and computing quality scores…")
        self.progress.emit(2)

        if self.file_type == 'ser':
            with open(self.filepath, 'rb') as f:
                header_bytes = f.read(178)
            header     = parse_ser_header(header_bytes)
            n_total    = header.frame_count
            fh, fw     = header.image_height, header.image_width
            frame_iter = _ser_frame_iter(self.filepath, header)
        else:
            cube, _    = read_fits_cube(self.filepath)
            n_total    = cube.shape[0]
            fh, fw     = cube.shape[1], cube.shape[2]
            frame_iter = iter(cube)

        scores = np.zeros(n_total, dtype=np.float32)
        frames = []
        for i, frame in enumerate(frame_iter):
            if self._stop:
                self.status.emit("Stopped."); return
            scores[i] = rms_contrast(frame)
            frames.append(frame)
            if i % 50 == 0 or i == n_total - 1:
                self.progress.emit(5 + int(35 * i / max(n_total - 1, 1)))
                self.status.emit(
                    f"Scoring… {i+1} / {n_total}  "
                    f"(best so far: {scores[:i+1].max():.4f})")

        self.quality.emit(scores.copy())
        self.status.emit(
            f"Scored {n_total} frames  —  "
            f"min {scores.min():.4f}  max {scores.max():.4f}  "
            f"mean {scores.mean():.4f}")
        self.progress.emit(40)

        threshold  = np.percentile(scores, 100.0 - self.best_pct)
        sel_mask   = scores >= threshold
        sel_idx    = np.where(sel_mask)[0]
        n_selected = len(sel_idx)
        self.status.emit(
            f"Keeping {n_selected} / {n_total} frames  "
            f"({self.best_pct:.0f}%,  Q ≥ {threshold:.4f})")
        self.progress.emit(43)

        max_roi = min(fw, fh)
        if self.roi_size > max_roi:
            old = self.roi_size
            self.roi_size = int(2 ** np.floor(np.log2(max_roi)))
            self.status.emit(
                f"WARNING: frame {fw}×{fh} — ROI clamped "
                f"{old} → {self.roi_size} px")

        all_crops = []
        shifts_px = []
        sel_set   = set(sel_idx.tolist())

        best_idx      = int(np.argmax(scores))
        best_centroid = find_centroid(frames[best_idx])
        best_crop     = register_and_crop(frames[best_idx],
                                          best_centroid, self.roi_size)
        self.preview.emit(best_crop.copy())
        self.status.emit(
            f"Best frame #{best_idx}  —  centroid "
            f"({best_centroid[0]:.1f}, {best_centroid[1]:.1f})  "
            f"ROI {self.roi_size}×{self.roi_size} px")
        self.progress.emit(46)

        output_cube = np.zeros((n_selected, self.roi_size, self.roi_size),
                               dtype=np.float32)
        out_i = 0
        for src_i, frame in enumerate(frames):
            if self._stop:
                self.status.emit("Stopped."); return
            cr, cc    = find_centroid(frame)
            shift_mag = np.hypot(cr - fh / 2.0, cc - fw / 2.0)
            shifts_px.append(shift_mag)
            crop = register_and_crop(frame, (cr, cc), self.roi_size)
            all_crops.append(crop)
            if src_i in sel_set:
                output_cube[out_i] = crop
                out_i += 1
            if src_i % 50 == 0 or src_i == n_total - 1:
                pct = 46 + int(48 * src_i / max(n_total - 1, 1))
                self.progress.emit(pct)
                self.status.emit(
                    f"Registering… {src_i+1} / {n_total}  "
                    f"(centroid: {cr:.1f}, {cc:.1f}  "
                    f"shift: {shift_mag:.1f} px)")

        max_shift = float(np.max(shifts_px)) if shifts_px else 0.0
        self.progress.emit(96)

        self.status.emit("Writing FITS cube…")
        from astropy.io import fits as _fits
        hdu = _fits.PrimaryHDU(output_cube)
        hdu.header['NFRAMES'] = (n_selected,        'Frames in cube')
        hdu.header['NTOTAL']  = (n_total,            'Total frames in source')
        hdu.header['BESTPCT'] = (self.best_pct,      'Frames kept [%]')
        hdu.header['QTHRESH'] = (float(threshold),   'Quality score threshold')
        hdu.header['ROISIZE'] = (self.roi_size,       'ROI size [px]')
        hdu.header['FRMH']    = (fh,                  'Original frame height [px]')
        hdu.header['FRMW']    = (fw,                  'Original frame width [px]')
        hdu.header['MAXSHFT'] = (max_shift,           'Max centroid shift [px]')
        hdu.header['SRCFILE'] = (Path(self.filepath).name, 'Source file')
        _fits.HDUList([hdu]).writeto(self.output_path, overwrite=True)

        self.progress.emit(100)
        self.status.emit(
            f"Done — {n_selected} frames → {Path(self.output_path).name}")
        self.finished.emit({
            'n_total':     n_total,
            'n_selected':  n_selected,
            'best_pct':    self.best_pct,
            'roi_size':    self.roi_size,
            'max_shift':   max_shift,
            'threshold':   float(threshold),
            'scores':      scores,
            'sel_mask':    sel_mask,
            'all_crops':   all_crops,
            'output_path': self.output_path,
        })


# ═══════════════════════════════════════════════════════════════════════════
#  ── DRIFT CALIBRATION BACKEND ──────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════

def read_ser_header_and_timestamps(filepath: str
    ) -> tuple[SERHeader, Optional[np.ndarray]]:
    """
    Read only the SER header (178 bytes) and the timestamp trailer
    (8 bytes × frame_count at the end of the file).

    Returns (header, timestamps_sec) where timestamps_sec is a float64
    array of elapsed seconds from the first frame, or None if the trailer
    is absent or invalid.

    This is intentionally a lightweight pre-flight call — no frame data
    is read into memory.
    """
    with open(filepath, 'rb') as f:
        header = parse_ser_header(f.read(178))

        bytes_per_pixel = 2 if header.pixel_depth > 8 else 1
        frame_size      = header.image_width * header.image_height * bytes_per_pixel
        trailer_offset  = 178 + header.frame_count * frame_size

        timestamps_sec = None
        try:
            f.seek(trailer_offset)
            raw_ts = f.read(header.frame_count * 8)
            if len(raw_ts) == header.frame_count * 8:
                ts = np.frombuffer(raw_ts, dtype=np.int64).copy()
                if np.all(ts > 0) and np.all(np.diff(ts) > 0):
                    # Windows FILETIME: 100-ns ticks → seconds
                    timestamps_sec = (ts - ts[0]).astype(np.float64) * 1e-7
        except Exception:
            pass

    return header, timestamps_sec


def stream_ser_centroids(filepath: str,
                         header: SERHeader,
                         timestamps_sec: np.ndarray,
                         t_start: float,
                         t_stop: float,
                         progress_cb=None,
                         stop_flag=None
                         ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Stream a SER file frame-by-frame, computing centroids on the fly.
    Never holds more than one frame in memory.

    Parameters
    ----------
    filepath       : path to the .ser file
    header         : already-parsed SERHeader
    timestamps_sec : (N,) elapsed-seconds array from read_ser_header_and_timestamps
    t_start        : skip frames before this time  [s]
    t_stop         : skip frames after this time   [s]
    progress_cb    : optional callable(int 0–100)
    stop_flag      : optional callable() → bool; returns True to abort

    Returns
    -------
    cx, cy   : centroid positions [px]
    ct       : centroid timestamps [s]
    """
    bytes_per_pixel = 2 if header.pixel_depth > 8 else 1
    frame_size      = header.image_width * header.image_height * bytes_per_pixel
    n_total         = header.frame_count
    is_colour       = header.color_id in (COLOR_BGR, COLOR_RGB)

    centroids_x, centroids_y, centroid_times = [], [], []

    with open(filepath, 'rb') as f:
        f.seek(178)   # skip header, go to first frame

        for i in range(n_total):
            # Check stop flag (allows worker thread to be cancelled)
            if stop_flag and stop_flag():
                break

            t = timestamps_sec[i]

            # Skip frames outside the time window without decoding
            if t < t_start or t > t_stop:
                f.seek(frame_size, 1)   # seek relative to current pos
                if progress_cb and i % 100 == 0:
                    progress_cb(int(100 * i / n_total))
                continue

            raw = f.read(frame_size)
            if len(raw) < frame_size:
                break   # truncated file

            # Decode to native dtype — do NOT convert full frame to float32.
            # compute_centroid will convert only the small ROI it needs.
            dtype = np.uint16 if bytes_per_pixel == 2 else np.uint8
            arr   = np.frombuffer(raw, dtype=dtype).reshape(
                header.image_height, header.image_width)

            if is_colour:
                # Colour: must average channels → float32 unavoidable here,
                # but still only done for the in-window frames
                arr = arr.reshape(
                    header.image_height, header.image_width // 3, 3
                ).mean(axis=2).astype(np.float32)

            # Centroid and discard frame immediately
            c = compute_centroid(arr)
            if c is not None:
                centroids_x.append(c[0])
                centroids_y.append(c[1])
                centroid_times.append(t)

            if progress_cb and i % 50 == 0:
                progress_cb(int(100 * i / n_total))

    return (np.array(centroids_x),
            np.array(centroids_y),
            np.array(centroid_times))


# ─────────────────────────────────────────────
#  Centroid Computation
# ─────────────────────────────────────────────

def compute_centroid(frame: np.ndarray,
                     roi_size: int = 64) -> Optional[tuple[float, float]]:
    """
    Find the brightest star in a frame and return its sub-pixel centroid (x, y).

    Performance notes
    -----------------
    - `frame` may be uint8 or uint16 — we do NOT convert the full frame to
      float32.  argmax works on any numeric dtype.
    - Background is estimated as the minimum of the 64×64 ROI around the peak,
      not the full-frame median.  This avoids an O(N log N) sort over millions
      of pixels and is equally valid: within a small ROI the background is
      uniform, and its minimum is a robust sky estimate.
    - Float32 conversion is deferred to the ROI only (~4096 values vs millions).
    """
    # Find peak in native dtype — no conversion needed
    peak_idx = np.unravel_index(np.argmax(frame), frame.shape)
    py, px   = peak_idx

    # Extract ROI bounds
    half = roi_size // 2
    y0 = max(0, py - half);  y1 = min(frame.shape[0], py + half)
    x0 = max(0, px - half);  x1 = min(frame.shape[1], px + half)

    # Convert only the ROI to float32
    roi = frame[y0:y1, x0:x1].astype(np.float32)

    # Background = ROI minimum (robust, O(roi_size²) instead of O(N log N))
    bg  = roi.min()
    roi = roi - bg

    if roi.sum() == 0:
        return None

    # Centre of mass within the ROI
    yy, xx = np.mgrid[y0:y1, x0:x1]
    cx = float((xx * roi).sum() / roi.sum())
    cy = float((yy * roi).sum() / roi.sum())
    return cx, cy


# ─────────────────────────────────────────────
#  Drift Analysis Worker Thread
# ─────────────────────────────────────────────

@dataclass
class DriftResult:
    camera_angle_deg: float
    pixel_scale_arcsec: float
    seeing_indicator_arcsec: float
    centroids_x: np.ndarray
    centroids_y: np.ndarray
    mask: np.ndarray
    n_frames_used: int
    n_frames_rejected: int


# ─────────────────────────────────────────────
#  Companion text file parser
# ─────────────────────────────────────────────

def _parse_declination_from_txt(ser_path: str) -> Optional[float]:
    """
    Look for a companion text file alongside the SER file and try to extract
    the target declination.  Returns the declination in decimal degrees, or
    None if not found / not parseable.

    Handles the formats produced by the most common capture packages:

    FireCapture (.txt):
        Dec=+12°34'56.7"
        Dec=+12.5822°
        Declination=+12°34'56.7"

    SharpCap (.txt / .CameraSettings.txt):
        Dec (J2000)=+12:34:56.7
        Declination=+12:34:56.7

    Genika (.txt):
        DEC: +12 34 56.7
        DEC: +12.5822

    Generic key=value or key: value with any of:
        dec, declination, de, δ
    followed by a value in any of:
        ±DD°MM'SS.s"   ±DD:MM:SS.s   ±DD MM SS.s   ±DD.dddd
    """
    import re

    ser_p = Path(ser_path)

    # Candidate companion files — same stem, various extensions
    candidates = [
        ser_p.with_suffix('.txt'),
        ser_p.with_suffix('.TXT'),
        ser_p.parent / (ser_p.stem + '.CameraSettings.txt'),
        ser_p.parent / (ser_p.stem + '_info.txt'),
        ser_p.parent / (ser_p.stem + '.log'),
    ]

    txt_path = next((p for p in candidates if p.exists()), None)
    if txt_path is None:
        return None

    try:
        text = txt_path.read_text(encoding='utf-8', errors='replace')
    except OSError:
        return None

    # Key pattern: dec / declination / de / δ  (case-insensitive)
    key_pat = r'(?:declination|decl?|de|δ)'

    # Value patterns for the coordinate part
    # 1.  ±DD°MM′SS.s″   or  ±DD°MM'SS.s"
    dms_deg  = r'([+\-]?\d{1,3})[°d](\d{1,2})[\'′m](\d{1,2}(?:\.\d+)?)[\"″s]?'
    # 2.  ±DD:MM:SS.s
    dms_col  = r'([+\-]?\d{1,3}):(\d{2}):(\d{2}(?:\.\d+)?)'
    # 3.  ±DD MM SS.s   (space-separated)
    dms_spc  = r'([+\-]?\d{1,3})\s+(\d{1,2})\s+(\d{1,2}(?:\.\d+)?)'
    # 4.  ±DD.dddd°   or just ±DD.dddd
    dec_dec  = r'([+\-]?\d{1,3}\.\d+)°?'
    # 5.  ±DDd  (integer degrees, less precise but valid)
    dec_int  = r'([+\-]?\d{1,3})°'

    def dms_to_deg(d, m, s):
        sign = -1 if str(d).strip().startswith('-') else 1
        return sign * (abs(float(d)) + float(m) / 60.0 + float(s) / 3600.0)

    sep = r'\s*[=:]\s*'   # separator between key and value

    patterns = [
        (re.compile(key_pat + sep + dms_deg,  re.IGNORECASE), 'dms'),
        (re.compile(key_pat + sep + dms_col,  re.IGNORECASE), 'dms'),
        (re.compile(key_pat + sep + dms_spc,  re.IGNORECASE), 'dms'),
        (re.compile(key_pat + sep + dec_dec,  re.IGNORECASE), 'dec'),
        (re.compile(key_pat + sep + dec_int,  re.IGNORECASE), 'dec'),
    ]

    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        for pat, kind in patterns:
            m = pat.search(line)
            if m:
                try:
                    if kind == 'dms':
                        return dms_to_deg(m.group(1), m.group(2), m.group(3))
                    else:
                        return float(m.group(1))
                except (ValueError, IndexError):
                    continue

    return None   # found the file but no parseable declination


class DriftWorker(QThread):
    """Background worker: streams SER file, computes centroids, emits result."""

    progress = pyqtSignal(int)
    status   = pyqtSignal(str)
    finished = pyqtSignal(object)
    error    = pyqtSignal(str)

    def __init__(self, filepath: str,
                 declination_deg: float):
        super().__init__()
        self.filepath         = filepath
        self.declination_deg  = declination_deg
        self._stop            = False

    def stop(self):
        self._stop = True

    def run(self):
        try:
            # ── 1. Read header + timestamps only (no frame data) ──────
            self.status.emit("Reading SER header and timestamps…")
            self.progress.emit(2)

            header, timestamps_sec = read_ser_header_and_timestamps(
                self.filepath)

            self.status.emit(
                f"{header.frame_count} frames  "
                f"({header.image_width}×{header.image_height}, "
                f"{header.pixel_depth}-bit)"
            )
            self.progress.emit(5)

            # ── 2. Validate timestamps — hard stop if absent ───────────
            if timestamps_sec is None:
                self.error.emit(
                    "No valid per-frame timestamps found in this SER file.\n\n"
                    "Timestamps are required to compute the frame interval and "
                    "therefore the pixel scale. Without them the result would "
                    "be silently wrong.\n\n"
                    "Please use capture software that records per-frame "
                    "timestamps (e.g. FireCapture, SharpCap) and re-acquire "
                    "the drift sequence."
                )
                return

            # ── 3. Derive fps from timestamps ──────────────────────────
            dt          = np.diff(timestamps_sec)
            dt_positive = dt[dt > 0]
            if len(dt_positive) < 3:
                self.error.emit(
                    f"Only {len(dt_positive)} valid timestamp intervals found. "
                    f"File may be too short or timestamps unreliable. Aborting."
                )
                return

            median_dt  = float(np.median(dt_positive))
            fps_actual = 1.0 / median_dt
            fps_std    = float(np.std(dt_positive))
            t_total    = float(timestamps_sec[-1] - timestamps_sec[0])

            self.status.emit(
                f"Frame interval: {median_dt*1000:.3f} ms  "
                f"({fps_actual:.2f} fps)  total: {t_total:.1f} s"
            )
            self.progress.emit(8)

            # ── 4. Stream all frames, compute centroids on the fly ─────
            self.status.emit(
                f"Streaming {header.frame_count} frames  ({t_total:.1f} s)…"
            )
            self.progress.emit(10)

            def prog(p):
                self.progress.emit(10 + int(88 * p / 100))

            cx, cy, ct = stream_ser_centroids(
                filepath       = self.filepath,
                header         = header,
                timestamps_sec = timestamps_sec,
                t_start        = float(timestamps_sec[0]),
                t_stop         = float(timestamps_sec[-1]),
                progress_cb    = prog,
                stop_flag      = lambda: self._stop,
            )

            if self._stop:
                self.status.emit("Stopped.")
                return

            if len(cx) < 10:
                self.error.emit(
                    f"Only {len(cx)} valid centroids found — too few to fit. "
                    f"Check that the star is visible and in focus."
                )
                return

            self.progress.emit(100)
            self.status.emit(
                f"Done — {len(cx)} centroids over {ct[-1]-ct[0]:.1f} s.  "
                f"Adjust σ slider then save."
            )
            self.finished.emit({
                'centroids_x':    cx,
                'centroids_y':    cy,
                'times_sec':      ct,
                'fps':            fps_actual,
                'median_dt_ms':   median_dt * 1000.0,
                'fps_std_ms':     fps_std   * 1000.0,
                'declination_deg': self.declination_deg,
            })

        except Exception as e:
            import traceback
            self.error.emit(f"{e}\n{traceback.format_exc()}")


# ─────────────────────────────────────────────
#  Interactive drift fitting (called live from UI)
# ─────────────────────────────────────────────

def _tls_fit(cx: np.ndarray, cy: np.ndarray) -> dict:
    """
    Total Least Squares (TLS) line fit via SVD on 2D centroid cloud.

    Unlike OLS (which minimises vertical residuals), TLS minimises the
    perpendicular (orthogonal) distance from each point to the line —
    the correct model when both X and Y carry measurement noise.

    Returns:
        direction   : unit vector (dx, dy) along the drift direction
        centroid    : mean point (cx_mean, cy_mean) — line passes through this
        perp_resid  : signed perpendicular residuals for each point  [px]
        para_resid  : signed parallel residuals (along drift)        [px]
        fitted_x/y  : projected points onto the line (all input points)
    """
    cx_mean = cx.mean()
    cy_mean = cy.mean()

    # Centre the data
    X = np.column_stack([cx - cx_mean, cy - cy_mean])   # (n, 2)

    # SVD: largest singular vector = drift direction
    _, _, Vt = np.linalg.svd(X, full_matrices=False)
    direction = Vt[0]   # unit vector (dx, dy) — dominant variance direction

    # Decompose residuals into parallel and perpendicular components
    # para_resid[i] = projection of point i onto the drift direction
    # perp_resid[i] = distance from point i to the line (signed)
    para_resid = X @ direction                          # (n,)
    perp_unit  = np.array([-direction[1], direction[0]])  # 90° rotation
    perp_resid = X @ perp_unit                          # (n,)

    # Reconstruct fitted positions (projections back onto the line)
    fitted_x = cx_mean + para_resid * direction[0]
    fitted_y = cy_mean + para_resid * direction[1]

    return {
        'direction':  direction,        # (dx, dy) unit vector
        'centroid':   (cx_mean, cy_mean),
        'perp_resid': perp_resid,       # perpendicular residuals [px]
        'para_resid': para_resid,       # parallel residuals [px]
        'fitted_x':   fitted_x,
        'fitted_y':   fitted_y,
    }


def fit_drift(cx: np.ndarray, cy: np.ndarray,
              declination_deg: float, fps: float,
              sigma_threshold: float,
              times_sec: Optional[np.ndarray] = None,
              start_trim_sec: float = 0.0,
              stop_trim_sec:  float = 0.0) -> dict:
    """
    TLS/SVD drift line fit with time-window trimming and sigma outlier rejection.

    start_trim_sec / stop_trim_sec clip the beginning and end of the sequence
    before any fitting.  They are applied as a hard time mask — clipped points
    appear as rejected (mask=False) in the returned mask so the plot can show
    them in a distinct colour.

    sigma_threshold then rejects geometric outliers among the remaining points.
    """
    n = len(cx)
    if times_sec is not None and len(times_sec) == n:
        t = times_sec
    else:
        t = np.arange(n, dtype=float) / fps

    # ── Time window mask (trim) ──────────────────────────────────────────
    t0 = t[0]  + start_trim_sec
    t1 = t[-1] - stop_trim_sec
    time_mask = (t >= t0) & (t <= t1)

    if time_mask.sum() < 5:
        time_mask = np.ones(n, dtype=bool)   # safety: never trim everything

    cx_win  = cx[time_mask]
    cy_win  = cy[time_mask]

    # ── Pass 1: unconstrained TLS fit on time-window points ─────────────
    tls0    = _tls_fit(cx_win, cy_win)
    resid0  = np.abs(tls0['perp_resid'])
    rms0    = float(np.sqrt(np.mean(resid0**2)))

    if sigma_threshold > 0 and rms0 > 0:
        sigma_mask_win = resid0 <= sigma_threshold * rms0
    else:
        sigma_mask_win = np.ones(len(cx_win), dtype=bool)

    if sigma_mask_win.sum() < 5:
        sigma_mask_win = np.ones(len(cx_win), dtype=bool)

    sigma_mask            = np.zeros(n, dtype=bool)
    sigma_mask[time_mask] = sigma_mask_win
    mask                  = time_mask & sigma_mask

    # ── Pass 2: TLS fit on inliers only ─────────────────────────────────
    tls     = _tls_fit(cx[mask], cy[mask])

    # ── Pass 3 (one refinement): re-reject using pass-2 residuals ────────
    # Reproject ALL time-window points onto the pass-2 line to get
    # consistent residuals for both the mask and the histogram.
    cx_mean2, cy_mean2 = tls['centroid']
    dx2, dy2           = tls['direction']
    perp_unit2         = np.array([-dy2, dx2])
    X_win2             = np.column_stack([cx_win - cx_mean2, cy_win - cy_mean2])
    resid2             = np.abs(X_win2 @ perp_unit2)
    rms2               = float(np.sqrt(np.mean(resid2[sigma_mask_win]**2)))  # inlier RMS

    if sigma_threshold > 0 and rms2 > 0:
        sigma_mask_win2 = resid2 <= sigma_threshold * rms2
    else:
        sigma_mask_win2 = np.ones(len(cx_win), dtype=bool)

    if sigma_mask_win2.sum() < 5:
        sigma_mask_win2 = sigma_mask_win   # fall back to pass-1 mask

    sigma_mask2            = np.zeros(n, dtype=bool)
    sigma_mask2[time_mask] = sigma_mask_win2
    mask                   = time_mask & sigma_mask2

    # Final TLS fit on refined inlier set
    tls     = _tls_fit(cx[mask], cy[mask])
    dx, dy  = tls['direction']

    # Reconstruct fitted positions for ALL points (for plotting)
    cx_mean, cy_mean = tls['centroid']
    X_all     = np.column_stack([cx - cx_mean, cy - cy_mean])
    para_all  = X_all @ tls['direction']
    perp_all  = X_all @ np.array([-dy, dx])
    fitted_x  = cx_mean + para_all * dx
    fitted_y  = cy_mean + para_all * dy

    # Final orthogonal residuals (all points, relative to inlier fit)
    resid_final = np.abs(perp_all)

    # ── Physical quantities ──────────────────────────────────────────────
    # Drift angle in image frame (degrees from +X axis)
    drift_angle_image    = np.degrees(np.arctan2(dy, dx))

    # Drift speed: total arc length / total elapsed time [px/s]
    para_inliers         = tls['para_resid']
    drift_length_px      = para_inliers.max() - para_inliers.min()   # [px]
    n_inliers            = int(mask.sum())
    t_inliers            = t[mask]
    elapsed_sec          = float(t_inliers[-1] - t_inliers[0])
    if elapsed_sec <= 0:
        elapsed_sec = n_inliers / fps   # fallback — should not happen
    drift_speed_px_sec   = drift_length_px / elapsed_sec             # [px/s]

    dec_rad              = np.radians(declination_deg)
    sidereal_arcsec_sec  = 15.041 * np.cos(dec_rad)                  # ["/s]
    pixel_scale          = (sidereal_arcsec_sec / drift_speed_px_sec
                            if drift_speed_px_sec > 0 else 0.0)      # ["/px]
    camera_angle         = (drift_angle_image + 90.0) % 360.0        # [°]

    rms_perp = float(np.sqrt(np.mean(tls['perp_resid']**2)))
    rms_para = float(np.sqrt(np.mean(tls['para_resid']**2)))
    sigma_perp = rms_perp
    sigma_para = rms_para
    sqrt_n      = np.sqrt(max(n_inliers, 1))

    # σ_angle [degrees]
    half_length = drift_length_px / 2.0
    if half_length > 0:
        sigma_angle_rad = np.arctan2(sigma_perp, half_length) / sqrt_n
    else:
        sigma_angle_rad = 0.0
    sigma_angle_deg = float(np.degrees(sigma_angle_rad))

    # σ_scale [arcsec/px]
    # Fractional uncertainty: σ_scale/scale = σ_para / drift_length_px / sqrt_n
    # (uncertainty decreases with more inlier points)
    if drift_length_px > 0 and sqrt_n > 0:
        sigma_scale = pixel_scale * sigma_para / drift_length_px / sqrt_n
    else:
        sigma_scale = 0.0

    return {
        # Geometry
        'mask':             mask,
        'time_mask':        time_mask,
        'direction':        (dx, dy),            # TLS unit direction vector
        'line_centroid':    (cx_mean, cy_mean),  # point on the line
        'fitted_x':         fitted_x,
        'fitted_y':         fitted_y,
        'perp_resid':       perp_all,
        'resid_abs':        resid_final,
        'rms_perp':         rms_perp,
        'rms_para':         rms_para,
        't':                t,
        # Physical results
        'camera_angle':     camera_angle,
        'pixel_scale':      pixel_scale,
        'drift_speed_px_s': drift_speed_px_sec,
        'drift_length_px':  drift_length_px,
        # Uncertainties (1σ)
        'sigma_angle_deg':  sigma_angle_deg,
        'sigma_scale':      sigma_scale,
        # Counts
        'n_used':           n_inliers,
        'n_rejected':       int((~mask).sum()),
    }




# ─────────────────────────────────────────────
#  Theme system
# ─────────────────────────────────────────────
#
# INTEGRATION NOTE — when all modules are merged into a single application:
#
#  1. Move THEMES, build_stylesheet(), _refresh_theme_aliases(), and the
#     active _theme state into a shared  theme.py  (or config.py) module.
#     Every panel imports its colour tokens from there.
#
#  2. The main application window owns the "View → Theme" menu/combo.
#     It should expose a QObject singleton with a ThemeChanged signal:
#
#       class ThemeManager(QObject):
#           changed = pyqtSignal(str)   # emits theme name: 'dark'|'red'|'light'
#           _instance = None
#           @classmethod
#           def instance(cls): ...
#
#  3. Each module window connects to that signal and implements a lightweight
#     _on_theme_changed(name) that:
#       a. calls _refresh_theme_aliases()          — updates colour tokens
#       b. repaints pyqtgraph plot backgrounds     — Qt stylesheets don't reach
#          into pg canvas backgrounds, so this must be done explicitly
#       c. does NOT call QApplication.setStyleSheet() — the main window does
#          that once, and it applies to the whole app automatically.
#
#  4. Inline widget stylesheets like setStyleSheet(f"color:{ACCENT}...")
#     that are set once at construction time will NOT update on theme switch.
#     Prefer assigning Qt object names (setObjectName) and handling colours
#     purely through the global stylesheet rules in build_stylesheet() —
#     that approach is zero-cost to re-theme.
#
# The current _apply_theme() already calls QApplication.instance().setStyleSheet()
# which is correct global behaviour; it just needs to be delegated to the
# main window's ThemeManager at integration time.
#


# ─────────────────────────────────────────────
#  Simbad name resolver worker
# ─────────────────────────────────────────────

class SimbadWorker(QThread):
    """Resolve a target name via the Simbad TAP service and return declination."""

    result  = pyqtSignal(float, str)   # (dec_deg, canonical_name)
    error   = pyqtSignal(str)

    def __init__(self, name: str):
        super().__init__()
        self.name = name.strip()

    def run(self):
        try:
            import urllib.request, urllib.parse, json as _json

            # Use the Simbad name-resolver (sim-id) endpoint — simplest approach.
            # Returns ADQL result as VOTable-like JSON via the TAP sync interface.
            adql = (
                "SELECT ra, dec, main_id "
                "FROM basic "
                "JOIN ident ON ident.oidref = basic.oid "
                f"WHERE ident.id = '{self.name}'"
            )
            params = urllib.parse.urlencode({
                "REQUEST": "doQuery",
                "LANG":    "ADQL",
                "FORMAT":  "json",
                "QUERY":   adql,
            })
            url = "https://simbad.u-strasbg.fr/simbad/sim-tap/sync?" + params
            req = urllib.request.Request(url, headers={"User-Agent": "DriftCalibration/1.0"})
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = _json.loads(resp.read().decode())

            rows = data.get("data", [])
            if not rows:
                self.error.emit(f"Object '{self.name}' not found in Simbad.")
                return

            # data columns: [ra, dec, main_id]
            ra_deg, dec_deg, main_id = rows[0]
            self.result.emit(float(dec_deg), str(main_id))

        except Exception as e:
            self.error.emit(f"Simbad query failed: {e}")


# ═══════════════════════════════════════════════════════════════════════════
#  ── DRIFT CALIBRATION TAB ───────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════

class DriftTab(QWidget):

    def __init__(self, parent=None):
        super().__init__(parent)
        self.worker: Optional[DriftWorker] = None
        self._simbad_worker: Optional[SimbadWorker] = None
        self.result: Optional[DriftResult] = None
        self._raw_data: Optional[dict] = None
        self._file_loaded: bool = False
        self._fit_uncertainties: dict = {}
        self._last_plate_scale: Optional[float] = None
        # batch navigator
        self._nav_paths:   list = []
        self._nav_idx:     int  = 0
        self._nav_pending: list = []
        self._nav_memory:  dict = {}

        self._build_ui()
        self._apply_graph_theme()

    # ── UI Construction ──────────────────────────────────────────────────

    def _build_ui(self):
        root = QHBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(12)

        # Left panel: controls
        left = QWidget()
        left.setFixedWidth(400)
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 8, 0)
        left_layout.setSpacing(8)

        sep = QFrame(); sep.setObjectName("separator"); sep.setFrameShape(QFrame.Shape.HLine)
        left_layout.addWidget(sep)

        # ── Target group (file + Simbad lookup + declination) ───────────
        target_group  = QGroupBox("Target")
        target_layout = QGridLayout(target_group)
        target_layout.setVerticalSpacing(8)
        target_layout.setHorizontalSpacing(8)

        # Row 0 — SER file
        ser_lbl = QLabel("SER file")
        ser_lbl.setStyleSheet(f"color: {TEXT_MUTED};")
        self.file_edit = QLineEdit()
        self.file_edit.setPlaceholderText("No file selected…")
        self.file_edit.setReadOnly(True)
        browse_btn = QPushButton("Browse")
        browse_btn.clicked.connect(self._browse_file)
        browse_btn.setMinimumWidth(80)
        target_layout.addWidget(ser_lbl,        0, 0)
        target_layout.addWidget(self.file_edit, 0, 1)
        target_layout.addWidget(browse_btn,     0, 2)

        # File info (spans all columns)
        self.file_info_label = QLabel("")
        self.file_info_label.setStyleSheet(f"color: {TEXT_MUTED}; font-size: 10px;")
        target_layout.addWidget(self.file_info_label, 1, 0, 1, 3)

        # Thin separator line
        sep2 = QFrame()
        sep2.setFrameShape(QFrame.Shape.HLine)
        sep2.setObjectName("separator")
        target_layout.addWidget(sep2, 2, 0, 1, 3)

        # Row 3 — Target name + Simbad resolve
        name_lbl = QLabel("Target name")
        name_lbl.setStyleSheet(f"color: {TEXT_MUTED};")
        name_lbl.setToolTip("Enter a Simbad-resolvable name to auto-fill declination")
        self.target_name_edit = QLineEdit()
        self.target_name_edit.setPlaceholderText("e.g.  eta Cas,  STF 60,  ADS 671 …")
        self.target_name_edit.setToolTip("Simbad object name — press Resolve or hit Enter")
        self.target_name_edit.returnPressed.connect(self._resolve_simbad)
        self.simbad_btn = QPushButton("Resolve")
        self.simbad_btn.setMinimumWidth(80)
        self.simbad_btn.setToolTip("Query Simbad for the declination of this target")
        self.simbad_btn.clicked.connect(self._resolve_simbad)
        target_layout.addWidget(name_lbl,              3, 0)
        target_layout.addWidget(self.target_name_edit, 3, 1)
        target_layout.addWidget(self.simbad_btn,       3, 2)

        # Row 4 — Declination
        dec_lbl = QLabel("Declination")
        dec_lbl.setStyleSheet(f"color: {TEXT_MUTED};")
        dec_lbl.setToolTip("Declination of the drift star (degrees)")

        self.dec_spin = QDoubleSpinBox()
        self.dec_spin.setRange(-90.0, 90.0)
        self.dec_spin.setValue(45.0)
        self.dec_spin.setSuffix(" °")
        self.dec_spin.setDecimals(4)
        self.dec_spin.setSpecialValueText("")
        self.dec_spin.lineEdit().setReadOnly(False)
        self.dec_spin.installEventFilter(self)
        self.dec_spin.setToolTip("Declination of the drift star (degrees)")

        self.dec_auto_lbl = QLabel("")
        self.dec_auto_lbl.setStyleSheet("font-size: 10px; border: none;")

        dec_widget = QWidget()
        dec_inner  = QHBoxLayout(dec_widget)
        dec_inner.setContentsMargins(0, 0, 0, 0)
        dec_inner.setSpacing(6)
        dec_inner.addWidget(self.dec_spin)
        dec_inner.addWidget(self.dec_auto_lbl)

        target_layout.addWidget(dec_lbl,    4, 0)
        target_layout.addWidget(dec_widget, 4, 1, 1, 2)

        target_layout.setColumnStretch(1, 1)
        left_layout.addWidget(target_group)

        # Run / Stop — side by side, 3:1 width ratio
        run_stop_row = QHBoxLayout()
        run_stop_row.setSpacing(6)
        self.run_btn = QPushButton("▶  Run Drift Analysis")
        self.run_btn.setObjectName("primary")
        self.run_btn.setFixedHeight(38)
        self.run_btn.clicked.connect(self._run_analysis)
        self.run_btn.setEnabled(False)
        self.kill_btn = QPushButton("■  Stop")
        self.kill_btn.setObjectName("primary")
        self.kill_btn.setFixedHeight(38)
        self.kill_btn.clicked.connect(self._kill_worker)
        self.kill_btn.setEnabled(False)
        run_stop_row.addWidget(self.run_btn, 3)
        run_stop_row.addWidget(self.kill_btn, 1)
        left_layout.addLayout(run_stop_row)

        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        left_layout.addWidget(self.progress_bar)

        optics_group  = QGroupBox("Optics  (optional)")
        optics_layout = QGridLayout(optics_group)
        optics_layout.setSpacing(6)

        def _optics_field(placeholder):
            w = QLineEdit()
            w.setPlaceholderText(placeholder)
            w.setMinimumHeight(26)
            w.textChanged.connect(lambda _: self._update_optics_labels())
            return w

        px_lbl = QLabel("Pixel size (µm)")
        px_lbl.setStyleSheet(f"color:{TEXT_MUTED};")
        self.pixel_size_edit = _optics_field("e.g. 4.65")

        ap_lbl = QLabel("Aperture (mm)")
        ap_lbl.setStyleSheet(f"color:{TEXT_MUTED};")
        self.aperture_edit = _optics_field("e.g. 200")

        wl_lbl = QLabel("Wavelength (nm)")
        wl_lbl.setStyleSheet(f"color:{TEXT_MUTED};")
        self.wavelength_edit = _optics_field("e.g. 550")
        self.wavelength_edit.setText("550")

        fl_lbl = QLabel("Focal length")
        fl_lbl.setStyleSheet(f"color:{TEXT_MUTED};")
        self.fl_value_lbl = QLabel("—")
        self.fl_value_lbl.setStyleSheet(f"color:{ACCENT}; font-weight:bold; border:none;")

        fratio_lbl = QLabel("f-ratio")
        fratio_lbl.setStyleSheet(f"color:{TEXT_MUTED};")
        self.fratio_value_lbl = QLabel("—")
        self.fratio_value_lbl.setStyleSheet(f"color:{ACCENT}; font-weight:bold; border:none;")

        sampling_lbl = QLabel("Sampling")
        sampling_lbl.setStyleSheet(f"color:{TEXT_MUTED};")
        self.sampling_value_lbl = QLabel("—")
        self.sampling_value_lbl.setStyleSheet(f"color:{ACCENT}; font-weight:bold; border:none;")

        optics_layout.addWidget(px_lbl,                  0, 0)
        optics_layout.addWidget(self.pixel_size_edit,    0, 1)
        optics_layout.addWidget(ap_lbl,                  1, 0)
        optics_layout.addWidget(self.aperture_edit,      1, 1)
        optics_layout.addWidget(wl_lbl,                  2, 0)
        optics_layout.addWidget(self.wavelength_edit,    2, 1)
        optics_layout.addWidget(fl_lbl,                  3, 0)
        optics_layout.addWidget(self.fl_value_lbl,       3, 1)
        optics_layout.addWidget(fratio_lbl,              4, 0)
        optics_layout.addWidget(self.fratio_value_lbl,   4, 1)
        optics_layout.addWidget(sampling_lbl,            5, 0)
        optics_layout.addWidget(self.sampling_value_lbl, 5, 1)
        optics_layout.setColumnStretch(1, 1)
        left_layout.addWidget(optics_group)

        self.save_json_btn = QPushButton("Save Calibration (.json)")
        self.save_json_btn.setObjectName("primary")
        self.save_json_btn.setFixedHeight(38)
        self.save_json_btn.clicked.connect(self._save_json)
        self.save_json_btn.setEnabled(False)
        left_layout.addWidget(self.save_json_btn)

        self.status_label = QLabel("Load a SER drift file to begin.")
        self.status_label.setStyleSheet(f"color: {TEXT_MUTED}; font-size: 10px;")
        self.status_label.setWordWrap(True)
        left_layout.addWidget(self.status_label)

        # Log
        log_group = QGroupBox("Log")
        log_layout = QVBoxLayout(log_group)
        self.log_edit = QTextEdit()
        self.log_edit.setReadOnly(True)
        self.log_edit.setMinimumHeight(110)
        self.log_edit.setMaximumHeight(180)
        log_layout.addWidget(self.log_edit)
        left_layout.addWidget(log_group)

        left_layout.addStretch()
        root.addWidget(left)

        # Right panel: plots + results
        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(10)

        # Results cards — two rows of 3
        cards_col = QVBoxLayout()
        cards_row1 = QHBoxLayout()
        cards_row2 = QHBoxLayout()
        self.card_angle       = ResultCard("Camera Angle",   "degrees")
        self.card_scale       = ResultCard("Pixel Scale",    "arcsec / pixel")
        self.card_frames      = ResultCard("Frames Used",    "")
        self.card_sigma_angle = ResultCard("σ angle",        "degrees (1σ)")
        self.card_sigma_scale = ResultCard("σ scale",        "arcsec/px (1σ)")
        self.card_rms         = ResultCard("RMS ⊥ residual", "px")
        for card in (self.card_angle, self.card_scale, self.card_frames):
            cards_row1.addWidget(card)
        for card in (self.card_sigma_angle, self.card_sigma_scale, self.card_rms):
            cards_row2.addWidget(card)
        cards_col.addLayout(cards_row1)
        cards_col.addLayout(cards_row2)
        right_layout.addLayout(cards_col)

        # ── View toggle ────────────────────────────────────────────────
        toggle_row = QHBoxLayout()
        toggle_row.setSpacing(16)
        toggle_lbl = QLabel("View:")
        toggle_lbl.setStyleSheet(f"color:{TEXT_MUTED};")
        self.radio_xy   = QRadioButton("X vs Y  (trajectory)")
        self.radio_xt   = QRadioButton("X vs time")
        self.radio_yt   = QRadioButton("Y vs time")
        self.radio_xt.setChecked(True)
        self._view_group = QButtonGroup()
        self._view_group.addButton(self.radio_xy, 0)
        self._view_group.addButton(self.radio_xt, 1)
        self._view_group.addButton(self.radio_yt, 2)
        self._view_group.buttonClicked.connect(self._on_view_toggled)
        toggle_row.addWidget(toggle_lbl)
        toggle_row.addWidget(self.radio_xt)
        toggle_row.addWidget(self.radio_yt)
        toggle_row.addWidget(self.radio_xy)

        # ── Scale toggle ────────────────────────────────────────────────
        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.VLine)
        sep.setStyleSheet(f"color:{BORDER_COLOR};")
        scale_lbl = QLabel("Scale:")
        scale_lbl.setStyleSheet(f"color:{TEXT_MUTED};")
        self.radio_scale_used = QRadioButton("Used points")
        self.radio_scale_all  = QRadioButton("All points")
        self.radio_scale_used.setChecked(True)
        self._scale_group = QButtonGroup()
        self._scale_group.addButton(self.radio_scale_used, 0)
        self._scale_group.addButton(self.radio_scale_all,  1)
        self._scale_group.buttonClicked.connect(self._on_view_toggled)
        toggle_row.addSpacing(8)
        toggle_row.addWidget(sep)
        toggle_row.addSpacing(8)
        toggle_row.addWidget(scale_lbl)
        toggle_row.addWidget(self.radio_scale_used)
        toggle_row.addWidget(self.radio_scale_all)
        toggle_row.addStretch()
        right_layout.addLayout(toggle_row)

        # ── Single main plot (content switches with toggle) ────────────
        # Navigator bar (multi-SER batch)
        self.drift_nav_bar = QWidget()
        _dnr = QHBoxLayout(self.drift_nav_bar)
        _dnr.setContentsMargins(4, 2, 4, 2); _dnr.setSpacing(6)
        self.drift_nav_prev = QPushButton("◄◄")
        self.drift_nav_prev.setFixedWidth(52); self.drift_nav_prev.setFixedHeight(28)
        self.drift_nav_prev.setStyleSheet("font-size:14px; font-weight:bold;")
        self.drift_nav_prev.clicked.connect(self._drift_nav_prev)
        self.drift_nav_next = QPushButton("►►")
        self.drift_nav_next.setFixedWidth(52); self.drift_nav_next.setFixedHeight(28)
        self.drift_nav_next.setStyleSheet("font-size:14px; font-weight:bold;")
        self.drift_nav_next.clicked.connect(self._drift_nav_next)
        self.drift_nav_lbl = QLabel("")
        self.drift_nav_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.drift_nav_lbl.setStyleSheet(
            f"color:{TEXT_PRIMARY}; font-size:13px; font-weight:bold;")
        self.drift_nav_lbl.setFixedWidth(60)
        self.drift_nav_file_lbl = QLabel("")
        self.drift_nav_file_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.drift_nav_file_lbl.setStyleSheet(f"color:{TEXT_MUTED}; font-size:10px;")
        self.drift_nav_file_lbl.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        _dnr.addWidget(self.drift_nav_prev)
        _dnr.addWidget(self.drift_nav_lbl)
        _dnr.addWidget(self.drift_nav_file_lbl, 1)
        _dnr.addWidget(self.drift_nav_next)
        self.drift_nav_bar.setVisible(False)
        right_layout.addWidget(self.drift_nav_bar)

        self.plot_main = pg.PlotWidget()
        self.plot_main.showGrid(x=True, y=True, alpha=0.15)
        right_layout.addWidget(self.plot_main)

        # ── Bottom strip: sliders (¾) + residuals histogram (¼) ──────────
        bottom_widget = QWidget()
        bottom_widget.setFixedHeight(160)
        bottom_outer = QHBoxLayout(bottom_widget)
        bottom_outer.setContentsMargins(0, 0, 0, 0)
        bottom_outer.setSpacing(8)

        # Left ¾ — interactive controls
        ctrl_widget = QWidget()
        ctrl_widget.setStyleSheet(
            f"QWidget {{ background:{PANEL_BG}; border:1px solid {BORDER_COLOR};"
            f" border-radius:6px; }}")
        self._ctrl_widget = ctrl_widget
        ctrl_inner = QVBoxLayout(ctrl_widget)
        ctrl_inner.setContentsMargins(12, 10, 12, 10)
        ctrl_inner.setSpacing(8)

        def make_slider_row(label, unit, lo, hi, default, decimals=1):
            row = QHBoxLayout()
            lbl = QLabel(label)
            lbl.setStyleSheet(f"color:{TEXT_MUTED}; min-width:76px; font-size:11px; border:none;")
            sl  = QSlider(Qt.Orientation.Horizontal)
            sl.setRange(lo, hi)
            sl.setValue(default)
            sl.setEnabled(False)
            val = QLabel(f"{default / (10**decimals):.{decimals}f} {unit}")
            val.setStyleSheet(f"color:{ACCENT}; min-width:48px; font-size:11px; border:none;")
            val.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            row.addWidget(lbl)
            row.addWidget(sl)
            row.addWidget(val)
            return row, sl, val

        row_s,   self.start_slider,   self.start_val_lbl   = \
            make_slider_row("Start trim",  "s", 0, 300, 5)
        row_e,   self.stop_slider,    self.stop_val_lbl    = \
            make_slider_row("Stop trim",   "s", 0, 300, 5)
        row_sig, self.sigma_slider,   self.sigma_value_lbl = \
            make_slider_row("σ threshold", "σ", 5, 50,  20)

        self.trim_info = QLabel("")
        self.trim_info.setStyleSheet(f"color:{TEXT_MUTED}; font-size:10px; border:none;")

        self.rejection_info = QLabel("Accepted: —  /  Rejected: —")
        self.rejection_info.setStyleSheet(f"color:{TEXT_MUTED}; font-size:10px; border:none;")

        ctrl_inner.addLayout(row_s)
        ctrl_inner.addLayout(row_e)
        ctrl_inner.addLayout(row_sig)
        info_row = QHBoxLayout()
        info_row.addWidget(self.trim_info)
        info_row.addStretch()
        info_row.addWidget(self.rejection_info)
        ctrl_inner.addLayout(info_row)

        # Right ¼ — residuals histogram
        self.plot_resid = pg.PlotWidget()
        self.plot_resid.setLabel('bottom', '⊥ residual  (px)')
        self.plot_resid.setLabel('left', 'count')
        self.plot_resid.getAxis('left').setStyle(showValues=True)
        self.plot_resid.showGrid(x=True, y=False, alpha=0.15)

        bottom_outer.addWidget(ctrl_widget, 3)
        bottom_outer.addWidget(self.plot_resid, 1)

        right_layout.addWidget(bottom_widget)

        # Wire sliders
        self.start_slider.valueChanged.connect(self._on_trim_slider_moved)
        self.stop_slider.valueChanged.connect(self._on_trim_slider_moved)
        self.sigma_slider.valueChanged.connect(self._on_sigma_changed)
        self.dec_spin.valueChanged.connect(self._on_dec_manually_edited)

        root.addWidget(right)

    def _apply_graph_theme(self):
        pg.setConfigOption('background', PANEL_BG)
        pg.setConfigOption('foreground', TEXT_MUTED)
        for plot in (self.plot_main, self.plot_resid):
            plot.setBackground(PANEL_BG)
            plot.getPlotItem().getAxis('left').setPen(pg.mkPen(color=BORDER_COLOR))
            plot.getPlotItem().getAxis('bottom').setPen(pg.mkPen(color=BORDER_COLOR))
            plot.getPlotItem().titleLabel.setAttr('color', ACCENT)
            plot.showGrid(x=True, y=True, alpha=0.15)

    # ── Interactions ─────────────────────────────────────────────────────

    def refresh_styles(self):
        """Called by SpeckleMainWindow when the theme changes."""
        t = _theme
        for plot in (self.plot_main, self.plot_resid):
            plot.setBackground(t['PANEL_BG'])
            for axis in ('left', 'bottom', 'right', 'top'):
                ax = plot.getAxis(axis)
                ax.setPen(pg.mkPen(t['TEXT_MUTED']))
                ax.setTextPen(pg.mkPen(t['TEXT_PRIMARY']))
        # Re-theme result cards
        for card in (self.card_angle, self.card_scale, self.card_frames,
                     self.card_sigma_angle, self.card_sigma_scale, self.card_rms):
            card.refresh_style()
        # Re-theme slider control panel
        self._ctrl_widget.setStyleSheet(
            f"QWidget {{ background:{PANEL_BG}; border:1px solid {BORDER_COLOR};"
            f" border-radius:6px; }}")
        # Re-theme slider value labels
        for lbl in (self.start_val_lbl, self.stop_val_lbl, self.sigma_value_lbl):
            lbl.setStyleSheet(
                f"color:{ACCENT}; min-width:48px; font-size:11px; border:none;")
        if self._raw_data is not None:
            self._recompute()

    def eventFilter(self, obj, event):
        """Select all text in dec_spin on focus so the user can type immediately."""
        from PyQt6.QtCore import QEvent
        if obj is self.dec_spin and event.type() == QEvent.Type.FocusIn:
            QTimer.singleShot(0, self.dec_spin.selectAll)
        return super().eventFilter(obj, event)

    def _update_dec_label(self, source: str = ""):
        """Update the small badge next to the declination spinbox.
        source: '' = clear, 'file' = from companion file, 'simbad' = from Simbad.
        """
        if source == 'file':
            self.dec_auto_lbl.setText("✓ from file")
            self.dec_auto_lbl.setStyleSheet(f"font-size:10px; color:{ACCENT2}; border:none;")
        elif source == 'simbad':
            self.dec_auto_lbl.setText("✓ from Simbad")
            self.dec_auto_lbl.setStyleSheet(f"font-size:10px; color:{ACCENT2}; border:none;")
        else:
            self.dec_auto_lbl.setText("")

    def _resolve_simbad(self):
        """Launch a background Simbad name resolution query."""
        name = self.target_name_edit.text().strip()
        if not name:
            self._log("Enter a target name before resolving.", error=True)
            return
        self.simbad_btn.setEnabled(False)
        self.simbad_btn.setText("…")
        self._log(f"Querying Simbad for '{name}'…")
        self._simbad_worker = SimbadWorker(name)
        self._simbad_worker.result.connect(self._on_simbad_result)
        self._simbad_worker.error.connect(self._on_simbad_error)
        self._simbad_worker.finished.connect(
            lambda: (self.simbad_btn.setEnabled(True),
                     self.simbad_btn.setText("Resolve")))
        self._simbad_worker.start()

    def _on_simbad_result(self, dec_deg: float, main_id: str):
        """Simbad returned a result — fill in the declination."""
        self._dec_auto_filled = True
        self.dec_spin.blockSignals(True)
        self.dec_spin.setValue(dec_deg)
        self.dec_spin.blockSignals(False)
        self._update_dec_label('simbad')
        self._log(f"Simbad: {main_id}  →  δ = {dec_deg:+.4f}°")
        if self._raw_data is not None:
            self._recompute()

    def _on_simbad_error(self, msg: str):
        """Simbad query failed — show error in log."""
        self._log(f"Simbad: {msg}", error=True)

    def _on_dec_manually_edited(self):
        """User changed declination by hand — clear the auto-fill indicator and recompute."""
        if getattr(self, '_dec_auto_filled', False):
            self._dec_auto_filled = False
            self.dec_auto_lbl.setText("")
        if self._raw_data is not None:
            self._recompute()

    def _browse_file(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Open SER Drift File(s)", _working_dir(),
            "SER Files (*.ser);;All Files (*)"
        )
        if not paths:
            return
        n = len(paths); path = paths[0]
        self.file_edit.setText(path if n == 1 else f"{n} files selected")
        self._file_loaded = False
        self._nav_paths  = list(paths) if n > 1 else []
        self._nav_memory = {}
        self.drift_nav_bar.setVisible(False)
        self.run_btn.setEnabled(True)
        try:
            header, ts = read_ser_header_and_timestamps(path)
            duration_str = ""
            if ts is not None:
                dur = float(ts[-1])
                self._ser_duration_sec = dur
                fps_est = (header.frame_count - 1) / dur if dur > 0 else 0
                duration_str = f"  ·  {dur:.1f} s  ·  ~{fps_est:.1f} fps"

            # Configure start/stop sliders to actual sequence duration
            if ts is not None:
                max_val = max(1, int(dur * 10) // 2)  # max = half duration
                self.start_slider.setRange(0, max_val)
                self.stop_slider.setRange(0, max_val)
                self.start_slider.setValue(5)   # default 0.5 s
                self.stop_slider.setValue(5)
                self.start_val_lbl.setText("0.5 s")
                self.stop_val_lbl.setText("0.5 s")
                self.start_slider.setEnabled(True)
                self.stop_slider.setEnabled(True)
                self._update_trim_info()

            self.file_info_label.setText(
                f"{header.frame_count} frames  ·  "
                f"{header.image_width}×{header.image_height}  ·  "
                f"{header.pixel_depth}-bit{duration_str}"
            )
            if n > 1:
                self.file_info_label.setText(
                    f"{n} files · first: "
                    f"{header.frame_count} frames · "
                    f"{header.image_width}×{header.image_height} · "
                    f"{header.pixel_depth}-bit{duration_str}")
            self._log(f"Opened {n} file(s) — first: {Path(path).name}")
            self._log(
                f"  {header.frame_count} frames, "
                f"{header.image_width}×{header.image_height} px, "
                f"{header.pixel_depth}-bit{duration_str}"
            )
            if header.observer:
                self._log(f"  Observer: {header.observer.strip()}")

            # ── Try companion text file for declination ────────────────
            dec_found = _parse_declination_from_txt(path)
            if dec_found is not None:
                self.dec_spin.setValue(dec_found)
                self._dec_auto_filled = True
                self._log(f"  Declination auto-filled from companion file: {dec_found:+.4f}°")
                self._update_dec_label('file')
            else:
                self._dec_auto_filled = False
                self._update_dec_label('')
                # Check if companion file exists but had no parseable dec
                ser_p = Path(path)
                has_txt = any(
                    (ser_p.with_suffix(ext)).exists()
                    for ext in ('.txt', '.TXT', '.log')
                )
                if has_txt:
                    self._log("  Companion file found but no declination tag — enter manually.")

            self._file_loaded = True

        except Exception as e:
            self.file_info_label.setText(f"⚠ Could not read header: {e}")

    def _update_trim_info(self):
        dur = getattr(self, '_ser_duration_sec', None)
        if dur is None:
            self.trim_info.setText("")
            return
        start = self.start_slider.value() / 10.0
        stop  = self.stop_slider.value()  / 10.0
        used  = dur - start - stop
        if used <= 0:
            self.trim_info.setText(
                f"⚠ Offsets exceed duration ({dur:.1f} s)")
            self.trim_info.setStyleSheet(f"color:{DANGER}; font-size:10px;")
        else:
            self.trim_info.setText(f"→ {used:.1f} s used  of  {dur:.1f} s total")
            self.trim_info.setStyleSheet(f"color:{ACCENT2}; font-size:10px;")

    def _run_analysis(self):
        if not self._file_loaded:
            return
        self._raw_data  = None
        self.result     = None
        self._last_plate_scale = None
        self._clear_plots()
        self.run_btn.setEnabled(False)
        self.kill_btn.setEnabled(True)
        self.save_json_btn.setEnabled(False)
        self.sigma_slider.setEnabled(False)
        self.start_slider.setEnabled(False)
        self.stop_slider.setEnabled(False)
        self.progress_bar.setValue(0)
        self.drift_nav_bar.setVisible(False)
        if self._nav_paths:
            self._nav_memory  = {}
            self._nav_idx     = 0
            self._nav_pending = list(self._nav_paths)
            self._log(f"Batch: {len(self._nav_paths)} SER files — processing…")
            self._run_next_drift()
        else:
            self._launch_drift_worker(self.file_edit.text())

    def _launch_drift_worker(self, path: str):
        self.worker = DriftWorker(
            filepath        = path,
            declination_deg = self.dec_spin.value(),
        )
        self.worker.progress.connect(self.progress_bar.setValue)
        self.worker.status.connect(self._on_status)
        self.worker.finished.connect(self._on_finished)
        self.worker.error.connect(self._on_error)
        self.worker.start()

    def _run_next_drift(self):
        path = self._nav_pending[0]
        done = len(self._nav_paths) - len(self._nav_pending) + 1
        n    = len(self._nav_paths)
        self.status_label.setText(
            f"Processing {done} / {n}  —  {Path(path).name}")
        self.progress_bar.setValue(0)
        self._launch_drift_worker(path)

    def _update_drift_nav(self):
        n   = len(self._nav_paths)
        idx = self._nav_idx
        self.drift_nav_lbl.setText(f"{idx + 1} / {n}")
        self.drift_nav_file_lbl.setText(Path(self._nav_paths[idx]).name)
        self.drift_nav_prev.setEnabled(idx > 0)
        self.drift_nav_next.setEnabled(idx < n - 1)

    def _drift_nav_go(self, idx: int):
        self._nav_idx = idx
        self._update_drift_nav()
        path = self._nav_paths[idx]
        raw  = self._nav_memory.get(path)
        if raw is None:
            return
        self._raw_data = raw
        self.file_edit.setText(path)
        dur = raw['times_sec'][-1] - raw['times_sec'][0]
        max_val = max(1, int(dur * 10) // 2)
        self.start_slider.setRange(0, max_val)
        self.stop_slider.setRange(0, max_val)
        self._log(f"◄► {Path(path).name}")
        self._recompute()

    def _drift_nav_prev(self):
        if self._nav_idx > 0:
            self._drift_nav_go(self._nav_idx - 1)

    def _drift_nav_next(self):
        if self._nav_idx < len(self._nav_paths) - 1:
            self._drift_nav_go(self._nav_idx + 1)

    def _kill_worker(self):
        if self.worker and self.worker.isRunning():
            self.worker.stop()
            self.worker.wait(2000)
            self._log("⚠ Analysis stopped by user.")
        self.run_btn.setEnabled(True)
        self.kill_btn.setEnabled(False)
        self.start_slider.setEnabled(True)
        self.stop_slider.setEnabled(True)

    def _on_status(self, msg: str):
        self.status_label.setText(msg)
        self._log(msg)

    def _on_error(self, msg: str):
        self._log(f"ERROR: {msg}", error=True)
        self.status_label.setText(f"Error — see log.")
        self.run_btn.setEnabled(True)
        self.kill_btn.setEnabled(False)
        self.start_slider.setEnabled(True)
        self.stop_slider.setEnabled(True)

    def _on_finished(self, raw: dict):
        """Worker done — store raw centroids and enable interactive rejection."""
        self._raw_data = raw
        n   = len(raw['centroids_x'])
        dt  = raw['median_dt_ms']
        dur = raw['times_sec'][-1] - raw['times_sec'][0]
        self._log(f"✓ {n} centroids over {dur:.2f} s  ({dt:.3f} ms/frame)")

        if self._nav_pending:
            # Batch mode: store and continue
            path = self._nav_pending.pop(0)
            self._nav_memory[path] = raw
            done = len(self._nav_paths) - len(self._nav_pending)
            self._log(f"  [{done}/{len(self._nav_paths)}] {Path(path).name} ✓")
            if self._nav_pending:
                self._run_next_drift()
                return
            # All done — activate navigator
            self._log("═" * 40)
            self._log(f"✓ Batch complete — {len(self._nav_paths)} files. Use ◄► to navigate.")
            self.save_json_btn.setText("Save Batch Calibration (.json)")
            self._nav_idx = 0
            self._drift_nav_go(0)
            self._update_drift_nav()
            self.drift_nav_bar.setVisible(True)
        else:
            # Single file
            self._log("  Adjust σ slider to tune outlier rejection, then save.")
            self.save_json_btn.setText("Save Calibration (.json)")
            self._recompute()

        self.run_btn.setEnabled(True)
        self.kill_btn.setEnabled(False)
        self.sigma_slider.setEnabled(True)
        self.start_slider.setEnabled(True)
        self.stop_slider.setEnabled(True)
        self.save_json_btn.setEnabled(True)

    def _on_trim_slider_moved(self):
        """Trim slider moved — update label and recompute fit live."""
        self.start_val_lbl.setText(f"{self.start_slider.value() / 10.0:.1f} s")
        self.stop_val_lbl.setText( f"{self.stop_slider.value()  / 10.0:.1f} s")
        self._update_trim_info()
        if self._raw_data is not None:
            self._recompute()

    def _on_sigma_changed(self, value: int):
        """Sigma slider moved — update label and recompute fit live."""
        sigma = value / 10.0
        self.sigma_value_lbl.setText(f"{sigma:.1f} σ")
        if self._raw_data is not None:
            self._recompute()

    def _plot_color_pts(self) -> str:
        """Accepted data point color — black on light, ACCENT otherwise."""
        return '#222222' if _theme is THEMES['light'] else ACCENT

    def _plot_color_fit(self) -> str:
        """Fit line color — red on light, ACCENT2 otherwise."""
        return '#0055aa' if _theme is THEMES['light'] else ACCENT2

    def _plot_color_hist(self) -> str:
        """Histogram inlier bar color — dark blue on light, ACCENT otherwise."""
        return '#2255aa' if _theme is THEMES['light'] else ACCENT

    def _recompute(self):
        """
        Refit with TLS/SVD at current sigma threshold.
        Called on every slider move — updates plots and cards live.
        """
        if self._raw_data is None:
            return

        sigma = self.sigma_slider.value() / 10.0
        fit   = fit_drift(
            self._raw_data['centroids_x'],
            self._raw_data['centroids_y'],
            self.dec_spin.value(),          # always use current spinbox value
            self._raw_data['fps'],
            sigma,
            times_sec      = self._raw_data.get('times_sec'),
            start_trim_sec = self.start_slider.value() / 10.0,
            stop_trim_sec  = self.stop_slider.value()  / 10.0,
        )
        self.card_angle.set_value(f"{fit['camera_angle']:.4f}")
        self.card_scale.set_value(f"{fit['pixel_scale']:.6f}")
        self._last_plate_scale = fit['pixel_scale']
        self.card_sigma_angle.set_value(f"{fit['sigma_angle_deg']:.4f}")
        self.card_sigma_scale.set_value(f"{fit['sigma_scale']:.6f}")
        self.card_rms.set_value(f"{fit['rms_perp']:.3f}")
        self.card_frames.set_value(
            f"{fit['n_used']} / {fit['n_used'] + fit['n_rejected']}")
        self.rejection_info.setText(
            f"Accepted: {fit['n_used']}  ·  Rejected: {fit['n_rejected']}\n"
            f"RMS ⊥: {fit['rms_perp']:.3f} px  ·  RMS ∥: {fit['rms_para']:.3f} px"
        )
        self._update_optics_labels()

        self._draw_plots_interactive(fit)

    def _on_view_toggled(self):
        """Radio button changed — redraw main plot with current fit."""
        if self._raw_data is not None:
            self._recompute()

    def _draw_plots_interactive(self, fit: dict):
        """
        Redraw plots using current view mode toggle.
        Accepted points = blue, rejected = red ✕.
        Fit line drawn as two clean endpoints (always a true straight line).
        """
        cx   = self._raw_data['centroids_x']
        cy   = self._raw_data['centroids_y']
        mask = fit['mask']
        t    = fit['t']
        view = self._view_group.checkedId()   # 0=XY, 1=Xt, 2=Yt

        self.plot_main.clear()
        self.plot_resid.clear()

        # ── Aspect lock only for trajectory view ───────────────────
        self.plot_main.getViewBox().setAspectLocked(view == 0)

        rej_mask = ~mask

        # ── Build fit line endpoints ────────────────────────────────
        # X/Y trajectory view: use TLS direction + centroid (correct 2D line)
        dx, dy   = fit['direction']
        lx, ly   = fit['line_centroid']
        para_all = (cx - lx) * dx + (cy - ly) * dy
        para_in  = para_all[mask]
        p_min, p_max = para_in.min(), para_in.max()
        x0, x1 = lx + p_min * dx, lx + p_max * dx
        y0, y1 = ly + p_min * dy, ly + p_max * dy

        # Time-based views: fit cx~t and cy~t independently with polyfit.
        # This is the correct way to get the slope in each time-series view —
        # the TLS para projection is a spatial quantity, not a time quantity.
        t_in = t[mask]
        t0_fit, t1_fit = t_in.min(), t_in.max()
        cx_fit = np.polyval(np.polyfit(t_in, cx[mask], 1), [t0_fit, t1_fit])
        cy_fit = np.polyval(np.polyfit(t_in, cy[mask], 1), [t0_fit, t1_fit])

        # ── Choose axes based on view mode ─────────────────────────
        if view == 0:   # X vs Y trajectory
            self.plot_main.setTitle("Drift Trajectory  (X vs Y)")
            self.plot_main.setLabel('bottom', 'X  (px)')
            self.plot_main.setLabel('left',   'Y  (px)')
            xdata_acc = cx[mask];      ydata_acc = cy[mask]
            xdata_rej = cx[rej_mask];  ydata_rej = cy[rej_mask]
            xfit = np.array([x0, x1]); yfit = np.array([y0, y1])

        elif view == 1:   # X vs time
            self.plot_main.setTitle("Centroid X  vs  Time")
            self.plot_main.setLabel('bottom', 'Time  (s)')
            self.plot_main.setLabel('left',   'X  (px)')
            xdata_acc = t[mask];       ydata_acc = cx[mask]
            xdata_rej = t[rej_mask];   ydata_rej = cx[rej_mask]
            xfit = np.array([t0_fit, t1_fit]); yfit = cx_fit

        else:             # Y vs time
            self.plot_main.setTitle("Centroid Y  vs  Time")
            self.plot_main.setLabel('bottom', 'Time  (s)')
            self.plot_main.setLabel('left',   'Y  (px)')
            xdata_acc = t[mask];       ydata_acc = cy[mask]
            xdata_rej = t[rej_mask];   ydata_rej = cy[rej_mask]
            xfit = np.array([t0_fit, t1_fit]); yfit = cy_fit

        # ── Accepted points (blue) ─────────────────────────────────
        _pc = self._plot_color_pts()
        self.plot_main.plot(
            xdata_acc, ydata_acc, pen=None, symbol='o', symbolSize=4,
            symbolBrush=pg.mkBrush(_pc + "aa"),
            symbolPen=pg.mkPen(_pc, width=0))

        # ── Rejected points (red ✕) ────────────────────────────────
        if rej_mask.any():
            self.plot_main.plot(
                xdata_rej, ydata_rej, pen=None, symbol='x', symbolSize=7,
                symbolBrush=pg.mkBrush(DANGER + "99"),
                symbolPen=pg.mkPen(DANGER, width=1.5))

        # ── Fitted line ────────────────────────────────────────────
        self.plot_main.plot(xfit, yfit, pen=pg.mkPen(self._plot_color_fit(), width=2))

        # ── Residuals histogram — blue = inliers, red = outliers ───
        thr_px      = self.sigma_slider.value() / 10.0 * fit['rms_perp']
        all_resid   = fit['perp_resid']          # all points incl. trimmed
        inlier_res  = all_resid[mask]            # accepted (blue)
        outlier_res = all_resid[~mask]           # trimmed + sigma-rejected (red)

        if len(inlier_res) >= 3:
            all_valid = all_resid[fit['time_mask']]   # use time-window extent for bins
            n_bins  = max(10, min(40, len(all_valid) // 8))
            _, edges = np.histogram(all_valid, bins=n_bins)
            centres = (edges[:-1] + edges[1:]) / 2
            bar_w   = (edges[1] - edges[0]) * 0.85

            # Blue bars — inliers
            counts_in, _ = np.histogram(inlier_res, bins=edges)
            _hc = self._plot_color_hist()
            self.plot_resid.addItem(pg.BarGraphItem(
                x=centres, height=counts_in, width=bar_w,
                brush=pg.mkBrush(_hc + "aa"),
                pen=pg.mkPen(_hc, width=0.5)))

            # Red bars — outliers (stacked on top of blue)
            if len(outlier_res):
                counts_out, _ = np.histogram(outlier_res, bins=edges)
                self.plot_resid.addItem(pg.BarGraphItem(
                    x=centres, height=counts_out, width=bar_w,
                    brush=pg.mkBrush(DANGER + "99"),
                    pen=pg.mkPen(DANGER, width=0.5)))

            # Sigma threshold lines
            for pos in (-thr_px, thr_px):
                self.plot_resid.addItem(pg.InfiniteLine(
                    pos=pos, angle=90,
                    pen=pg.mkPen(WARNING, width=1.5,
                                 style=Qt.PenStyle.DashLine)))

        # ── Apply axis scaling ──────────────────────────────────────
        scale_used = self._scale_group.checkedId() == 0   # 0=used, 1=all
        xs = xdata_acc if scale_used else np.concatenate([xdata_acc, xdata_rej])
        ys = ydata_acc if scale_used else np.concatenate([ydata_acc, ydata_rej])
        if len(xs) and len(ys):
            pad_x = max((xs.max() - xs.min()) * 0.05, 0.5)
            pad_y = max((ys.max() - ys.min()) * 0.05, 0.5)
            self.plot_main.setXRange(xs.min() - pad_x, xs.max() + pad_x, padding=0)
            self.plot_main.setYRange(ys.min() - pad_y, ys.max() + pad_y, padding=0)

    def _build_result(self) -> Optional[dict]:
        """Run fit at current sigma and return the fit dict. Used by _save_json."""
        if self._raw_data is None:
            return None
        sigma = self.sigma_slider.value() / 10.0
        return fit_drift(
            self._raw_data['centroids_x'],
            self._raw_data['centroids_y'],
            self._raw_data['declination_deg'],
            self._raw_data['fps'],
            sigma,
            times_sec      = self._raw_data.get('times_sec'),
            start_trim_sec = self.start_slider.value() / 10.0,
            stop_trim_sec  = self.stop_slider.value()  / 10.0,
        )

    def _compute_optics(self, plate_scale: float):
        """Derive (focal_length_mm, f_ratio, sampling) from spin values.
        plate_scale in arcsec/px. Returns (None, None, None) if inputs missing.
        sampling = f_ratio * lambda_nm / (pixel_size_um * 1000)
        """
        try:
            px = float(self.pixel_size_edit.text())
        except ValueError:
            px = 0.0
        try:
            ap = float(self.aperture_edit.text())
        except ValueError:
            ap = 0.0
        try:
            wl = float(self.wavelength_edit.text())
        except ValueError:
            wl = 550.0
        if px <= 0 or plate_scale <= 0:
            return None, None, None
        fl = 206265.0 * px / (plate_scale * 1000.0)
        fr = (fl / ap) if ap > 0 else None
        sampling = (fr * wl / (px * 1000.0)) if fr is not None else None
        return fl, fr, sampling

    def _update_optics_labels(self):
        """Refresh focal length / f-ratio display whenever fit or spin values change."""
        if self._last_plate_scale is None:
            return
        fl, fr, sampling = self._compute_optics(self._last_plate_scale)
        if fl is None:
            self.fl_value_lbl.setText("—")
            self.fratio_value_lbl.setText("—")
            self.sampling_value_lbl.setText("—")
        else:
            self.fl_value_lbl.setText(f"{fl:.1f} mm")
            self.fratio_value_lbl.setText(
                f"f/{fr:.1f}" if fr is not None else "— (no aperture)")
            if sampling is not None:
                self.sampling_value_lbl.setText(f"{sampling:.2f}")
            else:
                self.sampling_value_lbl.setText("—")

    def _fit_from_memory(self, mem: dict):
        """Re-run fit for a nav_memory entry using its stored slider values."""
        raw = mem["raw"]
        return fit_drift(
            raw['centroids_x'], raw['centroids_y'],
            raw['declination_deg'], raw['fps'],
            mem["sigma"] / 10.0,
            times_sec      = raw.get('times_sec'),
            start_trim_sec = mem["start"] / 10.0,
            stop_trim_sec  = mem["stop"]  / 10.0,
        )

    def _make_single_cal(self, fit: dict, sigma: float, source: str = "",
                         fl_mm=None, f_ratio=None, sampling=None) -> dict:
        """Build the standard single-file calibration dict."""
        try:
            px = float(self.pixel_size_edit.text()) or None
        except ValueError:
            px = None
        try:
            ap = float(self.aperture_edit.text()) or None
        except ValueError:
            ap = None
        try:
            wl = float(self.wavelength_edit.text())
        except ValueError:
            wl = 550.0
        d = {
            "camera_angle_deg":     round(fit['camera_angle'],    6),
            "sigma_angle_deg":      round(fit['sigma_angle_deg'], 6),
            "pixel_scale_arcsec":   round(fit['pixel_scale'],     8),
            "sigma_scale_arcsec":   round(fit['sigma_scale'],     8),
            "n_frames_used":        fit['n_used'],
            "n_frames_rejected":    fit['n_rejected'],
            "sigma_threshold_used": round(sigma, 1),
            "rms_perp_px":          round(fit['rms_perp'],  4),
            "rms_para_px":          round(fit['rms_para'],  4),
            "drift_length_px":      round(fit['drift_length_px'], 2),
            "fit_method":           "TLS/SVD (Total Least Squares)",
            "wavelength_nm":        wl,
            "pixel_size_um":        round(px, 2) if px else None,
            "aperture_mm":          round(ap, 1) if ap else None,
            "focal_length_mm":      round(fl_mm,   1) if fl_mm   is not None else None,
            "f_ratio":              round(f_ratio, 2) if f_ratio is not None else None,
            "sampling_factor":      round(sampling, 3) if sampling is not None else None,
        }
        if source:
            d["source_file"] = source
        return d

    def _save_json(self):
        """Export calibration to JSON. Batch: aggregate stats + per-file measurements."""
        import json as _json_cal
        import numpy as _np
        if self._raw_data is None:
            self._log("⚠ Run analysis first.", error=True)
            return

        is_batch = bool(self._nav_paths and len(self._nav_memory) > 1)

        default_name = ("drift_calibration_batch.json" if is_batch
                        else "drift_calibration.json")
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Drift Calibration",
            str(Path(_working_dir()) / default_name),
            "JSON files (*.json);;All Files (*)"
        )
        if not path:
            return

        if is_batch:
            # Snapshot current file's sliders before collecting all
            self._drift_nav_save_sliders()
            measurements = []
            angles, scales = [], []
            for p, mem in self._nav_memory.items():
                fit = self._fit_from_memory(mem)
                if fit is None:
                    continue
                fl_i, fr_i, samp_i = self._compute_optics(fit['pixel_scale'])
                m = self._make_single_cal(fit, mem["sigma"] / 10.0, Path(p).name,
                                          fl_mm=fl_i, f_ratio=fr_i, sampling=samp_i)
                measurements.append(m)
                angles.append(fit['camera_angle'])
                scales.append(fit['pixel_scale'])
            if not measurements:
                self._log("⚠ No valid fits in batch.", error=True)
                return
            a = _np.array(angles); s = _np.array(scales)
            cal = {
                "n_files":                  len(measurements),
                "camera_angle_mean_deg":    round(float(a.mean()), 6),
                "camera_angle_std_deg":     round(float(a.std()),  6),
                "pixel_scale_mean_arcsec":  round(float(s.mean()), 8),
                "pixel_scale_std_arcsec":   round(float(s.std()),  8),
                "camera_angle_deg":         round(float(a.mean()), 6),
                "pixel_scale_arcsec":       round(float(s.mean()), 8),
                "fit_method":               "TLS/SVD (Total Least Squares)",
                "wavelength_nm":            (lambda t: float(t) if t else None)(self.wavelength_edit.text()),
                "pixel_size_um":            (lambda t: round(float(t), 2) if t else None)(self.pixel_size_edit.text()),
                "aperture_mm":              (lambda t: round(float(t), 1) if t else None)(self.aperture_edit.text()),
                "focal_length_mm":          round(float(_np.mean(
                    [m["focal_length_mm"] for m in measurements
                     if m["focal_length_mm"] is not None])), 1)
                    if any(m["focal_length_mm"] is not None
                           for m in measurements) else None,
                "f_ratio":                  round(float(_np.mean(
                    [m["f_ratio"] for m in measurements
                     if m["f_ratio"] is not None])), 2)
                    if any(m["f_ratio"] is not None
                           for m in measurements) else None,
                "sampling_factor":          round(float(_np.mean(
                    [m["sampling_factor"] for m in measurements
                     if m["sampling_factor"] is not None])), 3)
                    if any(m["sampling_factor"] is not None
                           for m in measurements) else None,
                "_note": (
                    "camera_angle_deg and pixel_scale_arcsec are mean values across "
                    "all SER files and are safe to use directly as calibration inputs. "
                    "The _std fields quantify systematic uncertainties across the batch."
                ),
                "measurements":             measurements,
            }
        else:
            fit = self._build_result()
            if fit is None:
                return
            sigma = self.sigma_slider.value() / 10.0
            fl, fr, sampling = self._compute_optics(fit['pixel_scale'])
            cal = self._make_single_cal(fit, sigma, fl_mm=fl, f_ratio=fr, sampling=sampling)
            cal["_note_optics"] = (
                "f_ratio and pixel_size_um are not computed by drift calibration. "
                "Fill them in manually or via the speckle reduction module.")
            cal["_note_uncertainties"] = (
                "sigma_angle_deg and sigma_scale_arcsec are 1-sigma formal "
                "uncertainties from TLS/SVD fit residual propagation.")

        try:
            with open(path, 'w') as f:
                _json_cal.dump(cal, f, indent=2)
            self._log("─" * 38)
            if is_batch:
                n = cal["n_files"]
                self._log(f"✓ Batch calibration saved → {Path(path).name}  ({n} files)")
                self._log(f"  Camera angle : {cal['camera_angle_mean_deg']:.4f}°"
                          f"  ±{cal['camera_angle_std_deg']:.4f}°  (1σ systematic)")
                self._log(f"  Pixel scale  : {cal['pixel_scale_mean_arcsec']:.6f} arcsec/px"
                          f"  \u00b1{cal['pixel_scale_std_arcsec']:.6f}")
            else:
                sigma = self.sigma_slider.value() / 10.0
                self._log(f"✓ Saved → {Path(path).name}  (σ = {sigma:.1f})")
                self._log(f"  Camera angle : {cal['camera_angle_deg']:.4f}°"
                          f"  ±{cal['sigma_angle_deg']:.4f}°")
                self._log(f"  Pixel scale  : {cal['pixel_scale_arcsec']:.6f} arcsec/px"
                          f"  \u00b1{cal['sigma_scale_arcsec']:.6f}")
        except Exception as e:
            self._log(f"⚠ Could not save: {e}", error=True)

    def _clear_plots(self):
        for plot in (self.plot_main, self.plot_resid):
            plot.clear()
        for card in (self.card_angle, self.card_scale,
                     self.card_sigma_angle, self.card_sigma_scale,
                     self.card_rms, self.card_frames):
            card.set_value("—")
        self.rejection_info.setText("Accepted: —  /  Rejected: —")

    def _log(self, msg: str, error: bool = False):
        color = DANGER if error else TEXT_MUTED
        self.log_edit.append(f'<span style="color:{color}">{msg}</span>')

    def get_calibration(self) -> Optional[tuple[float, float]]:
        """Returns (camera_angle_deg, pixel_scale_arcsec) if a result is available."""
        if self.result:
            return self.result.camera_angle_deg, self.result.pixel_scale_arcsec
        return None


# ─────────────────────────────────────────────
#  Entry point
# ─────────────────────────────────────────────


# ═══════════════════════════════════════════════════════════════════════════
#  ── ANALYSIS BACKEND ────────────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════

KMAX_DEFAULT  = 60
DKMAX_DEFAULT =  9


def build_offset_list(dk_max: int) -> np.ndarray:
    offsets = []
    for vy in range(-dk_max, dk_max + 1):
        for vx in range(-dk_max, dk_max + 1):
            if vy*vy + vx*vx <= dk_max*dk_max:
                offsets.append((vy, vx))
    return np.array(offsets, dtype=np.int32)


def accumulate_bispectrum(cube: np.ndarray,
                          k_max:  int = KMAX_DEFAULT,
                          dk_max: int = DKMAX_DEFAULT,
                          progress_cb=None) -> tuple:
    n_frames, H, W = cube.shape
    offsets   = build_offset_list(dk_max)
    n_off     = len(offsets)
    avg_power  = np.zeros((H, W),       dtype=np.float64)
    avg_bispec = np.zeros((H, W, n_off), dtype=np.complex128)

    ky_arr = np.arange(H)
    kx_arr = np.arange(W)
    fy = np.where(ky_arr > H // 2, ky_arr - H, ky_arr).astype(float)
    fx = np.where(kx_arr > W // 2, kx_arr - W, kx_arr).astype(float)
    fy2d, fx2d = np.meshgrid(fy, fx, indexing='ij')
    u_mask = (fy2d**2 + fx2d**2) <= k_max**2
    u_idx  = np.argwhere(u_mask)

    for i in range(n_frames):
        F = np.fft.fft2(np.fft.ifftshift(cube[i].astype(np.float64)))
        avg_power += np.abs(F) ** 2
        for oi, (vy, vx) in enumerate(offsets):
            ivy = int(vy) % H
            ivx = int(vx) % W
            Fv  = F[ivy, ivx]
            uvy = (u_idx[:, 0] + vy) % H
            uvx = (u_idx[:, 1] + vx) % W
            Fu  = F[u_idx[:, 0], u_idx[:, 1]]
            Fuv = F[uvy, uvx]
            avg_bispec[u_idx[:, 0], u_idx[:, 1], oi] += Fu * Fv * np.conj(Fuv)
        if progress_cb is not None:
            progress_cb(int(5 + 75 * (i + 1) / n_frames))

    avg_power  /= n_frames
    avg_bispec /= n_frames
    return avg_power, avg_bispec, offsets, n_frames


def iterative_reconstruct(avg_power:  np.ndarray,
                          avg_bispec: np.ndarray,
                          offsets:    np.ndarray,
                          k_max:      int = KMAX_DEFAULT,
                          n_iter:     int = 30,
                          progress_cb = None) -> tuple:
    H, W  = avg_power.shape
    n_off = len(offsets)

    ky = np.where(np.arange(H) > H // 2, np.arange(H) - H,
                  np.arange(H)).astype(float)
    kx = np.where(np.arange(W) > W // 2, np.arange(W) - W,
                  np.arange(W)).astype(float)
    fy2d, fx2d = np.meshgrid(ky, kx, indexing='ij')
    r2d    = np.sqrt(fy2d**2 + fx2d**2)
    r_soft = 0.9 * k_max
    apod   = np.where(r2d <= r_soft, 1.0,
             np.where(r2d <  k_max,
                      0.5 * (1.0 + np.cos(np.pi * (r2d - r_soft)
                                          / (k_max - r_soft))),
                      0.0))

    noise_region = np.concatenate([avg_power[:H//8, :].ravel(),
                                   avg_power[-H//8:, :].ravel()])
    bias      = float(np.median(noise_region))
    amplitude = np.sqrt(np.maximum(avg_power - bias, 0.0)) * apod
    k_mask    = apod > 0.0

    bispec_mag = np.abs(avg_bispec)
    bispec_arg = np.angle(avg_bispec)

    ov = np.array([int(vy) % H for vy, vx in offsets], dtype=np.intp)
    ou = np.array([int(vx) % W for vy, vx in offsets], dtype=np.intp)

    row_idx = np.arange(H, dtype=np.intp)
    col_idx = np.arange(W, dtype=np.intp)

    phase = np.zeros((H, W), dtype=np.float64)

    for it in range(n_iter):
        phasor = np.zeros((H, W), dtype=np.complex128)
        wt_sum = np.zeros((H, W), dtype=np.float64)
        for oi in range(n_off):
            vy_i  = ov[oi]
            vx_i  = ou[oi]
            phi_v = phase[vy_i, vx_i]
            wy = (row_idx[:, None] + vy_i) % H
            wx = (col_idx[None, :] + vx_i) % W
            w_val = bispec_mag[:, :, oi]
            est   = phase + phi_v - bispec_arg[:, :, oi]
            np.add.at(phasor, (wy, wx), w_val * np.exp(1j * est))
            np.add.at(wt_sum, (wy, wx), w_val)
        upd = (wt_sum > 0) & k_mask
        phase[upd] = np.angle(phasor[upd])
        if progress_cb is not None:
            progress_cb(82 + int(15 * (it + 1) / n_iter))

    F_final = amplitude * np.exp(1j * phase)
    img = np.real(np.fft.ifft2(F_final))
    img = np.fft.fftshift(img)
    img = np.maximum(img, 0.0)
    return img.astype(np.float32), phase


def compute_autocorrelogram(avg_power: np.ndarray) -> np.ndarray:
    H, W = avg_power.shape
    noise_region = np.concatenate([
        avg_power[:H//8, :].ravel(),
        avg_power[-H//8:, :].ravel(),
    ])
    bias    = float(np.median(noise_region))
    debiased = np.maximum(avg_power - bias, 0.0)
    acorr   = np.real(np.fft.ifft2(debiased))
    acorr   = np.fft.fftshift(acorr)
    return acorr.astype(np.float32)


def deconvolve_bispectrum(avg_bispec_tgt: np.ndarray,
                          avg_bispec_ref: np.ndarray,
                          epsilon: float = 0.01) -> np.ndarray:
    denom_sq = np.abs(avg_bispec_ref) ** 2
    reg      = epsilon * float(denom_sq.mean()) + 1e-30
    return avg_bispec_tgt * np.conj(avg_bispec_ref) / (denom_sq + reg)



class NpzReconWorker(QThread):
    """Reconstruct from a pre-computed bispectrum .npz (no accumulation)."""
    progress = pyqtSignal(int)
    status   = pyqtSignal(str)
    finished = pyqtSignal(object)
    error    = pyqtSignal(str)

    def __init__(self, filepath: str, k_max: int, n_iter: int,
                 ref_path: str = "", ref_bispec=None,
                 use_deconv: bool = False, epsilon: float = 0.01):
        super().__init__()
        self.filepath   = filepath
        self.k_max      = k_max
        self.n_iter     = n_iter
        self.ref_path   = ref_path
        self.ref_bispec = ref_bispec
        self.use_deconv = use_deconv
        self.epsilon    = epsilon
        self._stop      = False

    def stop(self):
        self._stop = True

    def run(self):
        try:
            self._process()
        except Exception as e:
            import traceback
            self.error.emit(f"{e}\n{traceback.format_exc()}")

    def _process(self):
        self.status.emit("Loading bispectrum .npz…")
        self.progress.emit(5)

        data       = np.load(self.filepath, allow_pickle=False)
        avg_bispec = data['avg_bispec']
        avg_power  = data['avg_power']
        offsets    = data['offsets']
        H, W       = avg_power.shape
        if self._stop: return

        ref_bispec_arr = None
        deconv_done    = False

        has_ref = bool(self.ref_path or self.ref_bispec is not None)
        if has_ref:
            if self.ref_bispec is not None:
                ref_bispec_arr = self.ref_bispec
                self.status.emit("Reference bispectrum loaded from .npz.")
                self.progress.emit(40)
            else:
                self.status.emit("Loading reference bispectrum…")
                ref_data       = np.load(self.ref_path, allow_pickle=False)
                ref_bispec_arr = ref_data['avg_bispec']
                self.progress.emit(40)
            if self._stop: return
            if self.use_deconv:
                self.status.emit(
                    f"Deconvolving bispectrum  (ε = {self.epsilon:.4f})…")
                avg_bispec  = deconvolve_bispectrum(
                    avg_bispec, ref_bispec_arr, self.epsilon)
                deconv_done = True
            self.progress.emit(50)
        if self._stop: return

        self.status.emit(
            f"Iterative reconstruction ({self.n_iter} iters)…")
        self.progress.emit(55)

        def _prog(p):
            if self._stop: return
            mapped = 55 + int((p - 82) / 15 * 40)
            self.progress.emit(max(55, min(95, mapped)))
            it_done = max(0, round((p - 82) / 15 * self.n_iter))
            self.status.emit(
                f"Iterative reconstruction…  iteration {it_done}/{self.n_iter}")

        recon, phase = iterative_reconstruct(
            avg_power, avg_bispec, offsets,
            k_max=self.k_max, n_iter=self.n_iter,
            progress_cb=_prog)
        if self._stop: return

        self.progress.emit(100)
        self.status.emit("Done.")
        self.finished.emit({
            'recon':             recon,
            'avg_power':         avg_power,
            'avg_bispec':        avg_bispec,
            'offsets':           offsets,
            'n_frames':          0,   # unknown from npz alone
            'roi_size':          H,
            'mean_bispec_mag':   float(np.mean(np.abs(avg_bispec))),
            'mean_abs_phase':    float(np.mean(np.abs(phase))),
            'nonzero_phase_pct': float(np.sum(phase != 0.0)) / phase.size * 100,
            'deconv_done':       deconv_done,
            'ref_bispec':        ref_bispec_arr,
        })


# ── Analysis worker ────────────────────────────────────────────────────────

class AnalysisWorker(QThread):
    progress = pyqtSignal(int)
    status   = pyqtSignal(str)
    finished = pyqtSignal(object)
    error    = pyqtSignal(str)

    def __init__(self, filepath: str,
                 k_max:    int   = KMAX_DEFAULT,
                 dk_max:   int   = DKMAX_DEFAULT,
                 n_iter:   int   = 30,
                 ref_path: str   = "",
                 ref_bispec: object = None,
                 use_deconv: bool  = False,
                 epsilon:  float = 0.01):
        super().__init__()
        self.filepath    = filepath
        self.k_max       = k_max
        self.dk_max      = dk_max
        self.n_iter      = n_iter
        self.ref_path    = ref_path
        self.ref_bispec  = ref_bispec
        self.use_deconv  = use_deconv
        self.epsilon     = epsilon
        self._stop       = False

    def stop(self):
        self._stop = True

    def run(self):
        try:
            self._process()
        except Exception as e:
            import traceback
            self.error.emit(f"{e}\n{traceback.format_exc()}")

    def _process(self):
        has_ref    = bool(self.ref_path or self.ref_bispec is not None)
        use_deconv = has_ref and self.use_deconv

        TGT_END  = 42 if has_ref else 82
        REF_END  = 78
        ITER_END = 93 if has_ref else 97

        self.status.emit("Loading FITS cube…")
        self.progress.emit(2)
        cube, hdr = read_fits_cube(self.filepath)
        n_frames, H, W = cube.shape
        self.status.emit(
            f"Loaded {n_frames} frames  ·  {H}×{W} px  ·  dtype {cube.dtype}")
        self.progress.emit(5)
        if self._stop: return

        n_offsets = len(build_offset_list(self.dk_max))
        self.status.emit(
            f"Accumulating target bispectrum…  "
            f"Kmax={self.k_max}  dKmax={self.dk_max}  ({n_offsets} offsets)")

        def _prog_tgt(p):
            if self._stop: return
            mapped = 5 + int((p - 5) / 77 * (TGT_END - 5))
            self.progress.emit(mapped)
            self.status.emit(f"Target bispectrum…  {p-5:.0f}% of frames done")

        avg_power, avg_bispec, offsets, _ = accumulate_bispectrum(
            cube, self.k_max, self.dk_max, _prog_tgt)
        if self._stop: return

        ref_bispec_arr = None
        deconv_done    = False

        if has_ref:
            if self.ref_bispec is not None:
                ref_bispec_arr = self.ref_bispec
                self.status.emit("Reference bispectrum loaded from .npz.")
                self.progress.emit(REF_END)
            else:
                self.status.emit("Loading reference FITS cube…")
                self.progress.emit(TGT_END + 1)
                ref_cube, _ = read_fits_cube(self.ref_path)
                rn, rH, rW = ref_cube.shape
                if (rH, rW) != (H, W):
                    raise ValueError(
                        f"Reference ROI {rH}×{rW} ≠ target ROI {H}×{W}.")
                self.status.emit(
                    f"Accumulating reference bispectrum…  ({rn} frames)")

                def _prog_ref(p):
                    if self._stop: return
                    mapped = TGT_END + int((p - 5) / 77 * (REF_END - TGT_END))
                    self.progress.emit(mapped)
                    self.status.emit(
                        f"Reference bispectrum…  {p-5:.0f}% of frames done")

                _, ref_bispec_arr, _, _ = accumulate_bispectrum(
                    ref_cube, self.k_max, self.dk_max, _prog_ref)

            if self._stop: return

            if use_deconv:
                self.status.emit(
                    f"Deconvolving bispectrum  (ε = {self.epsilon:.4f})…")
                avg_bispec  = deconvolve_bispectrum(
                    avg_bispec, ref_bispec_arr, self.epsilon)
                deconv_done = True
            self.progress.emit(REF_END + 1)
        if self._stop: return

        if deconv_done:
            msg = 'Deconvolved bispectrum.'
        elif has_ref:
            msg = 'Reference bispectrum computed (no deconvolution).'
        else:
            msg = 'Accumulation complete.'
        self.status.emit(
            f"{msg}  Starting iterative reconstruction ({self.n_iter} iters)…")
        self.progress.emit(REF_END + 2 if has_ref else 82)

        iter_start = REF_END + 2 if has_ref else 82
        iter_span  = ITER_END - iter_start

        def _iter_prog(p):
            if self._stop: return
            mapped = iter_start + int((p - 82) / 15 * iter_span)
            self.progress.emit(mapped)
            it_done = max(0, round((p - 82) / 15 * self.n_iter))
            self.status.emit(
                f"Iterative reconstruction…  iteration {it_done}/{self.n_iter}")

        recon, phase = iterative_reconstruct(
            avg_power, avg_bispec, offsets,
            k_max=self.k_max, n_iter=self.n_iter,
            progress_cb=_iter_prog)
        if self._stop: return

        acorr = compute_autocorrelogram(avg_power)

        self.progress.emit(100)
        self.status.emit("Done.")
        self.finished.emit({
            'recon':             recon,
            'acorr':             acorr,
            'avg_power':         avg_power,
            'avg_bispec':        avg_bispec,
            'offsets':           offsets,
            'n_frames':          n_frames,
            'roi_size':          H,
            'fits_hdr':          hdr,
            'mean_bispec_mag':   float(np.mean(np.abs(avg_bispec))),
            'mean_abs_phase':    float(np.mean(np.abs(phase))),
            'nonzero_phase_pct': float(np.sum(phase != 0.0)) / phase.size * 100,
            'deconv_done':       deconv_done,
            'ref_bispec':        ref_bispec_arr,
        })


# ═══════════════════════════════════════════════════════════════════════════
#  Colormaps  (analysis tab)
# ═══════════════════════════════════════════════════════════════════════════

COLORMAP_NAMES = ["Grey", "Inverted", "Hot", "Rainbow", "Viridis"]
_COLORMAPS = None

def _get_colormaps():
    global _COLORMAPS
    if _COLORMAPS is None:
        def cm(stops, colors):
            return pg.ColorMap(np.array(stops, dtype=float),
                               np.array(colors, dtype=np.uint8))
        _COLORMAPS = {
            "Grey":     cm([0, 1], [[0,0,0,255],[255,255,255,255]]),
            "Inverted": cm([0, 1], [[255,255,255,255],[0,0,0,255]]),
            "Hot":      cm([0, 0.33, 0.66, 1],
                           [[0,0,0,255],[200,0,0,255],[255,180,0,255],[255,255,255,255]]),
            "Rainbow":  cm([0, 0.25, 0.5, 0.75, 1],
                           [[0,0,180,255],[0,200,255,255],[0,220,0,255],
                            [255,200,0,255],[220,0,0,255]]),
            "Viridis":  cm([0, 0.25, 0.5, 0.75, 1],
                           [[68,1,84,255],[59,82,139,255],[33,145,140,255],
                            [94,201,98,255],[253,231,37,255]]),
        }
    return _COLORMAPS


# ═══════════════════════════════════════════════════════════════════════════
#  ── PREPROCESS TAB ──────────────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════

class PreprocessTab(QWidget):

    def __init__(self, parent=None):
        super().__init__(parent)
        self.worker:       Optional[PreprocessWorker] = None
        self._result:      Optional[dict] = None
        self._scores:      Optional[np.ndarray] = None
        self._all_crops:   list = []
        self._sel_mask:    Optional[np.ndarray] = None
        self._file_loaded: bool = False
        self._file_type:   str = 'ser'
        # batch
        self._queue:       list = []   # list of (filepath, file_type) tuples
        self._queue_total: int = 0
        self._queue_done:  int = 0
        self._build_ui()

    # ── UI ─────────────────────────────────────────────────────────────────

    def _build_ui(self):
        root = QHBoxLayout(self)
        root.setSpacing(0)
        root.setContentsMargins(0, 0, 0, 0)

        # ── Left panel ─────────────────────────────────────────────────────
        left = QWidget()
        left.setFixedWidth(380)
        left.setStyleSheet(
            f"background:{PANEL_BG}; border-right:1px solid {BORDER_COLOR};")
        self._left_panel = left
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(14, 14, 14, 14)
        left_layout.setSpacing(10)

        title = QLabel("SPECKLE PREPROCESSING")
        title.setStyleSheet(
            f"font-size:14px; font-weight:bold; color:{ACCENT}; letter-spacing:2px;")
        self._title_lbl = title
        subtitle = QLabel("Frame selection · ROI extraction · Recentring")
        subtitle.setStyleSheet(f"color:{TEXT_MUTED}; font-size:10px;")
        self._subtitle_lbl = subtitle
        left_layout.addWidget(title)
        left_layout.addWidget(subtitle)

        sep = QFrame(); sep.setObjectName("separator")
        sep.setFrameShape(QFrame.Shape.HLine)
        left_layout.addWidget(sep)

        # Input group
        input_group  = QGroupBox("Input")
        input_layout = QGridLayout(input_group)
        input_layout.setVerticalSpacing(8)
        input_layout.setHorizontalSpacing(8)

        def _muted(txt):
            l = QLabel(txt); l.setStyleSheet(f"color:{TEXT_MUTED};"); return l

        self.file_edit = QLineEdit()
        self.file_edit.setPlaceholderText("No files selected…")
        self.file_edit.setReadOnly(True)
        browse_btn = QPushButton("Browse")
        browse_btn.setMinimumWidth(80)
        browse_btn.clicked.connect(self._browse_file)
        input_layout.addWidget(_muted("Files"),     0, 0)
        input_layout.addWidget(self.file_edit,      0, 1)
        input_layout.addWidget(browse_btn,          0, 2)

        self.file_info_lbl = QLabel("")
        self.file_info_lbl.setStyleSheet(f"color:{TEXT_MUTED}; font-size:10px;")
        input_layout.addWidget(self.file_info_lbl,  1, 0, 1, 3)

        self.target_edit = QLineEdit()
        self.target_edit.setPlaceholderText("Optional label for FITS header…")
        input_layout.addWidget(_muted("Target"),    2, 0)
        input_layout.addWidget(self.target_edit,    2, 1, 1, 2)

        input_layout.setColumnStretch(1, 1)
        left_layout.addWidget(input_group)

        # Parameters group
        param_group  = QGroupBox("Parameters")
        param_layout = QGridLayout(param_group)
        param_layout.setVerticalSpacing(10)
        param_layout.setHorizontalSpacing(8)

        def param_lbl(text, tooltip=""):
            l = QLabel(text)
            l.setStyleSheet(f"color:{TEXT_MUTED};")
            if tooltip: l.setToolTip(tooltip)
            return l

        # Reject % slider
        pct_widget = QWidget()
        pct_row    = QHBoxLayout(pct_widget)
        pct_row.setContentsMargins(0, 0, 0, 0)
        pct_row.setSpacing(8)
        self.pct_slider = QSlider(Qt.Orientation.Horizontal)
        self.pct_slider.setRange(0, 50)
        self.pct_slider.setValue(10)
        self.pct_slider.setEnabled(False)
        self.pct_lbl = QLabel("10 %")
        self.pct_lbl.setStyleSheet(f"color:{ACCENT}; min-width:40px;")
        self.pct_lbl.setAlignment(
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self.pct_slider.valueChanged.connect(
            lambda v: self.pct_lbl.setText(f"{v} %"))
        pct_row.addWidget(self.pct_slider)
        pct_row.addWidget(self.pct_lbl)
        param_layout.addWidget(param_lbl("Reject worst",
            "Percentage of frames to discard (lowest RMS contrast).\n"
            "5–10 % recommended; 0 % keeps everything."), 0, 0)
        param_layout.addWidget(pct_widget, 0, 1)

        self.roi_combo = QComboBox()
        self.roi_combo.addItems(["32 × 32", "64 × 64", "128 × 128"])
        self.roi_combo.setCurrentIndex(0)
        self.roi_combo.setMinimumHeight(28)
        param_layout.addWidget(param_lbl("ROI size",
            "Square ROI centred on the brightest source"), 1, 0)
        param_layout.addWidget(self.roi_combo, 1, 1)

        out_widget = QWidget()
        out_row    = QHBoxLayout(out_widget)
        out_row.setContentsMargins(0, 0, 0, 0)
        out_row.setSpacing(8)
        self.out_edit = QLineEdit()
        self.out_edit.setPlaceholderText("Output directory…")
        self.out_edit.setMinimumHeight(26)
        out_btn = QPushButton("…")
        out_btn.setFixedWidth(32)
        out_btn.setMinimumHeight(26)
        out_btn.clicked.connect(self._choose_output)
        out_row.addWidget(self.out_edit)
        out_row.addWidget(out_btn)
        param_layout.addWidget(param_lbl("Output dir"), 2, 0)
        param_layout.addWidget(out_widget, 2, 1)
        param_layout.setColumnStretch(1, 1)
        left_layout.addWidget(param_group)

        # Run / Stop
        run_row = QHBoxLayout()
        run_row.setSpacing(6)
        self.run_btn  = QPushButton("▶  Run Preprocessing")
        self.run_btn.setStyleSheet(_primary_btn_style())
        self.run_btn.setFixedHeight(38)
        self.run_btn.setEnabled(False)
        self.run_btn.clicked.connect(self._run)
        self.stop_btn = QPushButton("■  Stop")
        self.stop_btn.setStyleSheet(_primary_btn_style())
        self.stop_btn.setFixedHeight(38)
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self._kill_worker)
        run_row.addWidget(self.run_btn, 3)
        run_row.addWidget(self.stop_btn, 1)
        left_layout.addLayout(run_row)

        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        left_layout.addWidget(self.progress_bar)

        self.status_lbl = QLabel("Load a SER or FITS file to begin.")
        self.status_lbl.setStyleSheet(f"color:{TEXT_MUTED}; font-size:10px;")
        self.status_lbl.setWordWrap(True)
        left_layout.addWidget(self.status_lbl)

        sep2 = QFrame(); sep2.setObjectName("separator")
        sep2.setFrameShape(QFrame.Shape.HLine)
        left_layout.addWidget(sep2)

        # Result cards
        cards_lbl = QLabel("RESULTS")
        cards_lbl.setStyleSheet(
            f"color:{TEXT_MUTED}; font-size:10px; letter-spacing:1px;")
        left_layout.addWidget(cards_lbl)

        cards_row1 = QHBoxLayout()
        self.card_total    = ResultCard("Total frames", "")
        self.card_selected = ResultCard("Selected",     "")
        self.card_pct_kept = ResultCard("Frames used",  "%")
        for c in (self.card_total, self.card_selected, self.card_pct_kept):
            cards_row1.addWidget(c)
        left_layout.addLayout(cards_row1)

        cards_row2 = QHBoxLayout()
        self.card_roi      = ResultCard("ROI",         "px")
        self.card_thresh   = ResultCard("Q threshold", "")
        self.card_maxshift = ResultCard("Max shift",   "px")
        for c in (self.card_roi, self.card_thresh, self.card_maxshift):
            cards_row2.addWidget(c)
        left_layout.addLayout(cards_row2)

        left_layout.addStretch()

        log_group  = QGroupBox("Log")
        log_layout = QVBoxLayout(log_group)
        self.log_edit = QTextEdit()
        self.log_edit.setReadOnly(True)
        self.log_edit.setFixedHeight(120)
        log_layout.addWidget(self.log_edit)
        left_layout.addWidget(log_group)

        root.addWidget(left)

        # ── Right panel ────────────────────────────────────────────────────
        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(10, 10, 10, 10)
        right_layout.setSpacing(8)

        preview_group  = QGroupBox("Best Frame Preview  (ROI)")
        preview_layout = QVBoxLayout(preview_group)
        self.preview_plot = pg.ImageView()
        self.preview_plot.ui.roiBtn.hide()
        self.preview_plot.ui.menuBtn.hide()
        self.preview_plot.ui.histogram.hide()
        preview_layout.addWidget(self.preview_plot)

        def _labeled_slider(label_txt, lo, hi, val, enabled=True):
            row = QHBoxLayout()
            row.setSpacing(6)
            lbl = QLabel(label_txt)
            lbl.setStyleSheet(f"color:{TEXT_MUTED}; font-size:10px; min-width:34px;")
            lbl.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            sl = QSlider(Qt.Orientation.Horizontal)
            sl.setRange(lo, hi)
            sl.setValue(val)
            sl.setEnabled(enabled)
            row.addWidget(lbl)
            row.addWidget(sl, 1)
            return row, sl

        # Level sliders (0–65535 range covers both 8-bit and 16-bit data)
        level_min_row, self.prev_min_slider = _labeled_slider("Min", 0, 65535, 0)
        level_max_row, self.prev_max_slider = _labeled_slider("Max", 0, 65535, 65535)
        frame_row,     self.frame_slider    = _labeled_slider("Frame", 0, 0, 0, enabled=False)

        self.frame_info_lbl = QLabel("—")
        self.frame_info_lbl.setStyleSheet(
            f"color:{TEXT_MUTED}; font-size:11px;")
        self.frame_info_lbl.setAlignment(
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        frame_row.addWidget(self.frame_info_lbl)

        self.frame_slider.valueChanged.connect(self._on_frame_slider)

        def _on_level_slider(_):
            lo = self.prev_min_slider.value()
            hi = self.prev_max_slider.value()
            if lo >= hi:
                return
            self.preview_plot.setLevels(lo, hi)
        self.prev_min_slider.valueChanged.connect(_on_level_slider)
        self.prev_max_slider.valueChanged.connect(_on_level_slider)

        preview_layout.addLayout(level_min_row)
        preview_layout.addLayout(level_max_row)
        preview_layout.addLayout(frame_row)

        right_layout.addWidget(preview_group)
        root.addWidget(right)

        self._apply_graph_theme()

    # ── Graph theme ────────────────────────────────────────────────────────

    def _apply_graph_theme(self):
        self.preview_plot.setStyleSheet(f"background:{DARK_BG};")

    def refresh_styles(self):
        """Called by main window after a theme change."""
        self.run_btn.setStyleSheet(_primary_btn_style())
        self.stop_btn.setStyleSheet(_primary_btn_style())
        self._apply_graph_theme()
        self._left_panel.setStyleSheet(
            f"background:{PANEL_BG}; border-right:1px solid {BORDER_COLOR};")
        self._title_lbl.setStyleSheet(
            f"font-size:14px; font-weight:bold; color:{ACCENT}; letter-spacing:2px;")
        self._subtitle_lbl.setStyleSheet(
            f"color:{TEXT_MUTED}; font-size:10px;")
        for card in (self.card_total, self.card_selected, self.card_pct_kept,
                     self.card_roi, self.card_thresh, self.card_maxshift):
            card.refresh_style()

    # ── File browsing ──────────────────────────────────────────────────────

    def _browse_file(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Open Speckle Sequence(s)",
            _working_dir(),
            "Speckle files (*.ser *.fit *.fits *.fts);;All Files (*)"
            )
        if not paths:
            return
        self._queue = []
        for p in paths:
            suffix = Path(p).suffix.lower()
            ftype  = 'fits' if suffix in ('.fit', '.fits', '.fts') else 'ser'
            self._queue.append((p, ftype))
        n = len(self._queue)
        if n == 1:
            self.file_edit.setText(paths[0])
        else:
            self.file_edit.setText(f"{n} files selected")
        # probe first file for info display
        self._file_type = self._queue[0][1]
        self._probe_file(self._queue[0][0])

    def _probe_file(self, path: str):
        try:
            if self._file_type == 'ser':
                with open(path, 'rb') as f:
                    hdr_bytes = f.read(178)
                hdr = parse_ser_header(hdr_bytes)
                n, h, w = hdr.frame_count, hdr.image_height, hdr.image_width
                depth   = hdr.pixel_depth
                size_mb = Path(path).stat().st_size / 1e6
                self.file_info_lbl.setText(
                    f"{n} frames  ·  {w}×{h}  ·  {depth}-bit  ·  {size_mb:.0f} MB")
            else:
                from astropy.io import fits as _fits
                with _fits.open(path, memmap=True) as hdul:
                    for hdu in hdul:
                        if hdu.data is not None and hdu.data.ndim == 3:
                            n, h, w = hdu.data.shape
                            depth   = hdu.data.dtype.itemsize * 8
                            size_mb = Path(path).stat().st_size / 1e6
                            self.file_info_lbl.setText(
                                f"{n} frames  ·  {w}×{h}  ·  {depth}-bit  ·  {size_mb:.0f} MB")
                            break

            # Default output dir = same directory as first file
            if not self.out_edit.text().strip():
                self.out_edit.setText(str(Path(path).parent))
            self._file_loaded = True
            self.run_btn.setEnabled(True)
            self.pct_slider.setEnabled(True)
            n = len(self._queue)
            label = f"{n} file{'s' if n != 1 else ''} selected" if n > 1 else Path(path).name
            self._log(f"Loaded: {label}")

        except Exception as e:
            self.file_info_lbl.setText(f"Error reading file: {e}")
            self._log(f"Error probing file: {e}", error=True)

    def _choose_output(self):
        directory = QFileDialog.getExistingDirectory(
            self, "Select Output Directory",
            self.out_edit.text() or _working_dir())
        if directory:
            self.out_edit.setText(directory)

    # ── Run / Stop ─────────────────────────────────────────────────────────

    def _run(self):
        if not self._file_loaded or not self._queue:
            return
        out_dir = self.out_edit.text().strip()
        if not out_dir:
            self._log("Please specify an output directory.", error=True)
            return

        roi_size   = int(self.roi_combo.currentText().split()[0])
        reject_pct = float(self.pct_slider.value())
        best_pct   = 100.0 - reject_pct

        self._queue_total = len(self._queue)
        self._queue_done  = 0
        self._pending     = list(self._queue)   # working copy

        self.run_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.progress_bar.setValue(0)

        n = self._queue_total
        self._log("═" * 40)
        self._log(f"Batch: {n} file{'s' if n != 1 else ''}  "
                  f"· reject {reject_pct:.0f}%  · ROI {roi_size}×{roi_size} px")
        self._log(f"Output dir: {out_dir}")

        self._roi_size = roi_size
        self._best_pct = best_pct
        self._out_dir  = out_dir
        self._launch_next()

    def _launch_next(self):
        if not self._pending:
            return
        filepath, file_type = self._pending.pop(0)
        self._queue_done += 1
        n = self._queue_total

        out_path = str(Path(self._out_dir) /
                       (Path(filepath).stem + ".fits"))

        self.frame_slider.setEnabled(False)
        self.frame_slider.setValue(0)
        self.frame_info_lbl.setText("—")
        self.frame_info_lbl.setStyleSheet(f"color:{TEXT_MUTED}; font-size:11px;")
        self._scores    = None
        self._result    = None
        self._all_crops = []
        self._sel_mask  = None
        for card in (self.card_total, self.card_selected, self.card_pct_kept,
                     self.card_roi, self.card_thresh, self.card_maxshift):
            card.set_value("—")

        self._log("─" * 40)
        self._log(f"[{self._queue_done}/{n}]  {Path(filepath).name}")
        self.status_lbl.setText(
            f"File {self._queue_done} / {n}  —  {Path(filepath).name}")

        self.worker = PreprocessWorker(
            filepath    = filepath,
            file_type   = file_type,
            best_pct    = self._best_pct,
            roi_size    = self._roi_size,
            output_path = out_path,
        )
        self.worker.progress.connect(self.progress_bar.setValue)
        self.worker.status.connect(self._on_status)
        self.worker.preview.connect(self._on_preview)
        self.worker.quality.connect(self._on_quality)
        self.worker.finished.connect(self._on_finished)
        self.worker.error.connect(self._on_error)
        self.worker.finished.connect(lambda _: self._after_file())
        self.worker.error.connect(lambda _: self._after_file())
        self.worker.start()

    def _after_file(self):
        if self._pending:
            self._launch_next()
        else:
            self._log("═" * 40)
            self._log(f"✓ Batch complete — {self._queue_total} file(s) processed.")
            self.status_lbl.setText(
                f"Done — {self._queue_total} file(s) processed.")
            self._cleanup_worker()

    def _kill_worker(self):
        if self.worker and self.worker.isRunning():
            self.worker.stop()
        self._pending = []
        self._cleanup_worker()
        self.status_lbl.setText("Stopped.")
        self._log("■ Batch stopped by user.")

    def _cleanup_worker(self):
        self.run_btn.setEnabled(self._file_loaded)
        self.stop_btn.setEnabled(False)

    # ── Worker callbacks ───────────────────────────────────────────────────

    def _on_status(self, msg: str):
        self.status_lbl.setText(msg)
        self._log(msg)

    def _on_error(self, msg: str):
        self._log(f"ERROR: {msg}", error=True)
        self.status_lbl.setText("Error — see log.")

    def _on_preview(self, roi: np.ndarray):
        self.preview_plot.setImage(roi.T, autoLevels=True, autoRange=True)
        lo, hi = self.preview_plot.levelMin, self.preview_plot.levelMax
        self.prev_min_slider.blockSignals(True)
        self.prev_max_slider.blockSignals(True)
        self.prev_min_slider.setValue(int(lo))
        self.prev_max_slider.setValue(int(hi))
        self.prev_min_slider.blockSignals(False)
        self.prev_max_slider.blockSignals(False)

    def _on_quality(self, scores: np.ndarray):
        self._scores = scores

    def _on_finished(self, result: dict):
        self._result    = result
        self._scores    = result['scores']
        self._all_crops = result['all_crops']
        self._sel_mask  = result['sel_mask']

        n_total  = result['n_total']
        n_sel    = result['n_selected']
        kept_pct = 100.0 * n_sel / max(n_total, 1)
        self.card_total.set_value(str(n_total))
        self.card_selected.set_value(str(n_sel))
        self.card_pct_kept.set_value(f"{kept_pct:.1f}")
        self.card_roi.set_value(str(result['roi_size']))
        self.card_thresh.set_value(f"{result['threshold']:.4f}")
        self.card_maxshift.set_value(f"{result['max_shift']:.2f}")

        self.frame_slider.setMaximum(n_total - 1)
        self.frame_slider.setValue(int(np.argmax(self._scores)))
        self.frame_slider.setEnabled(True)

        self._log("─" * 40)
        self._log(f"✓ Complete — {n_sel}/{n_total} frames kept ({kept_pct:.1f}%)")
        self._log(f"  Max centroid shift: {result['max_shift']:.2f} px")
        self._log(f"  Written: {Path(result['output_path']).name}")

    def _on_frame_slider(self, idx: int):
        if not self._all_crops or idx >= len(self._all_crops):
            return
        crop     = self._all_crops[idx]
        accepted = bool(self._sel_mask[idx])
        score    = float(self._scores[idx])

        display = crop.copy()
        if not accepted:
            h, w  = display.shape
            vmax  = float(display.max())
            size  = max(8, h // 8)
            thick = max(1, h // 48)
            r0 = h - size - 4
            c0 = w - size - 4
            for t in range(size):
                for k in range(-thick, thick + 1):
                    r, c = r0 + t + k, c0 + t
                    if 0 <= r < h and 0 <= c < w:
                        display[r, c] = vmax
                    r, c = r0 + t + k, c0 + size - 1 - t
                    if 0 <= r < h and 0 <= c < w:
                        display[r, c] = vmax

        self.preview_plot.setImage(display.T, autoLevels=True, autoRange=False)
        lo, hi = self.preview_plot.levelMin, self.preview_plot.levelMax
        self.prev_min_slider.blockSignals(True)
        self.prev_max_slider.blockSignals(True)
        self.prev_min_slider.setValue(int(lo))
        self.prev_max_slider.setValue(int(hi))
        self.prev_min_slider.blockSignals(False)
        self.prev_max_slider.blockSignals(False)
        status = "✓ accepted" if accepted else "✗ rejected"
        color  = ACCENT2 if accepted else DANGER
        self.frame_info_lbl.setText(
            f"Frame {idx + 1} / {len(self._all_crops)}   "
            f"Q = {score:.4f}   {status}")
        self.frame_info_lbl.setStyleSheet(
            f"color:{color}; font-size:11px; min-width:280px;")

    # ── Log ────────────────────────────────────────────────────────────────

    def _log(self, msg: str, error: bool = False, warning: bool = False):
        if error:
            color = DANGER
        elif warning:
            color = WARNING
        else:
            color = TEXT_MUTED
        self.log_edit.append(f'<span style="color:{color}">{msg}</span>')


# ═══════════════════════════════════════════════════════════════════════════
#  ── ANALYSIS TAB ────────────────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════

class ClickableLineEdit(QLineEdit):
    clicked = pyqtSignal()
    def mousePressEvent(self, event):
        super().mousePressEvent(event)
        self.clicked.emit()


class ClickableImageView(pg.ImageView):
    clicked = pyqtSignal(float, float)

    def mousePressEvent(self, ev):
        if ev.button() == Qt.MouseButton.LeftButton:
            vb  = self.getView()
            pos = vb.mapSceneToView(ev.position())
            self.clicked.emit(pos.x(), pos.y())
        super().mousePressEvent(ev)


class AnalysisTab(QWidget):

    def __init__(self, parent=None):
        super().__init__(parent)
        self.worker:   Optional[AnalysisWorker] = None
        self._result:  Optional[dict] = None
        self._primary_marker   = None
        self._companion_marker = None
        self._primary_pos:   Optional[tuple] = None
        self._companion_pos: Optional[tuple] = None
        self._click_mode = 'primary'
        self._meas_rho:   Optional[float] = None
        self._meas_theta: Optional[float] = None
        self._ref_bispec:  object = None
        self._input_type:  str   = 'fits'   # 'fits' | 'npz'
        self._cal_file:    str   = ""
        self._cal:        dict  = {}
        self._csv_path:   str   = ""
        self._meas_rho_sky:    Optional[float] = None
        self._meas_theta_sky:  Optional[float] = None
        self._meas_sigma_rho:  float = 0.0
        self._meas_sigma_theta: float = 0.0
        # batch
        self._queue:        list = []
        self._queue_total:  int  = 0
        self._queue_done:   int  = 0
        self._current_path: str  = ""
        # npz navigator
        self._nav_paths:    list = []
        self._nav_idx:      int  = 0
        self._nav_memory:   dict = {}
        self._build_ui()

    # ── UI ─────────────────────────────────────────────────────────────────

    def _build_ui(self):
        central = QWidget()
        root    = QSplitter(Qt.Orientation.Horizontal, central)
        outer   = QHBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(root)

        # ── Left panel ─────────────────────────────────────────────────────
        left = QWidget()
        left.setFixedWidth(420)
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(10, 10, 10, 10)
        left_layout.setSpacing(8)

        # Input
        input_group  = QGroupBox("Input")
        input_layout = QGridLayout(input_group)
        input_layout.setVerticalSpacing(6)
        input_layout.setHorizontalSpacing(8)
        input_layout.addWidget(QLabel("FITS / .npz"), 0, 0)
        self.file_edit = QLineEdit()
        self.file_edit.setPlaceholderText("Select FITS cube(s) or bispectrum .npz…")
        self.file_edit.setReadOnly(True)
        browse_btn = QPushButton("Browse")
        browse_btn.setMinimumWidth(80)
        browse_btn.clicked.connect(self._browse_file)
        input_layout.addWidget(self.file_edit, 0, 1)
        input_layout.addWidget(browse_btn,     0, 2)
        self.file_info_lbl = QLabel("")
        self.file_info_lbl.setStyleSheet(f"color:{TEXT_MUTED}; font-size:10px;")
        input_layout.addWidget(self.file_info_lbl, 1, 0, 1, 3)
        input_layout.setColumnStretch(1, 1)
        left_layout.addWidget(input_group)

        # Reference deconvolution
        ref_group  = QGroupBox("Reference Deconvolution")
        ref_vbox   = QVBoxLayout(ref_group)
        ref_vbox.setSpacing(4)
        ref_vbox.setContentsMargins(8, 6, 8, 6)
        ref_layout = QGridLayout()
        ref_layout.setVerticalSpacing(4)
        ref_layout.setHorizontalSpacing(6)
        ref_vbox.addLayout(ref_layout)

        ref_layout.addWidget(QLabel("Ref. .npz"), 0, 0)
        self.ref_edit = ClickableLineEdit()
        self.ref_edit.setPlaceholderText("Click to select reference bispectrum .npz…")
        self.ref_edit.setReadOnly(True)
        self.ref_edit.setCursor(Qt.CursorShape.PointingHandCursor)
        self.ref_edit.clicked.connect(self._browse_ref)
        ref_clear_btn = QPushButton("✕")
        ref_clear_btn.setFixedWidth(28)
        ref_clear_btn.setToolTip("Clear reference file")
        ref_clear_btn.clicked.connect(self._clear_ref)
        ref_layout.addWidget(self.ref_edit,    0, 1)
        ref_layout.addWidget(ref_clear_btn,    0, 2)

        ref_layout.addWidget(QLabel("ε (Wiener)"), 1, 0)
        self.epsilon_spin = QDoubleSpinBox()
        self.epsilon_spin.setRange(0.001, 0.5)
        self.epsilon_spin.setDecimals(3)
        self.epsilon_spin.setValue(0.01)
        self.epsilon_spin.setSingleStep(0.005)
        self.epsilon_spin.setFixedWidth(72)
        self.epsilon_spin.setEnabled(False)
        self.epsilon_spin.setToolTip(
            "Wiener regularisation factor ε.\n"
            "Higher = more regularisation.  Typical: 0.005–0.05.")
        ref_layout.addWidget(self.epsilon_spin, 1, 1, 1, 2)
        ref_layout.setColumnStretch(1, 1)

        self.ref_info_lbl = QLabel("")
        self.ref_info_lbl.setStyleSheet(f"color:{TEXT_MUTED}; font-size:9px;")
        self.ref_info_lbl.setWordWrap(True)
        self.ref_info_lbl.setVisible(False)
        ref_vbox.addWidget(self.ref_info_lbl)
        left_layout.addWidget(ref_group)

        # K-space / bispectrum parameters
        kspace_group = QGroupBox("Bispectrum Calculation")
        kspace_vbox  = QVBoxLayout(kspace_group)
        kspace_vbox.setSpacing(6)
        kspace_row   = QHBoxLayout()
        kspace_row.setSpacing(6)
        kspace_vbox.addLayout(kspace_row)

        def _kfield(label_txt, widget):
            col = QVBoxLayout()
            col.setSpacing(2)
            lbl = QLabel(label_txt)
            lbl.setStyleSheet(f"color:{TEXT_MUTED}; font-size:9px;")
            lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            col.addWidget(lbl); col.addWidget(widget)
            kspace_row.addLayout(col, 1)

        self.kmax_spin = QSpinBox()
        self.kmax_spin.setRange(4, 512)
        self.kmax_spin.setValue(KMAX_DEFAULT)
        self.kmax_spin.setSuffix(" px")
        self.kmax_spin.setMinimumHeight(28)
        self.kmax_spin.setToolTip("Maximum spatial frequency radius (pixels)")

        self.dkmax_spin = QSpinBox()
        self.dkmax_spin.setRange(1, 64)
        self.dkmax_spin.setValue(DKMAX_DEFAULT)
        self.dkmax_spin.setSuffix(" px")
        self.dkmax_spin.setMinimumHeight(28)
        self.dkmax_spin.setToolTip("Maximum bispectrum offset radius (pixels)")

        self.niter_spin = QSpinBox()
        self.niter_spin.setRange(1, 200)
        self.niter_spin.setValue(30)
        self.niter_spin.setMinimumHeight(28)
        self.niter_spin.setToolTip(
            "Number of iterative phase-retrieval cycles.\n"
            "Typical range: 20–60.")

        _kfield("Kmax",       self.kmax_spin)
        _kfield("dKmax",      self.dkmax_spin)
        _kfield("Iterations", self.niter_spin)

        # Run / Stop inside the group
        run_row = QHBoxLayout()
        run_row.setSpacing(6)
        self.run_btn = QPushButton("▶  Run Analysis")
        self.run_btn.setStyleSheet(_primary_btn_style())
        self.run_btn.setFixedHeight(38)
        self.run_btn.setEnabled(False)
        self.run_btn.clicked.connect(self._run)
        self.stop_btn = QPushButton("■  Stop")
        self.stop_btn.setStyleSheet(_primary_btn_style())
        self.stop_btn.setFixedHeight(38)
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self._kill_worker)
        run_row.addWidget(self.run_btn, 3)
        run_row.addWidget(self.stop_btn, 1)
        kspace_vbox.addLayout(run_row)

        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        kspace_vbox.addWidget(self.progress_bar)

        self.save_bispec_btn = QPushButton("💾  Save Bispectrum (.npz)")
        self.save_bispec_btn.setEnabled(False)
        self.save_bispec_btn.setToolTip(
            "Save the computed target bispectrum to .npz.\n"
            "If a reference FITS was processed this run, the reference\n"
            "bispectrum is also saved (key: 'ref_bispec') for later reuse.")
        self.save_bispec_btn.clicked.connect(self._save_bispec)
        kspace_vbox.addWidget(self.save_bispec_btn)

        self.status_lbl = QLabel("")
        self.status_lbl.setStyleSheet(f"color:{TEXT_MUTED}; font-size:10px;")
        self.status_lbl.setWordWrap(True)
        kspace_vbox.addWidget(self.status_lbl)

        left_layout.addWidget(kspace_group)

        # Calibration
        cal_group  = QGroupBox("Astrometric Calibration")
        cal_layout = QVBoxLayout(cal_group)
        cal_layout.setSpacing(5)

        cal_file_row = QHBoxLayout()
        self.cal_edit = QLineEdit()
        self.cal_edit.setPlaceholderText("drift_calibration.json")
        self.cal_edit.setReadOnly(True)
        self.cal_edit.setMinimumHeight(26)
        cal_browse_btn = QPushButton("Browse")
        cal_browse_btn.setMinimumWidth(70)
        cal_browse_btn.setMinimumHeight(26)
        cal_browse_btn.clicked.connect(self._load_cal_dialog)
        cal_file_row.addWidget(self.cal_edit, 1)
        cal_file_row.addWidget(cal_browse_btn)
        cal_layout.addLayout(cal_file_row)

        self.cal_status_lbl = QLabel(
            "No calibration loaded — sky coords unavailable.")
        self.cal_status_lbl.setStyleSheet(f"color:{TEXT_MUTED}; font-size:9px;")
        self.cal_status_lbl.setWordWrap(True)
        cal_layout.addWidget(self.cal_status_lbl)

        cal_sep = QFrame(); cal_sep.setFrameShape(QFrame.Shape.HLine)
        cal_sep.setStyleSheet(f"color:{BORDER_COLOR};")
        cal_layout.addWidget(cal_sep)

        def _cal_row(label_txt, lo, hi, decimals, default, has_err):
            row_w = QWidget()
            row   = QHBoxLayout(row_w)
            row.setContentsMargins(0, 0, 0, 0); row.setSpacing(4)
            lbl = QLabel(label_txt)
            lbl.setFixedWidth(88)
            lbl.setStyleSheet("font-size:10px;")
            spin = QDoubleSpinBox()
            spin.setRange(lo, hi); spin.setDecimals(decimals)
            spin.setValue(default); spin.setMinimumHeight(24)
            row.addWidget(lbl); row.addWidget(spin, 2)
            err = None
            if has_err:
                pm = QLabel("±"); pm.setFixedWidth(12)
                pm.setAlignment(Qt.AlignmentFlag.AlignCenter)
                err = QDoubleSpinBox()
                err.setRange(0, hi); err.setDecimals(decimals)
                err.setValue(0.0); err.setMinimumHeight(24)
                row.addWidget(pm); row.addWidget(err, 1)
            return row_w, spin, err

        row_scale, self.cal_scale_spin, self.cal_scale_err = _cal_row(
            "Pixel scale", 0.0001, 10.0, 6, 0.065, True)
        row_angle, self.cal_angle_spin, self.cal_angle_err = _cal_row(
            "Camera angle", 0.0, 360.0, 4, 0.0, True)

        for sp in (self.cal_scale_spin, self.cal_scale_err,
                   self.cal_angle_spin, self.cal_angle_err):
            sp.valueChanged.connect(self._update_measurement)

        cal_layout.addWidget(row_scale)
        cal_layout.addWidget(row_angle)
        left_layout.addWidget(cal_group)

        # Results
        cards_group  = QGroupBox("Results")
        cards_layout = QVBoxLayout(cards_group)
        cards_layout.setSpacing(3)

        def _result_row(label_txt, unit_txt):
            row_w = QWidget()
            row   = QHBoxLayout(row_w)
            row.setContentsMargins(0, 0, 0, 0); row.setSpacing(4)
            lbl = QLabel(label_txt)
            lbl.setFixedWidth(88); lbl.setStyleSheet("font-size:10px;")
            val = QLabel("—")
            val.setStyleSheet(f"font-size:10px; color:{TEXT_PRIMARY};")
            val.setAlignment(
                Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
            unit = QLabel(unit_txt)
            unit.setFixedWidth(20)
            unit.setStyleSheet(f"font-size:10px; color:{TEXT_MUTED};")
            sig = QLabel("")
            sig.setStyleSheet(f"font-size:10px; color:{TEXT_MUTED};")
            sig.setAlignment(
                Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
            def _set_value(v, sigma=None, _val=val, _sig=sig):
                _val.setText(v)
                _sig.setText(f" ±{sigma}" if sigma else "")
            row_w.set_value = _set_value
            row.addWidget(lbl); row.addWidget(val)
            row.addWidget(unit); row.addWidget(sig, 1)
            return row_w

        self.card_theta     = _result_row("θ  (image)", "°")
        self.card_rho       = _result_row("ρ  (image)", "px")
        self.card_theta_sky = _result_row("θ  (sky)",   "°")
        self.card_rho_sky   = _result_row("ρ  (sky)",   "″")
        for w in (self.card_theta, self.card_rho,
                  self.card_theta_sky, self.card_rho_sky):
            cards_layout.addWidget(w)
        # Star placement
        detect_group  = QGroupBox("Star Placement")
        detect_layout = QVBoxLayout(detect_group)
        detect_layout.setSpacing(6)

        self._mode_group = QButtonGroup(detect_group)
        self.primary_radio   = QRadioButton("Place Primary")
        self.companion_radio = QRadioButton("Place Secondary")
        self.primary_radio.setChecked(True)
        self.primary_radio.setEnabled(False)
        self.companion_radio.setEnabled(False)
        self.primary_radio.setStyleSheet(
            f"QRadioButton {{ color: {DANGER}; font-weight: bold; }}"
            f"QRadioButton::indicator:checked {{ background: {DANGER}; "
            f"border: 2px solid {DANGER}; border-radius: 6px; }}"
            f"QRadioButton:disabled {{ color: {TEXT_MUTED}; }}")
        self.companion_radio.setStyleSheet(
            f"QRadioButton {{ color: {ACCENT2}; font-weight: bold; }}"
            f"QRadioButton::indicator:checked {{ background: {ACCENT2}; "
            f"border: 2px solid {ACCENT2}; border-radius: 6px; }}"
            f"QRadioButton:disabled {{ color: {TEXT_MUTED}; }}")
        self._mode_group.addButton(self.primary_radio)
        self._mode_group.addButton(self.companion_radio)
        self.primary_radio.toggled.connect(
            lambda checked: checked and self._set_click_mode('primary'))
        self.companion_radio.toggled.connect(
            lambda checked: checked and self._set_click_mode('companion'))
        radio_row = QHBoxLayout(); radio_row.setSpacing(8)
        radio_row.addWidget(self.primary_radio)
        radio_row.addWidget(self.companion_radio)
        detect_layout.addLayout(radio_row)

        self.clear_btn = QPushButton("Clear All")
        self.clear_btn.setEnabled(False)
        self.clear_btn.clicked.connect(self._clear_all)
        detect_layout.addWidget(self.clear_btn)

        left_layout.addWidget(detect_group)
        left_layout.addWidget(cards_group)
        left_layout.addStretch()
        root.addWidget(left)

        # ── Right panel ────────────────────────────────────────────────────
        right = QWidget()
        right_layout = QHBoxLayout(right)
        right_layout.setContentsMargins(10, 10, 10, 10)
        right_layout.setSpacing(10)

        # Reconstructed image
        recon_group  = QGroupBox("Reconstructed Image  (click to place markers)")
        recon_layout = QVBoxLayout(recon_group)
        recon_layout.setSpacing(4)

        # Navigator bar (all-npz batches only)
        self.nav_bar = QWidget()
        nav_row = QHBoxLayout(self.nav_bar)
        nav_row.setContentsMargins(4, 2, 4, 2)
        nav_row.setSpacing(6)
        self.nav_prev_btn = QPushButton("◄◄")
        self.nav_prev_btn.setFixedWidth(52)
        self.nav_prev_btn.setFixedHeight(30)
        self.nav_prev_btn.setStyleSheet(
            f"font-size:14px; font-weight:bold; padding:2px 6px;")
        self.nav_prev_btn.clicked.connect(self._nav_prev)
        self.nav_next_btn = QPushButton("►►")
        self.nav_next_btn.setFixedWidth(52)
        self.nav_next_btn.setFixedHeight(30)
        self.nav_next_btn.setStyleSheet(
            f"font-size:14px; font-weight:bold; padding:2px 6px;")
        self.nav_next_btn.clicked.connect(self._nav_next)
        self.nav_label = QLabel("")
        self.nav_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.nav_label.setStyleSheet(
            f"color:{TEXT_PRIMARY}; font-size:13px; font-weight:bold;")
        self.nav_label.setFixedWidth(60)
        self.nav_file_label = QLabel("")
        self.nav_file_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.nav_file_label.setStyleSheet(
            f"color:{TEXT_MUTED}; font-size:10px;")
        self.nav_file_label.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        nav_row.addWidget(self.nav_prev_btn)
        nav_row.addWidget(self.nav_label)
        nav_row.addWidget(self.nav_file_label, 1)
        nav_row.addWidget(self.nav_next_btn)
        self.nav_bar.setVisible(False)
        recon_layout.addWidget(self.nav_bar)

        self.recon_view = ClickableImageView()
        self.recon_view.ui.roiBtn.hide()
        self.recon_view.ui.menuBtn.hide()
        self.recon_view.ui.histogram.hide()
        self.recon_view.clicked.connect(self._on_recon_click)
        recon_layout.addWidget(self.recon_view)

        # Level sliders
        slider_grid = QGridLayout()
        slider_grid.setVerticalSpacing(2)

        def _level_slider():
            s = QSlider(Qt.Orientation.Horizontal)
            s.setRange(0, 255); s.setFixedHeight(18)
            return s

        lbl_min = QLabel("Min")
        lbl_min.setStyleSheet(f"color:{TEXT_MUTED}; font-size:9px;")
        lbl_min.setFixedWidth(24)
        lbl_max = QLabel("Max")
        lbl_max.setStyleSheet(f"color:{TEXT_MUTED}; font-size:9px;")
        lbl_max.setFixedWidth(24)
        self.level_min_slider = _level_slider()
        self.level_max_slider = _level_slider()
        self.level_max_slider.setValue(255)
        self.level_min_lbl = QLabel("0")
        self.level_min_lbl.setStyleSheet(f"color:{TEXT_MUTED}; font-size:9px;")
        self.level_min_lbl.setFixedWidth(28)
        self.level_max_lbl = QLabel("255")
        self.level_max_lbl.setStyleSheet(f"color:{TEXT_MUTED}; font-size:9px;")
        self.level_max_lbl.setFixedWidth(28)

        def _on_min_changed(v):
            if v >= self.level_max_slider.value():
                v = self.level_max_slider.value() - 1
                self.level_min_slider.setValue(v)
            self.level_min_lbl.setText(str(v))
            self.recon_view.setLevels(v, self.level_max_slider.value())

        def _on_max_changed(v):
            if v <= self.level_min_slider.value():
                v = self.level_min_slider.value() + 1
                self.level_max_slider.setValue(v)
            self.level_max_lbl.setText(str(v))
            self.recon_view.setLevels(self.level_min_slider.value(), v)

        self.level_min_slider.valueChanged.connect(_on_min_changed)
        self.level_max_slider.valueChanged.connect(_on_max_changed)

        slider_grid.addWidget(lbl_min,                0, 0)
        slider_grid.addWidget(self.level_min_slider,  0, 1)
        slider_grid.addWidget(self.level_min_lbl,     0, 2)
        slider_grid.addWidget(lbl_max,                1, 0)
        slider_grid.addWidget(self.level_max_slider,  1, 1)
        slider_grid.addWidget(self.level_max_lbl,     1, 2)
        slider_grid.setColumnStretch(1, 1)
        recon_layout.addLayout(slider_grid)

        cmap_row_r = QHBoxLayout()
        cmap_row_r.setContentsMargins(4, 0, 4, 2)
        cmap_lbl_r = QLabel("Colormap")
        cmap_lbl_r.setStyleSheet(f"color:{TEXT_MUTED}; font-size:9px;")
        self.recon_cmap_combo = QComboBox()
        self.recon_cmap_combo.addItems(COLORMAP_NAMES)
        self.recon_cmap_combo.setFixedHeight(22)
        self.recon_cmap_combo.setStyleSheet("font-size:9px;")
        self.recon_cmap_combo.currentTextChanged.connect(self._apply_recon_cmap)
        cmap_row_r.addWidget(cmap_lbl_r)
        cmap_row_r.addWidget(self.recon_cmap_combo, 1)
        recon_layout.addLayout(cmap_row_r)
        right_layout.addWidget(recon_group, 3)

        # Right column: log + save buttons
        right_col = QWidget()
        right_col_layout = QVBoxLayout(right_col)
        right_col_layout.setContentsMargins(0, 0, 0, 0)
        right_col_layout.setSpacing(8)

        log_group  = QGroupBox("Log")
        log_layout = QVBoxLayout(log_group)
        self.log_edit = QTextEdit()
        self.log_edit.setReadOnly(True)
        self.log_edit.setMinimumHeight(160)
        log_layout.addWidget(self.log_edit)
        log_group.setSizePolicy(QSizePolicy.Policy.Expanding,
                                QSizePolicy.Policy.Expanding)
        right_col_layout.addWidget(log_group, 1)

        # Save buttons below the log
        self.save_result_btn = QPushButton("💾  Save Result (.json)")
        self.save_result_btn.setEnabled(False)
        self.save_result_btn.setToolTip(
            "Save full result (pixels + sky coords + calibration) to JSON")
        self.save_result_btn.clicked.connect(self._save_result)
        right_col_layout.addWidget(self.save_result_btn)

        output_row = QHBoxLayout()
        output_row.setSpacing(6)
        self.append_csv_btn = QPushButton("CSV Log")
        self.append_csv_btn.setEnabled(False)
        self.append_csv_btn.setToolTip("Append measurement to CSV log file")
        self.append_csv_btn.clicked.connect(self._append_csv)
        self.save_wds_btn = QPushButton("WDS Report")
        self.save_wds_btn.setEnabled(False)
        self.save_wds_btn.setToolTip("Save WDS-format astrometry report (.txt)")
        self.save_wds_btn.clicked.connect(self._save_wds)
        output_row.addWidget(self.append_csv_btn)
        output_row.addWidget(self.save_wds_btn)
        right_col_layout.addLayout(output_row)

        self.csv_path_lbl = QLabel("No CSV log set.")
        self.csv_path_lbl.setStyleSheet(f"color:{TEXT_MUTED}; font-size:9px;")
        self.csv_path_lbl.setWordWrap(True)
        right_col_layout.addWidget(self.csv_path_lbl)

        right_layout.addWidget(right_col, 2)
        root.addWidget(right)
        root.setSizes([420, 980])

        self._apply_graph_theme()

    # ── Graph theme ────────────────────────────────────────────────────────

    def _apply_graph_theme(self):
        self.recon_view.setStyleSheet(f"background:{DARK_BG};")

    def refresh_styles(self):
        """Called by main window after a theme change."""
        self.run_btn.setStyleSheet(_primary_btn_style())
        self.stop_btn.setStyleSheet(_primary_btn_style())
        self._apply_graph_theme()
        # Re-apply inline color styles that were baked at build time
        for card in (self.card_theta, self.card_rho,
                     self.card_theta_sky, self.card_rho_sky):
            # card is a QWidget; its children are lbl, val, unit, sig
            children = card.findChildren(QLabel)
            for lbl in children:
                t = lbl.text()
                # value labels hold measurement text or "—"
                fs = lbl.font().pointSize()
                w  = lbl.minimumWidth()
                if w == 88:  # label column
                    lbl.setStyleSheet("font-size:10px;")
                elif w == 20:  # unit
                    lbl.setStyleSheet(f"font-size:10px; color:{TEXT_MUTED};")
                else:  # value or sigma
                    lbl.setStyleSheet(f"font-size:10px; color:{TEXT_PRIMARY};")
        self.nav_label.setStyleSheet(
            f"color:{TEXT_PRIMARY}; font-size:13px; font-weight:bold;")
        self.nav_file_label.setStyleSheet(
            f"color:{TEXT_MUTED}; font-size:10px;")
        for s_lbl in (self.level_min_lbl, self.level_max_lbl):
            s_lbl.setStyleSheet(f"color:{TEXT_MUTED}; font-size:9px;")
        self.cal_status_lbl.setStyleSheet(
            f"color:{TEXT_MUTED}; font-size:9px;")

    # ── Colormap ───────────────────────────────────────────────────────────

    def _apply_recon_cmap(self, name: str):
        cm = _get_colormaps().get(name)
        if cm is not None:
            self.recon_view.setColorMap(cm)

    # ── Reference helpers ──────────────────────────────────────────────────

    def _browse_ref(self):
        """Load a pre-computed reference bispectrum .npz.
        Deconvolution is applied automatically whenever a ref is present."""
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Reference Bispectrum",
            _working_dir(),
            "NumPy archive (*.npz);;All Files (*)"
            )
        if not path:
            return
        try:
            data = np.load(path, allow_pickle=False)
            if 'avg_bispec' not in data:
                raise KeyError("Missing 'avg_bispec' key in .npz")
            self._ref_bispec = data['avg_bispec']
            shape = self._ref_bispec.shape
            self.ref_edit.setText(path)
            self.ref_info_lbl.setText(
                f"✓ shape={shape}  dtype={self._ref_bispec.dtype}")
            self.ref_info_lbl.setVisible(True)
            self.epsilon_spin.setEnabled(True)
            self._log(
                f"Ref. bispectrum loaded: {Path(path).name}  shape={shape}")
        except Exception as e:
            self.ref_info_lbl.setText(f"⚠ {e}")
            self.ref_info_lbl.setVisible(True)
            self._log(f"⚠ Failed to load ref. bispectrum: {e}", error=True)

    def _clear_ref(self):
        """Remove the loaded reference bispectrum."""
        self._ref_bispec = None
        self.ref_edit.clear()
        self.ref_info_lbl.setVisible(False)
        self.epsilon_spin.setEnabled(False)
        self._log("Reference bispectrum cleared.")

    # ── File browsing ──────────────────────────────────────────────────────

    def _browse_file(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Open FITS Cube(s) or Bispectrum .npz",
            _working_dir(),
            "Supported files (*.fit *.fits *.fts *.npz);;"
                "FITS files (*.fit *.fits *.fts);;"
                "Bispectrum (*.npz);;All Files (*)")
        if not paths:
            return
        self._queue = list(paths)
        n = len(self._queue)
        if n == 1:
            self.file_edit.setText(paths[0])
        else:
            self.file_edit.setText(f"{n} files selected")
        # Probe first file for info
        path = paths[0]
        suffix = Path(path).suffix.lower()
        self._input_type = 'npz' if suffix == '.npz' else 'fits'
        all_npz = all(Path(p).suffix.lower() == '.npz' for p in self._queue)
        try:
            if self._input_type == 'npz':
                data = np.load(path, allow_pickle=False)
                bs   = data['avg_bispec']
                if n == 1:
                    self.file_info_lbl.setText(
                        f"bispectrum  shape={bs.shape}  dtype={bs.dtype}")
                else:
                    self.file_info_lbl.setText(
                        f"{n} bispectrum files  ·  first shape={bs.shape}")
            else:
                cube, hdr = read_fits_cube(path)
                nf, H, W  = cube.shape
                roi_info  = f"{H}×{W}"
                src_name  = hdr.get('SRCFILE', Path(path).name)
                if n == 1:
                    self.file_info_lbl.setText(
                        f"{nf} frames  ·  ROI {roi_info}  ·  src: {src_name}")
                else:
                    self.file_info_lbl.setText(
                        f"{n} files  ·  first: {nf} frames  ·  ROI {roi_info}")
                self._log(f"Loaded: {n} file(s) — first: {Path(path).name}  "
                          f"({nf} frames, {roi_info} px)")
                if H > 32 or W > 32:
                    self._log(
                        f"⚠  ROI {roi_info}: 32×32 recommended for bispectrum. "
                        f"64×64 ≈ 5 min / 1700 frames — 128×128 is impractical.",
                        warning=True)
        except Exception as e:
            self.file_info_lbl.setText(f"Error: {e}")
        # npz: reconstruct immediately, no Run needed
        if all_npz:
            self.run_btn.setEnabled(False)
            self.nav_bar.setVisible(False)
            self._nav_paths  = list(self._queue)
            self._nav_idx    = 0
            self._nav_memory = {}
            self._log(f"Loading {n} bispectrum file(s)…")
            self.status_lbl.setText(f"Reconstructing 1 / {n}…")
            self._npz_load_queue = list(self._queue)
            self._reconstruct_next_npz()
        else:
            self._nav_paths = []
            self.nav_bar.setVisible(False)
            self.run_btn.setEnabled(True)

    def _reconstruct_next_npz(self):
        """Sequentially reconstruct npz files on browse, no Run needed."""
        if not self._npz_load_queue:
            # All done — activate navigator
            n = len(self._nav_paths)
            self._nav_idx = 0
            self._log(f"✓ {n} bispectrum file(s) ready.")
            if n > 1:
                self._update_nav_bar()
                self.nav_bar.setVisible(True)
                self._log("Use ◀ ▶ to navigate.")
            # Show first file's result (already in memory from last completed)
            first = self._nav_paths[0]
            mem   = self._nav_memory.get(first)
            if mem and mem['result'] is not None:
                self._on_finished(mem['result'])
            self.status_lbl.setText(
                f"Ready — {n} file(s) loaded.")
            return

        path = self._npz_load_queue.pop(0)
        idx  = len(self._nav_paths) - len(self._npz_load_queue) - 1
        n    = len(self._nav_paths)
        self.status_lbl.setText(
            f"Reconstructing {idx + 1} / {n}  —  {Path(path).name}")
        self.progress_bar.setValue(0)

        ref_path   = self.ref_edit.text()
        ref_bispec = self._ref_bispec if ref_path else None
        worker = NpzReconWorker(
            path,
            k_max      = self.kmax_spin.value(),
            n_iter     = self.niter_spin.value(),
            ref_path   = ref_path,
            ref_bispec = ref_bispec,
            use_deconv = bool(ref_path),
            epsilon    = self.epsilon_spin.value())
        worker.progress.connect(self.progress_bar.setValue)
        worker.status.connect(self._on_status)
        worker.error.connect(self._on_error)

        def _on_done(result, p=path, i=idx):
            self._nav_memory[p] = {
                'result':        result,
                'primary_pos':   None,
                'companion_pos': None,
            }
            self._log(f"  [{i+1}/{n}] {Path(p).name}  ✓")
            self._reconstruct_next_npz()

        worker.finished.connect(_on_done)
        self.worker = worker
        worker.start()

    # ── Run / Stop ─────────────────────────────────────────────────────────

    def _run(self):
        if not self._queue:
            return
        self._queue_total = len(self._queue)
        self._queue_done  = 0
        self._pending     = list(self._queue)

        ref_path = self.ref_edit.text()
        self._log("═" * 40)
        self._log(f"Batch: {self._queue_total} file(s)")
        if ref_path:
            self._log(f"Reference: {Path(ref_path).name}  "
                      f"ε = {self.epsilon_spin.value():.3f}")

        self.run_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self._launch_next_analysis()

    def _launch_next_analysis(self):
        if not self._pending:
            return
        path = self._pending.pop(0)
        self._queue_done += 1
        n = self._queue_total

        self.progress_bar.setValue(0)
        self.primary_radio.setEnabled(False)
        self.companion_radio.setEnabled(False)
        self.clear_btn.setEnabled(False)
        self._clear_all(silent=True)
        for card in (self.card_theta, self.card_rho,
                     self.card_theta_sky, self.card_rho_sky):
            card.set_value("—")
        self._result = None
        self._current_path = path
        self._log("─" * 40)
        self._log(f"[{self._queue_done}/{n}]  {Path(path).name}")
        self.status_lbl.setText(
            f"File {self._queue_done} / {n}  —  {Path(path).name}")

        ref_path   = self.ref_edit.text()
        use_deconv = bool(ref_path)
        ref_bispec = self._ref_bispec if ref_path else None
        is_npz     = Path(path).suffix.lower() == '.npz'

        if is_npz:
            self.worker = NpzReconWorker(
                path,
                k_max      = self.kmax_spin.value(),
                n_iter     = self.niter_spin.value(),
                ref_path   = ref_path,
                ref_bispec = ref_bispec,
                use_deconv = use_deconv,
                epsilon    = self.epsilon_spin.value())
        else:
            self.worker = AnalysisWorker(
                path,
                k_max      = self.kmax_spin.value(),
                dk_max     = self.dkmax_spin.value(),
                n_iter     = self.niter_spin.value(),
                ref_path   = ref_path,
                ref_bispec = ref_bispec,
                use_deconv = use_deconv,
                epsilon    = self.epsilon_spin.value())
        self.worker.progress.connect(self.progress_bar.setValue)
        self.worker.status.connect(self._on_status)
        self.worker.finished.connect(self._on_finished)
        self.worker.error.connect(self._on_error)
        self.worker.finished.connect(lambda _: self._after_analysis())
        self.worker.error.connect(lambda _: self._after_analysis())
        self.worker.start()

    def _after_analysis(self):
        if (self._result is not None
                and self._result.get('avg_bispec') is not None
                and self._result.get('avg_power') is not None
                and self._current_path
                and Path(self._current_path).suffix.lower() != '.npz'):
            self._autosave_npz(self._current_path, self._result)
        if self._nav_paths and self._result is not None:
            _slot = self._queue_done - 1
            if 0 <= _slot < len(self._nav_paths):
                self._nav_memory[self._nav_paths[_slot]] = {
                    'result':        self._result,
                    'primary_pos':   self._primary_pos,
                    'companion_pos': self._companion_pos,
                }
                self._nav_idx = _slot
        if self._pending:
            if self._nav_paths:
                pass  # _nav_idx updated above; will advance on next _after_analysis
            self._launch_next_analysis()
        else:
            self._log("═" * 40)
            self._log(f"✓ Batch complete — {self._queue_total} file(s) analysed.")
            self.status_lbl.setText(
                f"Done — {self._queue_total} file(s) analysed.")
            self.run_btn.setEnabled(True)
            self.stop_btn.setEnabled(False)
            if self._nav_paths and len(self._nav_paths) > 1:
                # _nav_idx still points to the last file — correct
                self._update_nav_bar()
                self.nav_bar.setVisible(True)
                self._log(f"Navigator ready — {len(self._nav_paths)} files. "  
                          f"Use ◄ ► to navigate.")

    def _kill_worker(self):
        if self.worker:
            self.worker.stop()
        self._pending = []
        self.stop_btn.setEnabled(False)
        self.run_btn.setEnabled(True)
        self.status_lbl.setText("Stopped.")
        self._log("■ Batch stopped by user.")

    # ── Worker callbacks ───────────────────────────────────────────────────

    def _on_status(self, msg: str):
        self.status_lbl.setText(msg)

    def _on_error(self, msg: str):
        self._log(f"ERROR: {msg}", error=True)

    def _autosave_npz(self, source_path: str, result: dict):
        """Auto-save bispectrum .npz alongside the source FITS file."""
        try:
            p        = Path(source_path)
            out_path = Path(_working_dir()) / (p.stem + '_bispec.npz')
            save_dict = {
                'avg_bispec': result['avg_bispec'],
                'avg_power':  result['avg_power'],
                'offsets':    result.get('offsets',
                              np.zeros((0, 2), dtype=np.int32)),
            }
            ref_arr = result.get('ref_bispec')
            if ref_arr is not None:
                save_dict['ref_bispec'] = ref_arr
            np.savez_compressed(str(out_path), **save_dict)
            self._log(f"  ✓ npz saved: {out_path.name}")
        except Exception as e:
            self._log(f"  ⚠ npz auto-save failed: {e}", warning=True)

    def _on_finished(self, result: dict):
        self._result = result
        self.primary_radio.setEnabled(True)
        self.companion_radio.setEnabled(True)
        self.clear_btn.setEnabled(True)
        self._set_click_mode('primary')

        n   = result['n_frames']
        roi = result['roi_size']
        n_str = str(n) if n > 0 else "n/a"

        _rc = result['recon'].T
        _rc_min, _rc_max = _rc.min(), _rc.max()
        if _rc_max > _rc_min:
            _rc = (_rc - _rc_min) / (_rc_max - _rc_min) * 255.0
        self.level_min_slider.setValue(0)
        self.level_max_slider.setValue(255)
        self.recon_view.setImage(
            _rc, autoLevels=False, autoRange=True, levels=(0, 255))

        deconv  = result.get('deconv_done', False)
        ref_arr = result.get('ref_bispec')
        self.save_bispec_btn.setEnabled(True)

        if deconv:
            self._log(
                f"✓ Complete (ref. deconvolution applied) — "
                f"{n_str} frames, ROI {roi}×{roi} px")
        elif ref_arr is not None:
            self._log(
                f"✓ Complete (ref. bispectrum computed, deconv OFF) — "
                f"{n_str} frames, ROI {roi}×{roi} px")
        else:
            self._log(f"✓ Complete — {n_str} frames, ROI {roi}×{roi} px")

        self._log(f"  Bispectrum mean |B| = {result['mean_bispec_mag']:.3e}")
        self._log(
            f"  Phase: mean |φ| = {result['mean_abs_phase']:.3f} rad  "
            f"non-zero: {result['nonzero_phase_pct']:.1f}%")
        self._log("Click the reconstructed image: first place the Primary "
                  "(red), then switch to Companion (green).")

    # ── Star placement ─────────────────────────────────────────────────────

    def _set_click_mode(self, mode: str):
        self._click_mode = mode
        btn = self.primary_radio if mode == 'primary' else self.companion_radio
        btn.blockSignals(True)
        btn.setChecked(True)
        btn.blockSignals(False)

    # ── NPZ Navigator ──────────────────────────────────────────────────────

    def _update_nav_bar(self):
        n   = len(self._nav_paths)
        idx = self._nav_idx
        self.nav_label.setText(f"{idx + 1} / {n}")
        self.nav_file_label.setText(Path(self._nav_paths[idx]).name)
        self.nav_prev_btn.setEnabled(idx > 0)
        self.nav_next_btn.setEnabled(idx < n - 1)

    def _nav_save_current(self):
        if not self._nav_paths:
            return
        self._nav_memory[self._nav_paths[self._nav_idx]] = {
            'result':        self._result,
            'primary_pos':   self._primary_pos,
            'companion_pos': self._companion_pos,
        }

    def _nav_go(self, idx: int):
        self._nav_save_current()
        self._nav_idx = idx
        self._update_nav_bar()
        path = self._nav_paths[idx]
        mem  = self._nav_memory.get(path)
        if mem and mem['result'] is not None:
            self._on_finished(mem['result'])
            if mem['primary_pos']:
                self._place_primary(*mem['primary_pos'])
            if mem['companion_pos']:
                self._place_companion(*mem['companion_pos'])
            if mem['primary_pos'] and mem['companion_pos']:
                self._update_measurement()
            self._log(f"\u25c4\u25ba {Path(path).name}")
        else:
            self._clear_all(silent=True)
            self._result = None
            self.progress_bar.setValue(0)
            self._log(f"\u25c4\u25ba Reconstructing: {Path(path).name}")
            self.status_lbl.setText(f"Reconstructing {Path(path).name}\u2026")
            ref_path   = self.ref_edit.text()
            ref_bispec = self._ref_bispec if ref_path else None
            self.worker = NpzReconWorker(
                path,
                k_max      = self.kmax_spin.value(),
                n_iter     = self.niter_spin.value(),
                ref_path   = ref_path,
                ref_bispec = ref_bispec,
                use_deconv = bool(ref_path),
                epsilon    = self.epsilon_spin.value())
            self.worker.progress.connect(self.progress_bar.setValue)
            self.worker.status.connect(self._on_status)
            self.worker.finished.connect(self._on_finished)
            self.worker.error.connect(self._on_error)
            self.worker.finished.connect(
                lambda _, p=path: self._nav_memory.update({p: {
                    'result':        self._result,
                    'primary_pos':   self._primary_pos,
                    'companion_pos': self._companion_pos,
                }}))
            self.worker.start()

    def _nav_prev(self):
        if self._nav_idx > 0:
            self._nav_go(self._nav_idx - 1)

    def _nav_next(self):
        if self._nav_idx < len(self._nav_paths) - 1:
            self._nav_go(self._nav_idx + 1)

    def _on_recon_click(self, x: float, y: float):
        if self._result is None:
            return
        if self._click_mode == 'primary':
            self._place_primary(x, y)
            self._set_click_mode('companion')
        else:
            self._place_companion(x, y)
        self._update_measurement()

    def _place_primary(self, x: float, y: float):
        if self._primary_marker is not None:
            for item in self._primary_marker:
                self.recon_view.getView().removeItem(item)
        self._primary_pos = (x, y)
        circle = pg.ScatterPlotItem([{
            'pos': (x, y), 'size': 14,
            'pen': pg.mkPen(DANGER, width=2),
            'brush': pg.mkBrush(None), 'symbol': 'o'}])
        cross = pg.ScatterPlotItem([{
            'pos': (x, y), 'size': 8,
            'pen': pg.mkPen(DANGER, width=1.5),
            'brush': pg.mkBrush(DANGER), 'symbol': '+'}])
        self.recon_view.getView().addItem(circle)
        self.recon_view.getView().addItem(cross)
        self._primary_marker = (circle, cross)
        if self._nav_paths:
            self._nav_save_current()

    def _place_companion(self, x: float, y: float):
        if self._companion_marker is not None:
            for item in self._companion_marker:
                self.recon_view.getView().removeItem(item)
        self._companion_pos = (x, y)
        circle = pg.ScatterPlotItem([{
            'pos': (x, y), 'size': 14,
            'pen': pg.mkPen(ACCENT2, width=2),
            'brush': pg.mkBrush(None), 'symbol': 'o'}])
        cross = pg.ScatterPlotItem([{
            'pos': (x, y), 'size': 8,
            'pen': pg.mkPen(ACCENT2, width=1.5),
            'brush': pg.mkBrush(ACCENT2), 'symbol': '+'}])
        self.recon_view.getView().addItem(circle)
        self.recon_view.getView().addItem(cross)
        self._companion_marker = (circle, cross)
        if self._nav_paths:
            self._nav_save_current()

    def _update_measurement(self):
        if self._primary_pos is None or self._companion_pos is None:
            self.card_theta.set_value("—")
            self.card_rho.set_value("—")
            self.card_theta_sky.set_value("—")
            self.card_rho_sky.set_value("—")
            self._meas_rho   = None
            self._meas_theta = None
            self.save_result_btn.setEnabled(False)
            self.append_csv_btn.setEnabled(False)
            self.save_wds_btn.setEnabled(False)
            return

        px, py = self._primary_pos
        cx, cy = self._companion_pos
        dx  = cx - px
        dy  = cy - py
        rho   = float(np.hypot(dx, dy))
        theta = float(np.degrees(np.arctan2(dx, -dy))) % 360.0
        self._meas_rho   = rho
        self._meas_theta = theta
        self.card_rho.set_value(f"{rho:.1f}")
        self.card_theta.set_value(f"{theta:.1f}")

        scale = self.cal_scale_spin.value()
        angle = self.cal_angle_spin.value()
        if scale > 0:
            rho_arcsec  = rho * scale
            theta_sky   = (theta + angle) % 360.0
            sigma_scale = self.cal_scale_err.value() if self.cal_scale_err else 0.0
            sigma_angle = self.cal_angle_err.value() if self.cal_angle_err else 0.0
            sigma_rho   = rho * sigma_scale
            sigma_theta = sigma_angle
            self._meas_rho_sky     = rho_arcsec
            self._meas_theta_sky   = theta_sky
            self._meas_sigma_rho   = sigma_rho
            self._meas_sigma_theta = sigma_theta
            sig_rho_str   = f"{sigma_rho:.4f}"   if sigma_rho   > 0 else None
            sig_theta_str = f"{sigma_theta:.2f}" if sigma_theta > 0 else None
            self.card_rho_sky.set_value(f"{rho_arcsec:.4f}",  sig_rho_str)
            self.card_theta_sky.set_value(f"{theta_sky:.2f}", sig_theta_str)
            self._log(
                f"Primary→Companion  ρ = {rho:.2f} px  ({rho_arcsec:.4f}″"
                + (f" ±{sigma_rho:.4f}" if sigma_rho > 0 else "") + ")"
                + f"   θ = {theta:.2f}° img  ({theta_sky:.2f}°"
                + (f" ±{sigma_theta:.2f}" if sigma_theta > 0 else "") + "° sky)")
        else:
            self.card_theta_sky.set_value("—")
            self.card_rho_sky.set_value("—")
            self._meas_rho_sky   = None
            self._meas_theta_sky = None
            self._meas_sigma_rho   = 0.0
            self._meas_sigma_theta = 0.0
            self._log(
                f"Primary→Companion  ρ = {rho:.2f} px   θ = {theta:.2f}°  "
                f"(load calibration for sky coords)")

        self.save_result_btn.setEnabled(True)
        has_sky = self._meas_rho_sky is not None
        self.append_csv_btn.setEnabled(has_sky)
        self.save_wds_btn.setEnabled(has_sky)

    def _clear_all(self, silent: bool = False):
        for marker in (self._primary_marker, self._companion_marker):
            if marker is not None:
                for item in marker:
                    self.recon_view.getView().removeItem(item)
        self._primary_marker   = None
        self._companion_marker = None
        self._primary_pos      = None
        self._companion_pos    = None
        self.card_theta.set_value("—")
        self.card_rho.set_value("—")
        self.card_theta_sky.set_value("—")
        self.card_rho_sky.set_value("—")
        self._meas_rho   = None
        self._meas_theta = None
        self.save_result_btn.setEnabled(False)
        self.append_csv_btn.setEnabled(False)
        self.save_wds_btn.setEnabled(False)
        self._click_mode = 'primary'
        self.primary_radio.blockSignals(True)
        self.primary_radio.setChecked(True)
        self.primary_radio.blockSignals(False)
        if not silent:
            self._log("Markers cleared.")

    # ── Save bispectrum ────────────────────────────────────────────────────

    def _save_bispec(self):
        if self._result is None or self._result.get('avg_bispec') is None:
            self._log("⚠ No bispectrum available — run the analysis first.",
                      error=True)
            return
        fits_stem = (Path(self.file_edit.text()).stem
                     if self.file_edit.text() else "target")
        default   = str(Path(_working_dir()) / f"{fits_stem}_bispec.npz")
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Bispectrum (.npz)", default,
            "NumPy archive (*.npz);;All Files (*)")
        if not path:
            return
        try:
            save_dict = {
                'avg_bispec': self._result['avg_bispec'],
                'avg_power':  self._result['avg_power'],
                'offsets':    self._result.get('offsets',
                              np.zeros((0, 2), dtype=np.int32)),
            }
            ref_arr = self._result.get('ref_bispec')
            if ref_arr is not None:
                save_dict['ref_bispec'] = ref_arr
            np.savez_compressed(path, **save_dict)
            shape = self._result['avg_bispec'].shape
            msg = f"✓ Bispectrum saved: {Path(path).name}  shape={shape}"
            if ref_arr is not None:
                msg += "  (+ ref_bispec)"
            self._log(msg)
        except Exception as e:
            self._log(f"⚠ Save failed: {e}", error=True)

    # ── Calibration ────────────────────────────────────────────────────────

    def _load_cal_dialog(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Load Calibration JSON",
            _working_dir(),
            "JSON files (*.json);;All Files (*)"
            )
        if path:
            self._load_cal(path)

    def _load_cal(self, path: str):
        try:
            with open(path) as f:
                cal = _json.load(f)
            for k in ('pixel_scale_arcsec', 'camera_angle_deg'):
                if k not in cal:
                    raise KeyError(f"Missing key: {k}")
            self._cal      = cal
            self._cal_file = path
            self.cal_edit.setText(Path(path).name)

            scale = cal['pixel_scale_arcsec']
            angle = cal['camera_angle_deg']
            s_sc  = cal.get('sigma_scale_arcsec', 0.0)
            s_ang = cal.get('sigma_angle_deg',    0.0)

            self.cal_scale_spin.setValue(scale)
            self.cal_angle_spin.setValue(angle)
            self.cal_scale_err.setValue(s_sc)
            self.cal_angle_err.setValue(s_ang)

            self.cal_status_lbl.setText(
                f"✓  scale={scale:.6f}″/px  ±{s_sc:.6f}   "
                f"angle={angle:.4f}°  ±{s_ang:.4f}°")
            self._log(f"Calibration loaded: {Path(path).name}")
            self._update_measurement()
        except Exception as e:
            self.cal_status_lbl.setText(f"⚠ {e}")
            self._log(f"⚠ Calibration load failed: {e}", error=True)

    # ── Save result / CSV / WDS ────────────────────────────────────────────

    def _save_result(self):
        if self._meas_rho is None or self._meas_theta is None:
            self._log("⚠ Place both markers first.", error=True)
            return
        r = self._result or {}
        fits_name = (Path(self.file_edit.text()).stem
                     if self.file_edit.text() else "result")
        default   = f"{fits_name}_speckle_result.json"
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Speckle Result",
            str(Path(_working_dir()) / default),
            "JSON files (*.json);;All Files (*)"
            )
        if not path:
            return
        payload = {
            "rho_px":             round(self._meas_rho,   4),
            "theta_img_deg":      round(self._meas_theta, 4),
            "rho_arcsec":         round(self._meas_rho_sky,   6)
                                  if self._meas_rho_sky   is not None else None,
            "theta_sky_deg":      round(self._meas_theta_sky, 4)
                                  if self._meas_theta_sky is not None else None,
            "sigma_rho_arcsec":   round(self._meas_sigma_rho,   6),
            "sigma_theta_deg":    round(self._meas_sigma_theta, 4),
            "pixel_scale_arcsec": self.cal_scale_spin.value(),
            "camera_angle_deg":   self.cal_angle_spin.value(),
            "cal_file":           Path(self._cal_file).name
                                  if self._cal_file else "manual",
            "roi_size_px":        int(r.get('roi_size', 0)),
            "n_frames":           int(r.get('n_frames', 0)),
            "fits_source":        Path(self.file_edit.text()).name
                                  if self.file_edit.text() else "",
        }
        try:
            with open(path, 'w') as f:
                _json.dump(payload, f, indent=2)
            sky_str = (
                f"  sky: {self._meas_rho_sky:.4f}″  {self._meas_theta_sky:.2f}°"
                if self._meas_rho_sky is not None else "")
            self._log(
                f"✓ Saved → {Path(path).name}  "
                f"ρ={self._meas_rho:.2f}px  θ={self._meas_theta:.2f}°{sky_str}")
        except Exception as e:
            self._log(f"⚠ Save failed: {e}", error=True)

    _CSV_HEADER = [
        'date', 'target', 'observer', 'filter',
        'theta_sky_deg', 'sigma_theta_deg',
        'rho_arcsec', 'sigma_rho_arcsec',
        'theta_img_deg', 'rho_px',
        'pixel_scale_arcsec_px', 'sigma_scale',
        'camera_angle_deg', 'sigma_angle_deg',
        'cal_file', 'fits_source',
    ]

    def _set_csv_dialog(self):
        import os
        path, _ = QFileDialog.getSaveFileName(
            self, "Set CSV Log File",
            str(Path(_working_dir()) / "speckle_log.csv"),
            "CSV files (*.csv);;All Files (*)"
            )
        if not path:
            return
        self._csv_path = path
        self.csv_path_lbl.setText(Path(path).name)
        if not os.path.exists(path) or os.path.getsize(path) == 0:
            with open(path, 'w', newline='') as f:
                csv.writer(f).writerow(self._CSV_HEADER)
            self._log(f"CSV log created: {Path(path).name}")
        else:
            self._log(f"CSV log set: {Path(path).name}")

    def _append_csv(self):
        if self._meas_rho is None or self._meas_rho_sky is None:
            self._log("⚠ Need both markers and calibration for CSV.", error=True)
            return
        if not self._csv_path:
            self._set_csv_dialog()
            if not self._csv_path:
                return
        r = self._result or {}
        row = [
            date.today().isoformat(), "", "", "",
            f"{self._meas_theta_sky:.6f}", f"{self._meas_sigma_theta:.6f}",
            f"{self._meas_rho_sky:.6f}",   f"{self._meas_sigma_rho:.6f}",
            f"{self._meas_theta:.4f}",     f"{self._meas_rho:.4f}",
            f"{self.cal_scale_spin.value():.8f}",
            f"{self.cal_scale_err.value():.8f}",
            f"{self.cal_angle_spin.value():.6f}",
            f"{self.cal_angle_err.value():.6f}",
            Path(self._cal_file).name if self._cal_file else "manual",
            Path(self.file_edit.text()).name if self.file_edit.text() else "",
        ]
        try:
            with open(self._csv_path, 'a', newline='') as f:
                csv.writer(f).writerow(row)
            self._log(f"✓ Appended to CSV: {Path(self._csv_path).name}")
        except Exception as e:
            self._log(f"⚠ CSV write error: {e}", error=True)

    _WDS_TEMPLATE = """────────────────────────────────────────────────────
  WDS ASTROMETRIC MEASUREMENT
────────────────────────────────────────────────────
  Target          : {target}
  Observer        : {observer}
  Date            : {obs_date}
  Filter          : {filter_name}

  Position Angle  : {theta_sky:.2f}°  ± {sigma_theta:.2f}°
  Separation      : {rho_arcsec:.4f}″  ± {sigma_rho:.4f}″

  ─── Source measurements ──────────────────────────
  ρ (pixels)      : {rho_px:.3f}
  θ (image)       : {theta_img:.2f}°

  ─── Calibration ──────────────────────────────────
  Pixel scale     : {pixel_scale:.6f} ″/px  ± {sigma_scale:.6f}
  Camera angle    : {camera_angle:.4f}°  ± {sigma_angle:.4f}°
  Calibration file: {cal_file}
────────────────────────────────────────────────────
"""

    def _save_wds(self):
        if self._meas_rho_sky is None:
            self._log("⚠ Load calibration first.", error=True)
            return
        report = self._WDS_TEMPLATE.format(
            target       = "",
            observer     = "",
            obs_date     = date.today().isoformat(),
            filter_name  = "",
            theta_sky    = self._meas_theta_sky,
            sigma_theta  = self._meas_sigma_theta,
            rho_arcsec   = self._meas_rho_sky,
            sigma_rho    = self._meas_sigma_rho,
            rho_px       = self._meas_rho,
            theta_img    = self._meas_theta,
            pixel_scale  = self.cal_scale_spin.value(),
            sigma_scale  = self.cal_scale_err.value(),
            camera_angle = self.cal_angle_spin.value(),
            sigma_angle  = self.cal_angle_err.value(),
            cal_file     = Path(self._cal_file).name
                           if self._cal_file else "manual",
        )
        fits_stem = (Path(self.file_edit.text()).stem
                     if self.file_edit.text() else "result")
        default   = f"{fits_stem}_WDS_{date.today().isoformat()}.txt"
        path, _ = QFileDialog.getSaveFileName(
            self, "Save WDS Report",
            str(Path(_working_dir()) / default),
            "Text files (*.txt);;All Files (*)"
            )
        if not path:
            return
        try:
            with open(path, 'w') as f:
                f.write(report)
            self._log(f"✓ WDS report saved: {Path(path).name}")
        except Exception as e:
            self._log(f"⚠ Save error: {e}", error=True)

    # ── Log ────────────────────────────────────────────────────────────────

    def _log(self, msg: str, error: bool = False, warning: bool = False):
        if error:
            color = DANGER
        elif warning:
            color = WARNING
        else:
            color = TEXT_MUTED
        self.log_edit.append(f'<span style="color:{color}">{msg}</span>')


# ═══════════════════════════════════════════════════════════════════════════
#  ── MAIN WINDOW ─────────────────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════

# ═══════════════════════════════════════════════════════════════════════════
#  Settings  (persistent JSON preferences)
# ═══════════════════════════════════════════════════════════════════════════

import json as _json_settings
from pathlib import Path as _SPath

_SETTINGS_PATH = _SPath.home() / ".config" / "speckle_suite" / "settings.json"

_SETTINGS_DEFAULTS = {
    "theme":        "dark",
    "working_dir":  str(_SPath.home()),
    "preprocess": {"roi_index": 0},
    "analysis": {
        "k_max": 60, "dk_max": 9, "n_iter": 30,
        "epsilon": 0.01, "colormap": "Grey"
    }
}

def _load_settings() -> dict:
    try:
        if _SETTINGS_PATH.exists():
            with open(_SETTINGS_PATH) as f:
                data = _json_settings.load(f)
            merged = dict(_SETTINGS_DEFAULTS)
            merged["preprocess"] = {**_SETTINGS_DEFAULTS["preprocess"],
                                    **data.get("preprocess", {})}
            merged["analysis"]   = {**_SETTINGS_DEFAULTS["analysis"],
                                    **data.get("analysis", {})}
            for k in ("theme", "working_dir"):
                if k in data:
                    merged[k] = data[k]
            return merged
    except Exception:
        pass
    return dict(_SETTINGS_DEFAULTS)

def _save_settings(s: dict):
    try:
        _SETTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(_SETTINGS_PATH, "w") as f:
            _json_settings.dump(s, f, indent=2)
    except Exception as e:
        print(f"[Settings] Save failed: {e}")

SETTINGS = _load_settings()


def _working_dir() -> str:
    return SETTINGS.get("working_dir", str(_SPath.home()))


class SettingsDialog(QDialog):
    """Modal preferences window."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Preferences")
        self.setMinimumWidth(480)
        self.setModal(True)
        self._build_ui()
        self._load()

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setSpacing(16)
        root.setContentsMargins(20, 20, 20, 20)

        gen_group  = QGroupBox("General")
        gen_layout = QFormLayout(gen_group)
        gen_layout.setSpacing(8)
        self.theme_combo = QComboBox()
        self.theme_combo.addItems(["Dark", "Night (Red)", "Light"])
        self.theme_combo.setMinimumHeight(28)
        gen_layout.addRow("Theme", self.theme_combo)
        dir_row = QWidget()
        dir_hl  = QHBoxLayout(dir_row)
        dir_hl.setContentsMargins(0, 0, 0, 0); dir_hl.setSpacing(6)
        self.dir_edit = QLineEdit()
        self.dir_edit.setPlaceholderText("Working directory…")
        dir_browse = QPushButton("Browse…")
        dir_browse.setFixedHeight(28)
        dir_browse.clicked.connect(self._browse_dir)
        dir_hl.addWidget(self.dir_edit, 1); dir_hl.addWidget(dir_browse)
        gen_layout.addRow("Working directory", dir_row)
        root.addWidget(gen_group)

        pre_group  = QGroupBox("Preprocessing")
        pre_layout = QFormLayout(pre_group)
        pre_layout.setSpacing(8)
        self.roi_combo = QComboBox()
        self.roi_combo.addItems(["32 × 32", "64 × 64", "128 × 128"])
        self.roi_combo.setMinimumHeight(28)
        pre_layout.addRow("Default ROI size", self.roi_combo)
        root.addWidget(pre_group)

        ana_group  = QGroupBox("Analysis")
        ana_layout = QFormLayout(ana_group)
        ana_layout.setSpacing(8)
        self.kmax_spin = QSpinBox()
        self.kmax_spin.setRange(4, 512); self.kmax_spin.setSuffix(" px")
        self.kmax_spin.setMinimumHeight(28)
        ana_layout.addRow("K_max", self.kmax_spin)
        self.dkmax_spin = QSpinBox()
        self.dkmax_spin.setRange(1, 64); self.dkmax_spin.setSuffix(" px")
        self.dkmax_spin.setMinimumHeight(28)
        ana_layout.addRow("dK_max", self.dkmax_spin)
        self.niter_spin = QSpinBox()
        self.niter_spin.setRange(1, 200); self.niter_spin.setMinimumHeight(28)
        ana_layout.addRow("Iterations", self.niter_spin)
        self.epsilon_spin = QDoubleSpinBox()
        self.epsilon_spin.setRange(0.001, 0.5); self.epsilon_spin.setDecimals(3)
        self.epsilon_spin.setSingleStep(0.005); self.epsilon_spin.setMinimumHeight(28)
        ana_layout.addRow("Wiener ε", self.epsilon_spin)
        self.cmap_combo = QComboBox()
        self.cmap_combo.addItems(["Grey", "Inverted", "Hot", "Rainbow", "Viridis"])
        self.cmap_combo.setMinimumHeight(28)
        ana_layout.addRow("Default colormap", self.cmap_combo)
        root.addWidget(ana_group)

        btn_row = QHBoxLayout(); btn_row.addStretch()
        cancel_btn = QPushButton("Cancel")
        cancel_btn.setFixedHeight(30); cancel_btn.clicked.connect(self.reject)
        self.ok_btn = QPushButton("Apply & Close")
        self.ok_btn.setFixedHeight(30)
        self.ok_btn.setStyleSheet(_primary_btn_style())
        self.ok_btn.clicked.connect(self._apply)
        btn_row.addWidget(cancel_btn); btn_row.addWidget(self.ok_btn)
        root.addLayout(btn_row)

    def _load(self):
        theme_map = {"dark": 0, "red": 1, "light": 2}
        self.theme_combo.setCurrentIndex(
            theme_map.get(SETTINGS.get("theme", "dark"), 0))
        self.dir_edit.setText(SETTINGS.get("working_dir", str(_SPath.home())))
        pre = SETTINGS.get("preprocess", {})
        self.roi_combo.setCurrentIndex(pre.get("roi_index", 0))
        ana = SETTINGS.get("analysis", {})
        self.kmax_spin.setValue(ana.get("k_max",   60))
        self.dkmax_spin.setValue(ana.get("dk_max",  9))
        self.niter_spin.setValue(ana.get("n_iter",  30))
        self.epsilon_spin.setValue(ana.get("epsilon", 0.01))
        self.cmap_combo.setCurrentText(ana.get("colormap", "Grey"))

    def _browse_dir(self):
        d = QFileDialog.getExistingDirectory(
            self, "Select Working Directory",
            self.dir_edit.text() or str(_SPath.home()))
        if d:
            self.dir_edit.setText(d)

    def _apply(self):
        names = ["dark", "red", "light"]
        SETTINGS["theme"]       = names[self.theme_combo.currentIndex()]
        SETTINGS["working_dir"] = self.dir_edit.text().strip() or str(_SPath.home())
        SETTINGS["preprocess"]["roi_index"] = self.roi_combo.currentIndex()
        SETTINGS["analysis"]["k_max"]    = self.kmax_spin.value()
        SETTINGS["analysis"]["dk_max"]   = self.dkmax_spin.value()
        SETTINGS["analysis"]["n_iter"]   = self.niter_spin.value()
        SETTINGS["analysis"]["epsilon"]  = self.epsilon_spin.value()
        SETTINGS["analysis"]["colormap"] = self.cmap_combo.currentText()
        _save_settings(SETTINGS)
        self.accept()


class SpeckleMainWindow(QMainWindow):

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Double Star Speckle Astrometry Suite")
        self.resize(1400, 1020)
        self.setMinimumSize(1200, 800)

        global _theme
        _theme = THEMES.get(SETTINGS.get("theme", "dark"), THEMES["dark"])
        _refresh_theme_aliases()
        QApplication.instance().setStyleSheet(build_stylesheet(_theme))
        pg.setConfigOptions(antialias=True, imageAxisOrder='row-major')

        self._build_menu()
        self._build_tabs()
        self._apply_settings_to_tabs()

    def _build_tabs(self):
        self.tabs = QTabWidget()
        self.tabs.setTabPosition(QTabWidget.TabPosition.North)

        self.drift_tab      = DriftTab()
        self.preprocess_tab = PreprocessTab()
        self.analysis_tab   = AnalysisTab()

        self.tabs.addTab(self.drift_tab,      "🧭  Drift Alignment")
        self.tabs.addTab(self.preprocess_tab, "⚙  Preprocess")
        self.tabs.addTab(self.analysis_tab,   "🔭  Analysis")

        self.setCentralWidget(self.tabs)

    def _build_menu(self):
        mb = self.menuBar()

        file_menu = mb.addMenu("File")
        open_pre  = file_menu.addAction("Open Speckle Sequence…")
        open_pre.triggered.connect(self._open_preprocess)
        open_ana  = file_menu.addAction("Open FITS Cube (Analysis)…")
        open_ana.triggered.connect(self._open_analysis)
        file_menu.addSeparator()
        cal_act = file_menu.addAction("Load Calibration JSON…")
        cal_act.triggered.connect(
            lambda: self.analysis_tab._load_cal_dialog())
        csv_act = file_menu.addAction("Set CSV Log File…")
        csv_act.triggered.connect(
            lambda: self.analysis_tab._set_csv_dialog())
        file_menu.addSeparator()
        quit_act = file_menu.addAction("Quit")
        quit_act.triggered.connect(self.close)

        settings_act = mb.addAction("⚙  Settings")
        settings_act.triggered.connect(self._open_settings)

    def _open_preprocess(self):
        self.tabs.setCurrentWidget(self.preprocess_tab)
        self.preprocess_tab._browse_file()

    def _open_analysis(self):
        self.tabs.setCurrentWidget(self.analysis_tab)
        self.analysis_tab._browse_file()

    def _apply_settings_to_tabs(self):
        pre = SETTINGS.get("preprocess", {})
        ana = SETTINGS.get("analysis",   {})
        self.preprocess_tab.roi_combo.setCurrentIndex(pre.get("roi_index", 0))
        self.analysis_tab.kmax_spin.setValue(ana.get("k_max",   60))
        self.analysis_tab.dkmax_spin.setValue(ana.get("dk_max",  9))
        self.analysis_tab.niter_spin.setValue(ana.get("n_iter",  30))
        self.analysis_tab.epsilon_spin.setValue(ana.get("epsilon", 0.01))
        cmap = ana.get("colormap", "Grey")
        idx  = self.analysis_tab.recon_cmap_combo.findText(cmap)
        if idx >= 0:
            self.analysis_tab.recon_cmap_combo.setCurrentIndex(idx)

    def _open_settings(self):
        dlg = SettingsDialog(self)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            self._set_theme(SETTINGS["theme"])
            self._apply_settings_to_tabs()

    def _set_theme(self, name: str):

        global _theme
        _theme = THEMES[name]
        _refresh_theme_aliases()
        QApplication.instance().setStyleSheet(build_stylesheet(_theme))
        self.drift_tab.refresh_styles()
        self.preprocess_tab.refresh_styles()
        self.analysis_tab.refresh_styles()


# ═══════════════════════════════════════════════════════════════════════════
#  Entry point

def main():
    pg.setConfigOptions(antialias=True, imageAxisOrder='row-major')
    app = QApplication(sys.argv)
    app.setFont(QFont("JetBrains Mono", 10))
    win = SpeckleMainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()

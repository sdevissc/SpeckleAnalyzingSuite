"""
speckle_suite.preprocess_backend
=================================
Pure-Python / numpy backend for the Preprocess tab.

Public API
----------
rms_contrast()      frame quality metric
find_centroid()     sub-pixel centroid via weighted centre-of-mass
register_and_crop() shift + crop a frame to a centred ROI
PreprocessWorker    QThread: score → select → register → write FITS cube
"""

from pathlib import Path
from typing import Optional

import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal
from scipy.ndimage import shift as nd_shift

from speckle_suite.ser_io import (
    SERHeader, parse_ser_header, ser_frame_iter,
)
from speckle_suite.widgets import read_fits_cube


# ── Quality metric ─────────────────────────────────────────────────────────

def rms_contrast(frame: np.ndarray) -> float:
    """
    Normalised RMS contrast Q = σ(I) / μ(I).
    Higher value → sharper speckle pattern → preferred frame.
    """
    m = float(frame.mean())
    if m <= 0:
        return 0.0
    return float(frame.std()) / m


# ── Centroid and registration ──────────────────────────────────────────────

def find_centroid(frame: np.ndarray) -> tuple[float, float]:
    """
    Return the sub-pixel (row, col) centroid of the brightest source.

    Uses a 32×32 patch around the peak maximum, subtracts the local
    minimum as background, then computes the weighted centre-of-mass.
    """
    from scipy.ndimage import uniform_filter
    smoothed = uniform_filter(frame.astype(np.float32), size=5)
    pr, pc   = np.unravel_index(np.argmax(smoothed), smoothed.shape)
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


def register_and_crop(
        frame: np.ndarray,
        centroid_rc: tuple[float, float],
        roi_size: int,
) -> np.ndarray:
    """
    Shift *frame* so that *centroid_rc* lands at the frame centre,
    then return the central roi_size × roi_size crop.
    """
    h, w   = frame.shape
    cr, cc = centroid_rc
    dy = cr - h / 2.0
    dx = cc - w / 2.0
    shifted = nd_shift(frame.astype(np.float32),
                       shift=(-dy, -dx), order=3, mode='reflect')
    half = roi_size // 2
    mr, mc = h // 2, w // 2
    return shifted[mr - half: mr + half,
                   mc - half: mc + half].copy()


# ── Worker ─────────────────────────────────────────────────────────────────

class PreprocessWorker(QThread):
    """
    Background worker: score frames → select best % → register → write FITS cube.

    Signals
    -------
    progress(int)      0–100
    status(str)        human-readable status message
    preview(object)    best-frame crop as float32 ndarray
    quality(object)    all RMS-contrast scores as float32 ndarray
    finished(object)   result dict (see _process for keys)
    error(str)         exception message + traceback
    """

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
        self.file_type   = file_type   # 'ser' | 'fits'
        self.best_pct    = best_pct
        self.roi_size    = roi_size
        self.output_path = output_path
        self._stop       = False

    def stop(self) -> None:
        self._stop = True

    def run(self) -> None:
        try:
            self._process()
        except Exception as e:
            import traceback
            self.error.emit(f"{e}\n{traceback.format_exc()}")

    def _process(self) -> None:
        self.status.emit("Reading frames and computing quality scores…")
        self.progress.emit(2)

        if self.file_type == 'ser':
            with open(self.filepath, 'rb') as f:
                header = parse_ser_header(f.read(178))
            n_total    = header.frame_count
            fh, fw     = header.image_height, header.image_width
            frame_iter = ser_frame_iter(self.filepath, header)
        else:
            cube, _    = read_fits_cube(self.filepath)
            n_total    = cube.shape[0]
            fh, fw     = cube.shape[1], cube.shape[2]
            frame_iter = iter(cube)

        # ── Score all frames ──────────────────────────────────────────────
        scores = np.zeros(n_total, dtype=np.float32)
        frames: list[np.ndarray] = []
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

        # ── Select best frames ────────────────────────────────────────────
        threshold  = np.percentile(scores, 100.0 - self.best_pct)
        sel_mask   = scores >= threshold
        sel_idx    = np.where(sel_mask)[0]
        n_selected = len(sel_idx)
        self.status.emit(
            f"Keeping {n_selected} / {n_total} frames  "
            f"({self.best_pct:.0f}%,  Q ≥ {threshold:.4f})")
        self.progress.emit(43)

        # Clamp ROI to a power-of-two ≤ min(frame dimension)
        max_roi = min(fw, fh)
        if self.roi_size > max_roi:
            old = self.roi_size
            self.roi_size = int(2 ** np.floor(np.log2(max_roi)))
            self.status.emit(
                f"WARNING: frame {fw}×{fh} — ROI clamped "
                f"{old} → {self.roi_size} px")

        # ── Register all frames ───────────────────────────────────────────
        all_crops: list[np.ndarray] = []
        shifts_px: list[float]      = []
        sel_set = set(sel_idx.tolist())

        best_idx      = int(np.argmax(scores))
        best_centroid = find_centroid(frames[best_idx])
        best_crop     = register_and_crop(frames[best_idx], best_centroid, self.roi_size)
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
                self.progress.emit(46 + int(48 * src_i / max(n_total - 1, 1)))
                self.status.emit(
                    f"Registering… {src_i+1} / {n_total}  "
                    f"(centroid: {cr:.1f}, {cc:.1f}  shift: {shift_mag:.1f} px)")

        max_shift = float(np.max(shifts_px)) if shifts_px else 0.0
        self.progress.emit(96)

        # ── Write FITS cube ───────────────────────────────────────────────
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

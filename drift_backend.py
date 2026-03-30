"""
speckle_suite.drift_backend
============================
Backend for the Drift Alignment (calibration) tab.

Public API
----------
compute_centroid()                sub-pixel centroid (fast, ROI-only)
stream_ser_centroids()            stream a SER file computing centroids on the fly
read_ser_header_and_timestamps()  re-exported from ser_io for convenience
_parse_declination_from_txt()     auto-fill declination from capture-software sidecar
_tls_fit()                        Total Least Squares line fit via SVD
fit_drift()                       full drift fit with trimming + sigma clipping
DriftResult                       dataclass returned by DriftTab
DriftWorker                       QThread: stream → centroids → emit raw data
SimbadWorker                      QThread: resolve target name → declination
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal

from speckle_suite.ser_io import (
    SERHeader,
    COLOR_BGR, COLOR_RGB,
    read_ser_header_and_timestamps,
)


# ── Fast centroid (used by DriftWorker — does NOT load full frame) ─────────

def compute_centroid(
        frame: np.ndarray,
        roi_size: int = 64,
) -> Optional[tuple[float, float]]:
    """
    Return the sub-pixel (x, y) centroid of the brightest star.

    Works on uint8/uint16 arrays without converting the whole frame to float32:
    only the small ROI around the peak is converted.  Background is estimated
    as the ROI minimum (O(roi²) instead of a full-frame sort).

    Returns None if the ROI is completely flat.
    """
    peak_idx = np.unravel_index(np.argmax(frame), frame.shape)
    py, px   = peak_idx
    half = roi_size // 2
    y0 = max(0, py - half);  y1 = min(frame.shape[0], py + half)
    x0 = max(0, px - half);  x1 = min(frame.shape[1], px + half)
    roi = frame[y0:y1, x0:x1].astype(np.float32)
    roi = roi - roi.min()
    if roi.sum() == 0:
        return None
    yy, xx = np.mgrid[y0:y1, x0:x1]
    cx = float((xx * roi).sum() / roi.sum())
    cy = float((yy * roi).sum() / roi.sum())
    return cx, cy


# ── Centroid streaming ─────────────────────────────────────────────────────

def stream_ser_centroids(
        filepath: str,
        header: SERHeader,
        timestamps_sec: np.ndarray,
        t_start: float,
        t_stop: float,
        progress_cb=None,
        stop_flag=None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Stream a SER file frame-by-frame, computing a centroid per frame.

    Frames outside [t_start, t_stop] are skipped by seeking past them
    without decoding — minimises memory and CPU use.

    Returns
    -------
    cx, cy : centroid positions [px]
    ct     : centroid timestamps [s]
    """
    fsize    = header.frame_size
    n_total  = header.frame_count
    is_colour = header.is_colour
    dtype    = np.uint16 if header.bytes_per_pixel == 2 else np.uint8

    centroids_x: list[float] = []
    centroids_y: list[float] = []
    centroid_times: list[float] = []

    with open(filepath, 'rb') as f:
        f.seek(178)  # skip header
        for i in range(n_total):
            if stop_flag and stop_flag():
                break
            t = timestamps_sec[i]
            if t < t_start or t > t_stop:
                f.seek(fsize, 1)
                if progress_cb and i % 100 == 0:
                    progress_cb(int(100 * i / n_total))
                continue
            raw = f.read(fsize)
            if len(raw) < fsize:
                break
            arr = np.frombuffer(raw, dtype=dtype).reshape(
                header.image_height, header.image_width)
            if is_colour:
                arr = arr.reshape(
                    header.image_height, header.image_width // 3, 3
                ).mean(axis=2).astype(np.float32)
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


# ── Declination parser ─────────────────────────────────────────────────────

def _parse_declination_from_txt(ser_path: str) -> Optional[float]:
    """
    Look for a companion text file (FireCapture / SharpCap / Genika) and
    extract the target declination in decimal degrees.  Returns None if not
    found or not parseable.
    """
    ser_p = Path(ser_path)
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

    key_pat = r'(?<!\w)(?:declination|decl?|de|δ)(?!\w)'
    dms_deg = r'([+\-]?\d{1,3})[°d](\d{1,2})[\'′m](\d{1,2}(?:\.\d+)?)[\"″s]?'
    dms_col = r'([+\-]?\d{1,3}):(\d{2}):(\d{2}(?:\.\d+)?)'
    dms_spc = r'([+\-]?\d{1,3})\s+(\d{1,2})\s+(\d{1,2}(?:\.\d+)?)'
    dec_dec = r'([+\-]?\d{1,3}\.\d+)°?'
    dec_int = r'([+\-]?\d{1,3})°'

    def dms_to_deg(d, m, s):
        sign = -1 if str(d).strip().startswith('-') else 1
        return sign * (abs(float(d)) + float(m) / 60.0 + float(s) / 3600.0)

    sep = r'\s*[=:]\s*'
    patterns = [
        (re.compile(key_pat + sep + dms_deg, re.IGNORECASE), 'dms'),
        (re.compile(key_pat + sep + dms_col, re.IGNORECASE), 'dms'),
        (re.compile(key_pat + sep + dms_spc, re.IGNORECASE), 'dms'),
        (re.compile(key_pat + sep + dec_dec, re.IGNORECASE), 'dec'),
        (re.compile(key_pat + sep + dec_int, re.IGNORECASE), 'dec'),
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
    return None


# ── TLS / SVD drift fit ────────────────────────────────────────────────────

def _tls_fit(cx: np.ndarray, cy: np.ndarray) -> dict:
    """
    Total Least Squares line fit via SVD on 2-D centroid cloud.

    Minimises perpendicular (orthogonal) distances — correct when both
    X and Y carry measurement noise.
    """
    cx_mean = cx.mean()
    cy_mean = cy.mean()
    X = np.column_stack([cx - cx_mean, cy - cy_mean])
    _, _, Vt = np.linalg.svd(X, full_matrices=False)
    direction  = Vt[0]
    para_resid = X @ direction
    perp_unit  = np.array([-direction[1], direction[0]])
    perp_resid = X @ perp_unit
    fitted_x   = cx_mean + para_resid * direction[0]
    fitted_y   = cy_mean + para_resid * direction[1]
    return {
        'direction':  direction,
        'centroid':   (cx_mean, cy_mean),
        'perp_resid': perp_resid,
        'para_resid': para_resid,
        'fitted_x':   fitted_x,
        'fitted_y':   fitted_y,
    }


def fit_drift(
        cx: np.ndarray,
        cy: np.ndarray,
        declination_deg: float,
        fps: float,
        sigma_threshold: float,
        times_sec: Optional[np.ndarray] = None,
        start_trim_sec: float = 0.0,
        stop_trim_sec: float = 0.0,
) -> dict:
    """
    Three-pass TLS drift fit with time-window trimming and σ-clipping.

    Returns a dict with geometry, physical calibration results, 1-σ
    uncertainties, and frame counts.  All keys documented inline.
    """
    n = len(cx)
    t = times_sec if (times_sec is not None and len(times_sec) == n) \
        else np.arange(n, dtype=float) / fps

    # ── Time window ───────────────────────────────────────────────────────
    t0 = t[0]  + start_trim_sec
    t1 = t[-1] - stop_trim_sec
    time_mask = (t >= t0) & (t <= t1)
    if time_mask.sum() < 5:
        time_mask = np.ones(n, dtype=bool)

    cx_win = cx[time_mask]
    cy_win = cy[time_mask]

    # ── Pass 1 ────────────────────────────────────────────────────────────
    tls0  = _tls_fit(cx_win, cy_win)
    resid0 = np.abs(tls0['perp_resid'])
    rms0   = float(np.sqrt(np.mean(resid0**2)))
    sigma_mask_win = (resid0 <= sigma_threshold * rms0
                      if sigma_threshold > 0 and rms0 > 0
                      else np.ones(len(cx_win), dtype=bool))
    if sigma_mask_win.sum() < 5:
        sigma_mask_win = np.ones(len(cx_win), dtype=bool)

    sigma_mask = np.zeros(n, dtype=bool)
    sigma_mask[time_mask] = sigma_mask_win
    mask = time_mask & sigma_mask

    # ── Pass 2 ────────────────────────────────────────────────────────────
    tls = _tls_fit(cx[mask], cy[mask])

    # ── Pass 3 (refinement) ───────────────────────────────────────────────
    cx_mean2, cy_mean2 = tls['centroid']
    dx2, dy2           = tls['direction']
    perp_unit2         = np.array([-dy2, dx2])
    X_win2             = np.column_stack([cx_win - cx_mean2, cy_win - cy_mean2])
    resid2             = np.abs(X_win2 @ perp_unit2)
    rms2               = float(np.sqrt(np.mean(resid2[sigma_mask_win]**2)))
    sigma_mask_win2 = (resid2 <= sigma_threshold * rms2
                       if sigma_threshold > 0 and rms2 > 0
                       else np.ones(len(cx_win), dtype=bool))
    if sigma_mask_win2.sum() < 5:
        sigma_mask_win2 = sigma_mask_win

    sigma_mask2 = np.zeros(n, dtype=bool)
    sigma_mask2[time_mask] = sigma_mask_win2
    mask = time_mask & sigma_mask2

    # ── Final fit ─────────────────────────────────────────────────────────
    tls    = _tls_fit(cx[mask], cy[mask])
    dx, dy = tls['direction']

    cx_mean, cy_mean = tls['centroid']
    X_all    = np.column_stack([cx - cx_mean, cy - cy_mean])
    para_all = X_all @ tls['direction']
    perp_all = X_all @ np.array([-dy, dx])
    fitted_x = cx_mean + para_all * dx
    fitted_y = cy_mean + para_all * dy

    # ── Physical quantities ───────────────────────────────────────────────
    drift_angle_image  = np.degrees(np.arctan2(dy, dx))
    para_inliers       = tls['para_resid']
    drift_length_px    = para_inliers.max() - para_inliers.min()
    n_inliers          = int(mask.sum())
    t_inliers          = t[mask]
    elapsed_sec        = float(t_inliers[-1] - t_inliers[0])
    if elapsed_sec <= 0:
        elapsed_sec = n_inliers / fps
    drift_speed_px_sec = drift_length_px / elapsed_sec
    sidereal_arcsec_s  = 15.041 * np.cos(np.radians(declination_deg))
    pixel_scale        = (sidereal_arcsec_s / drift_speed_px_sec
                         if drift_speed_px_sec > 0 else 0.0)
    camera_angle       = (drift_angle_image + 90.0) % 360.0

    rms_perp  = float(np.sqrt(np.mean(tls['perp_resid']**2)))
    rms_para  = float(np.sqrt(np.mean(tls['para_resid']**2)))
    sqrt_n    = np.sqrt(max(n_inliers, 1))
    half_len  = drift_length_px / 2.0
    sigma_angle_rad = (np.arctan2(rms_perp, half_len) / sqrt_n
                       if half_len > 0 else 0.0)
    sigma_scale = (pixel_scale * rms_para / drift_length_px / sqrt_n
                   if drift_length_px > 0 and sqrt_n > 0 else 0.0)

    return {
        'mask':             mask,
        'time_mask':        time_mask,
        'direction':        (dx, dy),
        'line_centroid':    (cx_mean, cy_mean),
        'fitted_x':         fitted_x,
        'fitted_y':         fitted_y,
        'perp_resid':       perp_all,
        'resid_abs':        np.abs(perp_all),
        'rms_perp':         rms_perp,
        'rms_para':         rms_para,
        't':                t,
        'camera_angle':     camera_angle,
        'pixel_scale':      pixel_scale,
        'drift_speed_px_s': drift_speed_px_sec,
        'drift_length_px':  drift_length_px,
        'sigma_angle_deg':  float(np.degrees(sigma_angle_rad)),
        'sigma_scale':      sigma_scale,
        'n_used':           n_inliers,
        'n_rejected':       int((~mask).sum()),
    }


# ── DriftResult dataclass ──────────────────────────────────────────────────

@dataclass
class DriftResult:
    camera_angle_deg:       float
    pixel_scale_arcsec:     float
    seeing_indicator_arcsec: float
    centroids_x:            np.ndarray
    centroids_y:            np.ndarray
    mask:                   np.ndarray
    n_frames_used:          int
    n_frames_rejected:      int


# ── DriftWorker ────────────────────────────────────────────────────────────

class DriftWorker(QThread):
    """
    Stream a SER drift file, compute per-frame centroids, emit the raw data.

    The UI calls fit_drift() interactively once this worker finishes.
    """

    progress = pyqtSignal(int)
    status   = pyqtSignal(str)
    finished = pyqtSignal(object)
    error    = pyqtSignal(str)

    def __init__(self, filepath: str, declination_deg: float):
        super().__init__()
        self.filepath        = filepath
        self.declination_deg = declination_deg
        self._stop           = False

    def stop(self) -> None:
        self._stop = True

    def run(self) -> None:
        try:
            self._process()
        except Exception as e:
            import traceback
            self.error.emit(f"{e}\n{traceback.format_exc()}")

    def _process(self) -> None:
        self.status.emit("Reading SER header and timestamps…")
        self.progress.emit(2)

        header, timestamps_sec = read_ser_header_and_timestamps(self.filepath)
        self.status.emit(
            f"{header.frame_count} frames  "
            f"({header.image_width}×{header.image_height}, "
            f"{header.pixel_depth}-bit)")
        self.progress.emit(5)

        if timestamps_sec is None:
            self.error.emit(
                "No valid per-frame timestamps found in this SER file.\n\n"
                "Timestamps are required to compute the frame interval and "
                "therefore the pixel scale. Without them the result would "
                "be silently wrong.\n\n"
                "Please use capture software that records per-frame "
                "timestamps (e.g. FireCapture, SharpCap) and re-acquire "
                "the drift sequence.")
            return

        dt = np.diff(timestamps_sec)
        dt_positive = dt[dt > 0]
        if len(dt_positive) < 3:
            self.error.emit(
                f"Only {len(dt_positive)} valid timestamp intervals found. "
                f"File may be too short or timestamps unreliable. Aborting.")
            return

        median_dt  = float(np.median(dt_positive))
        fps_actual = 1.0 / median_dt
        fps_std    = float(np.std(dt_positive))
        t_total    = float(timestamps_sec[-1] - timestamps_sec[0])
        self.status.emit(
            f"Frame interval: {median_dt*1000:.3f} ms  "
            f"({fps_actual:.2f} fps)  total: {t_total:.1f} s")
        self.progress.emit(8)

        self.status.emit(
            f"Streaming {header.frame_count} frames  ({t_total:.1f} s)…")
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
            self.status.emit("Stopped."); return

        if len(cx) < 10:
            self.error.emit(
                f"Only {len(cx)} valid centroids found — too few to fit. "
                f"Check that the star is visible and in focus.")
            return

        self.progress.emit(100)
        self.status.emit(
            f"Done — {len(cx)} centroids over {ct[-1]-ct[0]:.1f} s.  "
            f"Adjust σ slider then save.")
        self.finished.emit({
            'centroids_x':     cx,
            'centroids_y':     cy,
            'times_sec':       ct,
            'fps':             fps_actual,
            'median_dt_ms':    median_dt * 1000.0,
            'fps_std_ms':      fps_std * 1000.0,
            'declination_deg': self.declination_deg,
        })


# ── SimbadWorker ───────────────────────────────────────────────────────────

class SimbadWorker(QThread):
    """Resolve a target name via the Simbad TAP service → declination."""

    result = pyqtSignal(float, str)   # (dec_deg, canonical_name)
    error  = pyqtSignal(str)

    def __init__(self, name: str):
        super().__init__()
        self.name = name.strip()

    def run(self) -> None:
        try:
            import urllib.request, urllib.parse, json as _json
            adql = (
                "SELECT ra, dec, main_id FROM basic "
                "JOIN ident ON ident.oidref = basic.oid "
                f"WHERE ident.id = '{self.name}'"
            )
            params = urllib.parse.urlencode({
                "REQUEST": "doQuery", "LANG": "ADQL",
                "FORMAT": "json", "QUERY": adql,
            })
            url = "https://simbad.u-strasbg.fr/simbad/sim-tap/sync?" + params
            req = urllib.request.Request(
                url, headers={"User-Agent": "SpeckleSuite/2.0"})
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = _json.loads(resp.read().decode())
            rows = data.get("data", [])
            if not rows:
                self.error.emit(f"Object '{self.name}' not found in Simbad.")
                return
            ra_deg, dec_deg, main_id = rows[0]
            self.result.emit(float(dec_deg), str(main_id))
        except Exception as e:
            self.error.emit(f"Simbad query failed: {e}")

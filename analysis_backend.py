"""
speckle_suite.analysis_backend
================================
Pure-Python / numpy backend for the Analysis tab.

Public API
----------
KMAX_DEFAULT / DKMAX_DEFAULT    sensible defaults for the UI spinboxes
build_offset_list()             list of (vy, vx) bispectrum offsets in a disc
accumulate_bispectrum()         sum B(u,v) over all frames
iterative_reconstruct()         Knox-Thompson phase retrieval → image
compute_autocorrelogram()       inverse FFT of the power spectrum
deconvolve_bispectrum()         Wiener division in bispectrum space
AnalysisWorker                  QThread: FITS cube → bispectrum → image
NpzReconWorker                  QThread: .npz bispectrum → image (no accumulation)
"""

from __future__ import annotations

import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal

from speckle_suite.widgets import read_fits_cube

# ── Constants ──────────────────────────────────────────────────────────────

KMAX_DEFAULT:  int = 60
DKMAX_DEFAULT: int = 9


# ── Bispectrum accumulation ────────────────────────────────────────────────

def build_offset_list(dk_max: int) -> np.ndarray:
    """Return (N, 2) int32 array of (vy, vx) offsets within a disc of radius dk_max."""
    offsets = [
        (vy, vx)
        for vy in range(-dk_max, dk_max + 1)
        for vx in range(-dk_max, dk_max + 1)
        if vy * vy + vx * vx <= dk_max * dk_max
    ]
    return np.array(offsets, dtype=np.int32)


def accumulate_bispectrum(
        cube: np.ndarray,
        k_max:  int = KMAX_DEFAULT,
        dk_max: int = DKMAX_DEFAULT,
        progress_cb=None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """
    Accumulate the averaged bispectrum B(u, v) = <F(u)·F(v)·F*(u+v)>
    and the averaged power spectrum |F|² over all frames in *cube*.

    Returns
    -------
    avg_power  : (H, W) float64
    avg_bispec : (H, W, N_offsets) complex128
    offsets    : (N_offsets, 2) int32
    n_frames   : int
    """
    n_frames, H, W = cube.shape
    offsets    = build_offset_list(dk_max)
    n_off      = len(offsets)
    avg_power  = np.zeros((H, W),        dtype=np.float64)
    avg_bispec = np.zeros((H, W, n_off), dtype=np.complex128)

    ky = np.where(np.arange(H) > H // 2, np.arange(H) - H, np.arange(H)).astype(float)
    kx = np.where(np.arange(W) > W // 2, np.arange(W) - W, np.arange(W)).astype(float)
    fy2d, fx2d = np.meshgrid(ky, kx, indexing='ij')
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


# ── Phase retrieval ────────────────────────────────────────────────────────

def iterative_reconstruct(
        avg_power:  np.ndarray,
        avg_bispec: np.ndarray,
        offsets:    np.ndarray,
        k_max:      int = KMAX_DEFAULT,
        n_iter:     int = 30,
        progress_cb=None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Knox-Thompson iterative phase retrieval.

    Returns
    -------
    img   : (H, W) float32  reconstructed image (non-negative, fftshifted)
    phase : (H, W) float64  recovered Fourier phase
    """
    H, W  = avg_power.shape
    n_off = len(offsets)

    ky = np.where(np.arange(H) > H // 2, np.arange(H) - H, np.arange(H)).astype(float)
    kx = np.where(np.arange(W) > W // 2, np.arange(W) - W, np.arange(W)).astype(float)
    fy2d, fx2d = np.meshgrid(ky, kx, indexing='ij')
    r2d    = np.sqrt(fy2d**2 + fx2d**2)
    r_soft = 0.9 * k_max
    apod   = np.where(
        r2d <= r_soft, 1.0,
        np.where(r2d < k_max,
                 0.5 * (1.0 + np.cos(np.pi * (r2d - r_soft) / (k_max - r_soft))),
                 0.0))

    noise_region = np.concatenate([
        avg_power[:H//8, :].ravel(), avg_power[-H//8:, :].ravel()])
    bias      = float(np.median(noise_region))
    amplitude = np.sqrt(np.maximum(avg_power - bias, 0.0)) * apod
    k_mask    = apod > 0.0

    bispec_mag = np.abs(avg_bispec)
    bispec_arg = np.angle(avg_bispec)

    ov = np.array([int(vy) % H for vy, vx in offsets], dtype=np.intp)
    ou = np.array([int(vx) % W for vy, vx in offsets], dtype=np.intp)
    row_idx = np.arange(H, dtype=np.intp)
    col_idx = np.arange(W, dtype=np.intp)
    phase   = np.zeros((H, W), dtype=np.float64)

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
    """Return the debias-corrected autocorrelogram (fftshifted)."""
    H, W = avg_power.shape
    noise = np.concatenate([
        avg_power[:H//8, :].ravel(), avg_power[-H//8:, :].ravel()])
    bias     = float(np.median(noise))
    debiased = np.maximum(avg_power - bias, 0.0)
    acorr    = np.real(np.fft.ifft2(debiased))
    return np.fft.fftshift(acorr).astype(np.float32)


def deconvolve_bispectrum(
        avg_bispec_tgt: np.ndarray,
        avg_bispec_ref: np.ndarray,
        epsilon: float = 0.01,
) -> np.ndarray:
    """Wiener deconvolution in bispectrum space."""
    denom_sq = np.abs(avg_bispec_ref) ** 2
    reg      = epsilon * float(denom_sq.mean()) + 1e-30
    return avg_bispec_tgt * np.conj(avg_bispec_ref) / (denom_sq + reg)


# ── NpzReconWorker ─────────────────────────────────────────────────────────

class NpzReconWorker(QThread):
    """Reconstruct from a pre-computed bispectrum .npz (no accumulation step)."""

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

    def stop(self) -> None:
        self._stop = True

    def run(self) -> None:
        try:
            self._process()
        except Exception as e:
            import traceback
            self.error.emit(f"{e}\n{traceback.format_exc()}")

    def _process(self) -> None:
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
                self.status.emit("Reference bispectrum loaded from memory.")
            else:
                self.status.emit("Loading reference bispectrum…")
                ref_bispec_arr = np.load(self.ref_path, allow_pickle=False)['avg_bispec']
            self.progress.emit(40)
            if self._stop: return
            if self.use_deconv:
                self.status.emit(f"Deconvolving  (ε = {self.epsilon:.4f})…")
                avg_bispec  = deconvolve_bispectrum(
                    avg_bispec, ref_bispec_arr, self.epsilon)
                deconv_done = True
            self.progress.emit(50)
        if self._stop: return

        self.status.emit(f"Iterative reconstruction ({self.n_iter} iters)…")
        self.progress.emit(55)

        def _prog(p):
            if self._stop: return
            self.progress.emit(max(55, min(95, 55 + int((p - 82) / 15 * 40))))
            it_done = max(0, round((p - 82) / 15 * self.n_iter))
            self.status.emit(
                f"Iterative reconstruction…  iteration {it_done}/{self.n_iter}")

        recon, phase = iterative_reconstruct(
            avg_power, avg_bispec, offsets,
            k_max=self.k_max, n_iter=self.n_iter, progress_cb=_prog)
        if self._stop: return

        self.progress.emit(100)
        self.status.emit("Done.")
        self.finished.emit({
            'recon':             recon,
            'avg_power':         avg_power,
            'avg_bispec':        avg_bispec,
            'offsets':           offsets,
            'n_frames':          0,
            'roi_size':          H,
            'mean_bispec_mag':   float(np.mean(np.abs(avg_bispec))),
            'mean_abs_phase':    float(np.mean(np.abs(phase))),
            'nonzero_phase_pct': float(np.sum(phase != 0.0)) / phase.size * 100,
            'deconv_done':       deconv_done,
            'ref_bispec':        ref_bispec_arr,
        })


# ── AnalysisWorker ─────────────────────────────────────────────────────────

class AnalysisWorker(QThread):
    """FITS cube → bispectrum accumulation → phase retrieval → reconstructed image."""

    progress = pyqtSignal(int)
    status   = pyqtSignal(str)
    finished = pyqtSignal(object)
    error    = pyqtSignal(str)

    def __init__(self, filepath: str,
                 k_max:    int   = KMAX_DEFAULT,
                 dk_max:   int   = DKMAX_DEFAULT,
                 n_iter:   int   = 30,
                 ref_path: str   = "",
                 ref_bispec=None,
                 use_deconv: bool  = False,
                 epsilon: float = 0.01):
        super().__init__()
        self.filepath   = filepath
        self.k_max      = k_max
        self.dk_max     = dk_max
        self.n_iter     = n_iter
        self.ref_path   = ref_path
        self.ref_bispec = ref_bispec
        self.use_deconv = use_deconv
        self.epsilon    = epsilon
        self._stop      = False

    def stop(self) -> None:
        self._stop = True

    def run(self) -> None:
        try:
            self._process()
        except Exception as e:
            import traceback
            self.error.emit(f"{e}\n{traceback.format_exc()}")

    def _process(self) -> None:
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
            self.progress.emit(5 + int((p - 5) / 77 * (TGT_END - 5)))
            self.status.emit(f"Target bispectrum…  {p-5:.0f}% of frames done")

        avg_power, avg_bispec, offsets, _ = accumulate_bispectrum(
            cube, self.k_max, self.dk_max, _prog_tgt)
        if self._stop: return

        ref_bispec_arr = None
        deconv_done    = False

        if has_ref:
            if self.ref_bispec is not None:
                ref_bispec_arr = self.ref_bispec
                self.status.emit("Reference bispectrum loaded from memory.")
                self.progress.emit(REF_END)
            else:
                self.status.emit("Loading reference FITS cube…")
                self.progress.emit(TGT_END + 1)
                ref_cube, _ = read_fits_cube(self.ref_path)
                rn, rH, rW  = ref_cube.shape
                if (rH, rW) != (H, W):
                    raise ValueError(
                        f"Reference ROI {rH}×{rW} ≠ target ROI {H}×{W}.")
                self.status.emit(
                    f"Accumulating reference bispectrum…  ({rn} frames)")

                def _prog_ref(p):
                    if self._stop: return
                    self.progress.emit(
                        TGT_END + int((p - 5) / 77 * (REF_END - TGT_END)))
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

        iter_start = REF_END + 2 if has_ref else 82
        iter_span  = ITER_END - iter_start
        self.status.emit(
            f"Starting iterative reconstruction ({self.n_iter} iters)…")
        self.progress.emit(iter_start)

        def _iter_prog(p):
            if self._stop: return
            self.progress.emit(iter_start + int((p - 82) / 15 * iter_span))
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

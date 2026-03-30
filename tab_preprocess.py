"""
speckle_suite.tab_preprocess
=============================
Tab 2 — Preprocess: frame selection, registration, ROI crop → FITS cube.
"""

from __future__ import annotations

import numpy as np
from pathlib import Path
from typing import Optional

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QLabel, QPushButton,
    QLineEdit, QGroupBox, QFileDialog, QProgressBar, QComboBox,
    QTextEdit, QCheckBox, QSlider, QFrame,
)
from PyQt6.QtCore import Qt, pyqtSignal

import pyqtgraph as pg

import speckle_suite.theme as theme
from speckle_suite.settings import working_dir
from speckle_suite.widgets import ResultCard, primary_btn_style
from speckle_suite.ser_io import parse_ser_header
from speckle_suite.preprocess_backend import PreprocessWorker

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
            f"background:{theme.PANEL_BG}; border-right:1px solid {theme.BORDER_COLOR};")
        self._left_panel = left
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(14, 14, 14, 14)
        left_layout.setSpacing(10)

        title = QLabel("SPECKLE PREPROCESSING")
        title.setStyleSheet(
            f"font-size:14px; font-weight:bold; color:{theme.ACCENT}; letter-spacing:2px;")
        self._title_lbl = title
        subtitle = QLabel("Frame selection · ROI extraction · Recentring")
        subtitle.setStyleSheet(f"color:{theme.TEXT_MUTED}; font-size:10px;")
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
            l = QLabel(txt); l.setStyleSheet(f"color:{theme.TEXT_MUTED};"); return l

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
        self.file_info_lbl.setStyleSheet(f"color:{theme.TEXT_MUTED}; font-size:10px;")
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
            l.setStyleSheet(f"color:{theme.TEXT_MUTED};")
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
        self.pct_lbl.setStyleSheet(f"color:{theme.ACCENT}; min-width:40px;")
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
        self.run_btn.setStyleSheet(primary_btn_style())
        self.run_btn.setFixedHeight(38)
        self.run_btn.setEnabled(False)
        self.run_btn.clicked.connect(self._run)
        self.stop_btn = QPushButton("■  Stop")
        self.stop_btn.setStyleSheet(primary_btn_style())
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
        self.status_lbl.setStyleSheet(f"color:{theme.TEXT_MUTED}; font-size:10px;")
        self.status_lbl.setWordWrap(True)
        left_layout.addWidget(self.status_lbl)

        sep2 = QFrame(); sep2.setObjectName("separator")
        sep2.setFrameShape(QFrame.Shape.HLine)
        left_layout.addWidget(sep2)

        # Result cards
        cards_lbl = QLabel("RESULTS")
        cards_lbl.setStyleSheet(
            f"color:{theme.TEXT_MUTED}; font-size:10px; letter-spacing:1px;")
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
            lbl.setStyleSheet(f"color:{theme.TEXT_MUTED}; font-size:10px; min-width:34px;")
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
            f"color:{theme.TEXT_MUTED}; font-size:11px;")
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
        self.preview_plot.setStyleSheet(f"background:{theme.DARK_BG};")

    def refresh_styles(self):
        """Called by main window after a theme change."""
        self.run_btn.setStyleSheet(primary_btn_style())
        self.stop_btn.setStyleSheet(primary_btn_style())
        self._apply_graph_theme()
        self._left_panel.setStyleSheet(
            f"background:{theme.PANEL_BG}; border-right:1px solid {theme.BORDER_COLOR};")
        self._title_lbl.setStyleSheet(
            f"font-size:14px; font-weight:bold; color:{theme.ACCENT}; letter-spacing:2px;")
        self._subtitle_lbl.setStyleSheet(
            f"color:{theme.TEXT_MUTED}; font-size:10px;")
        for card in (self.card_total, self.card_selected, self.card_pct_kept,
                     self.card_roi, self.card_thresh, self.card_maxshift):
            card.refresh_style()

    # ── File browsing ──────────────────────────────────────────────────────

    def _browse_file(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Open Speckle Sequence(s)",
            working_dir(),
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
            self.out_edit.text() or working_dir())
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
        self.frame_info_lbl.setStyleSheet(f"color:{theme.TEXT_MUTED}; font-size:11px;")
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
        color  = theme.ACCENT2 if accepted else theme.DANGER
        self.frame_info_lbl.setText(
            f"Frame {idx + 1} / {len(self._all_crops)}   "
            f"Q = {score:.4f}   {status}")
        self.frame_info_lbl.setStyleSheet(
            f"color:{color}; font-size:11px; min-width:280px;")

    # ── Log ────────────────────────────────────────────────────────────────

    def _log(self, msg: str, error: bool = False, warning: bool = False):
        if error:
            color = theme.DANGER
        elif warning:
            color = theme.WARNING
        else:
            color = theme.TEXT_MUTED
        self.log_edit.append(f'<span style="color:{color}">{msg}</span>')


# ═══════════════════════════════════════════════════════════════════════════
#  ── ANALYSIS TAB ────────────────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════


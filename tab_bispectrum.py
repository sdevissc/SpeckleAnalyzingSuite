"""
speckle_suite.tab_bispectrum
============================
Tab 3 -- Bispectrum: FITS/NPZ loading, bispectrum accumulation, phase
retrieval, image reconstruction and bispectrum archiving.
"""

from __future__ import annotations

import numpy as np
from pathlib import Path
from typing import Optional

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QLabel, QPushButton,
    QLineEdit, QGroupBox, QFileDialog, QProgressBar, QSizePolicy,
    QComboBox, QTextEdit, QSplitter, QCheckBox, QSpinBox,
    QRadioButton, QButtonGroup, QDoubleSpinBox, QFrame, QSlider,
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer

import pyqtgraph as pg

import speckle_suite.theme as theme
from speckle_suite.settings import SETTINGS, working_dir
from speckle_suite.widgets import primary_btn_style, get_colormaps, COLORMAP_NAMES, read_fits_cube
from speckle_suite.analysis_backend import (
    KMAX_DEFAULT, DKMAX_DEFAULT,
    build_offset_list, iterative_reconstruct, compute_autocorrelogram,
    deconvolve_bispectrum, AnalysisWorker, NpzReconWorker,
)

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


class BispectrumTab(QWidget):

    def __init__(self, parent=None):
        super().__init__(parent)
        self.worker:   Optional[AnalysisWorker] = None
        self._result:  Optional[dict] = None
        self._ref_bispec:  object = None
        self._input_type:  str   = 'fits'   # 'fits' | 'npz'
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
        left_layout.setContentsMargins(8, 6, 8, 6)
        left_layout.setSpacing(4)

        # Input
        input_group  = QGroupBox("Input")
        input_layout = QGridLayout(input_group)
        input_layout.setVerticalSpacing(4)
        input_layout.setHorizontalSpacing(6)
        input_layout.setContentsMargins(8, 4, 8, 4)
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
        self.file_info_lbl.setStyleSheet(f"color:{theme.TEXT_MUTED}; font-size:10px;")
        input_layout.addWidget(self.file_info_lbl, 1, 0, 1, 3)
        input_layout.setColumnStretch(1, 1)
        left_layout.addWidget(input_group)

        # Reference deconvolution
        ref_group  = QGroupBox("Reference Deconvolution")
        ref_vbox   = QVBoxLayout(ref_group)
        ref_vbox.setSpacing(3)
        ref_vbox.setContentsMargins(8, 4, 8, 4)
        ref_layout = QGridLayout()
        ref_layout.setVerticalSpacing(3)
        ref_layout.setHorizontalSpacing(6)
        ref_vbox.addLayout(ref_layout)

        ref_layout.addWidget(QLabel("Ref. .npz"), 0, 0)
        self.ref_edit = ClickableLineEdit()
        self.ref_edit.setPlaceholderText("Click to select reference bispectrum .npz…")
        self.ref_edit.setReadOnly(True)
        self.ref_edit.setCursor(Qt.CursorShape.PointingHandCursor)
        self.ref_edit.clicked.connect(self._browse_ref)
        ref_clear_btn = QPushButton("Clear")
        ref_clear_btn.setFixedHeight(26)
        ref_clear_btn.setMinimumWidth(52)
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
        self.ref_info_lbl.setStyleSheet(f"color:{theme.TEXT_MUTED}; font-size:9px;")
        self.ref_info_lbl.setWordWrap(True)
        self.ref_info_lbl.setVisible(False)
        ref_vbox.addWidget(self.ref_info_lbl)
        left_layout.addWidget(ref_group)

        # K-space / bispectrum parameters
        kspace_group = QGroupBox("Bispectrum Calculation")
        kspace_vbox  = QVBoxLayout(kspace_group)
        kspace_vbox.setSpacing(4)
        kspace_vbox.setContentsMargins(8, 4, 8, 6)
        kspace_row   = QHBoxLayout()
        kspace_row.setSpacing(6)
        kspace_vbox.addLayout(kspace_row)

        def _kfield(label_txt, widget):
            col = QVBoxLayout()
            col.setSpacing(2)
            lbl = QLabel(label_txt)
            lbl.setStyleSheet(f"color:{theme.TEXT_MUTED}; font-size:9px;")
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
        self.status_lbl.setStyleSheet(f"color:{theme.TEXT_MUTED}; font-size:10px;")
        self.status_lbl.setWordWrap(True)
        kspace_vbox.addWidget(self.status_lbl)

        left_layout.addWidget(kspace_group)
        left_layout.addStretch()

        root.addWidget(left)

        # ── Right panel ────────────────────────────────────────────────────
        right = QWidget()
        right_layout = QHBoxLayout(right)
        right_layout.setContentsMargins(10, 10, 10, 10)
        right_layout.setSpacing(10)

        # Reconstructed image
        recon_group  = QGroupBox("Reconstructed Image")
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
            f"color:{theme.TEXT_PRIMARY}; font-size:13px; font-weight:bold;")
        self.nav_label.setFixedWidth(60)
        self.nav_file_label = QLabel("")
        self.nav_file_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.nav_file_label.setStyleSheet(
            f"color:{theme.TEXT_MUTED}; font-size:10px;")
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
        recon_layout.addWidget(self.recon_view)

        # Level sliders
        slider_grid = QGridLayout()
        slider_grid.setVerticalSpacing(2)

        def _level_slider():
            s = QSlider(Qt.Orientation.Horizontal)
            s.setRange(0, 255); s.setFixedHeight(18)
            return s

        lbl_min = QLabel("Min")
        lbl_min.setStyleSheet(f"color:{theme.TEXT_MUTED}; font-size:9px;")
        lbl_min.setFixedWidth(24)
        lbl_max = QLabel("Max")
        lbl_max.setStyleSheet(f"color:{theme.TEXT_MUTED}; font-size:9px;")
        lbl_max.setFixedWidth(24)
        self.level_min_slider = _level_slider()
        self.level_max_slider = _level_slider()
        self.level_max_slider.setValue(255)
        self.level_min_lbl = QLabel("0")
        self.level_min_lbl.setStyleSheet(f"color:{theme.TEXT_MUTED}; font-size:9px;")
        self.level_min_lbl.setFixedWidth(28)
        self.level_max_lbl = QLabel("255")
        self.level_max_lbl.setStyleSheet(f"color:{theme.TEXT_MUTED}; font-size:9px;")
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
        cmap_lbl_r.setStyleSheet(f"color:{theme.TEXT_MUTED}; font-size:9px;")
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


        right_layout.addWidget(right_col, 2)
        root.addWidget(right)
        root.setSizes([420, 980])

        self._apply_graph_theme()

    # ── Graph theme ────────────────────────────────────────────────────────

    def _apply_graph_theme(self):
        self.recon_view.setStyleSheet(f"background:{theme.DARK_BG};")

    def refresh_styles(self):
        """Called by main window after a theme change."""
        self.run_btn.setStyleSheet(primary_btn_style())
        self.stop_btn.setStyleSheet(primary_btn_style())
        self._apply_graph_theme()
        self.nav_label.setStyleSheet(
            f"color:{theme.TEXT_PRIMARY}; font-size:13px; font-weight:bold;")
        self.nav_file_label.setStyleSheet(
            f"color:{theme.TEXT_MUTED}; font-size:10px;")
        for s_lbl in (self.level_min_lbl, self.level_max_lbl):
            s_lbl.setStyleSheet(f"color:{theme.TEXT_MUTED}; font-size:9px;")


    # ── Colormap ───────────────────────────────────────────────────────────

    def _apply_recon_cmap(self, name: str):
        cm = get_colormaps().get(name)
        if cm is not None:
            self.recon_view.setColorMap(cm)

    # ── Reference helpers ──────────────────────────────────────────────────

    def _browse_ref(self):
        """Load a pre-computed reference bispectrum .npz.
        Deconvolution is applied automatically whenever a ref is present."""
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Reference Bispectrum",
            working_dir(),
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
            working_dir(),
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
            self.status_lbl.setText(f"Ready — {n} file(s) loaded.")
            # Re-enable Run so the user can re-run with different parameters
            self.run_btn.setEnabled(True)
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

        # Reset navigator — a fresh run always starts clean
        self._nav_paths  = []
        self._nav_idx    = 0
        self._nav_memory = {}
        self.nav_bar.setVisible(False)


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
        self._clear_all(silent=True)
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
            out_path = Path(working_dir()) / (p.stem + '_bispec.npz')
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
            'result': self._result,
        }

    def _nav_go(self, idx: int):
        self._nav_save_current()
        self._nav_idx = idx
        self._update_nav_bar()
        path = self._nav_paths[idx]
        mem  = self._nav_memory.get(path)
        if mem and mem['result'] is not None:
            self._on_finished(mem['result'])
            self._log(f"\u25c4\u25ba {Path(path).name}")
        else:
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
                lambda _, p=path: self._nav_memory.update({
                    p: {'result': self._result}}))
            self.worker.start()

    def _nav_prev(self):
        if self._nav_idx > 0:
            self._nav_go(self._nav_idx - 1)

    def _nav_next(self):
        if self._nav_idx < len(self._nav_paths) - 1:
            self._nav_go(self._nav_idx + 1)

    def _clear_all(self, silent: bool = False):
        if not silent:
            self._log("Display cleared.")

    # ── Save bispectrum ────────────────────────────────────────────────────

    def _save_bispec(self):
        if self._result is None or self._result.get('avg_bispec') is None:
            self._log("⚠ No bispectrum available — run the analysis first.",
                      error=True)
            return
        fits_stem = (Path(self.file_edit.text()).stem
                     if self.file_edit.text() else "target")
        default   = str(Path(working_dir()) / f"{fits_stem}_bispec.npz")
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

    # ── Log ────────────────────────────────────────────────────────────────

    def _log(self, msg: str, error: bool = False, warning: bool = False):
        if error:
            color = theme.DANGER
        elif warning:
            color = theme.WARNING
        else:
            color = theme.TEXT_MUTED
        self.log_edit.append(f'<span style="color:{color}">{msg}</span>')



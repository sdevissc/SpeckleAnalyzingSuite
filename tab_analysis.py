"""
speckle_suite.tab_analysis
============================
Tab 3 — Analysis: bispectrum accumulation, phase retrieval,
image reconstruction, and (ρ, θ) astrometry.
"""

from __future__ import annotations

import csv
import json as _json
import numpy as np
from datetime import date
from pathlib import Path
from typing import Optional

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QLabel, QPushButton,
    QLineEdit, QGroupBox, QFileDialog, QProgressBar, QSizePolicy,
    QComboBox, QTextEdit, QSplitter, QCheckBox, QSpinBox,
    QRadioButton, QButtonGroup, QDoubleSpinBox, QSlider,
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
        self.file_info_lbl.setStyleSheet(f"color:{theme.TEXT_MUTED}; font-size:10px;")
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
        self.ref_info_lbl.setStyleSheet(f"color:{theme.TEXT_MUTED}; font-size:9px;")
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
        self.cal_status_lbl.setStyleSheet(f"color:{theme.TEXT_MUTED}; font-size:9px;")
        self.cal_status_lbl.setWordWrap(True)
        cal_layout.addWidget(self.cal_status_lbl)

        cal_sep = QFrame(); cal_sep.setFrameShape(QFrame.Shape.HLine)
        cal_sep.setStyleSheet(f"color:{theme.BORDER_COLOR};")
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
            val.setStyleSheet(f"font-size:10px; color:{theme.TEXT_PRIMARY};")
            val.setAlignment(
                Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
            unit = QLabel(unit_txt)
            unit.setFixedWidth(20)
            unit.setStyleSheet(f"font-size:10px; color:{theme.TEXT_MUTED};")
            sig = QLabel("")
            sig.setStyleSheet(f"font-size:10px; color:{theme.TEXT_MUTED};")
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
            f"QRadioButton {{ color: {theme.DANGER}; font-weight: bold; }}"
            f"QRadioButton::indicator:checked {{ background: {theme.DANGER}; "
            f"border: 2px solid {theme.DANGER}; border-radius: 6px; }}"
            f"QRadioButton:disabled {{ color: {theme.TEXT_MUTED}; }}")
        self.companion_radio.setStyleSheet(
            f"QRadioButton {{ color: {theme.ACCENT2}; font-weight: bold; }}"
            f"QRadioButton::indicator:checked {{ background: {theme.ACCENT2}; "
            f"border: 2px solid {theme.ACCENT2}; border-radius: 6px; }}"
            f"QRadioButton:disabled {{ color: {theme.TEXT_MUTED}; }}")
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
        self.csv_path_lbl.setStyleSheet(f"color:{theme.TEXT_MUTED}; font-size:9px;")
        self.csv_path_lbl.setWordWrap(True)
        right_col_layout.addWidget(self.csv_path_lbl)

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
                    lbl.setStyleSheet(f"font-size:10px; color:{theme.TEXT_MUTED};")
                else:  # value or sigma
                    lbl.setStyleSheet(f"font-size:10px; color:{theme.TEXT_PRIMARY};")
        self.nav_label.setStyleSheet(
            f"color:{theme.TEXT_PRIMARY}; font-size:13px; font-weight:bold;")
        self.nav_file_label.setStyleSheet(
            f"color:{theme.TEXT_MUTED}; font-size:10px;")
        for s_lbl in (self.level_min_lbl, self.level_max_lbl):
            s_lbl.setStyleSheet(f"color:{theme.TEXT_MUTED}; font-size:9px;")
        self.cal_status_lbl.setStyleSheet(
            f"color:{theme.TEXT_MUTED}; font-size:9px;")

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
            'pen': pg.mkPen(theme.DANGER, width=2),
            'brush': pg.mkBrush(None), 'symbol': 'o'}])
        cross = pg.ScatterPlotItem([{
            'pos': (x, y), 'size': 8,
            'pen': pg.mkPen(theme.DANGER, width=1.5),
            'brush': pg.mkBrush(theme.DANGER), 'symbol': '+'}])
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
            'pen': pg.mkPen(theme.ACCENT2, width=2),
            'brush': pg.mkBrush(None), 'symbol': 'o'}])
        cross = pg.ScatterPlotItem([{
            'pos': (x, y), 'size': 8,
            'pen': pg.mkPen(theme.ACCENT2, width=1.5),
            'brush': pg.mkBrush(theme.ACCENT2), 'symbol': '+'}])
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

    # ── Calibration ────────────────────────────────────────────────────────

    def _load_cal_dialog(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Load Calibration JSON",
            working_dir(),
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
            str(Path(working_dir()) / default),
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
            str(Path(working_dir()) / "speckle_log.csv"),
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
            str(Path(working_dir()) / default),
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
            color = theme.DANGER
        elif warning:
            color = theme.WARNING
        else:
            color = theme.TEXT_MUTED
        self.log_edit.append(f'<span style="color:{color}">{msg}</span>')



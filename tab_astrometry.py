"""
speckle_suite.tab_astrometry
=============================
Tab 4 -- Astrometry & Photometry: load a reconstructed image (.npz or the
result dict from the Bispectrum tab), place primary/secondary markers,
apply astrometric calibration, perform aperture photometry (delta-magnitude),
and export results as JSON / CSV / WDS report.
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
    QRadioButton, QButtonGroup, QDoubleSpinBox, QFrame, QSlider,
)
from PyQt6.QtCore import Qt, pyqtSignal, QTimer

import pyqtgraph as pg
import numpy as np

import speckle_suite.theme as theme
from speckle_suite.settings import working_dir
from speckle_suite.widgets import (
    primary_btn_style, get_colormaps, COLORMAP_NAMES, read_fits_cube,
)
from speckle_suite.analysis_backend import (
    iterative_reconstruct, NpzReconWorker,
)


# ── Clickable helpers (also used by BispectrumTab) ────────────────────────

class ClickableLineEdit(QLineEdit):
    clicked = pyqtSignal()
    def mousePressEvent(self, event):
        super().mousePressEvent(event)
        self.clicked.emit()


class ClickableImageView(pg.ImageView):
    clicked = pyqtSignal(float, float)
    hovered = pyqtSignal(float, float)

    def mousePressEvent(self, ev):
        if ev.button() == Qt.MouseButton.LeftButton:
            vb  = self.getView()
            pos = vb.mapSceneToView(ev.position())
            self.clicked.emit(pos.x(), pos.y())
        super().mousePressEvent(ev)

    def _connect_hover(self):
        scene = self.getView().scene()
        scene.sigMouseMoved.connect(self._on_scene_mouse_moved)

    def _on_scene_mouse_moved(self, scene_pos):
        vb = self.getView()
        if vb.sceneBoundingRect().contains(scene_pos):
            pos = vb.mapSceneToView(scene_pos)
            self.hovered.emit(pos.x(), pos.y())


# ── AstrometryTab ─────────────────────────────────────────────────────────

class AstrometryTab(QWidget):
    """
    Tab 4 -- Astrometry & Photometry.

    Workflow
    --------
    1. Browse a .npz bispectrum file (produced by Tab 3 or auto-saved).
    2. The reconstructed image is displayed immediately.
    3. Click to place Primary (red) and Secondary (green) markers.
    4. Load a calibration JSON to convert pixel measurements to sky coords.
    5. Optionally run aperture photometry to measure delta-magnitude.
    6. Export result as JSON / CSV / WDS report.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        # Image state
        self._recon:       Optional[np.ndarray] = None
        self._avg_bispec:  Optional[np.ndarray] = None
        self._avg_power:   Optional[np.ndarray] = None
        self._npz_path:    str = ""
        self._worker:      Optional[NpzReconWorker] = None

        # Multi-file navigator
        self._nav_paths:  list = []
        self._nav_idx:    int  = 0
        self._nav_memory: dict = {}   # path -> {result, primary_pos, companion_pos, meas}
        self._nav_load_queue: list = []

        # Marker state
        self._primary_marker:   object = None
        self._companion_marker: object = None
        self._primary_pos:   Optional[tuple[float, float]] = None
        self._companion_pos: Optional[tuple[float, float]] = None
        self._click_mode = 'primary'
        self._cursor_circle_item:    object = None
        self._aperture_circle_items: list   = []

        # Measurement state
        self._meas_rho:            Optional[float] = None
        self._meas_theta:          Optional[float] = None
        self._meas_rho_sky:        Optional[float] = None
        self._meas_theta_sky:      Optional[float] = None
        self._meas_sigma_rho_cal:  float = 0.0
        self._meas_sigma_theta_cal: float = 0.0
        self._meas_delta_mag:      Optional[float] = None

        # Calibration state
        self._cal_file: str  = ""
        self._cal:      dict = {}
        self._csv_path: str  = ""

        # Background
        self._bg_level: Optional[float] = None

        self._build_ui()

    # ── UI ─────────────────────────────────────────────────────────────────

    def _build_ui(self):
        root = QSplitter(Qt.Orientation.Horizontal)
        outer = QHBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(root)

        # ── Left panel ─────────────────────────────────────────────────────
        left = QWidget()
        left.setFixedWidth(420)
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(10, 10, 10, 10)
        left_layout.setSpacing(8)

        # Input
        input_group  = QGroupBox("Input (.npz bispectrum)")
        input_layout = QGridLayout(input_group)
        input_layout.setVerticalSpacing(6)
        input_layout.setHorizontalSpacing(8)
        input_layout.addWidget(QLabel("File"), 0, 0)
        self.file_edit = QLineEdit()
        self.file_edit.setPlaceholderText("Select bispectrum .npz…")
        self.file_edit.setReadOnly(True)
        browse_btn = QPushButton("Browse")
        browse_btn.setMinimumWidth(80)
        browse_btn.clicked.connect(self._browse_file)
        input_layout.addWidget(self.file_edit, 0, 1)
        input_layout.addWidget(browse_btn,     0, 2)
        self.file_info_lbl = QLabel("")
        self.file_info_lbl.setStyleSheet(
            f"color:{theme.TEXT_MUTED}; font-size:10px;")
        input_layout.addWidget(self.file_info_lbl, 1, 0, 1, 3)
        input_layout.setColumnStretch(1, 1)
        left_layout.addWidget(input_group)

        # Reconstruction params (for re-run with different iterations)
        recon_group = QGroupBox("Reconstruction")
        recon_vbox  = QVBoxLayout(recon_group)
        recon_vbox.setSpacing(6)
        recon_row = QHBoxLayout()
        recon_row.setSpacing(6)

        def _field(label_txt, widget):
            col = QVBoxLayout(); col.setSpacing(2)
            lbl = QLabel(label_txt)
            lbl.setStyleSheet(f"color:{theme.TEXT_MUTED}; font-size:9px;")
            lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            col.addWidget(lbl); col.addWidget(widget)
            recon_row.addLayout(col, 1)

        self.kmax_spin = QSpinBox()
        self.kmax_spin.setRange(4, 512); self.kmax_spin.setValue(60)
        self.kmax_spin.setSuffix(" px"); self.kmax_spin.setMinimumHeight(28)
        self.niter_spin = QSpinBox()
        self.niter_spin.setRange(1, 200); self.niter_spin.setValue(30)
        self.niter_spin.setMinimumHeight(28)
        _field("Kmax",       self.kmax_spin)
        _field("Iterations", self.niter_spin)

        self.rerun_btn = QPushButton("Re-run reconstruction")
        self.rerun_btn.setStyleSheet(primary_btn_style())
        self.rerun_btn.setFixedHeight(32)
        self.rerun_btn.setEnabled(False)
        self.rerun_btn.clicked.connect(self._rerun_reconstruction)

        recon_vbox.addLayout(recon_row)
        recon_vbox.addWidget(self.rerun_btn)
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        recon_vbox.addWidget(self.progress_bar)
        left_layout.addWidget(recon_group)

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

        self.cal_status_lbl = QLabel("No calibration loaded.")
        self.cal_status_lbl.setStyleSheet(
            f"color:{theme.TEXT_MUTED}; font-size:9px;")
        self.cal_status_lbl.setWordWrap(True)
        cal_layout.addWidget(self.cal_status_lbl)

        sep = QFrame(); sep.setFrameShape(QFrame.Shape.HLine)
        sep.setStyleSheet(f"color:{theme.BORDER_COLOR};")
        cal_layout.addWidget(sep)

        def _cal_row(label_txt, lo, hi, decimals, default, has_err):
            row_w = QWidget()
            row   = QHBoxLayout(row_w)
            row.setContentsMargins(0, 0, 0, 0); row.setSpacing(4)
            lbl = QLabel(label_txt)
            lbl.setFixedWidth(88); lbl.setStyleSheet("font-size:10px;")
            spin = QDoubleSpinBox()
            spin.setRange(lo, hi); spin.setDecimals(decimals)
            spin.setValue(default); spin.setMinimumHeight(24)
            row.addWidget(lbl); row.addWidget(spin, 2)
            err = None
            if has_err:
                pm = QLabel("±"); pm.setFixedWidth(12)
                pm.setAlignment(Qt.AlignmentFlag.AlignCenter)
                err = QDoubleSpinBox()
                err.setRange(0, hi); err.setDecimals(max(decimals + 2, 8))
                err.setValue(0.0); err.setMinimumHeight(24)
                row.addWidget(pm); row.addWidget(err, 1)
            return row_w, spin, err

        row_scale, self.cal_scale_spin, self.cal_scale_err = _cal_row(
            "Pixel scale", 0.0001, 10.0, 6, 0.065, True)
        row_angle, self.cal_angle_spin, self.cal_angle_err = _cal_row(
            "Camera angle", 0.0, 360.0, 4, 0.0, True)
        for sp in (self.cal_scale_spin, self.cal_scale_err,
                   self.cal_angle_spin,  self.cal_angle_err):
            sp.valueChanged.connect(self._update_measurement)
        cal_layout.addWidget(row_scale)
        cal_layout.addWidget(row_angle)
        left_layout.addWidget(cal_group)

        # Star placement controls (radio buttons are placed below the image)
        detect_group  = QGroupBox("Star Placement")
        detect_layout = QVBoxLayout(detect_group)
        detect_layout.setSpacing(6)

        self.clear_btn = QPushButton("Clear All")
        self.clear_btn.setEnabled(False)
        self.clear_btn.clicked.connect(self._clear_all)
        detect_layout.addWidget(self.clear_btn)

        # Cursor aperture radius / background
        cursor_row = QHBoxLayout(); cursor_row.setSpacing(6)
        cursor_lbl = QLabel("Cursor r:")
        cursor_lbl.setStyleSheet("font-size:10px;")
        self.cursor_radius_spin = QSpinBox()
        self.cursor_radius_spin.setRange(1, 60)
        self.cursor_radius_spin.setValue(7)
        self.cursor_radius_spin.setSuffix(" px")
        self.cursor_radius_spin.setFixedWidth(62)
        self.cursor_radius_spin.setEnabled(False)
        self.cursor_radius_spin.valueChanged.connect(self._on_cursor_radius_changed)
        self.bg_sample_btn = QPushButton("Sample BG")
        self.bg_sample_btn.setCheckable(True)
        self.bg_sample_btn.setEnabled(False)
        self.bg_subtract_chk = QCheckBox("- BG")
        self.bg_subtract_chk.setEnabled(False)
        self.bg_subtract_chk.toggled.connect(self._on_bg_subtract_toggled)
        cursor_row.addWidget(cursor_lbl)
        cursor_row.addWidget(self.cursor_radius_spin)
        cursor_row.addWidget(self.bg_sample_btn)
        cursor_row.addWidget(self.bg_subtract_chk)
        cursor_row.addStretch()
        detect_layout.addLayout(cursor_row)
        left_layout.addWidget(detect_group)

        # Results
        results_group  = QGroupBox("Results")
        results_layout = QVBoxLayout(results_group)
        results_layout.setSpacing(3)

        def _result_row(label_txt, unit_txt):
            row_w = QWidget()
            row   = QHBoxLayout(row_w)
            row.setContentsMargins(0, 0, 0, 0); row.setSpacing(4)
            lbl = QLabel(label_txt)
            lbl.setFixedWidth(88); lbl.setStyleSheet("font-size:10px;")
            val = QLabel("—")
            val.setStyleSheet(f"font-size:10px; color:{theme.TEXT_PRIMARY};")
            val.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
            unit = QLabel(unit_txt)
            unit.setFixedWidth(20)
            unit.setStyleSheet(f"font-size:10px; color:{theme.TEXT_MUTED};")
            sig = QLabel("")
            sig.setStyleSheet(f"font-size:10px; color:{theme.TEXT_MUTED};")
            sig.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
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
        self.card_delta_mag = _result_row("ΔM",         "mag")
        for w in (self.card_theta, self.card_rho,
                  self.card_theta_sky, self.card_rho_sky, self.card_delta_mag):
            results_layout.addWidget(w)
        left_layout.addWidget(results_group)

        # Aperture photometry
        phot_group  = QGroupBox("Aperture Photometry")
        phot_layout = QGridLayout(phot_group)
        phot_layout.setSpacing(6)

        def _ap_spin(val, lo=1, hi=60):
            s = QSpinBox()
            s.setRange(lo, hi); s.setValue(val)
            s.setSuffix(" px"); s.setMinimumHeight(26)
            s.setEnabled(False)   # enabled once image is loaded
            return s

        phot_layout.addWidget(QLabel("Inner (star)"),   0, 0)
        self.ap_inner_spin = _ap_spin(7)
        self.ap_inner_spin.setToolTip(
            "Aperture radius enclosing the star flux")
        phot_layout.addWidget(self.ap_inner_spin, 0, 1)

        phot_layout.addWidget(QLabel("Sky inner"),      1, 0)
        self.ap_sky_in_spin = _ap_spin(10)
        self.ap_sky_in_spin.setToolTip(
            "Inner radius of the sky annulus (gap between star and sky)")
        phot_layout.addWidget(self.ap_sky_in_spin, 1, 1)

        phot_layout.addWidget(QLabel("Sky outer"),      2, 0)
        self.ap_sky_out_spin = _ap_spin(15)
        self.ap_sky_out_spin.setToolTip(
            "Outer radius of the sky annulus")
        phot_layout.addWidget(self.ap_sky_out_spin, 2, 1)

        # Enforce D1 < D2 < D3 live and redraw circles if overlay is on
        def _enforce_ap_order():
            r1 = self.ap_inner_spin.value()
            r2 = self.ap_sky_in_spin.value()
            r3 = self.ap_sky_out_spin.value()
            if r2 <= r1:
                self.ap_sky_in_spin.blockSignals(True)
                self.ap_sky_in_spin.setValue(r1 + 1)
                self.ap_sky_in_spin.blockSignals(False)
                r2 = r1 + 1
            if r3 <= r2:
                self.ap_sky_out_spin.blockSignals(True)
                self.ap_sky_out_spin.setValue(r2 + 1)
                self.ap_sky_out_spin.blockSignals(False)
            if self.show_apertures_chk.isChecked():
                self._draw_aperture_circles()
        self.ap_inner_spin.valueChanged.connect(_enforce_ap_order)
        self.ap_sky_in_spin.valueChanged.connect(_enforce_ap_order)
        self.ap_sky_out_spin.valueChanged.connect(_enforce_ap_order)

        self.measure_phot_btn = QPushButton("Measure ΔM")
        self.measure_phot_btn.setStyleSheet(primary_btn_style())
        self.measure_phot_btn.setFixedHeight(32)
        self.measure_phot_btn.setEnabled(False)
        self.measure_phot_btn.clicked.connect(self._measure_photometry)
        phot_layout.addWidget(self.measure_phot_btn, 3, 0, 1, 2)
        left_layout.addWidget(phot_group)

        left_layout.addStretch()
        root.addWidget(left)

        # ── Right panel ─────────────────────────────────────────────────────
        right = QWidget()
        right_layout = QHBoxLayout(right)
        right_layout.setContentsMargins(10, 10, 10, 10)
        right_layout.setSpacing(10)

        # Image
        recon_group2  = QGroupBox("Reconstructed Image  (click to place markers)")
        recon_layout2 = QVBoxLayout(recon_group2)
        recon_layout2.setSpacing(4)

        # Navigator bar (multi-npz batches)
        self.nav_bar = QWidget()
        nav_row = QHBoxLayout(self.nav_bar)
        nav_row.setContentsMargins(4, 2, 4, 2); nav_row.setSpacing(6)
        self.nav_prev_btn = QPushButton("◄◄")
        self.nav_prev_btn.setFixedWidth(52); self.nav_prev_btn.setFixedHeight(30)
        self.nav_prev_btn.setStyleSheet("font-size:14px; font-weight:bold; padding:2px 6px;")
        self.nav_prev_btn.clicked.connect(self._nav_prev)
        self.nav_next_btn = QPushButton("►►")
        self.nav_next_btn.setFixedWidth(52); self.nav_next_btn.setFixedHeight(30)
        self.nav_next_btn.setStyleSheet("font-size:14px; font-weight:bold; padding:2px 6px;")
        self.nav_next_btn.clicked.connect(self._nav_next)
        self.nav_label = QLabel("")
        self.nav_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.nav_label.setStyleSheet(f"color:{theme.TEXT_PRIMARY}; font-size:13px; font-weight:bold;")
        self.nav_label.setFixedWidth(60)
        self.nav_file_label = QLabel("")
        self.nav_file_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.nav_file_label.setStyleSheet(f"color:{theme.TEXT_MUTED}; font-size:10px;")
        self.nav_file_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        nav_row.addWidget(self.nav_prev_btn)
        nav_row.addWidget(self.nav_label)
        nav_row.addWidget(self.nav_file_label, 1)
        nav_row.addWidget(self.nav_next_btn)
        self.nav_bar.setVisible(False)
        recon_layout2.addWidget(self.nav_bar)

        self.recon_view = ClickableImageView()
        self.recon_view.ui.roiBtn.hide()
        self.recon_view.ui.menuBtn.hide()
        self.recon_view.ui.histogram.hide()
        self.recon_view.clicked.connect(self._on_recon_click)
        self.recon_view.hovered.connect(self._on_recon_hover)
        self.bg_sample_btn.clicked.connect(self._on_bg_sample_clicked)
        QTimer.singleShot(0, self.recon_view._connect_hover)
        recon_layout2.addWidget(self.recon_view)

        # Level sliders
        slider_grid = QGridLayout(); slider_grid.setVerticalSpacing(2)
        def _level_slider():
            s = QSlider(Qt.Orientation.Horizontal)
            s.setRange(0, 255); s.setFixedHeight(18)
            return s
        lbl_min = QLabel("Min"); lbl_min.setStyleSheet(f"color:{theme.TEXT_MUTED}; font-size:9px;"); lbl_min.setFixedWidth(24)
        lbl_max = QLabel("Max"); lbl_max.setStyleSheet(f"color:{theme.TEXT_MUTED}; font-size:9px;"); lbl_max.setFixedWidth(24)
        self.level_min_slider = _level_slider()
        self.level_max_slider = _level_slider(); self.level_max_slider.setValue(255)
        self.level_min_lbl = QLabel("0"); self.level_min_lbl.setStyleSheet(f"color:{theme.TEXT_MUTED}; font-size:9px;"); self.level_min_lbl.setFixedWidth(28)
        self.level_max_lbl = QLabel("255"); self.level_max_lbl.setStyleSheet(f"color:{theme.TEXT_MUTED}; font-size:9px;"); self.level_max_lbl.setFixedWidth(28)
        def _on_min_changed(v):
            if v >= self.level_max_slider.value(): v = self.level_max_slider.value() - 1; self.level_min_slider.setValue(v)
            self.level_min_lbl.setText(str(v)); self.recon_view.setLevels(v, self.level_max_slider.value())
        def _on_max_changed(v):
            if v <= self.level_min_slider.value(): v = self.level_min_slider.value() + 1; self.level_max_slider.setValue(v)
            self.level_max_lbl.setText(str(v)); self.recon_view.setLevels(self.level_min_slider.value(), v)
        self.level_min_slider.valueChanged.connect(_on_min_changed)
        self.level_max_slider.valueChanged.connect(_on_max_changed)
        slider_grid.addWidget(lbl_min,               0, 0); slider_grid.addWidget(self.level_min_slider, 0, 1); slider_grid.addWidget(self.level_min_lbl, 0, 2)
        slider_grid.addWidget(lbl_max,               1, 0); slider_grid.addWidget(self.level_max_slider, 1, 1); slider_grid.addWidget(self.level_max_lbl, 1, 2)
        slider_grid.setColumnStretch(1, 1)
        recon_layout2.addLayout(slider_grid)

        cmap_row = QHBoxLayout(); cmap_row.setContentsMargins(4, 0, 4, 2)
        cmap_lbl = QLabel("Colormap"); cmap_lbl.setStyleSheet(f"color:{theme.TEXT_MUTED}; font-size:9px;")
        self.recon_cmap_combo = QComboBox(); self.recon_cmap_combo.addItems(COLORMAP_NAMES)
        self.recon_cmap_combo.setFixedHeight(22); self.recon_cmap_combo.setStyleSheet("font-size:9px;")
        self.recon_cmap_combo.currentTextChanged.connect(self._apply_recon_cmap)
        cmap_row.addWidget(cmap_lbl); cmap_row.addWidget(self.recon_cmap_combo, 1)
        recon_layout2.addLayout(cmap_row)

        # Aperture overlay toggle
        self.show_apertures_chk = QCheckBox("Show apertures")
        self.show_apertures_chk.setEnabled(False)
        self.show_apertures_chk.toggled.connect(self._redraw_markers)
        recon_layout2.addWidget(self.show_apertures_chk)

        # ── Marker placement radio buttons ─────────────────────────────
        marker_row = QHBoxLayout()
        marker_row.setSpacing(12)
        marker_row.setContentsMargins(4, 4, 4, 2)

        self._mode_group     = QButtonGroup()
        self.primary_radio   = QRadioButton("Place Primary")
        self.companion_radio = QRadioButton("Place Secondary")
        self.primary_radio.setChecked(True)
        self.primary_radio.setEnabled(False)
        self.companion_radio.setEnabled(False)
        self.primary_radio.setStyleSheet(
            f"QRadioButton {{ color:{theme.DANGER}; font-weight:bold; }}"
            f"QRadioButton::indicator:checked {{ background:{theme.DANGER}; "
            f"border:2px solid {theme.DANGER}; border-radius:6px; }}"
            f"QRadioButton:disabled {{ color:{theme.TEXT_MUTED}; }}")
        self.companion_radio.setStyleSheet(
            f"QRadioButton {{ color:{theme.ACCENT2}; font-weight:bold; }}"
            f"QRadioButton::indicator:checked {{ background:{theme.ACCENT2}; "
            f"border:2px solid {theme.ACCENT2}; border-radius:6px; }}"
            f"QRadioButton:disabled {{ color:{theme.TEXT_MUTED}; }}")
        self._mode_group.addButton(self.primary_radio)
        self._mode_group.addButton(self.companion_radio)
        self.primary_radio.toggled.connect(
            lambda checked: checked and self._set_click_mode('primary'))
        self.companion_radio.toggled.connect(
            lambda checked: checked and self._set_click_mode('companion'))
        marker_row.addStretch()
        marker_row.addWidget(self.primary_radio)
        marker_row.addWidget(self.companion_radio)
        marker_row.addStretch()
        recon_layout2.addLayout(marker_row)

        right_layout.addWidget(recon_group2, 3)

        # Right column: log + save
        right_col = QWidget()
        right_col_layout = QVBoxLayout(right_col)
        right_col_layout.setContentsMargins(0, 0, 0, 0)
        right_col_layout.setSpacing(8)

        log_group = QGroupBox("Log")
        log_layout = QVBoxLayout(log_group)
        self.log_edit = QTextEdit(); self.log_edit.setReadOnly(True); self.log_edit.setMinimumHeight(160)
        log_layout.addWidget(self.log_edit)
        log_group.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        right_col_layout.addWidget(log_group, 1)

        self.save_png_btn = QPushButton("Save Image (.png)")
        self.save_png_btn.setEnabled(False)
        self.save_png_btn.setToolTip(
            "Export the reconstructed image as PNG.\n"
            "Markers and apertures are included.\n"
            "Can be used as an overlay in the History tab.")
        self.save_png_btn.clicked.connect(self._save_png)
        right_col_layout.addWidget(self.save_png_btn)

        self.save_result_btn = QPushButton("Save Result (.json)")
        self.save_result_btn.setEnabled(False)
        self.save_result_btn.clicked.connect(self._save_result)
        right_col_layout.addWidget(self.save_result_btn)

        output_row = QHBoxLayout(); output_row.setSpacing(6)
        self.append_csv_btn = QPushButton("CSV Log")
        self.append_csv_btn.setEnabled(False)
        self.append_csv_btn.clicked.connect(self._append_csv)
        self.save_wds_btn = QPushButton("WDS Report")
        self.save_wds_btn.setEnabled(False)
        self.save_wds_btn.clicked.connect(self._save_wds)
        output_row.addWidget(self.append_csv_btn); output_row.addWidget(self.save_wds_btn)
        right_col_layout.addLayout(output_row)

        self.csv_path_lbl = QLabel("No CSV log set.")
        self.csv_path_lbl.setStyleSheet(f"color:{theme.TEXT_MUTED}; font-size:9px;")
        self.csv_path_lbl.setWordWrap(True)
        right_col_layout.addWidget(self.csv_path_lbl)

        right_layout.addWidget(right_col, 2)
        root.addWidget(right)
        root.setSizes([420, 980])

        self._apply_graph_theme()

    # ── Theme ──────────────────────────────────────────────────────────────

    def _apply_graph_theme(self):
        self.recon_view.setStyleSheet(f"background:{theme.DARK_BG};")

    def refresh_styles(self):
        self.rerun_btn.setStyleSheet(primary_btn_style())
        self.measure_phot_btn.setStyleSheet(primary_btn_style())
        self._apply_graph_theme()
        self.cal_status_lbl.setStyleSheet(f"color:{theme.TEXT_MUTED}; font-size:9px;")
        for w in (self.card_theta, self.card_rho,
                  self.card_theta_sky, self.card_rho_sky, self.card_delta_mag):
            for lbl in w.findChildren(QLabel):
                if lbl.minimumWidth() == 88:
                    lbl.setStyleSheet("font-size:10px;")
                elif lbl.minimumWidth() == 20:
                    lbl.setStyleSheet(f"font-size:10px; color:{theme.TEXT_MUTED};")
                else:
                    lbl.setStyleSheet(f"font-size:10px; color:{theme.TEXT_PRIMARY};")

    # ── Colormap ───────────────────────────────────────────────────────────

    def _apply_recon_cmap(self, name: str):
        cm = get_colormaps().get(name)
        if cm is not None:
            self.recon_view.setColorMap(cm)

    # ── File loading ───────────────────────────────────────────────────────

    def _browse_file(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Open Bispectrum .npz file(s)",
            working_dir(),
            "Bispectrum (*.npz);;All Files (*)")
        if not paths:
            return

        n = len(paths)
        if n == 1:
            self.file_edit.setText(Path(paths[0]).name)
        else:
            self.file_edit.setText(f"{n} files selected")

        # Probe first file for info
        try:
            data = np.load(paths[0], allow_pickle=False)
            bs   = data['avg_bispec']
            self.file_info_lbl.setText(
                f"{n} file(s)  ·  first bispectrum shape={bs.shape}")
        except Exception as e:
            self.file_info_lbl.setText(f"Error: {e}")
            return

        # Initialise navigator state
        self._nav_paths      = list(paths)
        self._nav_idx        = 0
        self._nav_memory     = {}
        self._nav_load_queue = list(paths)
        self.nav_bar.setVisible(False)
        self.rerun_btn.setEnabled(False)
        self._clear_all(silent=True)

        self._log(f"Loading {n} file(s)...")
        self.file_info_lbl.setText(f"Reconstructing 1 / {n}...")
        self._reconstruct_next()

    def _reconstruct_next(self):
        """Pop the next file from the load queue and start a NpzReconWorker."""
        if not self._nav_load_queue:
            # All done
            n = len(self._nav_paths)
            self._log(f"✓ {n} file(s) ready.")
            self.rerun_btn.setEnabled(True)
            self._nav_idx = 0
            self._nav_goto(0)
            if n > 1:
                self._update_nav_bar()
                self.nav_bar.setVisible(True)
                self._log("Use ◄ ► to navigate between files.")
            return

        path  = self._nav_load_queue.pop(0)
        idx   = len(self._nav_paths) - len(self._nav_load_queue) - 1
        n     = len(self._nav_paths)
        self._npz_path = path
        self.file_info_lbl.setText(
            f"Reconstructing {idx + 1} / {n}  --  {Path(path).name}")
        self.progress_bar.setValue(0)

        worker = NpzReconWorker(
            path,
            k_max  = self.kmax_spin.value(),
            n_iter = self.niter_spin.value())
        worker.progress.connect(self.progress_bar.setValue)
        worker.status.connect(lambda msg: self.file_info_lbl.setText(msg))
        worker.error.connect(lambda msg: self._log(f"ERROR: {msg}", error=True))

        def _on_done(result, p=path, i=idx):
            self._nav_memory[p] = {
                'result':        result,
                'primary_pos':   None,
                'companion_pos': None,
                'meas':          {},
            }
            self._log(f"  [{i+1}/{n}] {Path(p).name}  OK")
            self._reconstruct_next()

        worker.finished.connect(_on_done)
        self._worker = worker
        worker.start()

    def _rerun_reconstruction(self):
        """Re-reconstruct the currently displayed file with current parameters."""
        if not self._npz_path:
            return
        self.rerun_btn.setEnabled(False)
        self.progress_bar.setValue(0)
        self._clear_all(silent=True)
        self._recon = None

        worker = NpzReconWorker(
            self._npz_path,
            k_max  = self.kmax_spin.value(),
            n_iter = self.niter_spin.value())
        worker.progress.connect(self.progress_bar.setValue)
        worker.status.connect(lambda msg: self.file_info_lbl.setText(msg))
        worker.finished.connect(self._on_finished)
        worker.error.connect(lambda msg: self._log(f"ERROR: {msg}", error=True))
        worker.finished.connect(lambda _: self.rerun_btn.setEnabled(True))
        worker.error.connect(   lambda _: self.rerun_btn.setEnabled(True))
        self._worker = worker
        worker.start()

    def _on_finished(self, result: dict):
        self._recon       = result['recon']
        self._avg_bispec  = result.get('avg_bispec')
        self._avg_power   = result.get('avg_power')
        self._bg_level    = None
        self.bg_subtract_chk.setChecked(False)
        self.bg_subtract_chk.setEnabled(False)
        self.clear_btn.setEnabled(True)
        self.cursor_radius_spin.setEnabled(True)
        self.bg_sample_btn.setEnabled(True)
        self.show_apertures_chk.setEnabled(True)
        self.save_png_btn.setEnabled(True)
        self.primary_radio.setEnabled(True)
        self.companion_radio.setEnabled(True)
        self.ap_inner_spin.setEnabled(True)
        self.ap_sky_in_spin.setEnabled(True)
        self.ap_sky_out_spin.setEnabled(True)
        self._set_click_mode('primary')
        self.level_min_slider.setValue(0)
        self.level_max_slider.setValue(255)
        self._apply_recon_display(auto_range=True)
        roi = result.get('roi_size', '?')
        self._log(f"  {Path(self._npz_path).name}  --  ROI {roi}x{roi} px")
        # Update nav memory with new result
        if self._npz_path in self._nav_memory:
            self._nav_memory[self._npz_path]['result'] = result

    # ── Navigator ──────────────────────────────────────────────────────────

    def _update_nav_bar(self):
        n   = len(self._nav_paths)
        idx = self._nav_idx
        self.nav_label.setText(f"{idx + 1} / {n}")
        self.nav_file_label.setText(Path(self._nav_paths[idx]).name)
        self.nav_prev_btn.setEnabled(idx > 0)
        self.nav_next_btn.setEnabled(idx < n - 1)

    def _nav_prev(self):
        if self._nav_idx > 0:
            self._save_current_to_memory()
            self._nav_goto(self._nav_idx - 1)

    def _nav_next(self):
        if self._nav_idx < len(self._nav_paths) - 1:
            self._save_current_to_memory()
            self._nav_goto(self._nav_idx + 1)

    def _save_current_to_memory(self):
        """Persist current marker positions and measurements before navigating."""
        path = self._nav_paths[self._nav_idx] if self._nav_paths else None
        if path and path in self._nav_memory:
            self._nav_memory[path]['primary_pos']   = self._primary_pos
            self._nav_memory[path]['companion_pos'] = self._companion_pos
            self._nav_memory[path]['meas'] = {
                'rho':       self._meas_rho,
                'theta':     self._meas_theta,
                'rho_sky':   self._meas_rho_sky,
                'theta_sky': self._meas_theta_sky,
                'delta_mag': self._meas_delta_mag,
            }

    def _nav_goto(self, idx: int):
        """Switch to file at position idx in _nav_paths."""
        self._nav_idx  = idx
        path = self._nav_paths[idx]
        self._npz_path = path

        if len(self._nav_paths) > 1:
            self._update_nav_bar()

        mem = self._nav_memory.get(path)
        if mem is None or mem.get('result') is None:
            return

        # Restore image
        result = mem['result']
        self._recon      = result['recon']
        self._avg_bispec = result.get('avg_bispec')
        self._avg_power  = result.get('avg_power')
        self._bg_level   = None
        self.bg_subtract_chk.setChecked(False)
        self.level_min_slider.setValue(0)
        self.level_max_slider.setValue(255)
        self._apply_recon_display(auto_range=True)

        # Enable controls
        self.clear_btn.setEnabled(True)
        self.cursor_radius_spin.setEnabled(True)
        self.bg_sample_btn.setEnabled(True)
        self.show_apertures_chk.setEnabled(True)
        self.save_png_btn.setEnabled(True)
        self.primary_radio.setEnabled(True)
        self.companion_radio.setEnabled(True)
        self.ap_inner_spin.setEnabled(True)
        self.ap_sky_in_spin.setEnabled(True)
        self.ap_sky_out_spin.setEnabled(True)

        # Restore markers (clear first, then redraw)
        self._primary_marker   = None
        self._companion_marker = None
        self._primary_pos   = mem.get('primary_pos')
        self._companion_pos = mem.get('companion_pos')
        if self._primary_pos:
            self._place_primary(*self._primary_pos)
        if self._companion_pos:
            self._place_companion(*self._companion_pos)

        # Restore measurement display
        meas = mem.get('meas', {})
        self._meas_rho         = meas.get('rho')
        self._meas_theta       = meas.get('theta')
        self._meas_rho_sky     = meas.get('rho_sky')
        self._meas_theta_sky   = meas.get('theta_sky')
        self._meas_delta_mag   = meas.get('delta_mag')
        if self._meas_rho is not None:
            self.card_rho.set_value(f"{self._meas_rho:.1f}")
            self.card_theta.set_value(f"{self._meas_theta:.1f}")
        if self._meas_rho_sky is not None:
            self.card_rho_sky.set_value(
                f"{self._meas_rho_sky:.4f}",
                f"{self._meas_sigma_rho_cal:.4f}" if self._meas_sigma_rho_cal > 0 else None)
            self.card_theta_sky.set_value(
                f"{self._meas_theta_sky:.2f}",
                f"{self._meas_sigma_theta_cal:.2f}" if self._meas_sigma_theta_cal > 0 else None)
        if self._meas_delta_mag is not None:
            self.card_delta_mag.set_value(f"{self._meas_delta_mag:+.3f}")

        has_meas = self._meas_rho is not None
        self.save_result_btn.setEnabled(has_meas)
        self.measure_phot_btn.setEnabled(has_meas)
        has_sky = self._meas_rho_sky is not None
        self.append_csv_btn.setEnabled(has_sky)
        self.save_wds_btn.setEnabled(has_sky)

        self.file_edit.setText(Path(path).name)
        self._log(f"-- {Path(path).name}")

    # ── Image display ──────────────────────────────────────────────────────

    def _apply_recon_display(self, auto_range: bool = False):
        if self._recon is None:
            return
        arr = self._recon.T.astype(np.float32)
        if self.bg_subtract_chk.isChecked() and self._bg_level is not None:
            arr = np.clip(arr - self._bg_level, 0, None)
        arr_min, arr_max = arr.min(), arr.max()
        if arr_max > arr_min:
            arr = (arr - arr_min) / (arr_max - arr_min) * 255.0
        self.recon_view.setImage(arr, autoLevels=False, autoRange=auto_range,
                                 levels=(self.level_min_slider.value(),
                                         self.level_max_slider.value()))

    # ── Cursor & background ────────────────────────────────────────────────

    def _on_cursor_radius_changed(self, _=None):
        if self._primary_pos or self._companion_pos:
            self._redraw_markers()

    def _draw_cursor_circle(self, x: float, y: float):
        self._clear_cursor_circle()
        if self._recon is None:
            return
        r = self.cursor_radius_spin.value()
        theta = np.linspace(0, 2 * np.pi, 64)
        self._cursor_circle_item = pg.PlotCurveItem(
            x + r * np.cos(theta), y + r * np.sin(theta),
            pen=pg.mkPen(theme.WARNING, width=1,
                         style=Qt.PenStyle.DashLine))
        self.recon_view.getView().addItem(self._cursor_circle_item)

    def _clear_cursor_circle(self):
        if self._cursor_circle_item is not None:
            try:
                self.recon_view.getView().removeItem(self._cursor_circle_item)
            except Exception:
                pass
            self._cursor_circle_item = None

    def _on_recon_hover(self, x: float, y: float):
        if self._recon is None:
            return
        self._draw_cursor_circle(x, y)

    def _on_bg_sample_clicked(self, checked: bool):
        if checked:
            self._log("Click on a signal-free region to sample background.")

    def _sample_background(self, x: float, y: float):
        if self._recon is None:
            return
        r  = self.cursor_radius_spin.value()
        img = self._recon.T
        W, H = img.shape
        ix = int(round(x)); iy = int(round(y))
        x0 = max(0, ix - r); x1 = min(W, ix + r + 1)
        y0 = max(0, iy - r); y1 = min(H, iy + r + 1)
        patch = img[x0:x1, y0:y1].astype(np.float64)
        xx, yy = np.mgrid[x0:x1, y0:y1]
        mask = (xx - x)**2 + (yy - y)**2 <= r**2
        vals = patch[mask]
        if vals.size == 0:
            return
        self._bg_level = float(vals.mean())
        self._log(f"Background sampled: {self._bg_level:.4f}  ({vals.size} px)")
        self.bg_subtract_chk.setEnabled(True)
        self.bg_sample_btn.setChecked(False)
        self.bg_subtract_chk.setChecked(True)

    def _on_bg_subtract_toggled(self, _):
        self._apply_recon_display()

    # ── Marker placement ───────────────────────────────────────────────────

    def _on_recon_click(self, x: float, y: float):
        if self._recon is None:
            return
        if self.bg_sample_btn.isChecked():
            self._sample_background(x, y)
            return
        if self._click_mode == 'primary':
            self._place_primary(x, y)
            self._set_click_mode('companion')
        else:
            self._place_companion(x, y)
        self._update_measurement()

    def _set_click_mode(self, mode: str):
        self._click_mode = mode
        btn = self.primary_radio if mode == 'primary' else self.companion_radio
        btn.blockSignals(True); btn.setChecked(True); btn.blockSignals(False)

    def _place_primary(self, x: float, y: float):
        if self._primary_marker is not None:
            for item in self._primary_marker:
                self.recon_view.getView().removeItem(item)
        self._primary_pos = (x, y)
        self._primary_marker = self._make_marker(x, y, theme.DANGER)

    def _place_companion(self, x: float, y: float):
        if self._companion_marker is not None:
            for item in self._companion_marker:
                self.recon_view.getView().removeItem(item)
        self._companion_pos = (x, y)
        self._companion_marker = self._make_marker(x, y, theme.ACCENT2)

    def _make_marker(self, x: float, y: float, color: str):
        circle = pg.ScatterPlotItem([{
            'pos': (x, y), 'size': 14,
            'pen': pg.mkPen(color, width=2),
            'brush': pg.mkBrush(None), 'symbol': 'o'}])
        cross  = pg.ScatterPlotItem([{
            'pos': (x, y), 'size': 8,
            'pen': pg.mkPen(color, width=1.5),
            'brush': pg.mkBrush(color), 'symbol': '+'}])
        self.recon_view.getView().addItem(circle)
        self.recon_view.getView().addItem(cross)
        return (circle, cross)

    def _redraw_markers(self):
        """Redraw markers (and optionally aperture circles) after a change."""
        if self._primary_pos:
            self._place_primary(*self._primary_pos)
        if self._companion_pos:
            self._place_companion(*self._companion_pos)
        if self.show_apertures_chk.isChecked():
            self._draw_aperture_circles()

    def _clear_aperture_circles(self):
        """Remove all aperture circle overlays from the view."""
        for item in self._aperture_circle_items:
            try:
                self.recon_view.getView().removeItem(item)
            except Exception:
                pass
        self._aperture_circle_items = []

    def _draw_aperture_circles(self):
        """Overlay aperture circles on both stars, clearing old ones first."""
        self._clear_aperture_circles()
        if not self._primary_pos and not self._companion_pos:
            return
        theta = np.linspace(0, 2 * np.pi, 64)
        r1 = self.ap_inner_spin.value()
        r2 = self.ap_sky_in_spin.value()
        r3 = self.ap_sky_out_spin.value()
        for pos in [p for p in (self._primary_pos, self._companion_pos) if p]:
            x, y = pos
            for r, color, style in [
                (r1, theme.ACCENT2,      Qt.PenStyle.SolidLine),
                (r2, theme.TEXT_MUTED,   Qt.PenStyle.DashLine),
                (r3, theme.TEXT_MUTED,   Qt.PenStyle.DashLine),
            ]:
                item = pg.PlotCurveItem(
                    x + r * np.cos(theta), y + r * np.sin(theta),
                    pen=pg.mkPen(color, width=1, style=style))
                self.recon_view.getView().addItem(item)
                self._aperture_circle_items.append(item)

    def _clear_all(self, silent: bool = False):
        for marker in (self._primary_marker, self._companion_marker):
            if marker is not None:
                for item in marker:
                    try:
                        self.recon_view.getView().removeItem(item)
                    except Exception:
                        pass
        self._clear_aperture_circles()
        self._primary_marker   = None
        self._companion_marker = None
        self._primary_pos      = None
        self._companion_pos    = None
        self._meas_rho         = None
        self._meas_theta       = None
        self._meas_delta_mag   = None
        for card in (self.card_theta, self.card_rho,
                     self.card_theta_sky, self.card_rho_sky, self.card_delta_mag):
            card.set_value("—")
        self.save_png_btn.setEnabled(False)
        self.save_result_btn.setEnabled(False)
        self.append_csv_btn.setEnabled(False)
        self.save_wds_btn.setEnabled(False)
        self.measure_phot_btn.setEnabled(False)
        self._click_mode = 'primary'
        self.primary_radio.blockSignals(True)
        self.primary_radio.setChecked(True)
        self.primary_radio.blockSignals(False)
        if not silent:
            self._log("Markers cleared.")

    # ── Measurement ────────────────────────────────────────────────────────

    def _update_measurement(self):
        if self._primary_pos is None or self._companion_pos is None:
            for card in (self.card_theta, self.card_rho,
                         self.card_theta_sky, self.card_rho_sky):
                card.set_value("—")
            self.save_result_btn.setEnabled(False)
            self.append_csv_btn.setEnabled(False)
            self.save_wds_btn.setEnabled(False)
            self.measure_phot_btn.setEnabled(False)
            return

        px, py = self._primary_pos
        cx, cy = self._companion_pos
        dx  = cx - px; dy = cy - py
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
            self._meas_rho_sky        = rho_arcsec
            self._meas_theta_sky      = theta_sky
            self._meas_sigma_rho_cal  = rho * sigma_scale
            self._meas_sigma_theta_cal = sigma_angle
            self.card_rho_sky.set_value(
                f"{rho_arcsec:.4f}",
                f"{self._meas_sigma_rho_cal:.4f}" if self._meas_sigma_rho_cal > 0 else None)
            self.card_theta_sky.set_value(
                f"{theta_sky:.2f}",
                f"{sigma_angle:.2f}" if sigma_angle > 0 else None)
            self._log(
                f"ρ = {rho:.2f} px  ({rho_arcsec:.4f}\")"
                + (f" ±{self._meas_sigma_rho_cal:.4f}" if self._meas_sigma_rho_cal > 0 else "")
                + f"   θ = {theta:.2f}°  ({theta_sky:.2f}°)")
        else:
            self.card_theta_sky.set_value("—")
            self.card_rho_sky.set_value("—")
            self._meas_rho_sky   = None
            self._meas_theta_sky = None

        self.save_result_btn.setEnabled(True)
        self.measure_phot_btn.setEnabled(True)
        has_sky = self._meas_rho_sky is not None
        self.append_csv_btn.setEnabled(has_sky)
        self.save_wds_btn.setEnabled(has_sky)

        # Persist to navigator memory
        if self._npz_path in self._nav_memory:
            self._nav_memory[self._npz_path]['primary_pos']   = self._primary_pos
            self._nav_memory[self._npz_path]['companion_pos'] = self._companion_pos
            self._nav_memory[self._npz_path]['meas'] = {
                'rho':       self._meas_rho,
                'theta':     self._meas_theta,
                'rho_sky':   self._meas_rho_sky,
                'theta_sky': self._meas_theta_sky,
                'delta_mag': self._meas_delta_mag,
            }

    # ── Aperture photometry ────────────────────────────────────────────────

    def _aperture_flux(self, img: np.ndarray, cx: float, cy: float,
                       r_star: int, r_sky_in: int, r_sky_out: int) -> float | None:
        """
        Compute the sky-subtracted flux within r_star pixels of (cx, cy).

        img      : 2-D float array (image coordinates, i.e. img[x, y])
        cx, cy   : centre in image coordinates
        r_star   : aperture radius [px]
        r_sky_in : sky annulus inner radius [px]
        r_sky_out: sky annulus outer radius [px]
        """
        H, W = img.shape
        # Build pixel grids
        xi = int(round(cx)); yi = int(round(cy))
        margin = r_sky_out + 2
        x0 = max(0, xi - margin); x1 = min(W, xi + margin + 1)
        y0 = max(0, yi - margin); y1 = min(H, yi + margin + 1)
        patch = img[x0:x1, y0:y1].astype(np.float64)
        xx, yy = np.mgrid[x0:x1, y0:y1]
        r2 = (xx - cx)**2 + (yy - cy)**2

        star_mask = r2 <= r_star**2
        sky_mask  = (r2 >= r_sky_in**2) & (r2 <= r_sky_out**2)

        if sky_mask.sum() == 0:
            return None

        sky_per_px = float(np.median(patch[sky_mask]))
        star_flux  = float(patch[star_mask].sum()) - sky_per_px * star_mask.sum()
        return star_flux

    def _measure_photometry(self):
        if self._recon is None or self._primary_pos is None or self._companion_pos is None:
            self._log("⚠ Place both markers before measuring photometry.", error=True)
            return

        r1  = self.ap_inner_spin.value()
        r2  = self.ap_sky_in_spin.value()
        r3  = self.ap_sky_out_spin.value()

        if not (r1 < r2 < r3):
            self._log("⚠ Aperture radii must satisfy: inner < sky_in < sky_out.", error=True)
            return

        # Work on the raw recon array (before level scaling)
        img = self._recon.T.astype(np.float64)   # img[x, y]

        px, py = self._primary_pos
        cx, cy = self._companion_pos

        f_prim = self._aperture_flux(img, px, py, r1, r2, r3)
        f_comp = self._aperture_flux(img, cx, cy, r1, r2, r3)

        if f_prim is None or f_comp is None:
            self._log("⚠ Sky annulus outside image bounds.", error=True)
            return
        if f_prim <= 0 or f_comp <= 0:
            self._log(
                f"⚠ Non-positive flux: primary={f_prim:.1f}  companion={f_comp:.1f}. "
                f"Check aperture placement.", error=True)
            return

        delta_m = -2.5 * np.log10(f_comp / f_prim)
        self._meas_delta_mag = float(delta_m)
        self.card_delta_mag.set_value(f"{delta_m:+.3f}")
        self._log(
            f"Photometry: F_prim={f_prim:.1f}  F_comp={f_comp:.1f}  "
            f"ΔM = {delta_m:+.3f} mag  (r={r1}, sky {r2}–{r3} px)")

        # Draw aperture circles
        if self.show_apertures_chk.isChecked():
            self._redraw_markers()

    # ── Calibration ────────────────────────────────────────────────────────

    def _load_cal_dialog(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Load Calibration JSON", working_dir(),
            "JSON files (*.json);;All Files (*)")
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
            s_sc  = cal.get('sigma_scale_arcsec',
                    cal.get('pixel_scale_std_arcsec', 0.0))
            s_ang = cal.get('sigma_angle_deg',
                    cal.get('camera_angle_std_deg', 0.0))
            self.cal_scale_spin.setValue(scale)
            self.cal_angle_spin.setValue(angle)
            self.cal_scale_err.setValue(s_sc)
            self.cal_angle_err.setValue(s_ang)
            self.cal_status_lbl.setText(
                f"scale={scale:.6f}\"/px ±{s_sc:.6f}  "
                f"angle={angle:.4f}° ±{s_ang:.4f}°")
            self._log(f"Calibration loaded: {Path(path).name}")
            self._update_measurement()
        except Exception as e:
            self.cal_status_lbl.setText(f"⚠ {e}")
            self._log(f"⚠ Calibration load failed: {e}", error=True)

    # ── Save result / CSV / WDS ────────────────────────────────────────────

    def _save_png(self):
        """
        Export the reconstructed image as a PNG file.

        The image is rendered at full resolution using the current colormap
        and min/max levels.  Primary and secondary markers are drawn on top
        if placed.  The PNG can be loaded in the History tab as an overlay.
        """
        if self._recon is None:
            self._log("⚠ No image to export.", error=True)
            return

        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches

        stem    = Path(self._npz_path).stem if self._npz_path else "recon"
        default = f"{stem}_recon.png"
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Reconstructed Image",
            str(Path(working_dir()) / default),
            "PNG images (*.png);;All Files (*)")
        if not path:
            return

        arr = self._recon.T.astype(np.float32)
        if self.bg_subtract_chk.isChecked() and self._bg_level is not None:
            arr = np.clip(arr - self._bg_level, 0, None)

        # Apply current levels (same as display)
        lo = self.level_min_slider.value() / 255.0
        hi = self.level_max_slider.value() / 255.0
        arr_min, arr_max = arr.min(), arr.max()
        if arr_max > arr_min:
            arr = (arr - arr_min) / (arr_max - arr_min)
        arr = np.clip((arr - lo) / max(hi - lo, 1e-6), 0, 1)

        # Map to colormap
        cmap_name = self.recon_cmap_combo.currentText().lower()
        mpl_cmap  = {
            "grey":     "gray",
            "inverted": "gray_r",
            "hot":      "hot",
            "rainbow":  "rainbow",
            "viridis":  "viridis",
        }.get(cmap_name, "gray")

        fig, ax = plt.subplots(figsize=(6, 6), dpi=150)
        ax.imshow(arr, origin="lower", cmap=mpl_cmap, vmin=0, vmax=1,
                  interpolation="nearest")
        ax.axis("off")
        fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

        # Draw markers
        N = arr.shape[0]
        if self._primary_pos:
            px, py = self._primary_pos
            ax.plot(px, py, 'o', ms=10, mfc='none',
                    mec='#ff3311', mew=2.0, zorder=5)
            ax.plot(px, py, '+', ms=6,
                    mec='#ff3311', mew=1.5, zorder=5)
        if self._companion_pos:
            cx, cy = self._companion_pos
            ax.plot(cx, cy, 'o', ms=10, mfc='none',
                    mec='#3fb950', mew=2.0, zorder=5)
            ax.plot(cx, cy, '+', ms=6,
                    mec='#3fb950', mew=1.5, zorder=5)

        # Draw aperture circles if overlay is on
        if self.show_apertures_chk.isChecked():
            theta = np.linspace(0, 2 * np.pi, 128)
            r1, r2, r3 = (self.ap_inner_spin.value(),
                          self.ap_sky_in_spin.value(),
                          self.ap_sky_out_spin.value())
            for pos in [p for p in (self._primary_pos, self._companion_pos) if p]:
                x, y = pos
                for r, col, ls in [
                    (r1, '#3fb950', '-'),
                    (r2, '#8b949e', '--'),
                    (r3, '#8b949e', '--'),
                ]:
                    ax.plot(x + r * np.cos(theta),
                            y + r * np.sin(theta),
                            color=col, lw=0.8, ls=ls, zorder=4)

        try:
            fig.savefig(path, dpi=150, bbox_inches='tight', pad_inches=0)
            plt.close(fig)
            self._log(f"PNG saved: {Path(path).name}")
        except Exception as e:
            plt.close(fig)
            self._log(f"⚠ PNG save failed: {e}", error=True)

    def _save_result(self):
        if self._meas_rho is None:
            self._log("⚠ Place both markers first.", error=True)
            return
        default = (Path(self._npz_path).stem if self._npz_path else "result") + \
                  "_astrometry.json"
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Astrometry Result",
            str(Path(working_dir()) / default),
            "JSON files (*.json);;All Files (*)")
        if not path:
            return
        payload = {
            "rho_px":             round(self._meas_rho,   4),
            "theta_img_deg":      round(self._meas_theta, 4),
            "rho_arcsec":         round(self._meas_rho_sky,   6) if self._meas_rho_sky   is not None else None,
            "theta_sky_deg":      round(self._meas_theta_sky, 4) if self._meas_theta_sky is not None else None,
            "sigma_rho_arcsec":   round(self._meas_sigma_rho_cal,   6),
            "sigma_theta_deg":    round(self._meas_sigma_theta_cal, 4),
            "delta_mag":          round(self._meas_delta_mag, 4) if self._meas_delta_mag is not None else None,
            "ap_inner_px":        self.ap_inner_spin.value(),
            "ap_sky_in_px":       self.ap_sky_in_spin.value(),
            "ap_sky_out_px":      self.ap_sky_out_spin.value(),
            "pixel_scale_arcsec": self.cal_scale_spin.value(),
            "camera_angle_deg":   self.cal_angle_spin.value(),
            "cal_file":           Path(self._cal_file).name if self._cal_file else "manual",
            "npz_source":         Path(self._npz_path).name if self._npz_path else "",
        }
        try:
            with open(path, 'w') as f:
                _json.dump(payload, f, indent=2)
            self._log(f"Saved: {Path(path).name}")
        except Exception as e:
            self._log(f"⚠ Save failed: {e}", error=True)

    _CSV_HEADER = [
        'date', 'target', 'observer', 'filter',
        'theta_sky_deg', 'sigma_theta_deg',
        'rho_arcsec',    'sigma_rho_arcsec',
        'delta_mag',
        'theta_img_deg', 'rho_px',
        'pixel_scale_arcsec_px', 'sigma_scale',
        'camera_angle_deg',      'sigma_angle_deg',
        'cal_file', 'npz_source',
    ]

    def _set_csv_dialog(self):
        import os
        path, _ = QFileDialog.getSaveFileName(
            self, "Set CSV Log File",
            str(Path(working_dir()) / "speckle_log.csv"),
            "CSV files (*.csv);;All Files (*)")
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
        dm = f"{self._meas_delta_mag:.4f}" if self._meas_delta_mag is not None else ""
        row = [
            date.today().isoformat(), "", "", "",
            f"{self._meas_theta_sky:.6f}",
            f"{self._meas_sigma_theta_cal:.6f}",
            f"{self._meas_rho_sky:.6f}",
            f"{self._meas_sigma_rho_cal:.6f}",
            dm,
            f"{self._meas_theta:.4f}",
            f"{self._meas_rho:.4f}",
            f"{self.cal_scale_spin.value():.8f}",
            f"{self.cal_scale_err.value():.8f}",
            f"{self.cal_angle_spin.value():.6f}",
            f"{self.cal_angle_err.value():.6f}",
            Path(self._cal_file).name if self._cal_file else "manual",
            Path(self._npz_path).name if self._npz_path else "",
        ]
        try:
            with open(self._csv_path, 'a', newline='') as f:
                csv.writer(f).writerow(row)
            self._log(f"Appended to CSV: {Path(self._csv_path).name}")
        except Exception as e:
            self._log(f"⚠ CSV write error: {e}", error=True)

    _WDS_TEMPLATE = """--------------------------------------------
  WDS ASTROMETRIC MEASUREMENT
--------------------------------------------
  Date            : {obs_date}
  Position Angle  : {theta_sky:.2f} deg
    +/- {sigma_theta:.4f} deg  (calibration)
  Separation      : {rho_arcsec:.4f} arcsec
    +/- {sigma_rho:.4f} arcsec  (calibration)
  Delta magnitude : {delta_mag}
  rho (pixels)    : {rho_px:.3f}
  theta (image)   : {theta_img:.2f} deg
  Pixel scale     : {pixel_scale:.6f} arcsec/px +/- {sigma_scale:.8f}
  Camera angle    : {camera_angle:.4f} deg +/- {sigma_angle:.6f} deg
  Calibration     : {cal_file}
  Source          : {npz_source}
--------------------------------------------
"""

    def _save_wds(self):
        if self._meas_rho_sky is None:
            self._log("⚠ Load calibration first.", error=True)
            return
        dm_str = (f"{self._meas_delta_mag:+.3f} mag"
                  if self._meas_delta_mag is not None else "not measured")
        report = self._WDS_TEMPLATE.format(
            obs_date     = date.today().isoformat(),
            theta_sky    = self._meas_theta_sky,
            sigma_theta  = self._meas_sigma_theta_cal,
            rho_arcsec   = self._meas_rho_sky,
            sigma_rho    = self._meas_sigma_rho_cal,
            delta_mag    = dm_str,
            rho_px       = self._meas_rho,
            theta_img    = self._meas_theta,
            pixel_scale  = self.cal_scale_spin.value(),
            sigma_scale  = self.cal_scale_err.value(),
            camera_angle = self.cal_angle_spin.value(),
            sigma_angle  = self.cal_angle_err.value(),
            cal_file     = Path(self._cal_file).name if self._cal_file else "manual",
            npz_source   = Path(self._npz_path).name if self._npz_path else "",
        )
        stem    = Path(self._npz_path).stem if self._npz_path else "result"
        default = f"{stem}_WDS_{date.today().isoformat()}.txt"
        path, _ = QFileDialog.getSaveFileName(
            self, "Save WDS Report",
            str(Path(working_dir()) / default),
            "Text files (*.txt);;All Files (*)")
        if not path:
            return
        try:
            with open(path, 'w') as f:
                f.write(report)
            self._log(f"WDS report saved: {Path(path).name}")
        except Exception as e:
            self._log(f"⚠ Save error: {e}", error=True)

    # ── Log ────────────────────────────────────────────────────────────────

    def _log(self, msg: str, error: bool = False, warning: bool = False):
        color = theme.DANGER if error else (theme.WARNING if warning else theme.TEXT_MUTED)
        self.log_edit.append(f'<span style="color:{color}">{msg}</span>')

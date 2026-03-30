"""
speckle_suite.tab_drift
========================
Tab 1 — Drift Alignment: derives camera angle and pixel scale from a
sidereal-drift SER recording.
"""

from __future__ import annotations

import json as _json
import numpy as np

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QLabel, QPushButton,
    QLineEdit, QGroupBox, QFileDialog, QProgressBar, QSizePolicy,
    QTextEdit, QSplitter, QRadioButton, QButtonGroup, QDoubleSpinBox,
    QFrame, QSlider,
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt6.QtGui import QFont
from pathlib import Path
from typing import Optional

import pyqtgraph as pg

import speckle_suite.theme as theme
from speckle_suite.settings import SETTINGS, save_settings, working_dir
from speckle_suite.widgets import ResultCard
from speckle_suite.ser_io import read_ser_header_and_timestamps
from speckle_suite.drift_backend import (
    DriftResult, DriftWorker, SimbadWorker,
    fit_drift, _parse_declination_from_txt,
)

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
        ser_lbl.setStyleSheet(f"color: {theme.TEXT_MUTED};")
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
        self.file_info_label.setStyleSheet(f"color: {theme.TEXT_MUTED}; font-size: 10px;")
        target_layout.addWidget(self.file_info_label, 1, 0, 1, 3)

        # Thin separator line
        sep2 = QFrame()
        sep2.setFrameShape(QFrame.Shape.HLine)
        sep2.setObjectName("separator")
        target_layout.addWidget(sep2, 2, 0, 1, 3)

        # Row 3 — Target name + Simbad resolve
        name_lbl = QLabel("Target name")
        name_lbl.setStyleSheet(f"color: {theme.TEXT_MUTED};")
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
        dec_lbl.setStyleSheet(f"color: {theme.TEXT_MUTED};")
        dec_lbl.setToolTip("Declination of the drift star (degrees)")

        self.dec_spin = QLineEdit()
        self.dec_spin.setPlaceholderText("e.g. +45.0")
        self.dec_spin.setMinimumHeight(26)
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
        px_lbl.setStyleSheet(f"color:{theme.TEXT_MUTED};")
        self.pixel_size_edit = _optics_field("e.g. 4.65")

        ap_lbl = QLabel("Aperture (mm)")
        ap_lbl.setStyleSheet(f"color:{theme.TEXT_MUTED};")
        self.aperture_edit = _optics_field("e.g. 200")

        wl_lbl = QLabel("Wavelength (nm)")
        wl_lbl.setStyleSheet(f"color:{theme.TEXT_MUTED};")
        self.wavelength_edit = _optics_field("e.g. 550")
        self.wavelength_edit.setText("550")

        fl_lbl = QLabel("Focal length")
        fl_lbl.setStyleSheet(f"color:{theme.TEXT_MUTED};")
        self.fl_value_lbl = QLabel("—")
        self.fl_value_lbl.setStyleSheet(f"color:{theme.ACCENT}; font-weight:bold; border:none;")

        fratio_lbl = QLabel("f-ratio")
        fratio_lbl.setStyleSheet(f"color:{theme.TEXT_MUTED};")
        self.fratio_value_lbl = QLabel("—")
        self.fratio_value_lbl.setStyleSheet(f"color:{theme.ACCENT}; font-weight:bold; border:none;")

        sampling_lbl = QLabel("Sampling")
        sampling_lbl.setStyleSheet(f"color:{theme.TEXT_MUTED};")
        self.sampling_value_lbl = QLabel("—")
        self.sampling_value_lbl.setStyleSheet(f"color:{theme.ACCENT}; font-weight:bold; border:none;")

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
        self.status_label.setStyleSheet(f"color: {theme.TEXT_MUTED}; font-size: 10px;")
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
        toggle_lbl.setStyleSheet(f"color:{theme.TEXT_MUTED};")
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
        sep.setStyleSheet(f"color:{theme.BORDER_COLOR};")
        scale_lbl = QLabel("Scale:")
        scale_lbl.setStyleSheet(f"color:{theme.TEXT_MUTED};")
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
            f"color:{theme.TEXT_PRIMARY}; font-size:13px; font-weight:bold;")
        self.drift_nav_lbl.setFixedWidth(60)
        self.drift_nav_file_lbl = QLabel("")
        self.drift_nav_file_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.drift_nav_file_lbl.setStyleSheet(f"color:{theme.TEXT_MUTED}; font-size:10px;")
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
            f"QWidget {{ background:{theme.PANEL_BG}; border:1px solid {theme.BORDER_COLOR};"
            f" border-radius:6px; }}")
        self._ctrl_widget = ctrl_widget
        ctrl_inner = QVBoxLayout(ctrl_widget)
        ctrl_inner.setContentsMargins(12, 10, 12, 10)
        ctrl_inner.setSpacing(8)

        def make_slider_row(label, unit, lo, hi, default, decimals=1):
            row = QHBoxLayout()
            lbl = QLabel(label)
            lbl.setStyleSheet(f"color:{theme.TEXT_MUTED}; min-width:76px; font-size:11px; border:none;")
            sl  = QSlider(Qt.Orientation.Horizontal)
            sl.setRange(lo, hi)
            sl.setValue(default)
            sl.setEnabled(False)
            val = QLabel(f"{default / (10**decimals):.{decimals}f} {unit}")
            val.setStyleSheet(f"color:{theme.ACCENT}; min-width:48px; font-size:11px; border:none;")
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
        self.trim_info.setStyleSheet(f"color:{theme.TEXT_MUTED}; font-size:10px; border:none;")

        self.rejection_info = QLabel("Accepted: —  /  Rejected: —")
        self.rejection_info.setStyleSheet(f"color:{theme.TEXT_MUTED}; font-size:10px; border:none;")

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
        self.dec_spin.textChanged.connect(lambda _: self._on_dec_manually_edited())

        root.addWidget(right)

    def _apply_graph_theme(self):
        pg.setConfigOption('background', theme.PANEL_BG)
        pg.setConfigOption('foreground', theme.TEXT_MUTED)
        for plot in (self.plot_main, self.plot_resid):
            plot.setBackground(theme.PANEL_BG)
            plot.getPlotItem().getAxis('left').setPen(pg.mkPen(color=theme.BORDER_COLOR))
            plot.getPlotItem().getAxis('bottom').setPen(pg.mkPen(color=theme.BORDER_COLOR))
            plot.getPlotItem().titleLabel.setAttr('color', theme.ACCENT)
            plot.showGrid(x=True, y=True, alpha=0.15)

    # ── Interactions ─────────────────────────────────────────────────────

    def refresh_styles(self):
        """Called by SpeckleMainWindow when the theme changes."""
        t = theme._theme
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
            f"QWidget {{ background:{theme.PANEL_BG}; border:1px solid {theme.BORDER_COLOR};"
            f" border-radius:6px; }}")
        # Re-theme slider value labels
        for lbl in (self.start_val_lbl, self.stop_val_lbl, self.sigma_value_lbl):
            lbl.setStyleSheet(
                f"color:{theme.ACCENT}; min-width:48px; font-size:11px; border:none;")
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
            self.dec_auto_lbl.setStyleSheet(f"font-size:10px; color:{theme.ACCENT2}; border:none;")
        elif source == 'simbad':
            self.dec_auto_lbl.setText("✓ from Simbad")
            self.dec_auto_lbl.setStyleSheet(f"font-size:10px; color:{theme.ACCENT2}; border:none;")
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
        self.dec_spin.setText(f"{dec_deg:+.4f}")
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
            self, "Open SER Drift File(s)", working_dir(),
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
                self.dec_spin.setText(f"{dec_found:+.4f}")
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
            self.trim_info.setStyleSheet(f"color:{theme.DANGER}; font-size:10px;")
        else:
            self.trim_info.setText(f"→ {used:.1f} s used  of  {dur:.1f} s total")
            self.trim_info.setStyleSheet(f"color:{theme.ACCENT2}; font-size:10px;")

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
            declination_deg = float(self.dec_spin.text()) if self.dec_spin.text().strip() else 0.0,
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
        # Persist current slider state before switching
        self._drift_nav_save_sliders()
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
            # Batch mode: store raw data plus current default slider values
            path = self._nav_pending.pop(0)
            raw["sigma"] = self.sigma_slider.value()
            raw["start"] = self.start_slider.value()
            raw["stop"]  = self.stop_slider.value()
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
        """Accepted data point color."""
        if theme._theme is theme.THEMES['light']:
            return '#222222'
        if theme._theme is theme.THEMES['red']:
            return '#ffcc88'   # warm amber on dark red background
        return theme.ACCENT    # dark theme: blue

    def _plot_color_rej(self) -> str:
        """Rejected data point color — always visually distinct from accepted."""
        if theme._theme is theme.THEMES['light']:
            return '#cc0000'   # red on light
        if theme._theme is theme.THEMES['red']:
            return '#44aaff'   # blue on red — maximum contrast
        return theme.DANGER    # dark theme: red

    def _plot_color_fit(self) -> str:
        """Fit line color."""
        if theme._theme is theme.THEMES['light']:
            return '#0055aa'
        if theme._theme is theme.THEMES['red']:
            return '#88ff88'   # green on red background
        return theme.ACCENT2   # dark theme: green

    def _plot_color_hist(self) -> str:
        """Histogram inlier bar color."""
        if theme._theme is theme.THEMES['light']:
            return '#2255aa'
        if theme._theme is theme.THEMES['red']:
            return '#ffcc88'   # same as accepted points
        return theme.ACCENT    # dark theme: blue

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
            float(self.dec_spin.text()) if self.dec_spin.text().strip() else 0.0,
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

        # ── Accepted points ────────────────────────────────────────
        _pc = self._plot_color_pts()
        _rc = self._plot_color_rej()
        self.plot_main.plot(
            xdata_acc, ydata_acc, pen=None, symbol='o', symbolSize=4,
            symbolBrush=pg.mkBrush(_pc + "aa"),
            symbolPen=pg.mkPen(_pc, width=0))

        # ── Rejected points (✕) ────────────────────────────────────
        if rej_mask.any():
            self.plot_main.plot(
                xdata_rej, ydata_rej, pen=None, symbol='x', symbolSize=7,
                symbolBrush=pg.mkBrush(_rc + "99"),
                symbolPen=pg.mkPen(_rc, width=1.5))

        # ── Fitted line ────────────────────────────────────────────
        self.plot_main.plot(xfit, yfit, pen=pg.mkPen(self._plot_color_fit(), width=2))

        # ── Residuals histogram — inliers vs outliers ──────────────
        thr_px      = self.sigma_slider.value() / 10.0 * fit['rms_perp']
        all_resid   = fit['perp_resid']
        inlier_res  = all_resid[mask]
        outlier_res = all_resid[~mask]

        if len(inlier_res) >= 3:
            all_valid = all_resid[fit['time_mask']]
            n_bins  = max(10, min(40, len(all_valid) // 8))
            _, edges = np.histogram(all_valid, bins=n_bins)
            centres = (edges[:-1] + edges[1:]) / 2
            bar_w   = (edges[1] - edges[0]) * 0.85

            # Inlier bars
            counts_in, _ = np.histogram(inlier_res, bins=edges)
            _hc = self._plot_color_hist()
            self.plot_resid.addItem(pg.BarGraphItem(
                x=centres, height=counts_in, width=bar_w,
                brush=pg.mkBrush(_hc + "aa"),
                pen=pg.mkPen(_hc, width=0.5)))

            # Outlier bars (stacked)
            if len(outlier_res):
                counts_out, _ = np.histogram(outlier_res, bins=edges)
                self.plot_resid.addItem(pg.BarGraphItem(
                    x=centres, height=counts_out, width=bar_w,
                    brush=pg.mkBrush(_rc + "99"),
                    pen=pg.mkPen(_rc, width=0.5)))

            # Sigma threshold lines
            for pos in (-thr_px, thr_px):
                self.plot_resid.addItem(pg.InfiniteLine(
                    pos=pos, angle=90,
                    pen=pg.mkPen(theme.WARNING, width=1.5,
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
                print(f"DEBUG set sampling label to: {sampling:.2f}, label now reads: {self.sampling_value_lbl.text()}")
            else:
                self.sampling_value_lbl.setText("—")
            self.sampling_value_lbl.repaint()

    def _drift_nav_save_sliders(self):
        """Snapshot the current slider values into nav_memory for the active file."""
        if not self._nav_paths:
            return
        path = self._nav_paths[self._nav_idx]
        mem  = self._nav_memory.get(path)
        if mem is not None:
            mem["sigma"] = self.sigma_slider.value()
            mem["start"] = self.start_slider.value()
            mem["stop"]  = self.stop_slider.value()

    def _fit_from_memory(self, mem: dict):
        """Re-run fit for a nav_memory entry using its stored slider values."""
        return fit_drift(
            mem['centroids_x'], mem['centroids_y'],
            mem['declination_deg'], mem['fps'],
            mem["sigma"] / 10.0,
            times_sec      = mem.get('times_sec'),
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
            str(Path(working_dir()) / default_name),
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
        color = theme.DANGER if error else theme.TEXT_MUTED
        self.log_edit.append(f'<span style="color:{color}">{msg}</span>')

    def get_calibration(self) -> Optional[tuple[float, float]]:
        """Returns (camera_angle_deg, pixel_scale_arcsec) if a result is available."""
        if self.result:
            return self.result.camera_angle_deg, self.result.pixel_scale_arcsec
        return None


# ─────────────────────────────────────────────
#  Entry point
# ─────────────────────────────────────────────


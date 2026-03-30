"""
speckle_suite.main_window
==========================
Top-level application window and preferences dialog.

SpeckleMainWindow  -- QMainWindow hosting all five tabs
SettingsDialog     -- modal preferences (theme, working dir, analysis defaults)
"""

from __future__ import annotations

from pathlib import Path

import pyqtgraph as pg
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QTabWidget, QDialog, QFormLayout, QComboBox, QLineEdit, QPushButton,
    QSpinBox, QDoubleSpinBox, QGroupBox, QFileDialog,
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont

import speckle_suite.theme as theme
from speckle_suite.settings import SETTINGS, save_settings, working_dir
from speckle_suite.widgets import primary_btn_style

from speckle_suite.tab_drift       import DriftTab
from speckle_suite.tab_preprocess  import PreprocessTab
from speckle_suite.tab_bispectrum  import BispectrumTab
from speckle_suite.tab_astrometry  import AstrometryTab
from speckle_suite.tab_history     import HistoryTab


# ── SettingsDialog ─────────────────────────────────────────────────────────

class SettingsDialog(QDialog):

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

        # General
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
        self.dir_edit.setPlaceholderText("Working directory...")
        dir_browse = QPushButton("Browse...")
        dir_browse.setFixedHeight(28)
        dir_browse.clicked.connect(self._browse_dir)
        dir_hl.addWidget(self.dir_edit, 1); dir_hl.addWidget(dir_browse)
        gen_layout.addRow("Working directory", dir_row)
        root.addWidget(gen_group)

        # Preprocessing
        pre_group  = QGroupBox("Preprocessing")
        pre_layout = QFormLayout(pre_group)
        pre_layout.setSpacing(8)
        self.roi_combo = QComboBox()
        self.roi_combo.addItems(["32 x 32", "64 x 64", "128 x 128"])
        self.roi_combo.setMinimumHeight(28)
        pre_layout.addRow("Default ROI size", self.roi_combo)
        root.addWidget(pre_group)

        # Bispectrum / Astrometry
        bis_group  = QGroupBox("Bispectrum & Astrometry (Tabs 3 & 4)")
        bis_layout = QFormLayout(bis_group)
        bis_layout.setSpacing(8)
        self.kmax_spin = QSpinBox()
        self.kmax_spin.setRange(4, 512); self.kmax_spin.setSuffix(" px")
        self.kmax_spin.setMinimumHeight(28)
        bis_layout.addRow("K_max", self.kmax_spin)
        self.dkmax_spin = QSpinBox()
        self.dkmax_spin.setRange(1, 64); self.dkmax_spin.setSuffix(" px")
        self.dkmax_spin.setMinimumHeight(28)
        bis_layout.addRow("dK_max", self.dkmax_spin)
        self.niter_spin = QSpinBox()
        self.niter_spin.setRange(1, 200); self.niter_spin.setMinimumHeight(28)
        bis_layout.addRow("Iterations", self.niter_spin)
        self.epsilon_spin = QDoubleSpinBox()
        self.epsilon_spin.setRange(0.001, 0.5); self.epsilon_spin.setDecimals(3)
        self.epsilon_spin.setSingleStep(0.005); self.epsilon_spin.setMinimumHeight(28)
        bis_layout.addRow("Wiener epsilon", self.epsilon_spin)
        self.cmap_combo = QComboBox()
        self.cmap_combo.addItems(["Grey", "Inverted", "Hot", "Rainbow", "Viridis"])
        self.cmap_combo.setMinimumHeight(28)
        bis_layout.addRow("Default colormap", self.cmap_combo)
        root.addWidget(bis_group)

        btn_row = QHBoxLayout(); btn_row.addStretch()
        cancel_btn = QPushButton("Cancel"); cancel_btn.setFixedHeight(30)
        cancel_btn.clicked.connect(self.reject)
        self.ok_btn = QPushButton("Apply & Close"); self.ok_btn.setFixedHeight(30)
        self.ok_btn.setStyleSheet(primary_btn_style())
        self.ok_btn.clicked.connect(self._apply)
        btn_row.addWidget(cancel_btn); btn_row.addWidget(self.ok_btn)
        root.addLayout(btn_row)

    def _load(self):
        theme_map = {"dark": 0, "red": 1, "light": 2}
        self.theme_combo.setCurrentIndex(
            theme_map.get(SETTINGS.get("theme", "dark"), 0))
        self.dir_edit.setText(SETTINGS.get("working_dir", str(Path.home())))
        pre = SETTINGS.get("preprocess", {})
        self.roi_combo.setCurrentIndex(pre.get("roi_index", 0))
        ana = SETTINGS.get("analysis", {})
        self.kmax_spin.setValue(  ana.get("k_max",    60))
        self.dkmax_spin.setValue( ana.get("dk_max",    9))
        self.niter_spin.setValue( ana.get("n_iter",   30))
        self.epsilon_spin.setValue(ana.get("epsilon", 0.01))
        self.cmap_combo.setCurrentText(ana.get("colormap", "Grey"))

    def _browse_dir(self):
        d = QFileDialog.getExistingDirectory(
            self, "Select Working Directory",
            self.dir_edit.text() or str(Path.home()))
        if d:
            self.dir_edit.setText(d)

    def _apply(self):
        names = ["dark", "red", "light"]
        SETTINGS["theme"]       = names[self.theme_combo.currentIndex()]
        SETTINGS["working_dir"] = self.dir_edit.text().strip() or str(Path.home())
        SETTINGS["preprocess"]["roi_index"] = self.roi_combo.currentIndex()
        SETTINGS["analysis"]["k_max"]      = self.kmax_spin.value()
        SETTINGS["analysis"]["dk_max"]     = self.dkmax_spin.value()
        SETTINGS["analysis"]["n_iter"]     = self.niter_spin.value()
        SETTINGS["analysis"]["epsilon"]    = self.epsilon_spin.value()
        SETTINGS["analysis"]["colormap"]   = self.cmap_combo.currentText()
        save_settings(SETTINGS)
        self.accept()


# ── SpeckleMainWindow ──────────────────────────────────────────────────────

class SpeckleMainWindow(QMainWindow):

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Double Star Speckle Astrometry Suite")
        self.resize(1400, 1020)
        self.setMinimumSize(1200, 800)

        theme.set_theme(SETTINGS.get("theme", "dark"))
        QApplication.instance().setStyleSheet(
            theme.build_stylesheet(theme._theme))
        pg.setConfigOptions(antialias=True, imageAxisOrder='row-major')

        self._build_menu()
        self._build_tabs()
        self._apply_settings_to_tabs()

    # ── Tabs ───────────────────────────────────────────────────────────────

    def _build_tabs(self):
        self.tabs = QTabWidget()
        self.tabs.setTabPosition(QTabWidget.TabPosition.North)

        self.drift_tab      = DriftTab()
        self.preprocess_tab = PreprocessTab()
        self.bispectrum_tab = BispectrumTab()
        self.astrometry_tab = AstrometryTab()
        self.history_tab    = HistoryTab()

        self.tabs.addTab(self.drift_tab,      "🧭  Drift Alignment")
        self.tabs.addTab(self.preprocess_tab, "⚙  Preprocess")
        self.tabs.addTab(self.bispectrum_tab, "🔬  Bispectrum")
        self.tabs.addTab(self.astrometry_tab, "📐  Astrometry")
        self.tabs.addTab(self.history_tab,    "📜  History")

        self.setCentralWidget(self.tabs)

    # ── Menu ───────────────────────────────────────────────────────────────

    def _build_menu(self):
        mb = self.menuBar()

        file_menu = mb.addMenu("File")
        open_pre  = file_menu.addAction("Open Speckle Sequence...")
        open_pre.triggered.connect(self._open_preprocess)
        open_bis  = file_menu.addAction("Open FITS Cube (Bispectrum)...")
        open_bis.triggered.connect(self._open_bispectrum)
        open_ast  = file_menu.addAction("Open .npz (Astrometry)...")
        open_ast.triggered.connect(self._open_astrometry)
        file_menu.addSeparator()
        cal_act = file_menu.addAction("Load Calibration JSON...")
        cal_act.triggered.connect(lambda: self.astrometry_tab._load_cal_dialog())
        csv_act = file_menu.addAction("Set CSV Log File...")
        csv_act.triggered.connect(lambda: self.astrometry_tab._set_csv_dialog())
        file_menu.addSeparator()
        quit_act = file_menu.addAction("Quit")
        quit_act.triggered.connect(self.close)

        settings_act = mb.addAction("Settings")
        settings_act.triggered.connect(self._open_settings)

    def _open_preprocess(self):
        self.tabs.setCurrentWidget(self.preprocess_tab)
        self.preprocess_tab._browse_file()

    def _open_bispectrum(self):
        self.tabs.setCurrentWidget(self.bispectrum_tab)
        self.bispectrum_tab._browse_file()

    def _open_astrometry(self):
        self.tabs.setCurrentWidget(self.astrometry_tab)
        self.astrometry_tab._browse_file()

    # ── Settings ───────────────────────────────────────────────────────────

    def _apply_settings_to_tabs(self):
        pre = SETTINGS.get("preprocess", {})
        ana = SETTINGS.get("analysis",   {})
        cmap = ana.get("colormap", "Grey")

        self.preprocess_tab.roi_combo.setCurrentIndex(pre.get("roi_index", 0))

        # Bispectrum tab
        self.bispectrum_tab.kmax_spin.setValue(   ana.get("k_max",    60))
        self.bispectrum_tab.dkmax_spin.setValue(  ana.get("dk_max",    9))
        self.bispectrum_tab.niter_spin.setValue(  ana.get("n_iter",   30))
        self.bispectrum_tab.epsilon_spin.setValue(ana.get("epsilon", 0.01))
        idx = self.bispectrum_tab.recon_cmap_combo.findText(cmap)
        if idx >= 0:
            self.bispectrum_tab.recon_cmap_combo.setCurrentIndex(idx)

        # Astrometry tab (shares kmax / niter / colormap)
        self.astrometry_tab.kmax_spin.setValue( ana.get("k_max",  60))
        self.astrometry_tab.niter_spin.setValue(ana.get("n_iter", 30))
        idx2 = self.astrometry_tab.recon_cmap_combo.findText(cmap)
        if idx2 >= 0:
            self.astrometry_tab.recon_cmap_combo.setCurrentIndex(idx2)

    def _open_settings(self):
        dlg = SettingsDialog(self)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            self._set_theme(SETTINGS["theme"])
            self._apply_settings_to_tabs()

    def _set_theme(self, name: str):
        theme.set_theme(name)
        QApplication.instance().setStyleSheet(
            theme.build_stylesheet(theme._theme))
        self.drift_tab.refresh_styles()
        self.preprocess_tab.refresh_styles()
        self.bispectrum_tab.refresh_styles()
        self.astrometry_tab.refresh_styles()
        self.history_tab.refresh_styles()

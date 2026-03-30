"""
speckle_suite.tab_history
==========================
Tab 4 — History: compare your measurements to INT4 historical data
and ORB6 published orbits, with a polar / Cartesian plot and inset zoom.

⚠  This comparison tool must not be used to bias your determination of ρ
   and θ.  It is for curiosity and verification only.
"""

from __future__ import annotations

import json as _json
import datetime
import re
import numpy as np
from pathlib import Path
from typing import Optional

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QLineEdit, QGroupBox, QFileDialog, QTextEdit, QButtonGroup,
)
from PyQt6.QtCore import Qt
from PyQt6.QtCore import Qt

import speckle_suite.theme as theme
from speckle_suite.settings import working_dir
from speckle_suite.widgets import primary_btn_style
from speckle_suite.history_catalog import (
    ORB6_EPHEM_FILE, ORB6_ORBITS_FILE, INT4_FILE, INT4_DB_FILE, WDS_SUMMARY_FILE,
    _int4_color, _int4_label,
    derive_wds_key,
    build_int4_db,
    query_int4, query_wds_summary, query_orb6_ephem, query_orb6_elements,
    search_wds_by_discoverer,
    CatalogWorker,
)
from speckle_suite.history_orbit import compute_orbit_curve


class HistoryTab(QWidget):

    def __init__(self, parent=None):
        super().__init__(parent)
        self._wds_key:   Optional[str] = None
        self._int4_data: list = []
        self._ephem_pts: list = []
        self._orb_elem:  Optional[dict] = None
        self._user_meas: list = []
        self._worker:    object = None
        self._build_ui()

    def _build_ui(self):
        root = QHBoxLayout(self)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(10)

        # ── Left panel ────────────────────────────────────────────────────
        left = QWidget()
        left.setFixedWidth(260)
        lv = QVBoxLayout(left)
        lv.setSpacing(8)
        lv.setContentsMargins(0, 0, 0, 0)

        name_group = QGroupBox("Star Lookup")
        ng = QVBoxLayout(name_group)
        ng.setSpacing(6)
        name_row = QHBoxLayout()
        self.name_edit = QLineEdit()
        self.name_edit.setPlaceholderText("e.g. gamma Vir, WDS 12417-0127")
        self.name_edit.returnPressed.connect(self._resolve_star)
        self.resolve_btn = QPushButton("Resolve")
        self.resolve_btn.setFixedWidth(70)
        self.resolve_btn.clicked.connect(self._resolve_star)
        name_row.addWidget(self.name_edit)
        name_row.addWidget(self.resolve_btn)
        ng.addLayout(name_row)
        self.wds_lbl = QLabel("WDS: —")
        self.wds_lbl.setStyleSheet(f"font-size:10px; color:{theme.TEXT_MUTED};")
        self.coord_lbl = QLabel("Coords: —")
        self.coord_lbl.setStyleSheet(f"font-size:10px; color:{theme.TEXT_MUTED};")
        self.pair_lbl = QLabel("")
        self.pair_lbl.setStyleSheet(f"font-size:10px; color:{theme.TEXT_PRIMARY};")
        self.pair_lbl.setWordWrap(True)
        self.info_lbl = QLabel("")
        self.info_lbl.setStyleSheet(f"font-size:10px; color:{theme.TEXT_MUTED};")
        self.info_lbl.setWordWrap(True)
        ng.addWidget(self.wds_lbl)
        ng.addWidget(self.coord_lbl)
        ng.addWidget(self.pair_lbl)
        ng.addWidget(self.info_lbl)
        lv.addWidget(name_group)

        meas_group = QGroupBox("Your Measurements")
        mg = QVBoxLayout(meas_group)
        mg.setSpacing(6)
        self.load_json_btn = QPushButton("Load Result JSON(s)…")
        self.load_json_btn.clicked.connect(self._load_json)
        self.meas_list = QTextEdit()
        self.meas_list.setReadOnly(True)
        self.meas_list.setMaximumHeight(90)
        self.meas_list.setStyleSheet(f"font-size:9px; color:{theme.TEXT_MUTED};")
        self.clear_meas_btn = QPushButton("Clear")
        self.clear_meas_btn.setFixedWidth(55)
        self.clear_meas_btn.clicked.connect(self._clear_meas)
        mb_row = QHBoxLayout()
        mb_row.addWidget(self.load_json_btn)
        mb_row.addWidget(self.clear_meas_btn)
        mg.addLayout(mb_row)
        mg.addWidget(self.meas_list)
        lv.addWidget(meas_group)


        cat_group = QGroupBox("Catalogs")
        cg = QVBoxLayout(cat_group)
        cg.setSpacing(6)
        self.cat_status_lbl = QLabel("")
        self.cat_status_lbl.setStyleSheet(f"font-size:9px; color:{theme.TEXT_MUTED};")
        self.cat_status_lbl.setWordWrap(True)
        self.download_btn = QPushButton("⬇  Download / Update Catalogs")
        self.download_btn.clicked.connect(self._download_catalogs)
        self.rebuild_db_btn = QPushButton("🔄  Rebuild INT4 Index")
        self.rebuild_db_btn.setToolTip("Re-parse the INT4 file and rebuild the local search index")
        self.rebuild_db_btn.clicked.connect(self._rebuild_int4_db)
        cg.addWidget(self.cat_status_lbl)
        cg.addWidget(self.download_btn)
        cg.addWidget(self.rebuild_db_btn)
        lv.addWidget(cat_group)
        self._refresh_cat_status()

        mode_row = QHBoxLayout()
        mode_row.setSpacing(4)
        self.polar_btn = QPushButton("Polar")
        self.polar_btn.setCheckable(True)
        self.polar_btn.setChecked(True)
        self.polar_btn.setFixedHeight(26)
        self.cartesian_btn = QPushButton("Cartesian")
        self.cartesian_btn.setCheckable(True)
        self.cartesian_btn.setFixedHeight(26)
        self._plot_mode_grp = QButtonGroup()
        self._plot_mode_grp.addButton(self.polar_btn)
        self._plot_mode_grp.addButton(self.cartesian_btn)
        self._plot_mode_grp.setExclusive(True)
        self.polar_btn.clicked.connect(lambda: self._result_available() and self._plot())
        self.cartesian_btn.clicked.connect(lambda: self._result_available() and self._plot())
        mode_row.addWidget(self.polar_btn)
        mode_row.addWidget(self.cartesian_btn)
        lv.addLayout(mode_row)

        self.plot_btn = QPushButton("🔭  Plot History")
        self.plot_btn.setStyleSheet(primary_btn_style())
        self.plot_btn.setEnabled(False)
        self.plot_btn.clicked.connect(self._plot)
        lv.addWidget(self.plot_btn)
        lv.addStretch()

        self.log_edit = QTextEdit()
        self.log_edit.setReadOnly(True)
        self.log_edit.setMaximumHeight(120)
        lv.addWidget(self.log_edit)
        root.addWidget(left)

        # ── Right panel ───────────────────────────────────────────────────
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
        import matplotlib.pyplot as plt
        self._fig, self._ax = plt.subplots(
            subplot_kw={"projection": "polar"}, figsize=(6, 6))
        self._fig.patch.set_facecolor(theme.DARK_BG)
        self._ax.set_facecolor(theme.PANEL_BG)

        right_panel = QWidget()
        rv = QVBoxLayout(right_panel)
        rv.setContentsMargins(4, 4, 4, 4)
        rv.setSpacing(4)

        disclaimer = QLabel(
            "⚠️  This comparison tool between your measurements and historical "
            "data / orbit must <b>not</b> be used to bias your determination "
            "of ρ and θ. It is for curiosity only — and because it is fun! 🔭")
        disclaimer.setWordWrap(True)
        disclaimer.setAlignment(Qt.AlignmentFlag.AlignCenter)
        disclaimer.setStyleSheet(
            f"color:{theme.TEXT_MUTED}; font-style:italic; font-size:10px; "
            f"padding:4px 8px; border:1px solid {theme.BORDER_COLOR}; "
            f"border-radius:4px; background:{theme.PANEL_BG};")
        rv.addWidget(disclaimer)

        self._canvas = FigureCanvasQTAgg(self._fig)
        self._canvas.setMinimumWidth(400)
        rv.addWidget(self._canvas, 1)
        root.addWidget(right_panel, 1)

    # ── Catalog status ─────────────────────────────────────────────────────

    def _refresh_cat_status(self):
        lines = []
        for f, label in [
            (ORB6_EPHEM_FILE,  "ORB6 ephemeris"),
            (ORB6_ORBITS_FILE, "ORB6 elements"),
            (INT4_FILE,        "INT4 catalog"),
            (INT4_DB_FILE,     "INT4 index"),
            (WDS_SUMMARY_FILE, "WDS summary"),
        ]:
            if f.exists():
                lines.append(f"✓ {label} ({f.stat().st_size // 1024} KB)")
            else:
                lines.append(f"✗ {label} missing")
        self.cat_status_lbl.setText("\n".join(lines))

    def _download_catalogs(self):
        self.download_btn.setEnabled(False)
        self._log("Starting catalog download…")
        self._worker = CatalogWorker("download_all")
        self._worker.status.connect(self._log)
        self._worker.finished.connect(self._on_catalog_done)
        self._worker.start()

    def _on_catalog_done(self, ok: bool):
        self.download_btn.setEnabled(True)
        self._refresh_cat_status()
        self._log("✓ Catalogs ready." if ok else "⚠ Some downloads failed.")

    def _rebuild_int4_db(self):
        if not INT4_FILE.exists():
            self._log("⚠ INT4 file not downloaded yet.")
            return
        if INT4_DB_FILE.exists():
            INT4_DB_FILE.unlink()
        self.rebuild_db_btn.setEnabled(False)
        self._log("Rebuilding INT4 index…")
        self._worker = CatalogWorker("build_db")
        self._worker.status.connect(self._log)
        self._worker.finished.connect(lambda ok: (
            self.rebuild_db_btn.setEnabled(True),
            self._refresh_cat_status(),
            self._log("✓ INT4 index rebuilt." if ok else "⚠ Rebuild failed.")
        ))
        self._worker.start()

    # ── Star resolution ────────────────────────────────────────────────────

    def _resolve_star(self):
        name = self.name_edit.text().strip()
        if not name:
            return
        self._log(f"Resolving '{name}'…")

        wds_direct = re.match(r'^(\d{4,5}[+-]\d{3,4})$', name.replace(' ', ''))
        if wds_direct:
            self._set_wds(wds_direct.group(1).zfill(9), source="direct input")
            return

        disc_match = re.match(r'^([A-Za-z]{1,6})\s*(\d+)', name)
        if disc_match:
            code = disc_match.group(1).upper()
            PREFIXES = {
                "STF","STT","STI","SHY","HU","HO","HJ","BU","BV","CHR",
                "WSI","MSN","TOK","ZIR","MCA","BAG","LDS","ES","KUI",
                "A","AG","B","D","H","I","J","S","T","WRH","GRB","COU",
                "FIN","FLD","JOY","MLR","MLO","RST","SEI","SEE","SLE",
            }
            if any(code == p or code.startswith(p) for p in PREFIXES):
                self._log(f"Trying discoverer lookup for '{name}'…")
                wds = search_wds_by_discoverer(name)
                if wds:
                    self._set_wds(wds, source=f"discoverer '{name}'")
                    return
                self._log("  Not found in local catalogs — trying Simbad…")

        try:
            from astropy.coordinates import SkyCoord
            import astropy.units as u
            from astroquery.simbad import Simbad
            result = Simbad.query_object(name)
            if result is None or len(result) == 0:
                self._log(f"⚠ '{name}' not found in Simbad or local catalogs.")
                return
            ra_str = result["RA"][0]
            dec_str = result["DEC"][0]
            coord = SkyCoord(ra=ra_str, dec=dec_str,
                             unit=(u.hourangle, u.deg), frame="icrs")
            wds = derive_wds_key(float(coord.ra.deg), float(coord.dec.deg))
            self.coord_lbl.setText(
                f"Coords: {coord.ra.to_string(unit=u.hour, sep=':', precision=1)}  "
                f"{coord.dec.to_string(sep=':', precision=0, alwayssign=True)}")
            self._set_wds(wds, source="Simbad")
        except ImportError:
            self._log("⚠ astroquery not installed. Run: pip install astroquery")
        except Exception as e:
            self._log(f"⚠ Resolution failed: {e}")

    def _set_wds(self, wds_key: str, source: str = ""):
        self._wds_key = wds_key
        self.wds_lbl.setText(f"WDS: {wds_key}")
        self._log(f"✓ WDS key: {wds_key}  (via {source})")
        self._user_meas.clear()
        self.meas_list.clear()
        self.pair_lbl.setText("")
        self.coord_lbl.setText("Coords: —")
        self._fig.clear()
        self._ax = self._fig.add_subplot(111, projection="polar")
        self._ax.set_facecolor(theme.PANEL_BG)
        self._fig.patch.set_facecolor(theme.DARK_BG)
        self._canvas.draw()
        self._fetch_catalog_data()
        self.plot_btn.setEnabled(True)

    def _fetch_catalog_data(self):
        if not self._wds_key:
            return
        self._ephem_pts = query_orb6_ephem(self._wds_key)
        self._orb_elem  = query_orb6_elements(self._wds_key)
        self._int4_data = query_int4(self._wds_key)

        # ── WDS summary: pair properties ──────────────────────────────────
        wds = query_wds_summary(self._wds_key)
        if wds:
            # Populate coord label from WDS summary if not already set by Simbad
            coord_raw = wds.get("coord", "")
            if coord_raw and self.coord_lbl.text() in ("Coords: —", ""):
                # Format HHMMSS.ss±DDMMSS.s → HH:MM:SS.s  ±DD:MM:SS.s
                try:
                    # Find the sign separating RA and Dec
                    sign_idx = max(coord_raw.rfind('+'), coord_raw.rfind('-'))
                    if sign_idx > 0:
                        ra_raw  = coord_raw[:sign_idx]
                        dec_raw = coord_raw[sign_idx:]
                        ra_fmt  = f"{ra_raw[0:2]}:{ra_raw[2:4]}:{ra_raw[4:]}"
                        dec_fmt = f"{dec_raw[0]}:{dec_raw[1:3]}:{dec_raw[3:]}"
                        self.coord_lbl.setText(f"Coords (J2000): {ra_fmt}  {dec_fmt}")
                    else:
                        self.coord_lbl.setText(f"Coords: {coord_raw}")
                except Exception:
                    self.coord_lbl.setText(f"Coords: {coord_raw}")
        if wds:
            parts = []
            # Magnitudes
            m1 = f"{wds['mag1']:.1f}" if wds['mag1'] is not None else "?"
            m2 = f"{wds['mag2']:.1f}" if wds['mag2'] is not None else "?"
            parts.append(f"m = {m1} / {m2}")
            # Spectral type
            if wds['spectral_type']:
                parts.append(f"Sp: {wds['spectral_type']}")
            # Discoverer
            if wds['disc']:
                parts.append(f"Disc: {wds['disc']}")
            # Observation span
            if wds['obs_first'] and wds['obs_last']:
                n = wds['n_obs'] or "?"
                parts.append(
                    f"Obs: {wds['obs_first']}–{wds['obs_last']} ({n} obs)")
            # First and last measured separation
            if wds['sep_first'] is not None and wds['sep_last'] is not None:
                pa1 = f"{wds['pa_first']}°" if wds['pa_first'] is not None else "?"
                pa2 = f"{wds['pa_last']}°"  if wds['pa_last']  is not None else "?"
                parts.append(
                    f"First: {pa1} {wds['sep_first']:.1f}\"  "
                    f"Last: {pa2} {wds['sep_last']:.1f}\"")
            self.pair_lbl.setText("  ·  ".join(parts[:3]) +
                                  ("\n" + "  ·  ".join(parts[3:]) if len(parts) > 3 else ""))
            # Log the pair info
            self._log("  ".join(parts))
        else:
            self.pair_lbl.setText("")

        # ── Catalog summary for info_lbl ──────────────────────────────────
        info = []
        if self._int4_data:
            n = len(self._int4_data)
            yr_min = min(r["epoch"] for r in self._int4_data)
            yr_max = max(r["epoch"] for r in self._int4_data)
            info.append(f"INT4: {n} measures ({yr_min:.1f}–{yr_max:.1f})")
        else:
            info.append("INT4: no data (download catalogs?)")
        if self._orb_elem:
            info.append(f"ORB6: orbit found (grade {self._orb_elem.get('grade','?')})")
            e = self._orb_elem
            sigma_P = f" ±{e['sigma_P']:.4f}" if e.get('sigma_P') else ""
            sigma_a = f" ±{e['sigma_a']:.5f}" if e.get('sigma_a') else ""
            self._log(
                f"Elements: P={e['P']:.2f}yr{sigma_P} "
                f"a={e['a']:.4f}\"{sigma_a} "
                f"i={e['i']:.1f} "
                f"e={e['e']:.3f}")
        elif self._ephem_pts:
            info.append(f"ORB6: ephemeris only ({len(self._ephem_pts)} pts)")
        else:
            info.append("ORB6: no orbit")
        self.info_lbl.setText("\n".join(info))

    # ── User measurements ──────────────────────────────────────────────────

    def _load_json(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Load Speckle Result JSON(s)", working_dir(),
            "JSON files (*.json);;All Files (*)")
        for p in paths:
            try:
                with open(p) as f:
                    d = _json.load(f)
                if "rho_arcsec" not in d or "theta_sky_deg" not in d:
                    self._log(f"⚠ {Path(p).name}: missing rho/theta fields")
                    continue
                d["_source"] = Path(p).name
                self._user_meas.append(d)
                self._log(f"✓ Loaded: {Path(p).name}")
            except Exception as e:
                self._log(f"⚠ {Path(p).name}: {e}")
        self._update_meas_list()

    def _clear_meas(self):
        self._user_meas.clear()
        self._update_meas_list()

    def _update_meas_list(self):
        if not self._user_meas:
            self.meas_list.setPlainText("(none)")
            return
        lines = [
            f"{m['_source']}: ρ={m.get('rho_arcsec','?'):.4f}\" "
            f"θ={m.get('theta_sky_deg','?'):.2f}°"
            for m in self._user_meas
        ]
        self.meas_list.setPlainText("\n".join(lines))

    # ── Plot ───────────────────────────────────────────────────────────────

    def _result_available(self) -> bool:
        return bool(self._wds_key and (
            self._int4_data or self._ephem_pts or
            self._orb_elem  or self._user_meas))

    def _plot(self):
        import matplotlib
        matplotlib.use("Agg")

        use_polar = self.polar_btn.isChecked()
        self._fig.clear()
        self._ax = self._fig.add_subplot(
            111, projection="polar" if use_polar else None)
        ax = self._ax
        ax.set_facecolor(theme.PANEL_BG)
        if use_polar:
            ax.set_theta_zero_location("N")
            ax.set_theta_direction(-1)

        def to_xy(theta_deg, rho):
            if use_polar:
                return np.radians(90 - np.asarray(theta_deg)), np.asarray(rho)
            th = np.radians(np.asarray(theta_deg))
            return np.asarray(rho) * np.sin(th), np.asarray(rho) * np.cos(th)

        plotted: set[str] = set()

        # INT4
        if self._int4_data:
            groups: dict = {}
            for row in self._int4_data:
                if row["theta"] is None or row["rho"] is None:
                    continue
                lbl = _int4_label(row.get("technique", "?"))
                col = _int4_color(row.get("technique", "?"))
                groups.setdefault(lbl, {"col": col, "th": [], "rh": []})
                groups[lbl]["th"].append(row["theta"])
                groups[lbl]["rh"].append(row["rho"])
            for lbl, g in groups.items():
                xs, ys = to_xy(g["th"], g["rh"])
                ax.scatter(xs, ys, c=g["col"], s=14, alpha=0.75,
                           label=lbl if lbl not in plotted else None, zorder=3)
                plotted.add(lbl)

        # ORB6
        th_curve = rh_curve = None
        cur_yr = (datetime.date.today().year
                  + datetime.date.today().timetuple().tm_yday / 365.25)
        if self._ephem_pts or self._orb_elem:
            th_curve, rh_curve = compute_orbit_curve(
                self._ephem_pts, self._orb_elem, n_pts=360)
            xs_c, ys_c = to_xy(th_curve, rh_curve)
            ax.plot(np.append(xs_c, xs_c[0]), np.append(ys_c, ys_c[0]),
                    color=theme.ACCENT, lw=2.0, label="ORB6 orbit", zorder=4)
            if self._ephem_pts:
                years = [p[0] for p in self._ephem_pts]
                idx   = int(np.argmin(np.abs(np.array(years) - cur_yr)))
                xn, yn = to_xy(self._ephem_pts[idx][1], self._ephem_pts[idx][2])
                ax.scatter([xn], [yn], c=theme.ACCENT, s=120, marker="*",
                           label=f"Predicted {years[idx]:.0f}", zorder=6)

        # User measurements
        first = True
        for m in self._user_meas:
            rho   = m.get("rho_arcsec")
            theta = m.get("theta_sky_deg")
            if rho is None or theta is None:
                continue
            sig_rho   = m.get("sigma_rho_total_arcsec", 0) or 0
            sig_theta = m.get("sigma_theta_total_deg",  0) or 0
            xm, ym = to_xy(theta, rho)
            ax.scatter([xm], [ym], c="#f85149", s=40, marker="o",
                       zorder=7, label="Your measurement" if first else None)
            first = False
            if use_polar:
                th_r = float(np.radians(90 - theta))
                if sig_rho > 0:
                    ax.errorbar([th_r], [rho], yerr=[[sig_rho], [sig_rho]],
                                fmt="none", ecolor="#f85149",
                                elinewidth=1.5, capsize=3, zorder=7)
                if sig_theta > 0 and rho > 0:
                    ax.errorbar([th_r], [rho],
                                xerr=[[np.radians(sig_theta)],
                                       [np.radians(sig_theta)]],
                                fmt="none", ecolor="#f85149",
                                elinewidth=1.5, capsize=3, zorder=7)
            else:
                th_r     = np.radians(theta)
                sig_th_r = np.radians(sig_theta)
                sx = float(np.hypot(np.cos(th_r) * rho * sig_th_r,
                                    np.sin(th_r) * sig_rho))
                sy = float(np.hypot(np.sin(th_r) * rho * sig_th_r,
                                    np.cos(th_r) * sig_rho))
                if sig_rho > 0 or sig_theta > 0:
                    ax.errorbar([float(xm)], [float(ym)],
                                xerr=[[sx], [sx]], yerr=[[sy], [sy]],
                                fmt="none", ecolor="#f85149",
                                elinewidth=1.5, capsize=3, zorder=7)

        # Styling
        grade_str = (f"  grade {self._orb_elem['grade']}"
                     if self._orb_elem and self._orb_elem.get("grade") else "")
        ax.set_title(f"{self._wds_key or '—'}{grade_str}",
                     color=theme.TEXT_PRIMARY, pad=12, fontsize=11)

        if use_polar:
            ax.tick_params(colors=theme.TEXT_MUTED, labelsize=8)
            ax.spines["polar"].set_color(theme.BORDER_COLOR)
            ax.grid(color=theme.BORDER_COLOR, linewidth=0.5)
            ax.set_rlabel_position(45)
            for lbl in ax.get_yticklabels():
                lbl.set_color(theme.TEXT_MUTED); lbl.set_fontsize(7)
            for lbl in ax.get_xticklabels():
                lbl.set_color(theme.TEXT_MUTED); lbl.set_fontsize(8)
        else:
            ax.set_xlabel("ΔRA (arcsec, E→right)",
                          color=theme.TEXT_MUTED, fontsize=9)
            ax.set_ylabel("ΔDec (arcsec, N→up)",
                          color=theme.TEXT_MUTED, fontsize=9)
            ax.tick_params(colors=theme.TEXT_MUTED, labelsize=8)
            for sp in ax.spines.values():
                sp.set_color(theme.BORDER_COLOR)
            ax.grid(color=theme.BORDER_COLOR, linewidth=0.5, alpha=0.5)
            ax.scatter([0], [0], c=theme.WARNING, s=80, marker="+",
                       zorder=8, linewidths=2.5, label="Primary")
            ax.annotate("N", xy=(0.05, 0.95), xycoords="axes fraction",
                        color=theme.TEXT_MUTED, fontsize=8)
            ax.annotate("E", xy=(0.95, 0.05), xycoords="axes fraction",
                        color=theme.TEXT_MUTED, fontsize=8)
            ax.set_aspect("equal")

        # Scale
        candidates: list[float] = []
        if th_curve is not None and rh_curve is not None and len(rh_curve):
            candidates.append(float(rh_curve.max()))
        candidates.extend(m["rho_arcsec"] for m in self._user_meas
                          if m.get("rho_arcsec"))
        candidates.extend(p[2] for p in self._ephem_pts)
        if self._int4_data:
            rhos_i4 = [r["rho"] for r in self._int4_data if r["rho"]]
            if rhos_i4:
                candidates.append(float(np.percentile(rhos_i4, 90)))
        max_rho = max(candidates) if candidates else 1.0
        if use_polar:
            ax.set_ylim(0, max_rho / 0.88)
        else:
            lim = max_rho / 0.82
            ax.set_xlim(-lim, lim)
            ax.set_ylim(-lim, lim)

        if plotted or self._ephem_pts or self._user_meas:
            ax.legend(loc="upper right", bbox_to_anchor=(1.38, 1.12),
                      fontsize=8, framealpha=0.5,
                      facecolor=theme.PANEL_BG,
                      edgecolor=theme.BORDER_COLOR,
                      labelcolor=theme.TEXT_PRIMARY)

        # Inset zoom (Cartesian + user measurements)
        user_valid = [(m["rho_arcsec"], m["theta_sky_deg"])
                      for m in self._user_meas
                      if m.get("rho_arcsec") is not None
                      and m.get("theta_sky_deg") is not None]
        if user_valid and not use_polar:
            inset = self._fig.add_axes([0.80, 0.10, 0.17, 0.17])
            rhos_u   = np.array([v[0] for v in user_valid])
            thetas_u = np.array([v[1] for v in user_valid])
            xys_u    = [to_xy(t, r) for t, r in zip(thetas_u, rhos_u)]
            cx = float(np.mean([xy[0] for xy in xys_u]))
            cy = float(np.mean([xy[1] for xy in xys_u]))
            sig_rhos = [m.get("sigma_rho_total_arcsec", 0) or 0
                        for m in self._user_meas if m.get("rho_arcsec") is not None]
            zoom_r = float(np.clip(
                (max(sig_rhos) if sig_rhos else 0) * 4, 0.05, 0.35))

            if th_curve is not None:
                xs_o, ys_o = to_xy(th_curve, rh_curve)
                inset.plot(xs_o, ys_o, color=theme.ACCENT, lw=1.5, zorder=3)
            for row in self._int4_data:
                if row["theta"] is None or row["rho"] is None:
                    continue
                xi, yi = to_xy(row["theta"], row["rho"])
                inset.scatter([xi], [yi],
                              c=_int4_color(row.get("technique", "?")),
                              s=8, alpha=0.7, zorder=3)
            if self._ephem_pts:
                years = [p[0] for p in self._ephem_pts]
                idx   = int(np.argmin(np.abs(np.array(years) - cur_yr)))
                xe, ye = to_xy(self._ephem_pts[idx][1], self._ephem_pts[idx][2])
                inset.scatter([xe], [ye], c=theme.ACCENT, s=60,
                              marker="*", zorder=5)
            for m in self._user_meas:
                rho_m   = m.get("rho_arcsec")
                theta_m = m.get("theta_sky_deg")
                if rho_m is None or theta_m is None:
                    continue
                sig_r = m.get("sigma_rho_total_arcsec", 0) or 0
                sig_t = m.get("sigma_theta_total_deg",  0) or 0
                xm_i, ym_i = to_xy(theta_m, rho_m)
                inset.scatter([float(xm_i)], [float(ym_i)],
                              c="#f85149", s=30, marker="o", zorder=7)
                th_r_m   = np.radians(theta_m)
                sig_th_r = np.radians(sig_t)
                sx = float(np.hypot(np.cos(th_r_m) * rho_m * sig_th_r,
                                    np.sin(th_r_m) * sig_r))
                sy = float(np.hypot(np.sin(th_r_m) * rho_m * sig_th_r,
                                    np.cos(th_r_m) * sig_r))
                if sig_r > 0 or sig_t > 0:
                    inset.errorbar([float(xm_i)], [float(ym_i)],
                                   xerr=[[sx], [sx]], yerr=[[sy], [sy]],
                                   fmt="none", ecolor="#f85149",
                                   elinewidth=1.2, capsize=2, zorder=7)
            inset.set_facecolor(theme.PANEL_BG)
            inset.tick_params(colors=theme.TEXT_MUTED, labelsize=5)
            for sp in inset.spines.values():
                sp.set_edgecolor("#f85149")
                sp.set_linewidth(1.2)
            inset.set_xlim(cx - zoom_r, cx + zoom_r)
            inset.set_ylim(cy - zoom_r, cy + zoom_r)
            inset.set_xlabel("ΔRA\"", color=theme.TEXT_MUTED,
                             fontsize=5, labelpad=1)
            inset.set_ylabel("ΔDec\"", color=theme.TEXT_MUTED,
                             fontsize=5, labelpad=1)
            inset.axhline(cy, color=theme.TEXT_MUTED, lw=0.4, alpha=0.4)
            inset.axvline(cx, color=theme.TEXT_MUTED, lw=0.4, alpha=0.4)
            inset.set_title("Zoom", color=theme.TEXT_MUTED,
                            fontsize=6, pad=2, style="italic")

        self._fig.patch.set_facecolor(theme.DARK_BG)
        try:
            self._fig.tight_layout(rect=[0, 0, 0.90, 1.0])
        except Exception:
            pass
        self._canvas.draw()
        self._log(
            f"Plot updated — {len(self._int4_data)} INT4 points, "
            f"{'orbit' if (self._ephem_pts or self._orb_elem) else 'no orbit'}, "
            f"{len(self._user_meas)} your measurement(s).")

    # ── Log ────────────────────────────────────────────────────────────────

    def _log(self, msg: str):
        self.log_edit.append(
            f'<span style="color:{theme.TEXT_MUTED}">{msg}</span>')

    # ── Theme refresh ──────────────────────────────────────────────────────

    def _repaint_empty_canvas(self):
        """Repaint figure/axes backgrounds when no data is loaded yet."""
        self._fig.patch.set_facecolor(theme.DARK_BG)
        self._ax.set_facecolor(theme.PANEL_BG)
        # spines exist on both polar and cartesian axes but the polar axis
        # spine is keyed as 'polar'; handle both gracefully.
        try:
            for sp in self._ax.spines.values():
                sp.set_color(theme.BORDER_COLOR)
        except Exception:
            pass
        self._ax.tick_params(colors=theme.TEXT_MUTED, labelsize=8)
        self._ax.title.set_color(theme.TEXT_PRIMARY)
        self._canvas.draw()

    def refresh_styles(self):
        """Called by SpeckleMainWindow after a theme switch."""
        self.plot_btn.setStyleSheet(primary_btn_style())
        for w in (self.cat_status_lbl, self.wds_lbl,
                  self.coord_lbl, self.info_lbl):
            w.setStyleSheet(f"font-size:10px; color:{theme.TEXT_MUTED};")
        self.pair_lbl.setStyleSheet(f"font-size:10px; color:{theme.TEXT_PRIMARY};")
        self.meas_list.setStyleSheet(
            f"font-size:9px; color:{theme.TEXT_MUTED};")
        # Always repaint the matplotlib canvas — even when no data is loaded
        # the figure/axes backgrounds must track the active theme.
        if self._result_available():
            self._plot()
        else:
            self._repaint_empty_canvas()

"""
speckle_suite.history_catalog
==============================
Catalog access for the History tab: INT4 interferometric measurements,
ORB6 orbital ephemeris / elements, and WDS summary catalogue.

Catalog files are stored in  ~/.speckle_suite/catalogs/  and downloaded
from the USNO / GSU on first use.  The INT4 flat file is parsed into a
local SQLite database for fast WDS-key lookup.

Public API
----------
CATALOG_DIR                 Path to the local catalog directory
ORB6_EPHEM_FILE / etc.      Paths to the individual catalog files
INT4_TECHNIQUES             dict: technique code → (label, colour-hex)
derive_wds_key()            (ra_deg, dec_deg) → WDS designation string
download_catalog()          download one file from the USNO/GSU
build_int4_db()             parse INT4 flat file → SQLite index
query_int4()                look up all INT4 measures for a WDS key
query_wds_summary()         look up pair properties from WDS summary
query_orb6_ephem()          look up ORB6 ephemeris (theta/rho per year)
query_orb6_elements()       look up ORB6 orbital elements (fixed-width)
search_wds_by_discoverer()  match a discoverer code against ORB6/INT4
CatalogWorker               QThread: download + build DB
"""

from __future__ import annotations

import re
import sqlite3
from pathlib import Path

from PyQt6.QtCore import QThread, pyqtSignal

# ── Paths ──────────────────────────────────────────────────────────────────

CATALOG_DIR = Path.home() / ".speckle_suite" / "catalogs"
CATALOG_DIR.mkdir(parents=True, exist_ok=True)

ORB6_EPHEM_URL   = "https://crf.usno.navy.mil/data_products/WDS/orb6/orb6ephem.txt"
ORB6_ORBITS_URL  = "https://crf.usno.navy.mil/data_products/WDS/orb6/orb6orbits.txt"
INT4_URL         = "https://crf.usno.navy.mil/data_products/WDS/int4/int4_all.txt"
WDS_SUMMARY_URL  = "https://www.astro.gsu.edu/wds/Webtextfiles/wdsweb_summ2.txt"

ORB6_EPHEM_FILE  = CATALOG_DIR / "orb6ephem.txt"
ORB6_ORBITS_FILE = CATALOG_DIR / "orb6orbits.txt"
INT4_FILE        = CATALOG_DIR / "int4_all.txt"
INT4_DB_FILE     = CATALOG_DIR / "int4.sqlite"
WDS_SUMMARY_FILE = CATALOG_DIR / "wdsweb_summ2.txt"

# ── Technique metadata ─────────────────────────────────────────────────────

INT4_TECHNIQUES: dict[str, tuple[str, str]] = {
    "S":  ("Speckle",       "#58a6ff"),
    "Su": ("Speckle",       "#58a6ff"),
    "Sc": ("Speckle CCD",   "#3fb950"),
    "A":  ("Adaptive opt",  "#d2a679"),
    "H":  ("Hipparcos",     "#d29922"),
    "Hh": ("Hipparcos",     "#d29922"),
    "Hf": ("Hipparcos",     "#d29922"),
    "Ht": ("Hipparcos",     "#d29922"),
    "Hw": ("Hipparcos",     "#d29922"),
    "M":  ("Micrometry",    "#8b949e"),
    "C":  ("CCD",           "#a5d6ff"),
    "E":  ("Eyepiece int",  "#c9d1d9"),
    "E2": ("Eyepiece int",  "#c9d1d9"),
}


def _int4_color(tech: str) -> str:
    for k, (_, c) in INT4_TECHNIQUES.items():
        if tech.strip().startswith(k):
            return c
    return "#8b949e"


def _int4_label(tech: str) -> str:
    for k, (l, _) in INT4_TECHNIQUES.items():
        if tech.strip().startswith(k):
            return l
    return "Other"


# ── WDS key derivation ─────────────────────────────────────────────────────

def derive_wds_key(ra_deg: float, dec_deg: float) -> str:
    """Derive the WDS designation string (HHMM±DDMM) from J2000 coords."""
    ra_h = ra_deg / 15.0
    h    = int(ra_h)
    m    = (ra_h - h) * 60.0
    hh   = f"{h:02d}"
    mm   = f"{m:04.1f}".replace(".", "")[:4]
    sign = "+" if dec_deg >= 0 else "-"
    ad   = abs(dec_deg)
    dd   = int(ad)
    dm   = int((ad - dd) * 60.0)
    return f"{hh}{mm}{sign}{dd:02d}{dm:02d}"


# ── Download helper ────────────────────────────────────────────────────────

def download_catalog(url: str, dest: Path, log_cb=None) -> bool:
    """Download a catalog file with basic progress logging. Returns True on success."""
    import urllib.request
    try:
        if log_cb:
            log_cb(f"Downloading {dest.name}…")
        urllib.request.urlretrieve(url, dest)
        if log_cb:
            log_cb(f"✓ {dest.name} ({dest.stat().st_size // 1024} KB)")
        return True
    except Exception as e:
        if log_cb:
            log_cb(f"⚠ Download failed: {e}")
        return False


# ── INT4 database ──────────────────────────────────────────────────────────

def build_int4_db(log_cb=None) -> bool:
    """
    Parse the INT4 flat file and build a SQLite search index.

    The file comes in two variants (HTML-wrapped and plain text); both are
    handled by the same header-line detector.
    """
    if log_cb:
        log_cb("Building INT4 index (first-time, may take ~30 s)…")

    con = sqlite3.connect(INT4_DB_FILE)
    cur = con.cursor()
    cur.execute("DROP TABLE IF EXISTS measures")
    cur.execute("""CREATE TABLE measures (
        wds_key TEXT, epoch REAL, theta REAL, sigma_theta REAL,
        rho REAL, sigma_rho REAL, technique TEXT, reference TEXT)""")
    cur.execute("CREATE INDEX idx_wds ON measures(wds_key)")

    coord_re = re.compile(r"^\[?\d{6}\.\d{2}[+-]\d{6}\.\d")
    wds_re   = re.compile(r"(\d{5}[+-]\d{4})")

    current_wds = None
    batch: list[tuple] = []

    def _f(s):
        try:
            return float(s.strip(":q<>"))
        except ValueError:
            return None

    with open(INT4_FILE, "r", errors="replace") as f:
        for line in f:
            if coord_re.match(line):
                m = wds_re.search(line)
                if m:
                    current_wds = m.group(1)
                continue
            if current_wds is None:
                continue
            parts = line.split()
            if not parts:
                continue
            try:
                epoch = float(parts[0].strip(":q"))
            except ValueError:
                continue
            if not (1700 < epoch < 2200):
                continue
            try:
                theta     = _f(parts[1]) if len(parts) > 1 else None
                sig_theta = _f(parts[2]) if len(parts) > 2 else None
                rho       = _f(parts[3]) if len(parts) > 3 else None
                sig_rho   = _f(parts[4]) if len(parts) > 4 else None
                technique = parts[-1]    if len(parts) > 1 else ""
                reference = parts[-2]    if len(parts) > 2 else ""
            except Exception:
                continue
            if theta is None and rho is None:
                continue
            batch.append((current_wds, epoch, theta, sig_theta,
                          rho, sig_rho, technique, reference))
            if len(batch) >= 5000:
                cur.executemany(
                    "INSERT INTO measures VALUES(?,?,?,?,?,?,?,?)", batch)
                batch.clear()

    if batch:
        cur.executemany("INSERT INTO measures VALUES(?,?,?,?,?,?,?,?)", batch)
    con.commit()
    con.close()
    if log_cb:
        log_cb(f"✓ INT4 index built: {INT4_DB_FILE.name}")
    return True


def query_int4(wds_key: str) -> list[dict]:
    """Return all INT4 measures for *wds_key*, ordered by epoch."""
    if not INT4_DB_FILE.exists():
        return []
    con = sqlite3.connect(INT4_DB_FILE)
    cur = con.cursor()
    cur.execute(
        "SELECT epoch,theta,sigma_theta,rho,sigma_rho,technique,reference "
        "FROM measures WHERE wds_key=? ORDER BY epoch",
        (wds_key,))
    rows = cur.fetchall()
    con.close()
    return [{"epoch": r[0], "theta": r[1], "sigma_theta": r[2],
             "rho": r[3], "sigma_rho": r[4],
             "technique": r[5], "reference": r[6]} for r in rows]


def query_wds_summary(wds_key: str) -> dict | None:
    """
    Look up a pair in the WDS summary catalogue (wdsweb_summ2.txt) using
    fixed-width column parsing.

    Column positions (1-based, inclusive) follow the astrolabium WDSParser
    definition by TheWand3rer (https://github.com/TheWand3rer/astrolabium):

      WDS       1–10    disc    11–17   obs_f   24–27   obs_l   29–32
      n_obs    34–37    pa1     39–41   pa2     43–45   sep1    47–51
      sep2     53–57    mag1    59–63   mag2    65–69   st      71–79
      coord   113–130

    Returns a dict with keys: disc, obs_first, obs_last, n_obs,
    pa_first, pa_last, sep_first, sep_last, mag1, mag2, spectral_type,
    coord; or None if not found.
    """
    if not WDS_SUMMARY_FILE.exists():
        return None

    def _col(line: str, start: int, end: int) -> str:
        return line[start - 1: end].strip()

    def _f(s: str) -> float | None:
        try:
            return float(s) if s else None
        except ValueError:
            return None

    def _i(s: str) -> int | None:
        try:
            return int(s) if s else None
        except ValueError:
            return None

    with open(WDS_SUMMARY_FILE, "r", errors="replace") as f:
        for line in f:
            if len(line) < 10:
                continue
            key = _col(line, 1, 10)
            if key != wds_key:
                continue
            return {
                "disc":          _col(line, 11,  17),
                "obs_first":     _i(_col(line, 24, 27)),
                "obs_last":      _i(_col(line, 29, 32)),
                "n_obs":         _i(_col(line, 34, 37)),
                "pa_first":      _i(_col(line, 39, 41)),
                "pa_last":       _i(_col(line, 43, 45)),
                "sep_first":     _f(_col(line, 47, 51)),
                "sep_last":      _f(_col(line, 53, 57)),
                "mag1":          _f(_col(line, 59, 63)),
                "mag2":          _f(_col(line, 65, 69)),
                "spectral_type": _col(line, 71, 79),
                "coord":         _col(line, 113, 130),
            }
    return None


# ── ORB6 queries ───────────────────────────────────────────────────────────

def query_orb6_ephem(wds_key: str) -> list[tuple[float, float, float]]:
    """
    Return [(year, theta_deg, rho_arcsec), …] from the ORB6 ephemeris file.
    Predictions start at 2025 and advance by 1 year per pair.
    """
    if not ORB6_EPHEM_FILE.exists():
        return []
    results: list[tuple] = []
    with open(ORB6_EPHEM_FILE, "r", errors="replace") as f:
        for line in f:
            if not line.startswith(wds_key):
                continue
            parts = line.split()
            # Find the first numeric pair that looks like (theta, rho)
            data_start = None
            for i in range(2, len(parts) - 1):
                try:
                    float(parts[i]); float(parts[i + 1])
                    if 0.0 <= float(parts[i]) <= 360.0 and float(parts[i+1]) >= 0.0:
                        prev = parts[i - 1]
                        try:
                            float(prev)
                            continue
                        except ValueError:
                            data_start = i
                            break
                except (ValueError, IndexError):
                    continue
            if data_start is None:
                continue
            year = 2025.0
            for i in range(data_start, len(parts) - 1, 2):
                try:
                    th = float(parts[i])
                    rh = float(parts[i + 1])
                    results.append((year, th, rh))
                    year += 1.0
                except ValueError:
                    break
    return results


def query_orb6_elements(wds_key: str) -> dict | None:
    """
    Parse Keplerian elements from the ORB6 orbital elements file using
    fixed-width column slicing.

    Column positions (1-based, inclusive) follow the astrolabium Orb6Parser
    definition by TheWand3rer (https://github.com/TheWand3rer/astrolabium):

      WDS      20–30    P        82–93    a       106–115
      P_err    95–105   a_err   117–125   i       126–134
      i_err   135–143   Omega   144–152   Omega_err 154–162
      T       163–175   T_err   177–187   e       188–196
      e_err   197–205   omega   206–214   omega_err 215–223
      grade   234–235

    Period unit suffix (y/d/h/m/c) and axis unit suffix (a/m/u) may be
    appended within the same field; they are stripped before conversion.

    Returns a dict with keys P, a, i, Omega, T, e, omega, grade,
    sigma_P, sigma_a, sigma_i, sigma_Omega, sigma_e, sigma_omega,
    and _raw_line; or None if the star is not found.
    """
    if not ORB6_ORBITS_FILE.exists():
        return None

    # Unit conversion factors relative to the base unit (years / arcsec)
    _period_units = {"y": 1.0, "d": 1.0/365.25, "h": 1.0/8766.0,
                     "m": 1.0/525960.0, "c": 100.0}
    _axis_units   = {"a": 1.0, "m": 1e-3, "u": 1e-6}

    def _col(line: str, start: int, end: int) -> str:
        """Extract a 1-based inclusive column slice and strip whitespace."""
        return line[start - 1: end].strip()

    def _parse_period(raw: str) -> tuple[float | None, float]:
        """
        Parse a period field that may carry a unit suffix (y/d/h/m/c).
        Returns (value_years, factor) where factor converts raw→years.
        """
        raw = raw.strip().rstrip(".")
        if not raw:
            return None, 1.0
        # Unit suffix is the last alphabetic character(s)
        suffix = ""
        for ch in reversed(raw):
            if ch.isalpha():
                suffix = ch + suffix
            else:
                break
        factor = _period_units.get(suffix, 1.0)  # default: already in years
        num_str = raw[: len(raw) - len(suffix)].strip()
        try:
            return float(num_str) * factor, factor
        except ValueError:
            return None, 1.0

    def _parse_axis(raw: str) -> float | None:
        """
        Parse a semi-major axis field that may carry a unit suffix (a/m/u).
        Returns value in arcseconds.
        """
        raw = raw.strip().rstrip(".")
        if not raw:
            return None
        suffix = ""
        for ch in reversed(raw):
            if ch.isalpha():
                suffix = ch + suffix
            else:
                break
        factor = _axis_units.get(suffix, 1.0)
        num_str = raw[: len(raw) - len(suffix)].strip()
        try:
            return float(num_str) * factor
        except ValueError:
            return None

    def _parse_T(raw: str) -> float | None:
        """Parse epoch of periastron: Besselian year, JD, or truncated JD."""
        raw = raw.strip().rstrip(".")
        if not raw:
            return None
        # Strip trailing alphabetic unit (rare 'y' suffix)
        num_str = raw.rstrip("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ")
        try:
            v = float(num_str)
        except ValueError:
            return None
        if v > 2400000:          # full JD
            return 2000.0 + (v - 2451545.0) / 365.25
        if v > 10000:            # truncated JD (add 2 400 000)
            return 2000.0 + (v + 2400000.0 - 2451545.0) / 365.25
        return v                 # already a year

    def _f(raw: str) -> float | None:
        """Parse a plain float field; return None if blank or invalid."""
        s = raw.strip().rstrip(".*?q")
        try:
            return float(s) if s else None
        except ValueError:
            return None

    with open(ORB6_ORBITS_FILE, "r", errors="replace") as f:
        for line in f:
            # WDS key lives at columns 20–30 (1-based)
            if len(line) < 30:
                continue
            key = _col(line, 20, 30)
            if key != wds_key:
                continue

            # ── Fixed-width extraction ────────────────────────────────────
            P_raw     = _col(line, 82,  93)
            P_err_raw = _col(line, 95, 105)
            a_raw     = _col(line, 106, 115)
            a_err_raw = _col(line, 117, 125)
            i_raw     = _col(line, 126, 134)
            i_err_raw = _col(line, 135, 143)
            Om_raw    = _col(line, 144, 152)
            Om_err_raw= _col(line, 154, 162)
            T_raw     = _col(line, 163, 175)
            # T_err   = _col(line, 177, 187)   # available but not used
            e_raw     = _col(line, 188, 196)
            e_err_raw = _col(line, 197, 205)
            om_raw    = _col(line, 206, 214)
            om_err_raw= _col(line, 215, 223)
            grade_raw = _col(line, 234, 235)

            P_yr, _factor = _parse_period(P_raw)
            if P_yr is None:
                continue                  # malformed line — skip

            a_as = _parse_axis(a_raw)
            if a_as is None:
                continue

            # Uncertainties: same unit suffix as the value field
            sigma_P  = _parse_period(P_err_raw)[0]
            sigma_a  = _parse_axis(a_err_raw)
            sigma_i  = _f(i_err_raw)
            sigma_Om = _f(Om_err_raw)
            sigma_e  = _f(e_err_raw)
            sigma_om = _f(om_err_raw)

            try:
                grade = int(grade_raw) if grade_raw else None
            except ValueError:
                grade = None

            return {
                "P":           P_yr,
                "a":           a_as,
                "i":           _f(i_raw)  or 0.0,
                "Omega":       _f(Om_raw) or 0.0,
                "T":           _parse_T(T_raw) or 2000.0,
                "e":           _f(e_raw)  or 0.0,
                "omega":       _f(om_raw) or 0.0,
                "grade":       grade,
                # 1-σ uncertainties (None when not published)
                "sigma_P":     sigma_P,
                "sigma_a":     sigma_a,
                "sigma_i":     sigma_i,
                "sigma_Omega": sigma_Om,
                "sigma_e":     sigma_e,
                "sigma_omega": sigma_om,
                "_raw_line":   line.rstrip(),
            }
    return None


# ── Discoverer search ──────────────────────────────────────────────────────

def search_wds_by_discoverer(discoverer: str) -> str | None:
    """
    Match a discoverer designation (e.g. "STF 1883", "HU 628") against
    ORB6 and INT4 header lines.  Returns the WDS key if found, else None.
    """
    def _norm(s: str) -> str:
        return re.sub(r"\s+", "", s.upper())

    def _strip_suffix(s: str) -> str:
        return re.sub(r"[,]?(AA|AB|AC|AD|BC|CD|Aa|Ab|Ac|[A-Z]{1,2})$", "", s)

    raw_base = _strip_suffix(_norm(discoverer))
    if not raw_base:
        return None

    wds_re = re.compile(r"(\d{5}[+-]\d{4})")

    # ── ORB6 ephemeris ───────────────────────────────────────────────────
    if ORB6_EPHEM_FILE.exists():
        with open(ORB6_EPHEM_FILE, "r", errors="replace") as f:
            for line in f:
                parts = line.split()
                if len(parts) < 2:
                    continue
                disc      = _norm(parts[1] + (parts[2] if len(parts) > 2 else ""))
                disc_base = _strip_suffix(disc)
                if disc_base == raw_base or disc == raw_base:
                    m = wds_re.match(parts[0])
                    if m:
                        return m.group(1)

    # ── ORB6 elements ────────────────────────────────────────────────────
    if ORB6_ORBITS_FILE.exists():
        with open(ORB6_ORBITS_FILE, "r", errors="replace") as f:
            for line in f:
                m_wds = wds_re.match(line.strip())
                if not m_wds:
                    continue
                parts     = line.split()
                if len(parts) < 3:
                    continue
                disc      = _norm(parts[1] + (parts[2] if len(parts) > 2 else ""))
                disc_base = _strip_suffix(disc)
                if disc_base == raw_base or disc == raw_base:
                    return m_wds.group(1)

    # ── INT4 header lines ────────────────────────────────────────────────
    if INT4_FILE.exists():
        with open(INT4_FILE, "r", errors="replace") as f:
            for line in f:
                if not line.startswith("["):
                    continue
                if raw_base not in _norm(line):
                    continue
                m = wds_re.search(line)
                if m:
                    return m.group(1)

    return None


# ── CatalogWorker ──────────────────────────────────────────────────────────

class CatalogWorker(QThread):
    """Background worker: download catalogs and/or rebuild the INT4 SQLite index."""

    status   = pyqtSignal(str)
    finished = pyqtSignal(bool)

    def __init__(self, action: str):
        super().__init__()
        self.action = action   # "download_all" | "build_db"

    def run(self) -> None:
        try:
            if self.action == "download_all":
                ok = True
                for url, dest in [
                    (ORB6_EPHEM_URL,  ORB6_EPHEM_FILE),
                    (ORB6_ORBITS_URL, ORB6_ORBITS_FILE),
                    (INT4_URL,        INT4_FILE),
                    (WDS_SUMMARY_URL, WDS_SUMMARY_FILE),
                ]:
                    ok = ok and download_catalog(url, dest, self.status.emit)
                if ok and INT4_FILE.exists():
                    self.status.emit("Building INT4 index…")
                    build_int4_db(self.status.emit)
                self.finished.emit(ok)
            elif self.action == "build_db":
                ok = build_int4_db(self.status.emit)
                self.finished.emit(ok)
        except Exception as e:
            self.status.emit(f"⚠ {e}")
            self.finished.emit(False)

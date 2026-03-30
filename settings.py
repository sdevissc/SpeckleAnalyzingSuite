"""
speckle_suite.settings
======================
Persistent JSON user preferences, loaded once at import time.

Usage
-----
    from speckle_suite.settings import SETTINGS, save_settings, working_dir
"""

import json
from pathlib import Path

# ── Defaults ───────────────────────────────────────────────────────────────

_SETTINGS_PATH = Path.home() / ".config" / "speckle_suite" / "settings.json"

_DEFAULTS: dict = {
    "theme":        "dark",
    "working_dir":  str(Path.home()),
    "preprocess": {"roi_index": 0},
    "analysis": {
        "k_max": 60, "dk_max": 9, "n_iter": 30,
        "epsilon": 0.01, "colormap": "Grey",
    },
}


# ── Load / save ────────────────────────────────────────────────────────────

def _load() -> dict:
    try:
        if _SETTINGS_PATH.exists():
            with open(_SETTINGS_PATH) as f:
                data = json.load(f)
            merged = dict(_DEFAULTS)
            merged["preprocess"] = {**_DEFAULTS["preprocess"],
                                    **data.get("preprocess", {})}
            merged["analysis"]   = {**_DEFAULTS["analysis"],
                                    **data.get("analysis", {})}
            for k in ("theme", "working_dir"):
                if k in data:
                    merged[k] = data[k]
            return merged
    except Exception:
        pass
    return dict(_DEFAULTS)


def save_settings(s: dict) -> None:
    """Persist settings dict to disk."""
    try:
        _SETTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(_SETTINGS_PATH, "w") as f:
            json.dump(s, f, indent=2)
    except Exception as e:
        print(f"[Settings] Save failed: {e}")


def working_dir() -> str:
    """Return the current working directory preference."""
    return SETTINGS.get("working_dir", str(Path.home()))


# ── Module-level singleton ─────────────────────────────────────────────────

SETTINGS: dict = _load()

"""
speckle_suite.theme
===================
Single source of truth for all colour tokens and the Qt stylesheet.

Usage in every other module
----------------------------
    import speckle_suite.theme as theme

    # Read a colour token — always go via the module reference so the
    # value stays current after a theme switch.
    color = theme.ACCENT

    # Build a style string
    style = f"color: {theme.ACCENT}; background: {theme.PANEL_BG};"

Theme switching (done once, by SpeckleMainWindow)
-------------------------------------------------
    theme.set_theme("red")
    QApplication.instance().setStyleSheet(theme.build_stylesheet(theme._theme))
    # then call refresh_styles() on each tab widget
"""

# ── Palette ────────────────────────────────────────────────────────────────

THEMES: dict[str, dict[str, str]] = {
    'dark': {
        'DARK_BG':      '#0d1117',
        'PANEL_BG':     '#161b22',
        'BORDER_COLOR': '#30363d',
        'ACCENT':       '#58a6ff',
        'ACCENT2':      '#3fb950',
        'TEXT_PRIMARY': '#e6edf3',
        'TEXT_MUTED':   '#8b949e',
        'WARNING':      '#d29922',
        'DANGER':       '#f85149',
    },
    'red': {
        'DARK_BG':      '#0e0500',
        'PANEL_BG':     '#1a0a00',
        'BORDER_COLOR': '#4a1500',
        'ACCENT':       '#ff6b35',
        'ACCENT2':      '#cc3300',
        'TEXT_PRIMARY': '#ffcba4',
        'TEXT_MUTED':   '#994422',
        'WARNING':      '#ff9900',
        'DANGER':       '#ff3311',
    },
    'light': {
        'DARK_BG':      '#e8e8e8',
        'PANEL_BG':     '#f3f3f3',
        'BORDER_COLOR': '#b0b8c4',
        'ACCENT':       '#0055aa',
        'ACCENT2':      '#006b2b',
        'TEXT_PRIMARY': '#1a1a1a',
        'TEXT_MUTED':   '#5a6370',
        'WARNING':      '#9a4f00',
        'DANGER':       '#bb0000',
    },
}

# ── Active theme state ─────────────────────────────────────────────────────

_theme: dict[str, str] = THEMES['dark']

# Module-level colour aliases — always read as  theme.ACCENT  etc.
DARK_BG:      str = _theme['DARK_BG']
PANEL_BG:     str = _theme['PANEL_BG']
BORDER_COLOR: str = _theme['BORDER_COLOR']
ACCENT:       str = _theme['ACCENT']
ACCENT2:      str = _theme['ACCENT2']
TEXT_PRIMARY: str = _theme['TEXT_PRIMARY']
TEXT_MUTED:   str = _theme['TEXT_MUTED']
WARNING:      str = _theme['WARNING']
DANGER:       str = _theme['DANGER']


def _refresh_aliases() -> None:
    """Update all module-level colour aliases from the current _theme dict."""
    global DARK_BG, PANEL_BG, BORDER_COLOR, ACCENT, ACCENT2
    global TEXT_PRIMARY, TEXT_MUTED, WARNING, DANGER
    DARK_BG      = _theme['DARK_BG']
    PANEL_BG     = _theme['PANEL_BG']
    BORDER_COLOR = _theme['BORDER_COLOR']
    ACCENT       = _theme['ACCENT']
    ACCENT2      = _theme['ACCENT2']
    TEXT_PRIMARY = _theme['TEXT_PRIMARY']
    TEXT_MUTED   = _theme['TEXT_MUTED']
    WARNING      = _theme['WARNING']
    DANGER       = _theme['DANGER']


def set_theme(name: str) -> None:
    """Switch to a named theme and refresh all colour aliases."""
    global _theme
    _theme = THEMES[name]
    _refresh_aliases()


# ── Qt stylesheet ──────────────────────────────────────────────────────────

def build_stylesheet(t: dict | None = None) -> str:
    """Return the full Qt stylesheet for the given theme dict (default: active)."""
    if t is None:
        t = _theme
    return f"""
QMainWindow, QWidget {{
    background-color: {t['DARK_BG']};
    color: {t['TEXT_PRIMARY']};
    font-family: 'JetBrains Mono', 'Fira Code', 'Courier New', monospace;
    font-size: 12px;
}}
QTabWidget::pane {{
    border: 1px solid {t['BORDER_COLOR']};
    border-radius: 4px;
}}
QTabBar::tab {{
    background: {t['PANEL_BG']};
    border: 1px solid {t['BORDER_COLOR']};
    border-bottom: none;
    border-radius: 4px 4px 0 0;
    padding: 6px 24px;
    color: {t['TEXT_MUTED']};
    font-weight: bold;
    letter-spacing: 1px;
}}
QTabBar::tab:selected {{
    background: {t['DARK_BG']};
    color: {t['ACCENT']};
    border-bottom: 2px solid {t['ACCENT']};
}}
QTabBar::tab:hover:!selected {{
    color: {t['TEXT_PRIMARY']};
}}
QGroupBox {{
    border: 1px solid {t['BORDER_COLOR']};
    border-radius: 6px;
    margin-top: 10px;
    padding: 8px;
    font-weight: bold;
    color: {t['ACCENT']};
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    left: 10px;
    padding: 0 4px;
    color: {t['ACCENT']};
    font-size: 11px;
    letter-spacing: 1px;
    text-transform: uppercase;
}}
QLineEdit, QSpinBox, QDoubleSpinBox {{
    background-color: {t['PANEL_BG']};
    border: 1px solid {t['BORDER_COLOR']};
    border-radius: 4px;
    padding: 4px 8px;
    color: {t['TEXT_PRIMARY']};
    selection-background-color: {t['ACCENT']};
}}
QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus {{
    border-color: {t['ACCENT']};
}}
QLineEdit:read-only {{ color: {t['TEXT_MUTED']}; }}
QPushButton {{
    background-color: {t['PANEL_BG']};
    border: 1px solid {t['BORDER_COLOR']};
    border-radius: 4px;
    padding: 6px 14px;
    color: {t['TEXT_PRIMARY']};
    font-weight: bold;
}}
QPushButton:hover {{ border-color: {t['ACCENT']}; color: {t['ACCENT']}; }}
QPushButton:disabled {{ background-color: {t['BORDER_COLOR']}; color: {t['TEXT_MUTED']}; }}
QProgressBar {{
    border: 1px solid {t['BORDER_COLOR']};
    border-radius: 4px;
    background-color: {t['PANEL_BG']};
    text-align: center;
    color: {t['TEXT_PRIMARY']};
    height: 14px;
}}
QProgressBar::chunk {{
    background-color: {t['ACCENT']};
    border-radius: 3px;
}}
QTextEdit {{
    background-color: {t['PANEL_BG']};
    border: 1px solid {t['BORDER_COLOR']};
    border-radius: 4px;
    color: {t['TEXT_MUTED']};
    font-size: 11px;
}}
QComboBox {{
    background-color: {t['PANEL_BG']};
    border: 1px solid {t['BORDER_COLOR']};
    border-radius: 4px;
    padding: 4px 8px;
    color: {t['TEXT_PRIMARY']};
}}
QComboBox:hover {{ border-color: {t['ACCENT']}; }}
QComboBox::drop-down {{ border: none; }}
QComboBox QAbstractItemView {{
    background-color: {t['PANEL_BG']};
    border: 1px solid {t['BORDER_COLOR']};
    color: {t['TEXT_PRIMARY']};
    selection-background-color: {t['ACCENT']};
}}
QMenuBar {{
    background-color: {t['PANEL_BG']};
    color: {t['TEXT_PRIMARY']};
    border-bottom: 1px solid {t['BORDER_COLOR']};
}}
QMenuBar::item:selected {{ background-color: {t['BORDER_COLOR']}; }}
QMenu {{
    background-color: {t['PANEL_BG']};
    border: 1px solid {t['BORDER_COLOR']};
    color: {t['TEXT_PRIMARY']};
}}
QMenu::item:selected {{ background-color: {t['ACCENT']}; color: {t['DARK_BG']}; }}
QSlider::groove:horizontal {{
    height: 4px;
    background: {t['BORDER_COLOR']};
    border-radius: 2px;
}}
QSlider::handle:horizontal {{
    background: {t['ACCENT']};
    border: none;
    width: 14px; height: 14px;
    margin: -5px 0;
    border-radius: 7px;
}}
QSlider::sub-page:horizontal {{
    background: {t['ACCENT']};
    border-radius: 2px;
}}
QFrame#separator {{
    background-color: {t['BORDER_COLOR']};
    max-height: 1px;
}}
QLabel#result_value {{
    color: {t['ACCENT2']};
    font-size: 20px;
    font-weight: bold;
}}
QLabel#result_label {{
    color: {t['TEXT_MUTED']};
    font-size: 10px;
}}
QCheckBox {{ color: {t['TEXT_MUTED']}; spacing: 6px; }}
QCheckBox::indicator {{
    width: 14px; height: 14px;
    border: 1px solid {t['BORDER_COLOR']};
    border-radius: 3px;
    background: {t['PANEL_BG']};
}}
QCheckBox::indicator:checked {{ background: {t['ACCENT']}; border-color: {t['ACCENT']}; }}
"""


# Initialise aliases at import time
_refresh_aliases()

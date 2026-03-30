"""
speckle_suite.__main__
=======================
Entry point.  Run as:

    python -m speckle_suite
"""

import sys

import pyqtgraph as pg
from PyQt6.QtWidgets import QApplication
from PyQt6.QtGui import QFont

from speckle_suite.main_window import SpeckleMainWindow


def main():
    pg.setConfigOptions(antialias=True, imageAxisOrder='row-major')
    app = QApplication(sys.argv)
    app.setFont(QFont("JetBrains Mono", 10))
    win = SpeckleMainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()

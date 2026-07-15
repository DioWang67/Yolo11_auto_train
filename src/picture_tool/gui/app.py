"""Picture Tool Application Entry Point.

This module initializes the QApplication and launches the MainWindow.
"""
import argparse
import sys
from PyQt5 import QtGui
from PyQt5.QtCore import QTimer
from PyQt5.QtWidgets import QApplication

from picture_tool.gui.main_window import MainWindow

def main() -> None:
    """Application Entry Point."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--handoff")
    args, qt_args = parser.parse_known_args()
    app = QApplication([sys.argv[0], *qt_args])
    
    # Set Global Font
    font = QtGui.QFont("Segoe UI", 9)
    app.setFont(font)
    
    # Launch Main Window
    window = MainWindow()
    window.show()
    if args.handoff:
        QTimer.singleShot(0, lambda: window.apply_operator_handoff(args.handoff))
    
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()

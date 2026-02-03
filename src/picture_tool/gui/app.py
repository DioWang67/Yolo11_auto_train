"""Picture Tool Application Entry Point.

This module initializes the QApplication and launches the MainWindow.
"""
import sys
from PyQt5 import QtGui
from PyQt5.QtWidgets import QApplication

from picture_tool.gui.main_window import MainWindow

def main() -> None:
    """Application Entry Point."""
    app = QApplication(sys.argv)
    
    # Set Global Font
    font = QtGui.QFont("Segoe UI", 9)
    app.setFont(font)
    
    # Launch Main Window
    window = MainWindow()
    window.show()
    
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()

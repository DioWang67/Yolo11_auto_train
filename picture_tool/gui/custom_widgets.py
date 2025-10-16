"""Reusable custom widgets for the YOLO auto-train GUI."""
from PyQt5.QtWidgets import QCheckBox, QPushButton


class CompactCheckBox(QCheckBox):
    def __init__(self, text, parent=None):
        super().__init__(text, parent)
        self.setStyleSheet("""
            QCheckBox {
                font-size: 9pt;
                color: #2c3e50;
                spacing: 5px;
                padding: 2px;
                margin: 1px;
            }
            QCheckBox::indicator {
                width: 14px;
                height: 14px;
                border: 1px solid #ced4da;
                border-radius: 3px;
                background-color: white;
            }
            QCheckBox::indicator:checked {
                background-color: #007bff;
                border-color: #0056b3;
            }
            QCheckBox::indicator:hover {
                border-color: #80bdff;
            }
        """)


class CompactButton(QPushButton):
    def __init__(self, text, color_theme="primary", parent=None):
        super().__init__(text, parent)
        self.color_theme = color_theme
        self._setup_style()

    def _setup_style(self):
        base_style = """
            QPushButton {
                border: none;
                padding: 6px 12px;
                border-radius: 4px;
                font-size: 9pt;
                font-weight: bold;
                min-height: 24px;
                text-align: center;
            }
            QPushButton:disabled {
                background-color: #6c757d;
                color: #dee2e6;
            }
        """
        color_styles = {
            "primary": """
                QPushButton {
                    background-color: #007bff;
                    color: white;
                }
                QPushButton:hover {
                    background-color: #0056b3;
                }
                QPushButton:pressed {
                    background-color: #004085;
                }
            """,
            "success": """
                QPushButton {
                    background-color: #28a745;
                    color: white;
                }
                QPushButton:hover {
                    background-color: #218838;
                }
                QPushButton:pressed {
                    background-color: #1e7e34;
                }
            """,
            "danger": """
                QPushButton {
                    background-color: #dc3545;
                    color: white;
                }
                QPushButton:hover {
                    background-color: #c82333;
                }
                QPushButton:pressed {
                    background-color: #bd2130;
                }
            """,
            "secondary": """
                QPushButton {
                    background-color: #6c757d;
                    color: white;
                }
                QPushButton:hover {
                    background-color: #5a6268;
                }
                QPushButton:pressed {
                    background-color: #545b62;
                }
            """,
        }
        self.setStyleSheet(base_style + color_styles.get(self.color_theme, ""))

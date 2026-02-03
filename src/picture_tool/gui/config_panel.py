"""Config Panel Module.

Extracted from app.py to handle configuration loading, saving, and new project creation.
Maintains 100% visual consistency.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Dict, Any

from PyQt5 import QtCore
from PyQt5.QtWidgets import (
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
    QDialog,
)
from PyQt5.QtCore import pyqtSignal

import yaml
from picture_tool.gui.wizards import NewProjectWizard

class ConfigPanel(QWidget):
    """
    Manages Configuration loading, saving, path selection, and status display.
    Emits signals when config is loaded or changed.
    """
    config_loaded = pyqtSignal(dict)  # Emitted with the new config dict
    log_message = pyqtSignal(str)     # Emitted for logging

    def __init__(self, manager, parent=None):
        super().__init__(parent)
        self.manager = manager
        self._build_ui()
        
        # Initial status update
        self._update_config_status()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)

        self.config_path_edit = QLineEdit()
        self.config_path_edit.setPlaceholderText("config.yaml path...")

        browse_btn = QPushButton("📁 瀏覽")
        browse_btn.setToolTip("選擇設定檔 (yaml)")
        browse_btn.clicked.connect(self.browse_config_file)

        new_proj_btn = QPushButton("✨ 新專案")
        new_proj_btn.setToolTip("建立並初始化新專案資料夾")
        new_proj_btn.clicked.connect(self.launch_new_project_wizard)

        reload_btn = QPushButton("🔄 重載")
        reload_btn.clicked.connect(self.load_config)

        default_btn = QPushButton("↺ 重設")
        default_btn.clicked.connect(self.load_default_config)

        save_btn = QPushButton("💾 存擋")
        save_btn.clicked.connect(self.save_config)

        self.product_override_edit = QLineEdit()
        self.product_override_edit.setPlaceholderText("選填: 產品名稱 (如 Cable1)")
        self.product_override_edit.setToolTip(
            "若填寫，將自動設定 data/raw/{產品}/images 為輸入並覆寫專案名稱。"
        )

        row1 = QHBoxLayout()
        row1.addWidget(self.config_path_edit)
        row1.addWidget(browse_btn)

        row_prod = QHBoxLayout()
        row_prod.addWidget(QLabel("Product:"))
        row_prod.addWidget(self.product_override_edit)

        row2 = QHBoxLayout()
        row2.addWidget(new_proj_btn)
        row2.addWidget(reload_btn)
        row2.addWidget(default_btn)
        row2.addWidget(save_btn)

        layout.addLayout(row1)
        layout.addLayout(row_prod)
        layout.addLayout(row2)

        self.config_status_label = QLabel("尚未載入設定")
        self.config_status_label.setStyleSheet("color: #aaaaaa; font-size: 9pt;")
        layout.addWidget(self.config_status_label)

    def _update_config_status(self) -> None:
        """Update status label based on current configuration."""
        if not self.manager.config:
            self.config_status_label.setText("尚未載入設定 (無效)")
            self.config_status_label.setStyleSheet("color: #ff4d4d; font-size: 9pt;")
            return

        proj_name = self.manager.config.get("project_name", "Unknown")
        run_name = self.manager.config.get("run_name", "Unknown")
        
        # Check if path is set
        path_str = "Memory"
        if self.manager.current_config_path:
            path_str = self.manager.current_config_path.name

        self.config_status_label.setText(
            f"✅ {proj_name} / {run_name} ({path_str})"
        )
        self.config_status_label.setStyleSheet("color: #4D96FF; font-size: 9pt;")

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------

    def browse_config_file(self) -> None:
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Config", str(Path.cwd()), "YAML (*.yml *.yaml)"
        )
        if file_path:
            self.config_path_edit.setText(file_path)
            self.load_config()

    def load_config(self) -> None:
        """Load config from path and update UI"""
        path = self.config_path_edit.text().strip()
        
        try:
            self.manager.load_config(path)
            
            # Update path display with the actually loaded config path
            if self.manager.current_config_path:
                self.config_path_edit.setText(str(self.manager.current_config_path))
            
            self._update_config_status()
            
            # Notify listeners
            self.config_loaded.emit(self.manager.config)
            self.log_message.emit(f"[INFO] Config loaded from {path or 'Default'}")
            
        except Exception as e:
            self.log_message.emit(f"[ERROR] Failed to load config: {e}")
            self.config_status_label.setText(f"載入失敗: {e}")
            self.config_status_label.setStyleSheet("color: #ff4d4d;")

    def load_default_config(self) -> None:
        """Load default config and update UI"""
        try:
            self.manager.load_config()  # Load default
            if self.manager.current_config_path:
                self.config_path_edit.setText(str(self.manager.current_config_path))
            
            self._update_config_status()
            self.config_loaded.emit(self.manager.config)
            self.log_message.emit("[INFO] Loaded default config")
            
        except Exception as e:
            self.log_message.emit(f"[ERROR] Failed to load default config: {e}")

    def save_config(self) -> None:
        """Save current config to file"""
        path = self.config_path_edit.text().strip()
        success = self.manager.save_config(path)
        if success:
            self._update_config_status()
            self.log_message.emit(f"[INFO] Config saved to {path}")
        else:
            self.log_message.emit("[ERROR] Failed to save config")

    def launch_new_project_wizard(self) -> None:
        """Launch the wizard to create a new project."""
        dlg = NewProjectWizard(self)
        if dlg.exec_() == QDialog.Accepted:
            if hasattr(dlg, "created_path") and dlg.created_path:
                self.config_path_edit.setText(str(dlg.created_path).replace("\\", "/"))
                self.load_config()
                QMessageBox.information(
                    self,
                    "Loaded",
                    f"Loaded new project config from:\n{dlg.created_path}",
                )

    def get_product_override(self) -> Optional[str]:
        text = self.product_override_edit.text().strip()
        return text if text else None

    def get_config_path(self) -> Optional[str]:
        text = self.config_path_edit.text().strip()
        return text if text else None


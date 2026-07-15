"""Config Panel Module.

Extracted from app.py to handle configuration loading, saving, and new project creation.
Maintains 100% visual consistency.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from PyQt5.QtWidgets import (
    QFileDialog,
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

from picture_tool.gui.wizards import NewProjectWizard
from picture_tool.gui.readiness import build_project_readiness, format_readiness_preview
from picture_tool.path_resolver import parse_project_area_override

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
            "若填寫，將自動對齊 data/{產品}/raw 為輸入並輸出至 runs/{產品}/。"
        )
        self.product_override_edit.textChanged.connect(self._on_product_changed)

        self.path_preview_label = QLabel("")
        self.path_preview_label.setStyleSheet("color: #6BCB77; font-size: 8pt; font-family: Consolas;")
        self.path_preview_label.setWordWrap(True)

        self.readiness_preview_label = QLabel("")
        self.readiness_preview_label.setStyleSheet(
            "color: #b5b5b5; font-size: 8pt; font-family: Consolas;"
        )
        self.readiness_preview_label.setWordWrap(True)

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
        layout.addWidget(self.path_preview_label)
        layout.addWidget(self.readiness_preview_label)
        layout.addLayout(row2)

        self.config_status_label = QLabel("尚未載入設定")
        self.config_status_label.setStyleSheet("color: #aaaaaa; font-size: 9pt;")
        layout.addWidget(self.config_status_label)

    def _on_product_changed(self, text: str) -> None:
        """Update path preview when product name changes."""
        text = text.strip()
        if not text:
            self.path_preview_label.setText("")
            self.readiness_preview_label.setText("")
            return

        try:
            parsed = parse_project_area_override(text)
        except ValueError as exc:
            self.path_preview_label.setStyleSheet(
                "color: #ff6b6b; font-size: 8pt; font-family: Consolas;"
            )
            self.path_preview_label.setText(f"Invalid product input: {exc}")
            self.readiness_preview_label.setText("")
            return

        self.path_preview_label.setStyleSheet(
            "color: #6BCB77; font-size: 8pt; font-family: Consolas;"
        )
        project = parsed.project
        area = parsed.area or "(config default)"
        data_root = f"data/{project}/{parsed.area}" if parsed.area else f"data/{project}"
        runs_root = f"runs/{project}/{parsed.area}" if parsed.area else f"runs/{project}"

        preview = (
            f"Resolved: product={project}, area={area}\n"
            f"   Raw:       {data_root}/raw/\n"
            f"   Processed: {data_root}/processed/\n"
            f"   QC:        {data_root}/qc/\n"
            f"   Runs:      {runs_root}/"
        )
        self.path_preview_label.setText(preview)
        self._update_readiness_preview(text)

    def _update_readiness_preview(self, product_override: str) -> None:
        """Update dataset/model readiness preview for the selected target."""
        if not self.manager.config:
            self.readiness_preview_label.setText("Load a config to check readiness.")
            return
        try:
            readiness = build_project_readiness(self.manager.config, product_override)
        except (OSError, ValueError) as exc:
            self.readiness_preview_label.setStyleSheet(
                "color: #ff6b6b; font-size: 8pt; font-family: Consolas;"
            )
            self.readiness_preview_label.setText(f"Readiness check failed: {exc}")
            return

        has_blocking_warning = not (
            readiness.is_ready_for_yolo or readiness.is_ready_for_anomalib
        )
        color = "#ff6b6b" if has_blocking_warning else "#b5b5b5"
        self.readiness_preview_label.setStyleSheet(
            f"color: {color}; font-size: 8pt; font-family: Consolas;"
        )
        self.readiness_preview_label.setText(format_readiness_preview(readiness))

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
            product = self.get_product_override()
            if product:
                self._update_readiness_preview(product)
            
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
            product = self.get_product_override()
            if product:
                self._update_readiness_preview(product)
            
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

    def set_product_override(self, product: str, area: str | None = None) -> None:
        """Set a validated product/area received from an operator handoff."""
        value = f"{product},{area}" if area else product
        parse_project_area_override(value)
        self.product_override_edit.setText(value)

    def get_config_path(self) -> Optional[str]:
        text = self.config_path_edit.text().strip()
        return text if text else None


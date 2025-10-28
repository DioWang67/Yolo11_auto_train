"""Dialog for configuring position validation settings."""
from __future__ import annotations

from pathlib import Path

from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QGridLayout, QLabel, QLineEdit, QHBoxLayout,
    QFileDialog, QDialogButtonBox, QFrame
)

from .custom_widgets import CompactButton


class PositionSettingsDialog(QDialog):
    def __init__(self, parent, settings: dict):
        super().__init__(parent)
        self.setWindowTitle("YOLO 位置檢查設定")
        self.setMinimumWidth(500)
        self._settings = settings or {}

        layout = QVBoxLayout(self)
        form = QGridLayout()
        form.setSpacing(8)

        # 基本設定
        self.product_edit = QLineEdit(self._settings.get("product") or "")
        self.area_edit = QLineEdit(self._settings.get("area") or "")

        form.addWidget(QLabel("產品:"), 0, 0)
        form.addWidget(self.product_edit, 0, 1)
        form.addWidget(QLabel("區域:"), 1, 0)
        form.addWidget(self.area_edit, 1, 1)

        # 模型參數
        imgsz_value = self._settings.get("imgsz")
        self.imgsz_edit = QLineEdit("" if imgsz_value in (None, "") else str(imgsz_value))
        conf_value = self._settings.get("conf")
        self.conf_edit = QLineEdit("" if conf_value in (None, "") else str(conf_value))
        tol_value = self._settings.get("tolerance_override")
        self.tolerance_edit = QLineEdit("" if tol_value in (None, "") else str(tol_value))

        form.addWidget(QLabel("影像尺寸:"), 2, 0)
        form.addWidget(self.imgsz_edit, 2, 1)
        form.addWidget(QLabel("信心閾值:"), 3, 0)
        form.addWidget(self.conf_edit, 3, 1)
        form.addWidget(QLabel("容差 (%):"), 4, 0)
        form.addWidget(self.tolerance_edit, 4, 1)

        # 分隔線
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setFrameShadow(QFrame.Sunken)
        form.addWidget(line, 5, 0, 1, 2)

        # 路徑設定
        self.sample_dir_edit = QLineEdit(self._settings.get("sample_dir") or "")
        sample_btn = CompactButton("瀏覽", "secondary")
        sample_btn.clicked.connect(self._browse_sample_dir)
        sample_layout = QHBoxLayout()
        sample_layout.addWidget(self.sample_dir_edit)
        sample_layout.addWidget(sample_btn)
        form.addWidget(QLabel("樣本資料夾:"), 6, 0)
        form.addLayout(sample_layout, 6, 1)

        self.config_path_edit = QLineEdit(self._settings.get("config_path") or "")
        config_btn = CompactButton("瀏覽", "secondary")
        config_btn.clicked.connect(self._browse_config_path)
        config_layout = QHBoxLayout()
        config_layout.addWidget(self.config_path_edit)
        config_layout.addWidget(config_btn)
        form.addWidget(QLabel("設定檔:"), 7, 0)
        form.addLayout(config_layout, 7, 1)

        self.output_dir_edit = QLineEdit(self._settings.get("output_dir") or "")
        output_btn = CompactButton("瀏覽", "secondary")
        output_btn.clicked.connect(self._browse_output_dir)
        output_layout = QHBoxLayout()
        output_layout.addWidget(self.output_dir_edit)
        output_layout.addWidget(output_btn)
        form.addWidget(QLabel("輸出資料夾:"), 8, 0)
        form.addLayout(output_layout, 8, 1)

        self.weights_path_edit = QLineEdit(self._settings.get("weights") or "")
        weights_btn = CompactButton("瀏覽", "secondary")
        weights_btn.clicked.connect(self._browse_weights_path)
        weights_layout = QHBoxLayout()
        weights_layout.addWidget(self.weights_path_edit)
        weights_layout.addWidget(weights_btn)
        form.addWidget(QLabel("權重檔:"), 9, 0)
        form.addLayout(weights_layout, 9, 1)

        layout.addLayout(form)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _browse_sample_dir(self):
        directory = QFileDialog.getExistingDirectory(
            self, "選擇樣本資料夾", self.sample_dir_edit.text() or ""
        )
        if directory:
            self.sample_dir_edit.setText(directory)

    def _browse_config_path(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "選擇位置設定檔", self.config_path_edit.text() or "",
            "YAML files (*.yaml *.yml)"
        )
        if file_path:
            self.config_path_edit.setText(file_path)

    def _browse_output_dir(self):
        directory = QFileDialog.getExistingDirectory(
            self, "選擇輸出資料夾", self.output_dir_edit.text() or ""
        )
        if directory:
            self.output_dir_edit.setText(directory)

    def _browse_weights_path(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "選擇權重檔", self.weights_path_edit.text() or "",
            "Model files (*.pt *.onnx *.engine);;All files (*.*)"
        )
        if file_path:
            self.weights_path_edit.setText(file_path)

    def get_settings(self):
        return {
            "product": self.product_edit.text().strip() or None,
            "area": self.area_edit.text().strip() or None,
            "imgsz": self._parse_int(self.imgsz_edit.text()),
            "tolerance_override": self._parse_float(self.tolerance_edit.text()),
            "conf": self._parse_float(self.conf_edit.text()),
            "sample_dir": self.sample_dir_edit.text().strip() or None,
            "config_path": self.config_path_edit.text().strip() or None,
            "output_dir": self.output_dir_edit.text().strip() or None,
            "weights": self.weights_path_edit.text().strip() or None,
        }

    @staticmethod
    def _parse_int(value: str):
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _parse_float(value: str):
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

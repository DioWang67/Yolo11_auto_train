from __future__ import annotations

from typing import Dict, Any, Optional
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QFormLayout, QLineEdit, QComboBox, 
    QSpinBox, QDoubleSpinBox, QScrollArea, QTabWidget, QCheckBox,
    QLabel
)

class ConfigEditor(QWidget):
    """
    A widget to edit the application configuration dict visually.
    Updates the internal config dictionary in real-time or on demand.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__()
        self.config = config or {}
        self._inputs: Dict[str, QWidget] = {}
        self._init_ui()

    def set_config(self, config: Dict[str, Any]):
        """Load a new configuration into the editor."""
        self.config = config
        self._refresh_ui_from_config()

    def get_config(self) -> Dict[str, Any]:
        """Return the current configuration state."""
        return self.config

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        self.tabs = QTabWidget()
        layout.addWidget(self.tabs)
        
        # We will build tabs dynamically in _refresh_ui_from_config or just once here
        # For validation plan, we support: Training, Augmentation, Format
        
        self.training_tab = QWidget()
        self.aug_tab = QWidget()
        self.general_tab = QWidget()
        
        self.tabs.addTab(self.general_tab, "General / Paths")
        self.tabs.addTab(self.training_tab, "YOLO Training")
        self.tabs.addTab(self.aug_tab, "Augmentation")
        
        self._setup_general_tab()
        self._setup_training_tab()
        self._setup_augmentation_tab()

    def _setup_general_tab(self):
        """Setup General / Format Conversion settings."""
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        content = QWidget()
        layout = QFormLayout(content)
        
        # Pipeline Logs
        self._add_input(layout, "pipeline.log_file", "Log File Path", 
                        self.config.get("pipeline", {}).get("log_file", "logs/pipeline.log"))

        # Format Conversion
        layout.addRow(QLabel("<b>Format Conversion</b>"))
        fc_cfg = self.config.get("format_conversion", {})
        self._add_input(layout, "format_conversion.input_dir", "Input Dir", fc_cfg.get("input_dir", ""))
        self._add_input(layout, "format_conversion.output_dir", "Output Dir", fc_cfg.get("output_dir", ""))
        self._add_input(layout, "format_conversion.quality", "Quality (1-100)", fc_cfg.get("quality", 95), int)
        
        scroll.setWidget(content)
        
        main_layout = QVBoxLayout(self.general_tab)
        main_layout.addWidget(scroll)

    def _setup_training_tab(self):
        """Setup YOLO Training settings."""
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        content = QWidget()
        layout = QFormLayout(content)
        
        yt_cfg = self.config.get("yolo_training", {})
        
        self._add_input(layout, "yolo_training.model", "Model Path", yt_cfg.get("model", "yolo11n.pt"))
        self._add_input(layout, "yolo_training.epochs", "Epochs", yt_cfg.get("epochs", 50), int)
        self._add_input(layout, "yolo_training.batch", "Batch Size", yt_cfg.get("batch", 16), int)
        self._add_input(layout, "yolo_training.imgsz", "Image Size", yt_cfg.get("imgsz", 640), int)
        
        # Device selection
        device_combo = QComboBox()
        device_combo.addItems(["cpu", "0", "1", "auto"])
        current_device = str(yt_cfg.get("device", "cpu"))
        idx = device_combo.findText(current_device)
        if idx >= 0:
            device_combo.setCurrentIndex(idx)
        else:
            device_combo.setCurrentText(current_device)
        device_combo.currentTextChanged.connect(lambda v: self._update_config_value("yolo_training.device", v))
        layout.addRow("Device", device_combo)
        self._inputs["yolo_training.device"] = device_combo

        # Project/Name
        self._add_input(layout, "yolo_training.project", "Save Project", yt_cfg.get("project", "runs/detect"))
        self._add_input(layout, "yolo_training.name", "Run Name", yt_cfg.get("name", "train"))

        scroll.setWidget(content)
        main_layout = QVBoxLayout(self.training_tab)
        main_layout.addWidget(scroll)

    def _setup_augmentation_tab(self):
        """Setup Augmentation settings."""
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        content = QWidget()
        layout = QFormLayout(content)
        
        aug_cfg = self.config.get("yolo_augmentation", {}).get("augmentation", {})
        self._add_input(layout, "yolo_augmentation.augmentation.num_images", "Num Images (per file)", 
                        aug_cfg.get("num_images", 10), int)
        
        # TODO: Add more granular augmentation controls if needed
        
        scroll.setWidget(content)
        main_layout = QVBoxLayout(self.aug_tab)
        main_layout.addWidget(scroll)

    def _add_input(self, layout: QFormLayout, key: str, label: str, value: Any, type_hint=str):
        """Helper to create an input widget and bind it to the config."""
        if type_hint is int:
            widget = QSpinBox()
            widget.setRange(0, 999999)
            widget.setValue(int(value) if value is not None else 0)
            widget.valueChanged.connect(lambda v: self._update_config_value(key, v))
        elif type_hint is float:
            widget = QDoubleSpinBox()
            widget.setRange(0.0, 999999.0)
            widget.setValue(float(value) if value is not None else 0.0)
            widget.valueChanged.connect(lambda v: self._update_config_value(key, v))
        elif type_hint is bool:
            widget = QCheckBox()
            widget.setChecked(bool(value))
            widget.stateChanged.connect(lambda v: self._update_config_value(key, bool(v)))
        else:
            widget = QLineEdit(str(value) if value is not None else "")
            widget.textChanged.connect(lambda v: self._update_config_value(key, v))
        
        layout.addRow(label, widget)
        self._inputs[key] = widget

    def _update_config_value(self, key: str, value: Any):
        """Updates the nested config dictionary."""
        keys = key.split('.')
        target = self.config
        for k in keys[:-1]:
            target = target.setdefault(k, {})
        target[keys[-1]] = value
        
    def _refresh_ui_from_config(self):
        """Reloads UI values from self.config."""
        # This is a simplified refresh that just updates values of existing widgets
        # Re-building the whole UI might be safer if structure changes, but slower.
        for key, widget in self._inputs.items():
            val = self._get_config_value(key)
            
            # Block signals to prevent feedback loop
            widget.blockSignals(True)
            if isinstance(widget, QSpinBox):
                if val is not None:
                    widget.setValue(int(val))
            elif isinstance(widget, QDoubleSpinBox):
                 if val is not None:
                     widget.setValue(float(val))
            elif isinstance(widget, QCheckBox):
                 if val is not None:
                     widget.setChecked(bool(val))
            elif isinstance(widget, QLineEdit):
                 if val is not None:
                     widget.setText(str(val))
            elif isinstance(widget, QComboBox):
                 if val is not None: 
                     idx = widget.findText(str(val))
                     if idx >= 0:
                         widget.setCurrentIndex(idx)
                     else:
                         widget.setCurrentText(str(val))
            widget.blockSignals(False)

    def _get_config_value(self, key: str) -> Any:
        keys = key.split('.')
        val = self.config
        try:
            for k in keys:
                val = val[k]
            return val
        except (KeyError, TypeError):
            return None

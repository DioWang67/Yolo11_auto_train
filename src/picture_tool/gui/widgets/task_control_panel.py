"""
Task Control Panel Widget.

This module provides the `TaskControlPanel` widget, which manages the selection
of pipeline tasks, presets, and displays the status of each task.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import (
    QCheckBox,
    QComboBox,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)


class TaskControlPanel(QGroupBox):
    """
    Widget for managing task selection and status display.

    Signals:
        selection_changed: Emitted when task selection changes.
        preset_applied (str): Emitted when a preset is applied.
    """

    selection_changed = pyqtSignal()
    preset_applied = pyqtSignal(str)

    # Define the grid layout for tasks (row, col)
    # This structure mirrors the original app.py layout
    TASK_LAYOUT_MAP = {
        "1. Check Dependencies": (0, 0),
        "2. Format Conversion": (0, 1),
        "3. Dataset Split": (0, 2),
        "4. Generate Configs": (1, 0),
        "5. YOLO Training": (1, 1),
        "6. Model Evaluation": (1, 2),
        "7. Export ONNX": (2, 0),
        "8. Bundle Model": (2, 1),
        "9. Batch Inference": (2, 2),
    }

    def __init__(self, title: str = "Task Execution Flow", parent: Optional[QWidget] = None):
        super().__init__(title, parent)
        self._logger = logging.getLogger("picture_tool.gui.task_panel")
        self.task_checkboxes: Dict[str, QCheckBox] = {}
        self.task_status_labels: Dict[str, QLabel] = {}
        self._init_ui()

    def _init_ui(self):
        main_layout = QVBoxLayout(self)

        # 1. Preset Selection Area
        preset_layout = QHBoxLayout()
        preset_label = QLabel("Preset:")
        self.preset_combo = QComboBox()
        self.preset_combo.addItems(
            [
                "Custom",
                "Full Pipeline",
                "Data Prep Only",
                "Train Only",
                "Export/Bundle Only",
            ]
        )
        self.preset_combo.currentTextChanged.connect(self._on_preset_changed)

        btn_select_all = QPushButton("Select All")
        btn_select_all.clicked.connect(self.select_all_tasks)
        btn_clear_all = QPushButton("Clear All")
        btn_clear_all.clicked.connect(self.clear_all_tasks)

        preset_layout.addWidget(preset_label)
        preset_layout.addWidget(self.preset_combo)
        preset_layout.addStretch()
        preset_layout.addWidget(btn_select_all)
        preset_layout.addWidget(btn_clear_all)

        main_layout.addLayout(preset_layout)

        # 2. Task Grid
        grid_layout = QGridLayout()
        grid_layout.setSpacing(10)

        for task_name, (row, col) in self.TASK_LAYOUT_MAP.items():
            # Container for each task cell
            cell_widget = QWidget()
            # Add a subtle background or border could be nice, but keep it simple for now
            cell_layout = QVBoxLayout(cell_widget)
            cell_layout.setContentsMargins(4, 4, 4, 4)
            cell_layout.setSpacing(4)

            # Checkbox
            chk = QCheckBox(task_name)
            chk.setCursor(Qt.PointingHandCursor)  # type: ignore[attr-defined]
            chk.stateChanged.connect(self._on_checkbox_changed)
            self.task_checkboxes[task_name] = chk

            # Status Label
            lbl = QLabel("Idle")
            lbl.setStyleSheet("color: gray;")
            self.task_status_labels[task_name] = lbl

            cell_layout.addWidget(chk)
            cell_layout.addWidget(lbl)

            grid_layout.addWidget(cell_widget, row, col)

        main_layout.addLayout(grid_layout)

    def get_selected_tasks(self) -> List[str]:
        """Return a list of currently selected task names."""
        selected = []
        # sort by original mapping order naturally if we iterate TASK_LAYOUT_MAP keys
        # or simply iterate installed checkboxes
        # To maintain order 1..9, we sort keys
        sorted_keys = sorted(self.TASK_LAYOUT_MAP.keys())
        for name in sorted_keys:
            if self.task_checkboxes[name].isChecked():
                selected.append(name)
        return selected

    def set_task_status(self, task_name: str, status: str, color: Optional[str] = None):
        """Update the status label for a specific task."""
        # Fuzzy match because task names might differ slightly if aliases used
        # But we assume strict mapping for now based on app.py
        target_key = None
        for key in self.task_status_labels:
            if key.startswith(task_name) or task_name in key:
                 target_key = key
                 break
        
        if not target_key:
            # Fallback direct lookup
            target_key = task_name 

        if target_key in self.task_status_labels:
            lbl = self.task_status_labels[target_key]
            lbl.setText(status)
            
            # Simple color mapping matching original feel
            s_lower = status.lower()
            if "running" in s_lower:
                lbl.setStyleSheet("color: blue; font-weight: bold;")
            elif "completed" in s_lower:
                lbl.setStyleSheet("color: green;")
            elif "error" in s_lower or "fail" in s_lower:
                lbl.setStyleSheet("color: red;")
            else:
                lbl.setStyleSheet("color: gray;")

    def reset_statuses(self):
        """Reset all status labels to Idle."""
        for lbl in self.task_status_labels.values():
            lbl.setText("Idle")
            lbl.setStyleSheet("color: gray;")

    def select_all_tasks(self):
        for chk in self.task_checkboxes.values():
            chk.setChecked(True)

    def clear_all_tasks(self):
        for chk in self.task_checkboxes.values():
            chk.setChecked(False)

    def _on_checkbox_changed(self):
        self.preset_combo.blockSignals(True)
        self.preset_combo.setCurrentText("Custom")
        self.preset_combo.blockSignals(False)
        self.selection_changed.emit()

    def _on_preset_changed(self, preset_name: str):
        self.blockSignals(True)  # Prevent individual checkbox signals
        for chk in self.task_checkboxes.values():
            chk.blockSignals(True)
        
        try:
            if preset_name == "Full Pipeline":
                self.select_all_tasks()
            elif preset_name == "Data Prep Only":
                self._apply_mask(["1.", "2.", "3.", "4."])
            elif preset_name == "Train Only":
                self._apply_mask(["5.", "6."])
            elif preset_name == "Export/Bundle Only":
                self._apply_mask(["7.", "8."])
            # Custom does nothing effectively
        finally:
            for chk in self.task_checkboxes.values():
                chk.blockSignals(False)
            self.blockSignals(False)
        
        self.preset_applied.emit(preset_name)
        self.selection_changed.emit()

    def _apply_mask(self, prefixes: List[str]):
        """Helper to check boxes that start with any of the prefixes."""
        for name, chk in self.task_checkboxes.items():
            should_check = any(name.startswith(p) for p in prefixes)
            chk.setChecked(should_check)

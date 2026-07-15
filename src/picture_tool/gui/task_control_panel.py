"""Task selection panel for purpose-oriented pipeline execution."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List

import yaml
from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import (
    QCheckBox,
    QComboBox,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from picture_tool.gui.constants import (
    DEFAULT_PRESETS,
    TASK_DESCRIPTIONS,
    TASK_LABEL_TO_KEY,
    TASK_OPTIONS,
    TASK_OPTIONS_MAP,
)
from picture_tool.gui.workflows import WORKFLOW_PRESET_MAP, ordered_task_keys


class TaskControlPanel(QWidget):
    """Panel for selecting concrete pipeline tasks via workflow presets."""

    tasks_changed = pyqtSignal(list)
    log_message = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.task_checkboxes: Dict[str, QCheckBox] = {}
        self.presets: Dict[str, List[str]] = {}
        self.preset_source: Path | None = None
        self.presets = self._load_presets()
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)

        preset_row = QHBoxLayout()
        self.preset_combo = QComboBox()
        self._populate_preset_combo()

        apply_preset_btn = QPushButton("Apply")
        apply_preset_btn.setToolTip("Apply the selected workflow preset.")
        apply_preset_btn.clicked.connect(self._apply_selected_preset)

        reload_preset_btn = QPushButton("Reload")
        reload_preset_btn.setToolTip("Reload configs/gui_presets.yaml.")
        reload_preset_btn.clicked.connect(self._reload_presets)

        preset_row.addWidget(QLabel("Workflow"))
        preset_row.addWidget(self.preset_combo, 1)
        preset_row.addWidget(apply_preset_btn)
        preset_row.addWidget(reload_preset_btn)
        layout.addLayout(preset_row)

        control_row = QHBoxLayout()
        select_all_btn = QPushButton("Select all")
        select_all_btn.setToolTip("Select every available task.")
        select_all_btn.clicked.connect(self._select_all_tasks)
        clear_all_btn = QPushButton("Clear")
        clear_all_btn.setToolTip("Clear task selection.")
        clear_all_btn.clicked.connect(self._clear_all_tasks)
        control_row.addWidget(select_all_btn)
        control_row.addWidget(clear_all_btn)
        control_row.addStretch()
        layout.addLayout(control_row)

        grid = QGridLayout()
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setVerticalSpacing(6)
        grid.setHorizontalSpacing(10)
        grid.setColumnStretch(0, 1)
        grid.setColumnStretch(1, 1)

        for index, (task_key, label) in enumerate(TASK_OPTIONS):
            description = TASK_DESCRIPTIONS.get(task_key, label)
            checkbox = QCheckBox(label)
            checkbox.setToolTip(f"<b>{label}</b><br/>{description}")
            checkbox.setStatusTip(description)
            checkbox.stateChanged.connect(self._on_tasks_changed)
            self.task_checkboxes[task_key] = checkbox
            grid.addWidget(checkbox, index // 2, index % 2)

        layout.addLayout(grid)

        self.workflow_description_label = QLabel("")
        self.workflow_description_label.setStyleSheet("color: #b5b5b5; font-size: 9pt;")
        self.workflow_description_label.setWordWrap(True)
        layout.addWidget(self.workflow_description_label)

        self.task_summary_label = QLabel("")
        self.task_summary_label.setStyleSheet("color: #aaaaaa; font-size: 9pt;")
        self.task_summary_label.setWordWrap(True)
        layout.addWidget(self.task_summary_label)

        self.task_feedback_label = QLabel("")
        self.task_feedback_label.setStyleSheet("color: #4D96FF; font-size: 9pt;")
        self.task_feedback_label.setWordWrap(True)
        layout.addWidget(self.task_feedback_label)

        self.dependency_label = QLabel("")
        self.dependency_label.setStyleSheet("color: #b5b5b5; font-size: 8pt;")
        self.dependency_label.setWordWrap(True)
        layout.addWidget(self.dependency_label)

        if self.preset_combo.count() > 0:
            self._apply_selected_preset()
        else:
            self._update_task_summary()

    def get_selected_tasks(self) -> List[str]:
        """Return selected concrete task keys in stable display order."""

        return ordered_task_keys(
            key for key, checkbox in self.task_checkboxes.items() if checkbox.isChecked()
        )

    def apply_workflow(self, name: str) -> bool:
        """Select and apply a named workflow for an external OP handoff."""
        index = self.preset_combo.findText(name)
        if index < 0:
            return False
        self.preset_combo.setCurrentIndex(index)
        self._apply_selected_preset()
        return True

    def show_dependency_chain(self, ordered: List[str], auto_added: set[str]) -> None:
        """Display the resolved execution order and auto-added dependencies."""

        if not ordered:
            self.dependency_label.setText("")
            return

        parts = []
        for task_name in ordered:
            label = TASK_OPTIONS_MAP.get(task_name, task_name)
            if task_name in auto_added:
                parts.append(f"<i style='color:#cca700;'>{label} (auto)</i>")
            else:
                parts.append(label)
        self.dependency_label.setText(f"<b>Execution order:</b> {' -> '.join(parts)}")

    def _on_tasks_changed(self) -> None:
        self._update_task_summary()
        self.tasks_changed.emit(self.get_selected_tasks())

    def _select_all_tasks(self) -> None:
        self._set_selected_tasks(self.task_checkboxes.keys())
        self._show_task_feedback("Selected all tasks.")

    def _clear_all_tasks(self) -> None:
        self._set_selected_tasks([])
        self._show_task_feedback("Task selection cleared.", color="#aaaaaa")

    def _set_selected_tasks(self, task_keys: Iterable[str]) -> None:
        selected = set(task_keys)
        for key, checkbox in self.task_checkboxes.items():
            checkbox.blockSignals(True)
            checkbox.setChecked(key in selected)
            checkbox.blockSignals(False)
        self._on_tasks_changed()

    def _update_task_summary(self) -> None:
        selected = self.get_selected_tasks()
        if not selected:
            self.task_summary_label.setText("No tasks selected.")
            return
        labels = [TASK_OPTIONS_MAP.get(key, key) for key in selected]
        self.task_summary_label.setText(
            f"Selected {len(selected)} task(s): {', '.join(labels)}"
        )

    def _show_task_feedback(self, message: str, color: str = "#4D96FF") -> None:
        self.task_feedback_label.setText(message)
        self.task_feedback_label.setStyleSheet(f"color: {color}; font-size: 9pt;")

    def _populate_preset_combo(self) -> None:
        self.preset_combo.clear()
        if not self.presets:
            self.presets = DEFAULT_PRESETS.copy()
            self.preset_source = None
        for name in self.presets:
            self.preset_combo.addItem(name)

    def _reload_presets(self) -> None:
        self.presets = self._load_presets()
        self._populate_preset_combo()
        self._show_task_feedback("Workflow presets reloaded.")

    def _apply_selected_preset(self) -> None:
        name = self.preset_combo.currentText()
        normalized = self._normalize_task_list(self.presets.get(name, []), name=name)
        self._set_selected_tasks(normalized)

        preset = WORKFLOW_PRESET_MAP.get(name)
        if preset:
            self.workflow_description_label.setText(preset.description)
        else:
            self.workflow_description_label.setText("Custom workflow preset.")

        labels = [TASK_OPTIONS_MAP.get(key, key) for key in normalized]
        source = self.preset_source.name if self.preset_source else "built-in"
        if labels:
            self._show_task_feedback(
                f"Applied {name}: {', '.join(labels)} ({source})."
            )
            self.log_message.emit(f"[INFO] Applied workflow {name}: {', '.join(normalized)}")
        else:
            self._show_task_feedback(f"Workflow {name} has no valid tasks.", "#cca700")

    def _normalize_task_list(self, tasks: Iterable[str], *, name: str) -> List[str]:
        normalized: List[str] = []
        seen: set[str] = set()
        for task_name in tasks:
            key = self._normalize_task_name(str(task_name))
            if not key:
                self.log_message.emit(
                    f"[WARNING] Workflow {name} references unknown task: {task_name}"
                )
                continue
            if key in seen:
                continue
            seen.add(key)
            normalized.append(key)
        return ordered_task_keys(normalized)

    def _normalize_task_name(self, name: str) -> str | None:
        trimmed = name.strip()
        if trimmed in self.task_checkboxes:
            return trimmed
        if trimmed in TASK_LABEL_TO_KEY:
            return TASK_LABEL_TO_KEY[trimmed]
        lowered = trimmed.lower()
        for label, key in TASK_LABEL_TO_KEY.items():
            if lowered == label.lower():
                return key
        return None

    def _load_presets(self) -> Dict[str, List[str]]:
        for path in self._preset_candidate_paths():
            if not path.exists():
                continue
            try:
                data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
            except (OSError, yaml.YAMLError) as exc:
                self.log_message.emit(f"[WARNING] Failed to read presets {path}: {exc}")
                continue
            raw = data.get("presets") if isinstance(data, dict) else None
            if not isinstance(raw, dict):
                continue
            presets: Dict[str, List[str]] = {}
            for preset_name, tasks in raw.items():
                if isinstance(tasks, list):
                    presets[str(preset_name)] = [str(task) for task in tasks]
            if presets:
                self.preset_source = path
                return presets
        self.preset_source = None
        return DEFAULT_PRESETS.copy()

    def _preset_candidate_paths(self) -> List[Path]:
        return [
            Path.cwd() / "configs" / "gui_presets.yaml",
            Path(__file__).resolve().parent.parent / "resources" / "gui_presets.yaml",
        ]

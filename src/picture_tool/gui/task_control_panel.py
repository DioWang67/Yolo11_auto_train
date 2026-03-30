"""Task Control Panel Module.

Manages task selection checkboxes and preset loading.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import yaml
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
from PyQt5.QtCore import pyqtSignal

from picture_tool.gui.constants import (
    TASK_OPTIONS,
    TASK_OPTIONS_MAP,
    TASK_LABEL_TO_KEY,
    TASK_DESCRIPTIONS,
    DEFAULT_PRESETS,
)

class TaskControlPanel(QWidget):
    """
    Panel for selecting pipeline tasks and applying presets.
    Emits `tasks_changed` when selection changes.
    """
    tasks_changed = pyqtSignal(list)  # Emits list of selected task keys
    log_message = pyqtSignal(str)     # For logging warnings/info

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

        # Control Row (Select/Clear All)
        control_row = QHBoxLayout()
        select_all_btn = QPushButton("☑ 全選")
        select_all_btn.setToolTip("勾選全部任務")
        select_all_btn.clicked.connect(self._select_all_tasks)

        clear_all_btn = QPushButton("☐ 清空")
        clear_all_btn.setToolTip("清除所有勾選")
        clear_all_btn.clicked.connect(self._clear_all_tasks)

        control_row.addWidget(select_all_btn)
        control_row.addWidget(clear_all_btn)
        control_row.addStretch()
        layout.addLayout(control_row)

        # Preset Row
        preset_row = QHBoxLayout()
        self.preset_combo = QComboBox()
        self._populate_preset_combo()
        apply_preset_btn = QPushButton("✓ 套用")
        apply_preset_btn.setToolTip("依需求模式勾選任務")
        apply_preset_btn.clicked.connect(self._apply_selected_preset)
        reload_preset_btn = QPushButton("🔄")
        reload_preset_btn.setToolTip("重新讀取 configs/gui_presets.yaml")
        reload_preset_btn.clicked.connect(self._reload_presets)

        preset_row.addWidget(QLabel("預設模式："))
        preset_row.addWidget(self.preset_combo)
        preset_row.addWidget(apply_preset_btn)
        preset_row.addWidget(reload_preset_btn)
        preset_row.addStretch()
        layout.addLayout(preset_row)

        # Task Grid (2 columns, equal stretch)
        grid = QGridLayout()
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setVerticalSpacing(6)
        grid.setHorizontalSpacing(10)
        grid.setColumnStretch(0, 1)
        grid.setColumnStretch(1, 1)

        for index, (task_key, label) in enumerate(TASK_OPTIONS):
            desc = TASK_DESCRIPTIONS.get(task_key, label)
            checkbox = QCheckBox(label)
            checkbox.setToolTip(f"<b>{label}</b><br/>{desc}")
            # Default state logic (moved from main)
            checkbox.setChecked(task_key in {"dataset_splitter", "yolo_train"})
            checkbox.setStatusTip(desc)
            checkbox.stateChanged.connect(self._on_tasks_changed)

            self.task_checkboxes[task_key] = checkbox
            # 2 columns layout
            grid.addWidget(checkbox, index // 2, index % 2)

        layout.addLayout(grid)

        # Feedback Labels
        self.task_summary_label = QLabel("尚未選擇任務")
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

        self._update_task_summary()

    def get_selected_tasks(self) -> List[str]:
        """Return list of selected task keys."""
        return [
            key for key, cb in self.task_checkboxes.items()
            if cb.isChecked()
        ]

    # ------------------------------------------------------------------
    # Internal Logic
    # ------------------------------------------------------------------

    def _on_tasks_changed(self) -> None:
        self._update_task_summary()
        self.tasks_changed.emit(self.get_selected_tasks())

    def _select_all_tasks(self) -> None:
        for checkbox in self.task_checkboxes.values():
            checkbox.blockSignals(True)
            checkbox.setChecked(True)
            checkbox.blockSignals(False)
        self._on_tasks_changed()
        self._show_task_feedback("已全選所有任務。")

    def _clear_all_tasks(self) -> None:
        for checkbox in self.task_checkboxes.values():
            checkbox.blockSignals(True)
            checkbox.setChecked(False)
            checkbox.blockSignals(False)
        self._on_tasks_changed()
        self._show_task_feedback("已清除所有勾選。", color="#aaaaaa")

    def _update_task_summary(self) -> None:
        selected = self.get_selected_tasks()
        count = len(selected)
        labels = [TASK_OPTIONS_MAP.get(k, k) for k in selected]
        summary_text = (
            "尚未選擇任務"
            if not labels
            else f"將執行 {count} 項：{', '.join(labels)}"
        )
        self.task_summary_label.setText(summary_text)

    def _show_task_feedback(self, message: str, color: str = "#4D96FF") -> None:
        self.task_feedback_label.setText(message)
        self.task_feedback_label.setStyleSheet(f"color: {color}; font-size: 9pt;")

    def show_dependency_chain(
        self, ordered: List[str], auto_added: "set[str]"
    ) -> None:
        """Display the resolved execution order with auto-added deps highlighted."""
        if not ordered:
            self.dependency_label.setText("")
            return
        parts = []
        for name in ordered:
            label = TASK_OPTIONS_MAP.get(name, name)
            if name in auto_added:
                parts.append(f"<i style='color:#cca700;'>{label} (自動加入)</i>")
            else:
                parts.append(label)
        self.dependency_label.setText(
            f"<b>執行順序：</b>{'  →  '.join(parts)}"
        )

    # ------------------------------------------------------------------
    # Preset Logic
    # ------------------------------------------------------------------
    def _populate_preset_combo(self) -> None:
        self.preset_combo.clear()
        if not self.presets:
            self.presets = DEFAULT_PRESETS.copy()
            self.preset_source = None
        for name in self.presets.keys():
            self.preset_combo.addItem(name)

    def _reload_presets(self) -> None:
        self.presets = self._load_presets()
        self._populate_preset_combo()
        self._show_task_feedback("已重新載入預設設定檔。", color="#4D96FF")

    def _apply_selected_preset(self) -> None:
        name = self.preset_combo.currentText()
        tasks = self.presets.get(name, [])
        normalized: List[str] = []
        for t in tasks:
            key = self._normalize_task_name(t)
            if key:
                normalized.append(key)
            else:
                self.log_message.emit(f"[WARNING] 預設「{name}」包含未知任務：{t}")
        
        for key, checkbox in self.task_checkboxes.items():
            checkbox.blockSignals(True)
            checkbox.setChecked(key in normalized)
            checkbox.blockSignals(False)

        self._on_tasks_changed()

        preset_labels = [TASK_OPTIONS_MAP.get(k, k) for k in normalized]
        src = (
            f"(來源: {self.preset_source.name})"
            if self.preset_source
            else "(來源: 內建)"
        )
        if preset_labels:
            self._show_task_feedback(
                f"已套用預設「{name}」：{', '.join(preset_labels)} {src}"
            )
            self.log_message.emit(f"[INFO] 套用預設 {name}: {', '.join(normalized)} {src}")
        else:
            self._show_task_feedback(f"預設「{name}」沒有有效任務", color="#cca700")

    def _normalize_task_name(self, name: str) -> str | None:
        trimmed = name.strip()
        if trimmed in self.task_checkboxes:
            return trimmed
        if trimmed in TASK_LABEL_TO_KEY:
            return TASK_LABEL_TO_KEY[trimmed]
        for label, key in TASK_LABEL_TO_KEY.items():
            if trimmed.lower() == label.lower():
                return key
        return None

    def _load_presets(self) -> Dict[str, List[str]]:
        for path in self._preset_candidate_paths():
            if not path.exists():
                continue
            try:
                data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
            except (OSError, ValueError) as exc:
                self.log_message.emit(f"[WARNING] 無法讀取預設檔 {path}: {exc}")
                continue
            raw = data.get("presets") if isinstance(data, dict) else None
            if not isinstance(raw, dict):
                continue
            presets: Dict[str, List[str]] = {}
            for name, tasks in raw.items():
                if not isinstance(tasks, list):
                    continue
                presets[name] = [str(t) for t in tasks]
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

"""Entry point for the Picture Tool desktop GUI."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QApplication,
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QPushButton,
    QProgressBar,
    QSplitter,
    QTextEdit,
    QVBoxLayout,
    QWidget,
    QCheckBox,
    QLineEdit,
)

from picture_tool.gui.pipeline_controller import PipelineControllerMixin


TASK_OPTIONS: List[tuple[str, str]] = [
    ("format_conversion", "Format Conversion"),
    ("anomaly_detection", "Anomaly Detection"),
    ("yolo_augmentation", "YOLO Augmentation"),
    ("image_augmentation", "Image Augmentation"),
    ("dataset_splitter", "Dataset Splitter"),
    ("yolo_train", "YOLO Training"),
    ("yolo_evaluation", "YOLO Evaluation"),
    ("generate_report", "Generate Report"),
    ("dataset_lint", "Dataset Lint"),
    ("aug_preview", "Augmentation Preview"),
    ("batch_inference", "Batch Inference"),
]


class PictureToolGUI(QMainWindow, PipelineControllerMixin):
    """Simple PyQt5 front-end around the pipeline controller mixin."""

    def __init__(self) -> None:
        super().__init__()
        self._init_pipeline_controller()
        self._log_history: List[str] = []
        self.task_checkboxes: Dict[str, QCheckBox] = {}
        self.task_status_items: Dict[str, QListWidgetItem] = {}

        self.setWindowTitle("Picture Tool – Pipeline Orchestrator")
        self.resize(1080, 720)

        self._build_ui()
        self.load_default_config()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------
    def _build_ui(self) -> None:
        central = QWidget()
        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(12, 12, 12, 12)
        main_layout.setSpacing(12)

        main_layout.addWidget(self._build_config_group())
        main_layout.addWidget(self._build_task_group())
        main_layout.addWidget(self._build_status_group())
        main_layout.addWidget(self._build_log_panel())

        self.setCentralWidget(central)

    def _build_config_group(self) -> QGroupBox:
        group = QGroupBox("Pipeline Configuration")
        layout = QHBoxLayout(group)
        layout.setSpacing(8)

        self.config_path_edit = QLineEdit()
        self.config_path_edit.setPlaceholderText("Path to pipeline configuration")

        browse_btn = QPushButton("Browse…")
        browse_btn.clicked.connect(self.browse_config_file)

        reload_btn = QPushButton("Load")
        reload_btn.clicked.connect(self.load_config)

        default_btn = QPushButton("Use Default")
        default_btn.clicked.connect(self.load_default_config)

        layout.addWidget(self.config_path_edit)
        layout.addWidget(browse_btn)
        layout.addWidget(reload_btn)
        layout.addWidget(default_btn)

        return group

    def _build_task_group(self) -> QGroupBox:
        group = QGroupBox("Tasks")
        layout = QVBoxLayout(group)
        layout.setSpacing(8)

        grid = QGridLayout()
        grid.setSpacing(6)

        for index, (task_key, label) in enumerate(TASK_OPTIONS):
            checkbox = QCheckBox(label)
            checkbox.setChecked(task_key in {"dataset_splitter", "yolo_train"})
            self.task_checkboxes[task_key] = checkbox
            row = index // 3
            col = index % 3
            grid.addWidget(checkbox, row, col)

        layout.addLayout(grid)

        self.status_list = QListWidget()
        self.status_list.setMaximumHeight(150)
        layout.addWidget(self.status_list)

        self._rebuild_status_items()
        return group

    def _build_status_group(self) -> QGroupBox:
        group = QGroupBox("Execution Controls")
        layout = QHBoxLayout(group)
        layout.setSpacing(12)

        self.start_btn = QPushButton("Start")
        self.start_btn.clicked.connect(self.start_pipeline)

        self.stop_btn = QPushButton("Stop")
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self.stop_pipeline)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)

        self.status_label = QLabel("Idle")

        layout.addWidget(self.start_btn)
        layout.addWidget(self.stop_btn)
        layout.addWidget(self.progress_bar, 1)
        layout.addWidget(self.status_label)

        return group

    def _build_log_panel(self) -> QWidget:
        splitter = QSplitter(Qt.Horizontal)

        self.config_text = QTextEdit()
        self.config_text.setReadOnly(True)
        self.config_text.setPlaceholderText("Loaded configuration will appear here.")

        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setPlaceholderText("Pipeline logs will appear here.")

        splitter.addWidget(self._wrap_with_label("Configuration Preview", self.config_text))
        splitter.addWidget(self._wrap_with_label("Logs", self.log_text))
        splitter.setSizes([400, 600])

        return splitter

    @staticmethod
    def _wrap_with_label(title: str, widget: QWidget) -> QWidget:
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        label = QLabel(title)
        label.setStyleSheet("font-weight: 600;")
        layout.addWidget(label)
        layout.addWidget(widget)
        return container

    # ------------------------------------------------------------------
    # Mix-in hooks
    # ------------------------------------------------------------------
    def log_message(self, message: str) -> None:
        self._log_history.append(message)
        self.log_text.append(message)
        self.log_text.verticalScrollBar().setValue(
            self.log_text.verticalScrollBar().maximum()
        )

    def reset_task_statuses(self, tasks):
        self._rebuild_status_items(default_state="pending", only=tasks)

    def _set_task_status(self, task: str, message: str, color=None) -> None:
        item = self.task_status_items.get(task)
        if item is not None:
            item.setText(f"{task} – {message}")

    def _validate_pipeline_configuration(self, tasks):
        missing_sections: List[str] = []
        pipeline_section = self.config.get("pipeline")
        if not isinstance(pipeline_section, dict):
            missing_sections.append("pipeline")
        for task in tasks:
            if task not in self.config:
                continue
            section = self.config.get(task)
            if not isinstance(section, dict):
                missing_sections.append(task)
        return [f"Invalid config section: {name}" for name in missing_sections]

    # ------------------------------------------------------------------
    # UI helpers
    # ------------------------------------------------------------------
    def _rebuild_status_items(self, default_state: str = "pending", only=None) -> None:
        if not hasattr(self, "status_list"):
            return
        self.status_list.clear()
        for task in self.task_checkboxes.keys():
            item = QListWidgetItem(f"{task} – {default_state}")
            self.task_status_items[task] = item
            self.status_list.addItem(item)

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------
    def browse_config_file(self) -> None:
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select configuration file", str(Path.cwd()), "YAML files (*.yml *.yaml)"
        )
        if file_path:
            self.config_path_edit.setText(file_path)
            self.load_config()

    # ------------------------------------------------------------------
    # Qt overrides
    # ------------------------------------------------------------------
    def closeEvent(self, event) -> None:  # pragma: no cover - UI integration
        if self.worker_thread and self.worker_thread.isRunning():
            self.stop_pipeline()
            self.worker_thread.wait(2000)
        super().closeEvent(event)


def main() -> None:
    app = QApplication(sys.argv)
    window = PictureToolGUI()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":  # pragma: no cover - manual launch
    main()

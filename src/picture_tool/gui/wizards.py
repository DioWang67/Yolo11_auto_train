from __future__ import annotations
from typing import Optional

from pathlib import Path
import yaml
from PyQt5.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QFormLayout,
    QLineEdit,
    QPushButton,
    QFileDialog,
    QMessageBox,
    QHBoxLayout,
)


class NewProjectWizard(QDialog):
    """Wizard to initialize a new Picture Tool project structure."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Create New Project")
        self.resize(500, 200)
        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout(self)

        form_layout = QFormLayout()

        self.name_edit = QLineEdit()
        self.name_edit.setPlaceholderText("MyNewProject")

        self.location_edit = QLineEdit()
        self.location_edit.setPlaceholderText("D:/Projects")
        self.location_edit.setText(str(Path.cwd().parent))  # Default suggestion

        browse_btn = QPushButton("...")
        browse_btn.setFixedWidth(30)
        browse_btn.clicked.connect(self._browse_location)

        loc_row = QHBoxLayout()
        loc_row.addWidget(self.location_edit)
        loc_row.addWidget(browse_btn)

        form_layout.addRow("Project Name:", self.name_edit)
        form_layout.addRow("Parent Directory:", loc_row)

        self.classes_edit = QLineEdit()
        self.classes_edit.setPlaceholderText("dog, cat, person")
        self.classes_edit.setText("object")  # Default value
        form_layout.addRow("Class Names:", self.classes_edit)

        layout.addLayout(form_layout)

        layout.addStretch()

        btn_layout = QHBoxLayout()
        create_btn = QPushButton("Create Project")
        create_btn.setObjectName("PrimaryBtn")  # Re-use app style if possible
        create_btn.clicked.connect(self.create_project)

        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)

        btn_layout.addStretch()
        btn_layout.addWidget(cancel_btn)
        btn_layout.addWidget(create_btn)

        layout.addLayout(btn_layout)

    def _browse_location(self):
        dir_path = QFileDialog.getExistingDirectory(self, "Select Parent Directory")
        if dir_path:
            self.location_edit.setText(dir_path)

    def create_project(self):
        name = self.name_edit.text().strip()
        location = self.location_edit.text().strip()

        if not name or not location:
            QMessageBox.warning(
                self, "Missing Info", "Please provide both project name and location."
            )
            return

        project_dir = Path(location) / name

        if project_dir.exists():
            QMessageBox.warning(
                self, "Exists", f"Directory {project_dir} already exists."
            )
            return

        try:
            self._create_structure(project_dir)
            QMessageBox.information(
                self,
                "Success",
                f"Project created at:\n{project_dir}\n\nA config.yaml has been generated.",
            )
            self.created_path = project_dir / "config.yaml"
            self.accept()
        except (OSError, ValueError, yaml.YAMLError) as e:
            QMessageBox.critical(self, "Error", f"Failed to create project: {e}")

    def _create_structure(self, root: Path):
        root.mkdir(parents=True, exist_ok=True)
        project_name = root.name

        # Standard Data Structure: data/<project>/...
        data_root = root / "data" / project_name
        (data_root / "raw" / "images").mkdir(parents=True)
        (data_root / "raw" / "labels").mkdir(parents=True)
        (data_root / "processed").mkdir(parents=True)
        (data_root / "split").mkdir(parents=True)
        (data_root / "qc" / "color_samples").mkdir(parents=True)
        (data_root / "qc" / "position_samples").mkdir(parents=True)
        
        # Standard Runs Structure: runs/<project>/...
        runs_root = root / "runs" / project_name
        (runs_root / "train").mkdir(parents=True)
        (runs_root / "infer").mkdir(parents=True)
        (runs_root / "quality" / "color").mkdir(parents=True)
        (runs_root / "quality" / "position").mkdir(parents=True)
        (runs_root / "reports").mkdir(parents=True)
        (runs_root / "logs").mkdir(parents=True)

        (root / "models").mkdir(parents=True)

        raw_classes = self.classes_edit.text().strip()
        class_list = [c.strip() for c in raw_classes.split(",") if c.strip()]
        if not class_list:
            class_list = ["object"]

        config_data = self._generate_default_config(root, class_names=class_list)

        with (root / "config.yaml").open("w", encoding="utf-8") as f:
            yaml.safe_dump(config_data, f, sort_keys=False, allow_unicode=True)

    def _generate_default_config(
        self, root: Path, class_names: Optional[list[str]] = None
    ) -> dict:
        """Generates a config dict with absolute project-centric paths for the new project."""
        if class_names is None:
            class_names = ["object"]
        
        project_name = root.name
        # Use absolute paths for the generated config to ensure stability
        data_p = (root / "data" / project_name).resolve().as_posix()
        runs_p = (root / "runs" / project_name).resolve().as_posix()
        
        return {
            "project_name": project_name,
            "run_name": "train",
            "pipeline": {
                "log_file": f"{runs_p}/logs/pipeline.log",
                "tasks": [
                    {"name": "dataset_splitter", "enabled": True, "dependencies": []},
                    {"name": "yolo_train", "enabled": True, "dependencies": ["dataset_splitter"]},
                    {"name": "yolo_evaluation", "enabled": True, "dependencies": ["yolo_train"]},
                    {"name": "generate_report", "enabled": True, "dependencies": []},
                ],
            },
            "format_conversion": {
                "input_dir": f"{data_p}/raw/images",
                "output_dir": f"{data_p}/processed/images_converted",
                "quality": 95,
            },
            "train_test_split": {
                "input": {
                    "image_dir": f"{data_p}/processed/images",
                    "label_dir": f"{data_p}/processed/labels",
                },
                "output": {"output_dir": f"{data_p}/split"},
                "split_ratios": {"train": 0.7, "val": 0.2, "test": 0.1},
                "input_formats": [".jpg", ".jpeg", ".png", ".bmp"],
                "label_format": ".txt",
            },
            "dataset_lint": {
                "image_dir": f"{data_p}/processed/images",
                "label_dir": f"{data_p}/processed/labels",
                "output_dir": f"{runs_p}/quality/lint",
            },
            "aug_preview": {
                "image_dir": f"{data_p}/processed/images",
                "label_dir": f"{data_p}/processed/labels",
                "output_dir": f"{runs_p}/quality/preview",
                "num_samples": 16,
                "cols": 4,
            },
            "color_inspection": {
                "enabled": True,
                "input_dir": f"{data_p}/qc/color_samples",
                "output_json": f"{runs_p}/quality/color/stats.json",
            },
            "color_verification": {
                "enabled": True,
                "input_dir": f"{data_p}/qc/color_samples",
                "color_stats": f"{runs_p}/quality/color/stats.json",
                "output_json": f"{runs_p}/quality/color/verification.json",
                "output_csv": f"{runs_p}/quality/color/verification.csv",
                "debug_dir": f"{runs_p}/quality/color/debug",
            },
            "yolo_training": {
                "dataset_dir": f"{data_p}/split",
                "model": "models/yolo11n.pt",
                "epochs": 50,
                "project": runs_p,
                "name": "train",
                "class_names": class_names,
                "position_validation": {
                    "enabled": True,
                    "auto_generate": True,
                    "output_dir": f"{runs_p}/quality/position",
                },
                "export_onnx": {"enabled": True},
                "export_detection_config": {"enabled": True},
                "artifact_bundle": {"enabled": True},
            },
            "yolo_augmentation": {
                "input": {
                    "image_dir": f"{data_p}/raw/images",
                    "label_dir": f"{data_p}/raw/labels",
                },
                "output": {
                    "image_dir": f"{data_p}/processed/images",
                    "label_dir": f"{data_p}/processed/labels",
                },
                "processing": {"num_workers": 2},
                "augmentation": {
                    "num_images": 20,
                    "num_operations": [2, 3],
                    "operations": {
                        "blur": {"kernel": [0, 1]},
                        "contrast": {"range": [0.9, 1.1]},
                        "flip": {"probability": 0.5},
                    },
                },
            },
            "yolo_evaluation": {
                "weights": None,
                "imgsz": 640,
                "device": "cpu",
                "conf": 0.25,
            },
            "batch_inference": {
                "input_dir": f"{data_p}/split/test/images",
                "output_dir": f"{runs_p}/infer",
                "weights": None,
                "imgsz": 640,
                "conf": 0.25,
            },
            "report": {
                "output_dir": f"{runs_p}/reports",
                "include_artifacts": True,
            },
        }

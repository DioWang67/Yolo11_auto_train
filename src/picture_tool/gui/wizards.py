from __future__ import annotations
from typing import Optional

from pathlib import Path
import yaml
from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QFormLayout, QLineEdit, QPushButton, 
    QFileDialog, QMessageBox, QHBoxLayout
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
        self.location_edit.setText(str(Path.cwd().parent)) # Default suggestion
        
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
        self.classes_edit.setText("object") # Default value
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
            QMessageBox.warning(self, "Missing Info", "Please provide both project name and location.")
            return
            
        project_dir = Path(location) / name
        
        if project_dir.exists():
             QMessageBox.warning(self, "Exists", f"Directory {project_dir} already exists.")
             return
             
        try:
            self._create_structure(project_dir)
            QMessageBox.information(self, "Success", f"Project created at:\n{project_dir}\n\nA config.yaml has been generated.")
            self.created_path = project_dir / "config.yaml"
            self.accept()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to create project: {e}")

    def _create_structure(self, root: Path):
        root.mkdir(parents=True, exist_ok=True)
        
        (root / "data" / "raw" / "images").mkdir(parents=True)
        (root / "data" / "raw" / "labels").mkdir(parents=True)
        (root / "data" / "split").mkdir(parents=True)
        (root / "models").mkdir(parents=True)
        (root / "reports").mkdir(parents=True)
        (root / "logs").mkdir(parents=True)
        
        raw_classes = self.classes_edit.text().strip()
        class_list = [c.strip() for c in raw_classes.split(",") if c.strip()]
        if not class_list:
            class_list = ["object"]

        config_data = self._generate_default_config(root, class_names=class_list)
        
        with (root / "config.yaml").open("w", encoding="utf-8") as f:
            yaml.safe_dump(config_data, f, sort_keys=False, allow_unicode=True)

    def _generate_default_config(self, root: Path, class_names: Optional[list[str]] = None) -> dict:
        """Generates a config dict with absolute paths for the new project."""
        if class_names is None:
            class_names = ["object"]
        # Note: We use .as_posix() to ensure forward slashes, which are safer in YAML/Python on Windows
        return {
            "pipeline": {
                "log_file": (root / "logs" / "pipeline.log").as_posix(),
                "tasks": [
                    {"name": "dataset_splitter", "enabled": True, "dependencies": []},
                    {"name": "yolo_train", "enabled": True, "dependencies": ["dataset_splitter"]},
                    {"name": "yolo_evaluation", "enabled": True, "dependencies": ["yolo_train"]},
                    {"name": "generate_report", "enabled": True, "dependencies": []}
                ]
            },
            "format_conversion": {
                "input_dir": (root / "data" / "raw" / "images").as_posix(),
                "output_dir": (root / "data" / "raw" / "images_converted").as_posix(),
                "quality": 95
            },
            "train_test_split": {
                "input": {
                    "image_dir": (root / "data" / "augmented" / "images").as_posix(),
                    "label_dir": (root / "data" / "augmented" / "labels").as_posix()
                },
                "output": {
                    "output_dir": (root / "data" / "split").as_posix()
                },
                "split_ratios": {"train": 0.7, "val": 0.2, "test": 0.1},
                "input_formats": [".jpg", ".jpeg", ".png", ".bmp"],
                "label_format": ".txt"
            },
            "dataset_lint": {
                "input_dir": (root / "data" / "raw" / "images").as_posix(),
                "label_dir": (root / "data" / "raw" / "labels").as_posix(),
                "extensions": [".jpg", ".jpeg", ".png", ".bmp"],
                "fix": True
            },
            "color_inspection": {
                "enabled": True,
                "input_dir": (root / "data" / "raw" / "images").as_posix(),
                "output_dir": (root / "reports" / "color_inspection").as_posix()
            },
            "color_verification": {
                "enabled": False, # Disabled by default as it requires reference image
                "input_dir": (root / "data" / "raw" / "images").as_posix(),
                "reference_image": None,
                "tolerance": 10,
                "method": "DeltaE"
            },
            "aug_preview": {
                "input_dir": (root / "data" / "raw" / "images").as_posix(),
                "output_dir": (root / "data" / "augmented" / "preview").as_posix(),
                "number": 5
            },
            "yolo_evaluation": {
                 "dataset_dir": (root / "data" / "split").as_posix(),
                 "imgsz": 640,
                 "batch": 16,
                 "device": "cpu",
                 "conf": 0.001,
                 "iou": 0.6
            },
            "yolo_training": {
                "dataset_dir": (root / "data" / "split").as_posix(),
                "model": "yolo11n.pt",
                "epochs": 50,
                "project": (root / "runs" / "detect").as_posix(),
                "name": "train",
                "class_names": class_names,
                "position_validation": {
                    "enabled": True,
                    "auto_generate": True,
                    "product": "ProductA",
                    "area": "Area1",
                    "device": "cpu",
                    "conf": 0.25
                },
                "export_onnx": {
                    "enabled": True,
                    "simplify": True,
                    "opset": 12,
                    "dynamic": False
                },
                "export_detection_config": {
                    "enabled": True,
                    "output_path": None, # default: runs/detect/{name}/detection_config.yaml
                    "conf_thres": 0.25,
                    "iou_thres": 0.45
                },
                "artifact_bundle": {
                    "enabled": True,
                    "dir_name": "bundle",
                    "include_weights": True,
                    "include_detection_config": True,
                    "include_onnx": True
                }
            },
            "yolo_augmentation": {
                "input": {
                    "image_dir": (root / "data" / "raw" / "images").as_posix(),
                    "label_dir": (root / "data" / "raw" / "labels").as_posix()
                },
                "output": {
                    "image_dir": (root / "data" / "augmented" / "images").as_posix(),
                    "label_dir": (root / "data" / "augmented" / "labels").as_posix()
                },
                "processing": {"num_workers": 2},
                "augmentation": {
                    "num_images": 20,
                    "num_operations": [2, 3],
                    "operations": {
                        "blur": {"kernel": [0, 1]},
                        "contrast": {"range": [0.9, 1.1]},
                        "flip": {"probability": 0.5},
                        "rotate": {"angle": [-10, 10]}
                    }
                }
            },
            "image_augmentation": {
                "input": {"image_dir": (root / "data" / "raw" / "images").as_posix()},
                "output": {"image_dir": (root / "data" / "augmented" / "only_images").as_posix()},
                "processing": {"num_workers": 2},
                "augmentation": {
                    "num_images": 20,
                    "num_operations": [1, 2],
                    "operations": {"blur": {"kernel": [1, 3]}}
                }
            },
            "batch_inference": {
                "input_dir": (root / "data" / "split" / "test" / "images").as_posix(),
                "output_dir": (root / "reports" / "inference").as_posix(),
                "weights": None,
                "imgsz": 640,
                "conf": 0.25
            },
            "report": {
                 "output_dir": (root / "reports").as_posix(),
                 "include_artifacts": True
            }
        }

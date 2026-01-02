
from pathlib import Path
from PyQt5.QtWidgets import QApplication
import sys

# Ensure QApplication exists
app = QApplication.instance() or QApplication(sys.argv)

from picture_tool.gui.wizards import NewProjectWizard

wizard = NewProjectWizard()
config = wizard._generate_default_config(Path("."))

print("\n--- Top Level Keys ---")
for k in config.keys():
    print(f"- {k}")

print("\n--- YOLO Training Keys ---")
if "yolo_training" in config:
    for k in config["yolo_training"].keys():
        print(f"- {k}")

print("\n--- CHECKING MISSING ---")
missing = []
if "dataset_lint" not in config: missing.append("dataset_lint")
if "color_inspection" not in config: missing.append("color_inspection")
if "aug_preview" not in config: missing.append("aug_preview")
if "yolo_evaluation" not in config: missing.append("yolo_evaluation")

yt = config.get("yolo_training", {})
if "position_validation" not in yt: missing.append("yolo_training.position_validation")
if "export_onnx" not in yt: missing.append("yolo_training.export_onnx")

if missing:
    print(f"STILL MISSING: {missing}")
else:
    print("ALL KEYS PRESENT.")

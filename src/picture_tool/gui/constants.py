"""Shared GUI labels, task descriptions, and fallback presets."""

from __future__ import annotations

from typing import Dict, List, Tuple

from picture_tool.gui.workflows import WORKFLOW_PRESETS


TASK_OPTIONS: List[Tuple[str, str]] = [
    ("format_conversion", "Format conversion"),
    ("dataset_lint", "Dataset lint"),
    ("yolo_augmentation", "YOLO augmentation"),
    ("image_augmentation", "Image augmentation"),
    ("aug_preview", "Augmentation preview"),
    ("dataset_splitter", "Dataset split"),
    ("yolo_train", "YOLO training"),
    ("yolo_evaluation", "YOLO evaluation"),
    ("position_validation", "Position validation"),
    ("artifact_bundle", "YOLO package zip"),
    ("deploy", "Deploy to inference"),
    ("anomalib_train", "Anomalib training"),
    ("anomalib_package", "Anomalib package zip"),
    ("batch_inference", "Batch inference"),
    ("anomaly_detection", "Anomaly mask generation"),
    ("color_inspection", "Color inspection"),
    ("color_verification", "Color verification"),
    ("qc_summary", "QC summary"),
    ("generate_report", "Generate report"),
]

TASK_OPTIONS_MAP: Dict[str, str] = {key: label for key, label in TASK_OPTIONS}
TASK_LABEL_TO_KEY: Dict[str, str] = {label: key for key, label in TASK_OPTIONS}

TASK_DESCRIPTIONS: Dict[str, str] = {
    "format_conversion": "Convert input images into the configured format.",
    "dataset_lint": "Check dataset folders, labels, and common data issues.",
    "yolo_augmentation": "Create YOLO image/label augmentation outputs.",
    "image_augmentation": "Create image-only augmentation outputs.",
    "aug_preview": "Render a small preview of augmentation results.",
    "dataset_splitter": "Create train/val/test folders for YOLO training.",
    "yolo_train": "Train YOLO and generate runtime config exports.",
    "yolo_evaluation": "Evaluate the trained YOLO run.",
    "position_validation": "Validate expected component positions using trained weights.",
    "artifact_bundle": "Create a zip that can be extracted under yolo11_inference/models.",
    "deploy": "Copy YOLO artifacts directly into yolo11_inference/models.",
    "anomalib_train": "Train an Anomalib model from normal/abnormal image folders.",
    "anomalib_package": "Package an Anomalib checkpoint for yolo11_inference.",
    "batch_inference": "Run inference on a folder of images.",
    "anomaly_detection": "Generate anomaly masks from images.",
    "color_inspection": "Build color statistics or inspection artifacts.",
    "color_verification": "Verify color constraints against configured stats.",
    "qc_summary": "Summarize inference and quality-control results.",
    "generate_report": "Generate a pipeline report.",
}

ANOMALIB_MODEL_DESCRIPTIONS: Dict[str, str] = {
    "efficientad": "Recommended first choice for industrial anomaly detection; fast and practical. Uses batch size 1 in Anomalib 1.2.",
    "padim": "Stable lightweight baseline; useful when data is limited or you want a simple comparison.",
    "patchcore": "Strong common baseline; can use more memory and is often slower on large datasets.",
}

DEFAULT_PRESETS: Dict[str, List[str]] = {
    preset.name: list(preset.tasks) for preset in WORKFLOW_PRESETS
}

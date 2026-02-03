"""Shared Constants for Picture Tool GUI.

Defines styles, task options, and default presets.
"""
from typing import Dict, List, Tuple

# ------------------------------------------------------------------
# 任務定義
# ------------------------------------------------------------------
TASK_OPTIONS: List[Tuple[str, str]] = [
    ("format_conversion", "Format Conversion"),
    ("anomaly_detection", "Anomaly Detection"),
    ("yolo_augmentation", "YOLO Augmentation"),
    ("image_augmentation", "Image Augmentation"),
    ("dataset_splitter", "Dataset Splitter"),
    ("yolo_train", "YOLO Training"),
    ("yolo_evaluation", "YOLO Evaluation"),
    ("position_validation", "Position Validation"),
    ("color_inspection", "顏色範本蒐集 (SAM)"),
    ("color_verification", "顏色批次驗證"),
    ("generate_report", "Generate Report"),
    ("dataset_lint", "Dataset Lint"),
    ("aug_preview", "Augmentation Preview"),
    ("batch_inference", "Batch Inference"),
    ("artifact_bundle", "Artifact Bundle (Zip)"),
]

TASK_OPTIONS_MAP: Dict[str, str] = {key: label for key, label in TASK_OPTIONS}
TASK_LABEL_TO_KEY: Dict[str, str] = {label: key for key, label in TASK_OPTIONS}

TASK_DESCRIPTIONS: Dict[str, str] = {
    "format_conversion": "格式轉換，整理輸入圖檔格式。",
    "anomaly_detection": "瑕疵檢測基準生成。",
    "yolo_augmentation": "YOLO 標註資料增強。",
    "image_augmentation": "無標註圖像增強。",
    "dataset_splitter": "切分 train/val/test。",
    "yolo_train": "訓練 YOLO 模型。",
    "yolo_evaluation": "驗證模型表現。",
    "position_validation": "位置驗證輔助。",
    "color_inspection": "SAM 顏色範本蒐集。",
    "color_verification": "顏色批次驗證。",
    "generate_report": "彙整訓練/推理報告。",
    "dataset_lint": "資料品質檢查。",
    "aug_preview": "增強結果預覽。",
    "batch_inference": "批次推理輸出。",
}

DEFAULT_PRESETS: Dict[str, List[str]] = {
    "常用流程": [
        "dataset_splitter",
        "yolo_train",
        "yolo_evaluation",
        "generate_report",
    ],
}

# ------------------------------------------------------------------
# 現代化樣式表
# ------------------------------------------------------------------
# Moved to resources/style.qss


"""Purpose-oriented GUI workflows for pipeline task selection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class WorkflowPreset:
    """A user-facing workflow mapped to concrete pipeline task keys.

    Args:
        name: Display name shown in the GUI.
        tasks: Concrete task names passed to the pipeline.
        description: Short explanation of the expected output.
    """

    name: str
    tasks: tuple[str, ...]
    description: str


TASK_DISPLAY_ORDER: tuple[str, ...] = (
    "format_conversion",
    "dataset_lint",
    "yolo_augmentation",
    "image_augmentation",
    "aug_preview",
    "dataset_splitter",
    "yolo_train",
    "yolo_evaluation",
    "position_validation",
    "artifact_bundle",
    "deploy",
    "anomalib_train",
    "anomalib_package",
    "batch_inference",
    "anomaly_detection",
    "color_inspection",
    "color_verification",
    "qc_summary",
    "generate_report",
)


WORKFLOW_PRESETS: tuple[WorkflowPreset, ...] = (
    WorkflowPreset(
        name="YOLO: train and package",
        tasks=("dataset_splitter", "yolo_train", "artifact_bundle"),
        description="Train YOLO and create a zip that can be dropped into inference models.",
    ),
    WorkflowPreset(
        name="YOLO: train and deploy",
        tasks=(
            "yolo_augmentation",
            "dataset_lint",
            "dataset_splitter",
            "yolo_train",
            "position_validation",
            "yolo_evaluation",
            "generate_report",
            "batch_inference",
            "qc_summary",
            "deploy",
        ),
        description=(
            "Augment and lint data, split by source family, train, run position "
            "and detection validation, create reports, smoke-test inference, and "
            "deploy YOLO to yolo11_inference."
        ),
    ),
    WorkflowPreset(
        name="YOLO: train only",
        tasks=("dataset_splitter", "yolo_train"),
        description="Train YOLO without packaging or deploying artifacts.",
    ),
    WorkflowPreset(
        name="YOLO: package existing run",
        tasks=("artifact_bundle",),
        description="Package the latest existing YOLO run without retraining.",
    ),
    WorkflowPreset(
        name="Anomalib: train and package",
        tasks=("anomalib_train", "anomalib_package"),
        description="Train an anomaly model and create an inference-ready zip.",
    ),
    WorkflowPreset(
        name="Anomalib: train only",
        tasks=("anomalib_train",),
        description="Train an Anomalib model without packaging it.",
    ),
    WorkflowPreset(
        name="Anomalib: package existing run",
        tasks=("anomalib_package",),
        description="Package the latest existing Anomalib checkpoint.",
    ),
    WorkflowPreset(
        name="Data prep only",
        tasks=("format_conversion", "dataset_lint", "yolo_augmentation", "dataset_splitter"),
        description="Prepare and validate the dataset without training.",
    ),
    WorkflowPreset(
        name="Inference smoke test",
        tasks=("batch_inference", "qc_summary"),
        description="Run batch inference and summarize results.",
    ),
)


WORKFLOW_PRESET_MAP = {preset.name: preset for preset in WORKFLOW_PRESETS}


def ordered_task_keys(task_keys: Iterable[str]) -> list[str]:
    """Return task keys in the GUI's stable display order."""

    selected = set(task_keys)
    ordered = [key for key in TASK_DISPLAY_ORDER if key in selected]
    ordered.extend(key for key in task_keys if key not in TASK_DISPLAY_ORDER)
    return ordered


def workflow_tasks(name: str) -> list[str]:
    """Return concrete task keys for a workflow preset name."""

    preset = WORKFLOW_PRESET_MAP.get(name)
    return list(preset.tasks) if preset else []

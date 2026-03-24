"""Project-centric path resolution for the picture-tool pipeline.

All functions in this module are **pure** — they return a new configuration
dictionary rather than mutating the input.  This makes them straightforward
to unit-test and reason about.
"""

from __future__ import annotations

import copy
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def resolve_project_paths(config: dict[str, Any], project: str) -> dict[str, Any]:
    """Return a **new** config dict with all paths standardised for *project*.

    The function follows the 'Project-Centric' directory layout::

        data/<project>/raw/          ← original images & labels
        data/<project>/processed/    ← converted / augmented
        data/<project>/split/        ← train / val / test
        data/<project>/qc/           ← quality-control samples
        runs/<project>/              ← training outputs, logs, reports

    Args:
        config: The current pipeline configuration (not mutated).
        project: Project identifier (e.g. ``"Cable1"``).

    Returns:
        A deep copy of *config* with resolved paths.
    """
    cfg = copy.deepcopy(config)

    raw_root = Path("data") / project / "raw"
    processed_root = Path("data") / project / "processed"
    split_root = Path("data") / project / "split"
    qc_root = Path("data") / project / "qc"

    runs_root = Path("runs") / project
    infer_root = runs_root / "infer"
    quality_root = runs_root / "quality"
    reports_root = runs_root / "reports"

    # --- Format Conversion ---
    if "format_conversion" in cfg:
        fc = cfg["format_conversion"]
        fc["input_dir"] = str(raw_root / "images")
        fc["output_dir"] = str(processed_root / "images_converted")

    # --- Anomaly Detection ---
    if "anomaly_detection" in cfg:
        ad = cfg["anomaly_detection"]
        ad["output_folder"] = str(processed_root / "anomaly_results")
        ad["reference_folder"] = str(raw_root / "good")
        ad["test_folder"] = str(raw_root / "test")

    # --- Pipeline Logs ---
    if "pipeline" in cfg:
        cfg["pipeline"]["log_file"] = str(runs_root / "logs" / "pipeline.log")

    # --- YOLO Augmentation ---
    if "yolo_augmentation" in cfg:
        ya = cfg["yolo_augmentation"]
        inp = ya.get("input", {})
        out = ya.get("output", {})
        inp["image_dir"] = str(raw_root / "images")
        inp["label_dir"] = str(raw_root / "labels")
        out["image_dir"] = str(processed_root / "images")
        out["label_dir"] = str(processed_root / "labels")
        ya["input"], ya["output"] = inp, out

    # --- Image Augmentation ---
    if "image_augmentation" in cfg:
        ia = cfg["image_augmentation"]
        ia.setdefault("input", {})["image_dir"] = str(raw_root / "images")
        ia.setdefault("output", {})["image_dir"] = str(
            processed_root / "augmented_preview"
        )

    # --- Dataset Splitter ---
    if "train_test_split" in cfg:
        tts = cfg["train_test_split"]
        tts.setdefault("input", {})["image_dir"] = str(processed_root / "images")
        tts.setdefault("input", {})["label_dir"] = str(processed_root / "labels")
        tts.setdefault("output", {})["output_dir"] = str(split_root)

    # --- YOLO Training ---
    if "yolo_training" in cfg:
        yt = cfg["yolo_training"]
        yt["project"] = str(runs_root)
        yt["name"] = "train"
        yt["dataset_dir"] = str(split_root)

        pv = yt.get("position_validation", {})
        pv["output_dir"] = str(quality_root / "position")
        pv["sample_dir"] = str(split_root / "val" / "images")
        pv["product"] = project
        yt["position_validation"] = pv

        edc = yt.get("export_detection_config", {})
        edc["current_product"] = project
        yt["export_detection_config"] = edc

    # --- Batch Inference ---
    if "batch_inference" in cfg:
        bi = cfg["batch_inference"]
        bi["output_dir"] = str(infer_root)
        if not bi.get("input_dir") or "/project/" in str(bi.get("input_dir")):
            bi["input_dir"] = str(split_root / "test" / "images")

    # --- Color Inspection ---
    if "color_inspection" in cfg:
        ci = cfg["color_inspection"]
        ci["input_dir"] = str(qc_root / "color_samples")
        ci["output_json"] = str(quality_root / "color" / "stats.json")

    # --- Color Verification ---
    if "color_verification" in cfg:
        cv = cfg["color_verification"]
        cv["input_dir"] = str(qc_root / "color_samples")
        cv["color_stats"] = str(quality_root / "color" / "stats.json")
        cv["output_json"] = str(quality_root / "color" / "verification.json")
        cv["output_csv"] = str(quality_root / "color" / "verification.csv")
        cv["debug_dir"] = str(quality_root / "color" / "debug")

    # --- Dataset Lint ---
    if "dataset_lint" in cfg:
        dl = cfg["dataset_lint"]
        dl["image_dir"] = str(processed_root / "images")
        dl["label_dir"] = str(processed_root / "labels")
        dl["output_dir"] = str(quality_root / "lint")

    # --- Augmentation Preview ---
    if "aug_preview" in cfg:
        ap = cfg["aug_preview"]
        ap["image_dir"] = str(processed_root / "images")
        ap["label_dir"] = str(processed_root / "labels")
        ap["output_dir"] = str(quality_root / "preview")

    # --- Reports ---
    if "report" in cfg:
        cfg["report"]["output_dir"] = str(reports_root)

    _check_for_placeholders(cfg)
    return cfg


def _check_for_placeholders(
    obj: Any, path: str = "",
) -> None:
    """Recursively scan for remaining 'project' placeholder strings."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            _check_for_placeholders(v, f"{path}.{k}" if path else k)
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            _check_for_placeholders(v, f"{path}[{i}]")
    elif isinstance(obj, str):
        if "/project/" in obj.lower() or "./data/project" in obj.lower():
            logger.warning(
                "Unresolved placeholder found in config at '%s': %s", path, obj
            )

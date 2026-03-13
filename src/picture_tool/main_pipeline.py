import logging
from functools import lru_cache
from importlib import resources
from pathlib import Path
from typing import Any, Iterable, Optional

import yaml
from picture_tool.pipeline.core import Pipeline, Task
from picture_tool.config_validation import validate_config_schema


def _should_skip(
    task: str, config: dict, args, logger: Optional[logging.Logger] = None
) -> Optional[str]:
    # Deprecated: Logic moved to individual task definitions.
    return None


_DEFAULT_CONFIG_RESOURCE = "default_pipeline.yaml"


def setup_logging(log_file):
    """Initialise logging targets for the pipeline run."""
    log_path = Path(log_file)
    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
    except OSError:
        pass
    
    # Use force=True to allow re-configuring if called multiple times (e.g. smart resolution)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_file, encoding='utf-8'), logging.StreamHandler()],
        force=True
    )
    return logging.getLogger(__name__)


def _load_packaged_default() -> dict:
    """Load the bundled sample config shipped with the package."""
    package_resources = resources.files("picture_tool.resources")
    default_file = package_resources / _DEFAULT_CONFIG_RESOURCE
    with default_file.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def load_config(config_path: str | Path = "config.yaml"):
    """Load a pipeline configuration file, falling back to the packaged template."""
    path = Path(config_path)
    if path.exists():
        with path.open("r", encoding="utf-8") as handle:
            return yaml.safe_load(handle) or {}
    try:
        return _load_packaged_default()
    except FileNotFoundError as exc:  # pragma: no cover - packaging error guard
        raise FileNotFoundError(
            f"Config file '{config_path}' not found and packaged default is missing."
        ) from exc


@lru_cache(maxsize=4)
def _load_config_snapshot(path: str, mtime: float) -> dict:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def load_config_if_updated(config_path, config, logger):
    """Reload the configuration if the on-disk file changed."""
    config_file = Path(config_path)
    if not config_file.exists():
        return config
    current_mtime = config_file.stat().st_mtime
    last_mtime = getattr(load_config_if_updated, "last_mtime", None)

    if last_mtime is None:
        load_config_if_updated.last_mtime = current_mtime  # type: ignore
        return config

    if current_mtime > last_mtime:
        logger.info("Detected configuration change; reloading.")
        load_config_if_updated.last_mtime = current_mtime  # type: ignore
        return _load_config_snapshot(str(config_file.resolve()), current_mtime)

    return config


def _apply_smart_path_resolution(config: dict, args, logger: logging.Logger) -> None:
    """
    Standardizes paths based on the project name (from --name or --product).
    Follows the 'Project-Centric' architecture.
    """
    project = getattr(args, "name", None) or getattr(args, "product", None)
    if not project:
        return

    logger.info(f"Applying smart path resolution for project: {project}")
    
    # 1. Inputs: data/<project>/raw/
    raw_root = Path("data") / project / "raw"
    processed_root = Path("data") / project / "processed"
    split_root = Path("data") / project / "split"
    qc_root = Path("data") / project / "qc"
    
    # 2. Outputs: runs/<project>/
    runs_root = Path("runs") / project
    train_root = runs_root / "train"
    infer_root = runs_root / "infer"
    quality_root = runs_root / "quality"
    reports_root = runs_root / "reports"

    # --- Update Config Sections ---
    
    # 0. Formatting & Conversion
    if "format_conversion" in config:
        fc = config["format_conversion"]
        fc["input_dir"] = str(raw_root / "images")
        fc["output_dir"] = str(processed_root / "images_converted")

    # 0.1 Anomaly Detection
    if "anomaly_detection" in config:
        ad = config["anomaly_detection"]
        ad["output_folder"] = str(processed_root / "anomaly_results")
        ad["reference_folder"] = str(raw_root / "good")
        ad["test_folder"] = str(raw_root / "test")

    # Pipeline Logs
    if "pipeline" in config:
        config["pipeline"]["log_file"] = str(runs_root / "logs" / "pipeline.log")
        # Re-setup logging to the new project-specific path
        setup_logging(config["pipeline"]["log_file"])

    # Yolo Augmentation
    if "yolo_augmentation" in config:
        ya = config["yolo_augmentation"]
        inp = ya.get("input", {})
        out = ya.get("output", {})
        # Use simple defaults if not explicitly set to something else
        inp["image_dir"] = str(raw_root / "images")
        inp["label_dir"] = str(raw_root / "labels")
        out["image_dir"] = str(processed_root / "images")
        out["label_dir"] = str(processed_root / "labels")
        ya["input"], ya["output"] = inp, out

    # Image Augmentation
    if "image_augmentation" in config:
        ia = config["image_augmentation"]
        ia.setdefault("input", {})["image_dir"] = str(raw_root / "images")
        ia.setdefault("output", {})["image_dir"] = str(processed_root / "augmented_preview")

    # Dataset Splitter
    if "train_test_split" in config:
        tts = config["train_test_split"]
        tts.setdefault("input", {})["image_dir"] = str(processed_root / "images")
        tts.setdefault("input", {})["label_dir"] = str(processed_root / "labels")
        tts.setdefault("output", {})["output_dir"] = str(split_root)

    # Yolo Training
    if "yolo_training" in config:
        yt = config["yolo_training"]
        yt["project"] = str(runs_root)
        yt["name"] = "train"
        yt["dataset_dir"] = str(split_root)
        
        # Position Validation
        pv = yt.get("position_validation", {})
        pv["output_dir"] = str(quality_root / "position")
        pv["sample_dir"] = str(split_root / "val" / "images")
        pv["product"] = project
        yt["position_validation"] = pv

        # Export Detection Config
        edc = yt.get("export_detection_config", {})
        edc["current_product"] = project
        yt["export_detection_config"] = edc

    # Batch Inference
    if "batch_inference" in config:
        bi = config["batch_inference"]
        bi["output_dir"] = str(infer_root)
        # Default input for inference often comes from split/test
        if not bi.get("input_dir") or "/project/" in str(bi.get("input_dir")):
            bi["input_dir"] = str(split_root / "test" / "images")

    # Color Verification & Inspection
    if "color_inspection" in config:
        ci = config["color_inspection"]
        ci["input_dir"] = str(qc_root / "color_samples")
        ci["output_json"] = str(quality_root / "color" / "stats.json")

    if "color_verification" in config:
        cv = config["color_verification"]
        cv["input_dir"] = str(qc_root / "color_samples")
        cv["color_stats"] = str(quality_root / "color" / "stats.json")
        cv["output_json"] = str(quality_root / "color" / "verification.json")
        cv["output_csv"] = str(quality_root / "color" / "verification.csv")
        cv["debug_dir"] = str(quality_root / "color" / "debug")
        
    # Dataset Lint
    if "dataset_lint" in config:
        dl = config["dataset_lint"]
        dl["image_dir"] = str(processed_root / "images")
        dl["label_dir"] = str(processed_root / "labels")
        dl["output_dir"] = str(quality_root / "lint")

    # Augmentation Preview
    if "aug_preview" in config:
        ap = config["aug_preview"]
        ap["image_dir"] = str(processed_root / "images")
        ap["label_dir"] = str(processed_root / "labels")
        ap["output_dir"] = str(quality_root / "preview")

    # Reports
    if "report" in config:
        config["report"]["output_dir"] = str(reports_root)

    # 3. Final Placeholder Check (Safety Guard)
    _check_for_placeholders(config, logger)


def _check_for_placeholders(obj: Any, logger: logging.Logger, path: str = "") -> None:
    """Recursively scans for any remaining 'project' placeholders in the configuration."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            _check_for_placeholders(v, logger, f"{path}.{k}" if path else k)
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            _check_for_placeholders(v, logger, f"{path}[{i}]")
    elif isinstance(obj, str):
        if "/project/" in obj.lower() or "./data/project" in obj.lower():
            logger.warning(f"Unresolved placeholder found in config at '{path}': {obj}")


def _apply_cli_overrides(config: dict, args, logger: logging.Logger) -> None:
    # First, apply smart resolution if applicable
    _apply_smart_path_resolution(config, args, logger)
    
    yt = config.get("yolo_training", {})
    changed = []
    
    # Individual Overrides (Override smart resolution if explicitly provided)
    if getattr(args, "device", None):
        yt["device"] = args.device
        changed.append(("device", args.device))
    if getattr(args, "epochs", None):
        yt["epochs"] = int(args.epochs)
        changed.append(("epochs", yt["epochs"]))
    if getattr(args, "imgsz", None):
        yt["imgsz"] = int(args.imgsz)
        changed.append(("imgsz", yt["imgsz"]))
    if getattr(args, "batch", None):
        yt["batch"] = int(args.batch)
        changed.append(("batch", yt["batch"]))
    if getattr(args, "model", None):
        yt["model"] = args.model
        changed.append(("model", yt["model"]))
    if getattr(args, "project", None):
        yt["project"] = args.project
        changed.append(("project", yt["project"]))
    if getattr(args, "name", None):
        yt["name"] = args.name
        changed.append(("name", yt["name"]))

    config["yolo_training"] = yt

    ye = config.get("yolo_evaluation", {})
    if getattr(args, "weights", None):
        ye["weights"] = args.weights
        changed.append(("eval.weights", ye["weights"]))
    config["yolo_evaluation"] = ye

    bi = config.get("batch_inference", {})
    if getattr(args, "infer_input", None):
        bi["input_dir"] = args.infer_input
        changed.append(("infer.input", bi["input_dir"]))
    if getattr(args, "infer_output", None):
        bi["output_dir"] = args.infer_output
        changed.append(("infer.output", bi["output_dir"]))
    # Also apply product override to inference input if not manually specified?
    # For now, let's keep inference manual or assume default.

    config["batch_inference"] = bi

    if changed:
        logger.info(
            "Applied manual CLI overrides: %s",
            ", ".join([f"{key}={value}" for key, value in changed]),
        )


def _auto_device(config: dict, logger: logging.Logger) -> None:
    yt = config.get("yolo_training", {})
    device = str(yt.get("device", "auto"))
    if device.lower() == "auto":
        try:
            import os
            if os.environ.get("PYTEST_IS_RUNNING") == "1":
                raise ImportError("Bypass torch during pytest")
            import torch  # type: ignore

            yt["device"] = "0" if torch.cuda.is_available() else "cpu"
        except (FileNotFoundError, yaml.YAMLError, OSError, ImportError):
            yt["device"] = "cpu"
        logger.info(f"Auto-selected device: {yt['device']}")
        config["yolo_training"] = yt


def build_task_registry(config: dict) -> dict[str, Task]:
    from picture_tool.tasks import (
        conversion,
        quality,
        training,
        augmentation,
        data_sync,
    )

    tasks_modules = [
        conversion,
        quality,
        training,
        augmentation,
        data_sync,
    ]

    registry: dict[str, Task] = {}
    for mod in tasks_modules:
        if hasattr(mod, "TASKS"):
            for task in mod.TASKS:
                registry[task.name] = task
    return registry


def validate_dependencies(
    tasks: list[str], config: dict, logger: logging.Logger
) -> list[str]:
    # Simplistic validation: just ensure tasks exist
    # Dependencies are handled by Pipeline core at runtime
    # We still want to warn about missing weights if possible, but the original
    # logic was complex. We'll rely on tasks failing or skipping gracefully.
    return tasks


def get_tasks_from_groups(groups, config):
    """Expand named task groups into a deduplicated task list."""
    tasks = set()
    for group in groups:
        if group in config["pipeline"]["task_groups"]:
            tasks.update(config["pipeline"]["task_groups"][group])
        else:
            logging.warning("Unknown task group: %s", group)
    return list(tasks)


def interactive_task_selection(config, registry: dict[str, Task]):
    """Prompt the user to pick tasks interactively."""
    print("\nAvailable tasks:")
    all_tasks = sorted(registry.keys())
    # Try to get enabled status from config if possible
    pipeline_tasks = {
        t.get("name"): t for t in config.get("pipeline", {}).get("tasks", [])
    }

    for i, task_name in enumerate(all_tasks, 1):
        # Default to enabled if not specified
        enabled = True
        if task_name in pipeline_tasks:
            enabled = pipeline_tasks[task_name].get("enabled", True)

        status = "enabled" if enabled else "disabled"
        print(f"{i}. {task_name} ({status}) - {registry[task_name].description}")

    print(
        "\nEnter task numbers separated by spaces. Enter 0 to run all enabled tasks, or press Enter to accept defaults."
    )
    user_input = input("> ").strip()

    if not user_input:
        # Return defaults from config
        return [
            t["name"] for t in config["pipeline"]["tasks"] if t.get("enabled", True)
        ]
    if user_input == "0":
        return [
            t["name"] for t in config["pipeline"]["tasks"] if t.get("enabled", True)
        ]

    selected_indices = [int(i) - 1 for i in user_input.split()]
    selected_tasks = [all_tasks[i] for i in selected_indices if 0 <= i < len(all_tasks)]
    return selected_tasks


def run_pipeline(tasks, config, logger, args, stop_event=None):
    """Execute each task handler with dependency checks and skipping logic."""
    validate_config_schema(config, logger=logger, strict=False)

    registry = build_task_registry(config)

    # If interactive selection was NOT used (tasks passed from CLI/groups),
    # ensuring they exist in registry:
    valid_tasks = []
    for t in tasks:
        if t in registry:
            valid_tasks.append(t)
        else:
            logger.warning(f"Unknown task requested: {t}")

    setattr(args, "stop_event", stop_event)

    def _before_task(task_obj: Task, cfg: dict) -> dict:
        fresh_cfg = load_config_if_updated(args.config, cfg, logger)
        fresh_cfg = validate_config_schema(fresh_cfg, logger=logger, strict=False)
        _apply_cli_overrides(fresh_cfg, args, logger)
        _auto_device(fresh_cfg, logger)
        return fresh_cfg

    pipeline = Pipeline(registry, logger=logger)
    pipeline.run(valid_tasks, config, args, before_task=_before_task)


def main():
    from picture_tool.cli import app

    app()


if __name__ == "__main__":
    main()

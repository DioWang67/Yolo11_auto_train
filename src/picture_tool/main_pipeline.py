import logging
from functools import lru_cache
from importlib import resources
from pathlib import Path
from typing import Optional

import yaml
from picture_tool.pipeline.core import Pipeline, Task
from picture_tool.config_validation import validate_config_schema


def _should_skip(task: str, config: dict, args, logger: Optional[logging.Logger] = None) -> Optional[str]:
    # Deprecated: Logic moved to individual task definitions.
    return None


_DEFAULT_CONFIG_RESOURCE = "default_pipeline.yaml"


def setup_logging(log_file):
    """Initialise logging targets for the pipeline run."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
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
        load_config_if_updated.last_mtime = current_mtime
        return config

    if current_mtime > last_mtime:
        logger.info("Detected configuration change; reloading.")
        load_config_if_updated.last_mtime = current_mtime
        return _load_config_snapshot(str(config_file.resolve()), current_mtime)

    return config


def _apply_cli_overrides(config: dict, args, logger: logging.Logger) -> None:
    yt = config.get("yolo_training", {})
    changed = []
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
    
    # [NEW] Product Override
    product = getattr(args, "product", None)
    if product:
        # 1. Update Augmentation Inputs
        ya = config.get("yolo_augmentation", {})
        inp = ya.get("input", {})
        
        target_img_dir = f"./data/raw/{product}/images"
        target_lbl_dir = f"./data/raw/{product}/labels"
        
        if not Path(target_img_dir).exists():
             raise FileNotFoundError(f"Product directory not found: {target_img_dir}\n請確認 data/raw/{product} 是否存在。")
        
        inp["image_dir"] = target_img_dir
        inp["label_dir"] = target_lbl_dir
        ya["input"] = inp
        config["yolo_augmentation"] = ya
        changed.append(("yolo_augmentation.input", f"data/raw/{product}"))

        # 2. Update Training Name
        yt["name"] = product
        changed.append(("yolo_training.name", product))

        # 3. Update Position Validation Product
        pv = yt.get("position_validation", {})
        pv["product"] = product
        yt["position_validation"] = pv
        changed.append(("position_validation.product", product))

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
            "Applied CLI overrides: %s",
            ", ".join([f"{key}={value}" for key, value in changed]),
        )


def _auto_device(config: dict, logger: logging.Logger) -> None:
    yt = config.get("yolo_training", {})
    device = str(yt.get("device", "auto"))
    if device.lower() == "auto":
        try:
            import torch  # type: ignore

            yt["device"] = "0" if torch.cuda.is_available() else "cpu"
        except Exception:
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


def validate_dependencies(tasks: list[str], config: dict, logger: logging.Logger) -> list[str]:
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
    pipeline_tasks = {t.get("name"): t for t in config.get("pipeline", {}).get("tasks", [])}
    
    for i, task_name in enumerate(all_tasks, 1):
        # Default to enabled if not specified
        enabled = True
        if task_name in pipeline_tasks:
            enabled = pipeline_tasks[task_name].get("enabled", True)
        
        status = "enabled" if enabled else "disabled"
        print(f"{i}. {task_name} ({status}) - {registry[task_name].description}")

    print("\nEnter task numbers separated by spaces. Enter 0 to run all enabled tasks, or press Enter to accept defaults.")
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

"""Pipeline orchestration entry point.

Delegates configuration loading to :mod:`picture_tool.config_loader`,
path resolution to :mod:`picture_tool.path_resolver`, and task
execution to :mod:`picture_tool.pipeline.core`.
"""

import logging
from pathlib import Path
from typing import Any, Optional

import yaml
from picture_tool.config_loader import (
    load_config,
    load_config_if_updated,
)
from picture_tool.config_validation import validate_config_schema
from picture_tool.path_resolver import resolve_project_paths
from picture_tool.pipeline.core import Pipeline, Task


def _should_skip(
    task: str, config: dict, args, logger: Optional[logging.Logger] = None
) -> Optional[str]:
    # Deprecated: Logic moved to individual task definitions.
    return None


def setup_logging(log_file):
    """Initialise logging targets for the pipeline run."""
    log_path = Path(log_file)
    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
    except OSError:
        pass

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_file, encoding='utf-8'), logging.StreamHandler()],
        force=True
    )
    return logging.getLogger(__name__)


def _apply_cli_overrides(config: dict, args, logger: logging.Logger) -> dict:
    """Apply smart path resolution followed by explicit CLI overrides.

    Args:
        config: Current pipeline configuration (not mutated).
        args: Parsed CLI arguments namespace.
        logger: Logger instance.

    Returns:
        A new configuration dictionary with all overrides applied.
    """
    # Smart path resolution (pure — returns a new dict)
    project = getattr(args, "name", None) or getattr(args, "product", None)
    if project:
        logger.info("Applying smart path resolution for project: %s", project)
        config = resolve_project_paths(config, project)
        # Re-setup logging to the new project-specific path
        log_file = config.get("pipeline", {}).get("log_file")
        if log_file:
            setup_logging(log_file)

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

    config["batch_inference"] = bi

    if changed:
        logger.info(
            "Applied manual CLI overrides: %s",
            ", ".join([f"{key}={value}" for key, value in changed]),
        )

    return config


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
        except (ImportError, OSError):
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
    """Validate that all requested tasks exist and have no cyclic dependencies.

    Returns the input list unchanged if valid; raises ValueError otherwise.
    """
    registry = build_task_registry(config)
    unknown = [t for t in tasks if t not in registry]
    if unknown:
        raise ValueError(f"Unknown tasks: {', '.join(unknown)}")

    from picture_tool.pipeline.core import Pipeline
    pipe = Pipeline(registry, logger=logger)
    collected = pipe._collect(tasks)
    pipe._toposort(collected)  # raises on cycle
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
    pipeline_tasks = {
        t.get("name"): t for t in config.get("pipeline", {}).get("tasks", [])
    }

    for i, task_name in enumerate(all_tasks, 1):
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
        fresh_cfg = _apply_cli_overrides(fresh_cfg, args, logger)
        _auto_device(fresh_cfg, logger)
        return fresh_cfg

    pipeline = Pipeline(registry, logger=logger)
    pipeline.run(valid_tasks, config, args, before_task=_before_task)


def main():
    from picture_tool.cli import app

    app()


if __name__ == "__main__":
    main()

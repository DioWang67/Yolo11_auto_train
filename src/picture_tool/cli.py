from pathlib import Path
from typing import Optional, List
import typer  # type: ignore
import yaml  # type: ignore
from picture_tool.main_pipeline import (
    load_config,
    setup_logging,
    run_pipeline,
)
from picture_tool.exceptions import ConfigurationError, PipelineError

app = typer.Typer(help="YOLO auto-train pipeline orchestration tools.")


def _load_config_or_exit(config_path: str):
    try:
        return load_config(config_path)
    except (FileNotFoundError, yaml.YAMLError, OSError) as e:
        typer.echo(f"Error loading config: {e}", err=True)
        raise typer.Exit(code=1) from e


@app.command()
def run(
    tasks: Optional[List[str]] = typer.Option(None, help="Specific tasks to run."),
    config: str = typer.Option("config.yaml", help="Path to the pipeline config."),
    exclude_tasks: Optional[List[str]] = typer.Option(None, help="Tasks to exclude."),
    task_groups: Optional[List[str]] = typer.Option(
        None, help="Named task groups to run."
    ),
    interactive: bool = typer.Option(False, help="Interactively select tasks."),
    force: bool = typer.Option(False, help="Force execution ignoring cache."),
    device: Optional[str] = typer.Option(
        None, help="Override device (e.g., '0' or 'cpu')."
    ),
    epochs: Optional[int] = typer.Option(None, help="Override training epochs."),
    imgsz: Optional[int] = typer.Option(None, help="Override training image size."),
    batch: Optional[int] = typer.Option(None, help="Override training batch size."),
    model: Optional[str] = typer.Option(None, help="Override model weight path/name."),
    project: Optional[str] = typer.Option(None, help="Override project path."),
    name: Optional[str] = typer.Option(None, help="Override run name."),
    weights: Optional[str] = typer.Option(
        None, help="Override weights for evaluation."
    ),
    infer_input: Optional[str] = typer.Option(
        None, help="Override batch inference input."
    ),
    infer_output: Optional[str] = typer.Option(
        None, help="Override batch inference output."
    ),
    product: Optional[str] = typer.Option(
        None, help="Override product name (e.g., Cable1)."
    ),
):
    """Run the pipeline with specified tasks and overrides."""
    logger = setup_logging("pipeline.log")
    cfg_data = _load_config_or_exit(config)

    # Construct args object to mimic argparse namespace for compatibility
    args = type(
        "Args",
        (),
        {
            "config": config,
            "tasks": tasks,
            "exclude_tasks": exclude_tasks,
            "task_groups": task_groups,
            "interactive": interactive,
            "force": force,
            "device": device,
            "epochs": epochs,
            "imgsz": imgsz,
            "batch": batch,
            "model": model,
            "project": project,
            "name": name,
            "weights": weights,
            "infer_input": infer_input,
            "infer_output": infer_output,
            "product": product,
            "input_format": None,  # Not exposed in this command yet
            "output_format": None,  # Not exposed in this command yet
        },
    )()

    # Task selection logic
    from picture_tool.main_pipeline import (
        interactive_task_selection,
        get_tasks_from_groups,
        build_task_registry,
    )

    registry = build_task_registry(cfg_data)

    selected_tasks = []
    if interactive:
        selected_tasks = interactive_task_selection(cfg_data, registry)
    elif tasks:
        selected_tasks = tasks
    elif task_groups:
        selected_tasks = get_tasks_from_groups(task_groups, cfg_data)
    else:
        selected_tasks = [
            t["name"] for t in cfg_data["pipeline"]["tasks"] if t.get("enabled", True)
        ]

    if exclude_tasks:
        selected_tasks = [t for t in selected_tasks if t not in exclude_tasks]

    if not selected_tasks:
        logger.warning("No tasks selected to run.")
        return

    args.tasks = selected_tasks
    logger.info(f"Starting pipeline with tasks: {selected_tasks}")
    try:
        run_pipeline(selected_tasks, cfg_data, logger, args)
    except (ConfigurationError, PipelineError, RuntimeError) as e:
        logger.exception("Pipeline execution failed.")
        raise typer.Exit(code=1) from e


@app.command()
def list_tasks(
    config: str = typer.Option(
        "config.yaml", help="Path to config for dependency check."
    ),
):
    """List all available tasks."""
    try:
        cfg = load_config(config)
    except (ConfigurationError, OSError, yaml.YAMLError):
        cfg = {}

    from picture_tool.main_pipeline import build_task_registry

    registry = build_task_registry(cfg)

    typer.echo("Available tasks:")
    for name, task in sorted(registry.items()):
        typer.echo(f"  {name:<25} : {task.description}")


@app.command()
def describe(
    task: str, config: str = typer.Option("config.yaml", help="Path to config.")
):
    """Show details for a specific task."""
    try:
        cfg = load_config(config)
    except (ConfigurationError, OSError, yaml.YAMLError):
        cfg = {}

    from picture_tool.main_pipeline import build_task_registry

    registry = build_task_registry(cfg)

    if task in registry:
        t = registry[task]
        typer.echo(f"Task: {t.name}")
        typer.echo(f"Description: {t.description}")
        deps = t.dependencies
        typer.echo(f"Dependencies: {', '.join(deps) if deps else 'None'}")
    else:
        typer.echo(f"Unknown task: {task}", err=True)
        raise typer.Exit(code=1)


@app.command("anomalib-train-folder")
def anomalib_train_folder(
    input_dir: Path = typer.Option(
        ..., "--input", exists=True, file_okay=False, help="Product/area folder or image folder."
    ),
    product: str = typer.Option(..., help="Product name, e.g. PCBA1."),
    area: str = typer.Option(..., help="Area name, e.g. B."),
    project: Optional[Path] = typer.Option(
        None, help="Output project directory. Defaults to runs/anomalib/<product>/<area>."
    ),
    model: str = typer.Option("padim", help="Anomalib model: padim, patchcore, or efficientad."),
    image_size: int = typer.Option(256, min=1, help="Square training image size."),
    batch_size: int = typer.Option(8, min=1, help="Train/eval batch size."),
    max_epochs: int = typer.Option(1, min=1, help="Maximum training epochs."),
    accelerator: str = typer.Option("cpu", help="Lightning accelerator, e.g. cpu, gpu, auto."),
    devices: str = typer.Option("1", help="Lightning devices value."),
    pre_trained: bool = typer.Option(False, help="Use pretrained backbone weights."),
    require_anomalous_validation: bool = typer.Option(
        False, help="Fail if no abnormal validation/test folder is found."
    ),
    force: bool = typer.Option(False, help="Retrain even if a checkpoint exists."),
    tmp_dir: Path = typer.Option(
        Path("runs/tmp"), help="Temp directory used during checkpoint writes."
    ),
):
    """Train Anomalib from a folder with automatic layout detection."""
    from picture_tool.train.anomalib_trainer import train_anomalib_folder

    logger = setup_logging("logs/anomalib_train_folder.log")
    try:
        result = train_anomalib_folder(
            input_dir=input_dir,
            product=product,
            area=area,
            project=project,
            model=model,
            image_size=image_size,
            batch_size=batch_size,
            max_epochs=max_epochs,
            accelerator=accelerator,
            devices=_parse_devices(devices),
            pre_trained=pre_trained,
            require_anomalous_validation=require_anomalous_validation,
            force=force,
            tmp_dir=tmp_dir,
            logger=logger,
        )
    except (ValueError, RuntimeError, ImportError) as exc:
        logger.exception("Anomalib folder training failed.")
        typer.echo(f"Anomalib training failed: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    typer.echo(f"Run directory: {result.run_dir}")
    typer.echo(f"Checkpoint: {result.checkpoint_path or 'not found'}")
    typer.echo(f"Report: {result.report_path}")
    typer.echo(f"Normal images: {result.normal_image_count}")
    typer.echo(f"Abnormal images: {result.abnormal_image_count}")
    if result.baseline_only:
        typer.echo("Status: baseline_only=true; threshold is not deployment-grade.")
    else:
        typer.echo("Status: validated layout detected.")


def _parse_devices(value: str) -> str | int:
    """Return an int for simple numeric device values, otherwise the raw string."""
    stripped = value.strip()
    if stripped.isdigit():
        return int(stripped)
    return stripped


if __name__ == "__main__":
    app()

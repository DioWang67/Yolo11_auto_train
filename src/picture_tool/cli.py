from typing import Optional, List
import typer  # type: ignore
from picture_tool.main_pipeline import (
    load_config,
    setup_logging,
    run_pipeline,
)

app = typer.Typer(help="YOLO auto-train pipeline orchestration tools.")

def _load_config_or_exit(config_path: str):
    try:
        return load_config(config_path)
    except Exception as e:
        typer.echo(f"Error loading config: {e}", err=True)
        raise typer.Exit(code=1)

@app.command()
def run(
    tasks: Optional[List[str]] = typer.Option(None, help="Specific tasks to run."),
    config: str = typer.Option("config.yaml", help="Path to the pipeline config."),
    exclude_tasks: Optional[List[str]] = typer.Option(None, help="Tasks to exclude."),
    task_groups: Optional[List[str]] = typer.Option(None, help="Named task groups to run."),
    interactive: bool = typer.Option(False, help="Interactively select tasks."),
    force: bool = typer.Option(False, help="Force execution ignoring cache."),
    device: Optional[str] = typer.Option(None, help="Override device (e.g., '0' or 'cpu')."),
    epochs: Optional[int] = typer.Option(None, help="Override training epochs."),
    imgsz: Optional[int] = typer.Option(None, help="Override training image size."),
    batch: Optional[int] = typer.Option(None, help="Override training batch size."),
    model: Optional[str] = typer.Option(None, help="Override model weight path/name."),
    project: Optional[str] = typer.Option(None, help="Override project path."),
    name: Optional[str] = typer.Option(None, help="Override run name."),
    weights: Optional[str] = typer.Option(None, help="Override weights for evaluation."),
    infer_input: Optional[str] = typer.Option(None, help="Override batch inference input."),
    infer_output: Optional[str] = typer.Option(None, help="Override batch inference output."),
    product: Optional[str] = typer.Option(None, help="Override product name (e.g., Cable1)."),
):
    """Run the pipeline with specified tasks and overrides."""
    logger = setup_logging("pipeline.log")
    cfg_data = _load_config_or_exit(config)
    
    # Construct args object to mimic argparse namespace for compatibility
    args = type("Args", (), {
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
        "input_format": None, # Not exposed in this command yet
        "output_format": None, # Not exposed in this command yet
    })()

    # Task selection logic
    from picture_tool.main_pipeline import (
        interactive_task_selection, 
        get_tasks_from_groups,
        build_task_registry
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
        selected_tasks = [t["name"] for t in cfg_data["pipeline"]["tasks"] if t.get("enabled", True)]

    if exclude_tasks:
        selected_tasks = [t for t in selected_tasks if t not in exclude_tasks]

    if not selected_tasks:
        logger.warning("No tasks selected to run.")
        return

    args.tasks = selected_tasks
    logger.info(f"Starting pipeline with tasks: {selected_tasks}")
    try:
        run_pipeline(selected_tasks, cfg_data, logger, args)
    except Exception:
        logger.exception("Pipeline execution failed.")
        raise typer.Exit(code=1)

@app.command()
def list_tasks(config: str = typer.Option("config.yaml", help="Path to config for dependency check.")):
    """List all available tasks."""
    try:
        cfg = load_config(config)
    except Exception:
        cfg = {}
        
    from picture_tool.main_pipeline import build_task_registry
    registry = build_task_registry(cfg)
    
    typer.echo("Available tasks:")
    for name, task in sorted(registry.items()):
        typer.echo(f"  {name:<25} : {task.description}")

@app.command()
def describe(task: str, config: str = typer.Option("config.yaml", help="Path to config.")):
    """Show details for a specific task."""
    try:
        cfg = load_config(config)
    except Exception:
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

if __name__ == "__main__":
    app()

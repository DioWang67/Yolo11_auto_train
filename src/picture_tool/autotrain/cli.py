"""Command line for the autonomous training path.

Every command refuses while ``autonomous_training.enabled`` is false, and no
command can deploy anything. Steps are individually runnable so a stuck cycle
can be nudged forward one stage at a time.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import typer

from picture_tool.autotrain import AutoTrainDisabledError, AutoTrainError
from picture_tool.autotrain import golden as golden_module
from picture_tool.autotrain.config import AutoTrainConfig
from picture_tool.autotrain.labeling import (
    export_request,
    import_request,
    request_statistics,
)
from picture_tool.autotrain.orchestrator import TrainingCycle, run_training_cycle
from picture_tool.autotrain.paths import AutoTrainPaths
from picture_tool.autotrain.service import AutoTrainService

app = typer.Typer(
    help=(
        "Autonomous training: collect, select, version, train a challenger and "
        "compare it with the deployed champion. Never deploys."
    )
)


def _service(config_path: Optional[str], product: str, area: str) -> AutoTrainService:
    config = AutoTrainConfig.load(config_path)
    if product or area:
        from dataclasses import replace

        from picture_tool.autotrain.config import StationConfig

        config = replace(
            config,
            station=StationConfig(
                product=product or config.station.product,
                area=area or config.station.area,
            ),
        )
    return AutoTrainService(config, AutoTrainPaths.discover())


def _run(action, *, quiet: bool = False):
    """Run one action, turning the domain errors into clean CLI output."""
    logging.basicConfig(level=logging.WARNING if quiet else logging.INFO)
    try:
        return action()
    except AutoTrainDisabledError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=2) from exc
    except AutoTrainError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1) from exc


def _echo_json(payload) -> None:
    typer.echo(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))


# ---------------------------------------------------------------------------
# The cycle


@app.command()
def cycle(
    product: str = typer.Option("", help="Override the configured product."),
    area: str = typer.Option("", help="Override the configured area."),
    config: Optional[str] = typer.Option(None, help="Path to the settings file."),
    cycle_id: Optional[str] = typer.Option(None, help="Resume this cycle id."),
):
    """Run a full training cycle and print the report."""

    def action():
        service = _service(config, product, area)
        result = run_training_cycle(
            service.config, service.paths, cycle_id=cycle_id
        )
        typer.echo(result.report_text)
        if result.report_path:
            typer.echo(f"Report: {result.report_path}")
        return result

    result = _run(action)
    raise typer.Exit(code=0 if result.status == "COMPLETED" else 3)


@app.command()
def collect(
    product: str = typer.Option("", help="Override the configured product."),
    area: str = typer.Option("", help="Override the configured area."),
    config: Optional[str] = typer.Option(None, help="Path to the settings file."),
    cycle_id: Optional[str] = typer.Option(None, help="Cycle to record into."),
):
    """Read recent production inspections. Read-only."""

    def action():
        service = _service(config, product, area)
        step = TrainingCycle(service.config, service.paths, cycle_id=cycle_id)
        summary = step.collect()
        _echo_json({"cycle_id": step.cycle_id, **summary})

    _run(action)


@app.command()
def select(
    product: str = typer.Option("", help="Override the configured product."),
    area: str = typer.Option("", help="Override the configured area."),
    config: Optional[str] = typer.Option(None, help="Path to the settings file."),
    cycle_id: str = typer.Option(..., help="Cycle whose collection to select from."),
):
    """Run the enabled selectors and add their picks to the candidate pool."""

    def action():
        service = _service(config, product, area)
        step = TrainingCycle(service.config, service.paths, cycle_id=cycle_id)
        _echo_json({"cycle_id": step.cycle_id, **step.select()})

    _run(action)


@app.command("dataset")
def dataset_create(
    product: str = typer.Option("", help="Override the configured product."),
    area: str = typer.Option("", help="Override the configured area."),
    config: Optional[str] = typer.Option(None, help="Path to the settings file."),
    description: str = typer.Option("", help="Why this version was created."),
):
    """Freeze the verified candidates into a new immutable dataset version."""

    def action():
        service = _service(config, product, area)
        _echo_json(service.create_dataset_version(description=description))

    _run(action)


# ---------------------------------------------------------------------------
# Labelling


labeling_app = typer.Typer(help="Export and import the human labelling queue.")
app.add_typer(labeling_app, name="labeling")


@labeling_app.command("export")
def labeling_export(
    classes: str = typer.Option(..., help="Comma-separated class names, in order."),
    product: str = typer.Option("", help="Override the configured product."),
    area: str = typer.Option("", help="Override the configured area."),
    config: Optional[str] = typer.Option(None, help="Path to the settings file."),
    limit: Optional[int] = typer.Option(None, help="Maximum images to export."),
):
    """Write the pending candidates out for annotation."""

    def action():
        service = _service(config, product, area)
        service.config.require_enabled()
        names = [name.strip() for name in classes.split(",") if name.strip()]
        request = export_request(
            service._pool(),
            service.paths.labeling_root,
            product=service.product,
            area=service.area,
            class_names=names,
            limit=limit,
        )
        _echo_json(
            {
                "request_id": request.request_id,
                "root": str(request.root),
                "images": len(request.sample_ids),
                "next": "annotate the images, then run 'labeling import'",
            }
        )

    _run(action)


@labeling_app.command("import")
def labeling_import(
    request: str = typer.Argument(..., help="Path to the exported request."),
    product: str = typer.Option("", help="Override the configured product."),
    area: str = typer.Option("", help="Override the configured area."),
    config: Optional[str] = typer.Option(None, help="Path to the settings file."),
):
    """Validate annotated labels and mark their candidates verified."""

    def action():
        service = _service(config, product, area)
        service.config.require_enabled()
        result = import_request(service._pool(), request)
        _echo_json(result.summary())
        return result

    result = _run(action)
    raise typer.Exit(code=1 if result.errors else 0)


@labeling_app.command("status")
def labeling_status(
    request: str = typer.Argument(..., help="Path to the exported request."),
):
    """Show how much of a labelling request has been annotated."""

    def action():
        from picture_tool.autotrain.labeling import load_request

        _echo_json(dict(request_statistics(load_request(request))))

    _run(action)


# ---------------------------------------------------------------------------
# Golden dataset


golden_app = typer.Typer(help="Register and check the locked evaluation dataset.")
app.add_typer(golden_app, name="golden")


@golden_app.command("register")
def golden_register(
    path: str = typer.Argument(..., help="Directory a person has assembled."),
    registered_by: str = typer.Option(..., help="Who is registering this set."),
    description: str = typer.Option("", help="What this set covers."),
    overwrite: bool = typer.Option(False, help="Replace an existing registration."),
):
    """Lock a directory as the golden evaluation set."""

    def action():
        dataset = golden_module.register(
            path,
            registered_by=registered_by,
            description=description,
            overwrite=overwrite,
        )
        _echo_json(
            {
                "root": str(dataset.root),
                "image_count": dataset.image_count,
                "manifest_sha256": dataset.manifest_sha256,
                "next": "put dataset_path and manifest_sha256 in the settings file",
            }
        )

    _run(action)


@golden_app.command("check")
def golden_check(
    config: Optional[str] = typer.Option(None, help="Path to the settings file."),
):
    """Report the configured golden set's status."""

    def action():
        settings = AutoTrainConfig.load(config)
        status = golden_module.resolve(
            settings.golden.dataset_path, settings.golden.manifest_sha256
        )
        _echo_json(status.to_dict())

    _run(action)


# ---------------------------------------------------------------------------
# Inspection


@app.command()
def status(
    product: str = typer.Option("", help="Override the configured product."),
    area: str = typer.Option("", help="Override the configured area."),
    config: Optional[str] = typer.Option(None, help="Path to the settings file."),
):
    """Show champion, candidates, pool and golden status."""

    def action():
        _echo_json(_service(config, product, area).get_model_health())

    _run(action, quiet=True)


@app.command()
def registry(
    product: str = typer.Option("", help="Override the configured product."),
    area: str = typer.Option("", help="Override the configured area."),
    config: Optional[str] = typer.Option(None, help="Path to the settings file."),
):
    """List the deployed champion and every candidate model."""

    def action():
        _echo_json(_service(config, product, area).get_model_registry())

    _run(action, quiet=True)


@app.command()
def history(
    product: str = typer.Option("", help="Override the configured product."),
    area: str = typer.Option("", help="Override the configured area."),
    config: Optional[str] = typer.Option(None, help="Path to the settings file."),
    limit: int = typer.Option(20, help="How many cycles to list."),
):
    """List past training cycles and their decisions."""

    def action():
        _echo_json(_service(config, product, area).get_training_history(limit=limit))

    _run(action, quiet=True)


@app.command()
def report(
    cycle_id: str = typer.Argument(..., help="Cycle whose report to print."),
    config: Optional[str] = typer.Option(None, help="Path to the settings file."),
):
    """Print a cycle's report."""

    def action():
        paths = AutoTrainPaths.discover()
        path = Path(paths.cycle_dir(cycle_id)) / "report.md"
        if not path.is_file():
            raise AutoTrainError(f"No report for {cycle_id} at {path}")
        typer.echo(path.read_text(encoding="utf-8"))

    _run(action, quiet=True)


if __name__ == "__main__":  # pragma: no cover
    app()

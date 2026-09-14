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
from picture_tool.autotrain.class_schema import (
    resolve_class_schema,
    schema_from_station_config,
)
from picture_tool.autotrain.config import AutoTrainConfig
from picture_tool.autotrain import label_review, review_pack
from picture_tool.autotrain.golden_candidates import read_group_assignments
from picture_tool.autotrain.labeling import (
    export_request,
    import_request,
    request_statistics,
)
from picture_tool.autotrain.orchestrator import TrainingCycle, run_training_cycle
from picture_tool.autotrain.paths import AutoTrainPaths
from picture_tool.autotrain.registry import (
    read_champion,
    read_champion_class_schema,
)
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
    groups: Optional[str] = typer.Option(
        None,
        help=(
            "Candidate report (directory or candidates.csv) whose "
            "representative/hard_case split should be recorded with this set. "
            "Without it the set is registered unsplit."
        ),
    ),
    config: Optional[str] = typer.Option(None, help="Path to the settings file."),
    product: str = typer.Option("", help="Station product."),
    area: str = typer.Option("", help="Station area."),
):
    """Lock a directory as the golden evaluation set."""

    def action():
        # Resolved from the station rather than asked for on the command
        # line: the contract the golden labels must agree with is whatever
        # the deployed model uses, not whatever the person typing remembers.
        service = _service(config, product, area)
        model_dir = service.paths.production_model_dir(
            service.product, service.area
        )
        champion = read_champion(model_dir)
        schema = resolve_class_schema(
            [
                read_champion_class_schema(champion) if champion else None,
                schema_from_station_config(model_dir),
            ],
            context=f"{service.product}/{service.area}",
        )
        assignments = (
            read_group_assignments(groups) if groups else None
        )
        dataset = golden_module.register(
            path,
            registered_by=registered_by,
            class_schema=schema,
            description=description,
            groups=assignments,
            overwrite=overwrite,
        )
        _echo_json(
            {
                "root": str(dataset.root),
                "image_count": dataset.image_count,
                "manifest_sha256": dataset.manifest_sha256,
                "class_schema": schema.to_dict(),
                "groups": dataset.group_counts(),
                # Counted, not assumed: per-group metrics only describe the
                # samples that carry a label, so a set that is mostly
                # ungrouped must say so at the moment it is registered.
                "ungrouped": len(dataset.ungrouped_sample_ids),
                "group_assignments_read": len(assignments) if assignments else 0,
                "next": "put dataset_path and manifest_sha256 in the settings file",
            }
        )

    _run(action)


def _review_context(config, product, area):
    """Everything validation needs about the station, resolved once."""
    service = _service(config, product, area)
    model_dir = service.paths.production_model_dir(service.product, service.area)
    champion = read_champion(model_dir)
    schema = resolve_class_schema(
        [
            read_champion_class_schema(champion) if champion else None,
            schema_from_station_config(model_dir),
        ],
        context=f"{service.product}/{service.area}",
    )
    return service, model_dir, schema


def _expected_counts(model_dir: Path, product: str, area: str) -> dict:
    """The station's expected object multiset, counted.

    ``expected_items`` lists Black twice because the station has two black
    wires. It is not a class schema and is never used as one here --- only
    counted, which is what it is actually for.
    """
    import yaml

    config_path = model_dir / "config.yaml"
    try:
        payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    except (OSError, UnicodeDecodeError, yaml.YAMLError):
        return {}
    items = (payload.get("expected_items") or {}).get(product, {}).get(area)
    if not isinstance(items, list):
        return {}
    counts: dict[str, int] = {}
    for item in items:
        name = str(item).strip()
        if name:
            counts[name] = counts.get(name, 0) + 1
    return counts


def _validate(path, pack, config, product, area):
    """Run a full validation pass. Never reads a cached result."""
    service, model_dir, schema = _review_context(config, product, area)
    pack_samples = label_review.read_pack_samples(pack) if pack else None
    report = label_review.validate_labels(
        path,
        schema=schema,
        expected_counts=_expected_counts(
            model_dir, service.product, service.area
        ),
        pack_samples=pack_samples,
        expected_schema_hash=schema.schema_hash,
    )
    return service, schema, report


@golden_app.command("validate-labels")
def golden_validate_labels(
    path: str = typer.Argument(..., help="Directory holding images/ and labels/."),
    pack: Optional[str] = typer.Option(
        None,
        help="The review pack these images came from. Strongly recommended: "
        "it supplies the group split and is where the contamination checks "
        "were made, and an image absent from it is flagged.",
    ),
    config: Optional[str] = typer.Option(None, help="Path to the settings file."),
    product: str = typer.Option("", help="Station product."),
    area: str = typer.Option("", help="Station area."),
):
    """Check human labels. Validates only --- never registers, never edits."""

    def action():
        _, _, report = _validate(path, pack, config, product, area)
        payload = report.to_dict()
        payload["next"] = (
            "approve the LABELED samples with 'golden approve-labels'; "
            "nothing becomes golden until someone does"
        )
        _echo_json(payload)

    _run(action)


@golden_app.command("approve-labels")
def golden_approve_labels(
    path: str = typer.Argument(..., help="Directory holding images/ and labels/."),
    reviewed_by: str = typer.Option(..., help="Who is approving these."),
    sample_id: Optional[list[str]] = typer.Option(
        None, help="Source image id to approve. Repeatable."
    ),
    group: Optional[str] = typer.Option(
        None, help="Approve every clean sample in this group instead."
    ),
    note: str = typer.Option("", help="Anything the next reader should know."),
    pack: Optional[str] = typer.Option(None, help="The review pack."),
    config: Optional[str] = typer.Option(None, help="Path to the settings file."),
    product: str = typer.Option("", help="Station product."),
    area: str = typer.Option("", help="Station area."),
):
    """Record that a person judged these labels correct."""

    def action():
        _, _, report = _validate(path, pack, config, product, area)
        targets = list(sample_id or [])
        if group:
            targets += [
                s.source_image_id
                for s in report.samples
                if s.group == group and s.state == label_review.LABELED
            ]
        if not targets:
            raise AutoTrainError(
                "Nothing to approve. Pass --sample-id, or --group to take "
                "every sample in a group that validation has already cleared."
            )
        recorded, refused = label_review.record_decision(
            path,
            report,
            source_image_ids=sorted(set(targets)),
            state=label_review.APPROVED,
            reviewed_by=reviewed_by,
            note=note,
        )
        _echo_json({"approved": recorded, "refused": refused})

    _run(action)


@golden_app.command("reject-labels")
def golden_reject_labels(
    path: str = typer.Argument(..., help="Directory holding images/ and labels/."),
    reviewed_by: str = typer.Option(..., help="Who is rejecting these."),
    sample_id: list[str] = typer.Option(..., help="Source image id. Repeatable."),
    note: str = typer.Option("", help="Why."),
    pack: Optional[str] = typer.Option(None, help="The review pack."),
    config: Optional[str] = typer.Option(None, help="Path to the settings file."),
    product: str = typer.Option("", help="Station product."),
    area: str = typer.Option("", help="Station area."),
):
    """Record that a person judged these labels unusable."""

    def action():
        _, _, report = _validate(path, pack, config, product, area)
        recorded, refused = label_review.record_decision(
            path,
            report,
            source_image_ids=sorted(set(sample_id)),
            state=label_review.REJECTED,
            reviewed_by=reviewed_by,
            note=note,
        )
        _echo_json({"rejected": recorded, "refused": refused})

    _run(action)


@golden_app.command("coverage")
def golden_coverage(
    path: str = typer.Argument(..., help="Directory holding images/ and labels/."),
    pack: Optional[str] = typer.Option(None, help="The review pack."),
    config: Optional[str] = typer.Option(None, help="Path to the settings file."),
    product: str = typer.Option("", help="Station product."),
    area: str = typer.Option("", help="Station area."),
):
    """What the currently approved samples would cover as a golden set."""

    def action():
        _, _, report = _validate(path, pack, config, product, area)
        settings = AutoTrainConfig.load(config)
        _echo_json(
            label_review.coverage(
                report.samples,
                groups=list(review_pack.PACK_GROUPS),
                min_group_samples=settings.golden.min_group_samples,
                critical_group=review_pack.RED_ORANGE_CRITICAL,
            )
        )

    _run(action)


@golden_app.command("register-from-review")
def golden_register_from_review(
    path: str = typer.Argument(..., help="The labelling directory."),
    out: str = typer.Option(..., help="Where the golden set is assembled."),
    registered_by: str = typer.Option(..., help="Who is registering this set."),
    pack: Optional[str] = typer.Option(None, help="The review pack."),
    description: str = typer.Option("", help="What this set covers."),
    overwrite: bool = typer.Option(False, help="Replace an existing set."),
    config: Optional[str] = typer.Option(None, help="Path to the settings file."),
    product: str = typer.Option("", help="Station product."),
    area: str = typer.Option("", help="Station area."),
):
    """Assemble and lock a golden set from the approved labels.

    Every check runs again from the files on disk. Nothing is taken from a
    previous validation: an approval recorded yesterday says a person judged
    those bytes, not that the bytes are still there or still clean, and the
    gap between the two is where a contaminated or half-finished sample
    would get in.

    Partial sets are allowed on purpose --- waiting for all 250 images before
    any evaluation is possible would stall everything --- so the coverage
    report says which groups are still INSUFFICIENT rather than describing a
    thin set as complete.
    """

    def action():
        _, schema, report = _validate(path, pack, config, product, area)
        eligible = report.eligible()
        if not eligible:
            raise AutoTrainError(
                "No sample is both approved and clean. Run "
                "'golden validate-labels' to see what each one is waiting for."
            )
        staged, groups = label_review.stage_approved(
            report, out, schema=schema, overwrite=overwrite
        )
        dataset = golden_module.register(
            staged,
            registered_by=registered_by,
            class_schema=schema,
            description=description,
            groups=groups,
            overwrite=overwrite,
        )
        settings = AutoTrainConfig.load(config)
        report_payload = label_review.coverage(
            report.samples,
            groups=list(review_pack.PACK_GROUPS),
            min_group_samples=settings.golden.min_group_samples,
            critical_group=review_pack.RED_ORANGE_CRITICAL,
        )
        (staged / "coverage.json").write_text(
            json.dumps(report_payload, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        _echo_json(
            {
                "root": str(dataset.root),
                "image_count": dataset.image_count,
                "manifest_sha256": dataset.manifest_sha256,
                "groups": dataset.group_counts(),
                "ungrouped": len(dataset.ungrouped_sample_ids),
                "coverage": report_payload,
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

import logging
from types import SimpleNamespace

from picture_tool.config_loader import load_config_if_updated
from picture_tool.main_pipeline import run_pipeline
from picture_tool.pipeline.core import Task
from picture_tool.tasks.deployment_target import resolve_yolo_deployment_target


def test_run_pipeline_invokes_shared_task_callbacks(monkeypatch):
    """GUI and CLI should share the same runner hooks."""
    import picture_tool.main_pipeline as main_pipeline

    events: list[str] = []

    monkeypatch.setattr(
        main_pipeline,
        "build_task_registry",
        lambda config: {
            "demo": Task(
                name="demo",
                run=lambda config, args: events.append(
                    f"run:{config['yolo_training']['name']}"
                ),
            )
        },
    )
    monkeypatch.setattr(main_pipeline, "validate_config_schema", lambda cfg, **_: cfg)
    monkeypatch.setattr(main_pipeline, "_auto_device", lambda cfg, logger: None)

    run_pipeline(
        ["demo"],
        {"yolo_training": {"name": "train"}},
        logging.getLogger("test"),
        SimpleNamespace(config="", force=False, product=None, name="PCBA1"),
        on_task_start=lambda task: events.append(f"start:{task.name}"),
        on_task_complete=lambda task, index, total: events.append(
            f"done:{task.name}:{index}:{total}"
        ),
        reload_config=False,
    )

    assert events == ["start:demo", "run:PCBA1", "done:demo:0:1"]


def test_load_config_if_updated_tracks_each_config_path_independently(
    tmp_path, monkeypatch
):
    """Hot reload state should not leak between different config files."""
    logger = logging.getLogger("test")
    first = tmp_path / "first.yaml"
    second = tmp_path / "second.yaml"
    first.write_text("value: 1\n", encoding="utf-8")
    second.write_text("value: 10\n", encoding="utf-8")

    if hasattr(load_config_if_updated, "_last_mtimes"):
        monkeypatch.setattr(load_config_if_updated, "_last_mtimes", {})

    assert load_config_if_updated(first, {"value": 1}, logger) == {"value": 1}
    assert load_config_if_updated(second, {"value": 10}, logger) == {"value": 10}

    first.write_text("value: 2\n", encoding="utf-8")
    second.write_text("value: 20\n", encoding="utf-8")
    future_mtime = first.stat().st_mtime + 5
    import os

    os.utime(first, (future_mtime, future_mtime))
    os.utime(second, (future_mtime, future_mtime))

    assert load_config_if_updated(first, {"value": 1}, logger) == {"value": 2}
    assert load_config_if_updated(second, {"value": 10}, logger) == {"value": 20}


def test_resolve_yolo_deployment_target_prefers_gui_product_override():
    """A GUI product override should consistently affect deploy and bundle."""
    config = {
        "yolo_training": {
            "name": "train",
            "deploy": {"product": "OLD", "area": "A"},
            "artifact_bundle": {"product": "BUNDLE", "area": "C"},
        }
    }

    target = resolve_yolo_deployment_target(
        config,
        SimpleNamespace(product="PCBA1,B"),
        artifact_config_key="artifact_bundle",
    )

    assert target.product == "PCBA1"
    assert target.area == "B"


def test_resolve_yolo_deployment_target_uses_artifact_then_deploy_fallback():
    """Bundle-specific config should win before deploy defaults."""
    config = {
        "yolo_training": {
            "name": "train",
            "deploy": {"product": "DEPLOY", "area": "A"},
            "artifact_bundle": {"product": "BUNDLE", "area": "C"},
        }
    }

    bundle_target = resolve_yolo_deployment_target(
        config, SimpleNamespace(product=None), artifact_config_key="artifact_bundle"
    )
    deploy_target = resolve_yolo_deployment_target(
        config, SimpleNamespace(product=None)
    )

    assert bundle_target.product == "BUNDLE"
    assert bundle_target.area == "C"
    assert deploy_target.product == "DEPLOY"
    assert deploy_target.area == "A"


def test_bundle_and_deploy_tasks_are_owned_by_their_modules():
    """Registry ownership should match task implementation modules."""
    from picture_tool.main_pipeline import build_task_registry
    from picture_tool.tasks import bundle, deploy, training

    registry = build_task_registry({})

    assert registry["artifact_bundle"] in bundle.TASKS
    assert registry["deploy"] in deploy.TASKS
    assert all(task.name != "artifact_bundle" for task in training.TASKS)
    assert all(task.name != "deploy" for task in training.TASKS)

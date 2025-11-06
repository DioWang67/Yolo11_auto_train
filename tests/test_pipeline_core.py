import logging
from types import SimpleNamespace

import pytest
import yaml

import picture_tool.main_pipeline as pipeline
from picture_tool.train import yolo_trainer


class DummyLogger:
    def __init__(self):
        self.records: list[tuple[str, str]] = []

    def info(self, message):
        self.records.append(("info", str(message)))

    def warning(self, message):
        self.records.append(("warning", str(message)))

    def error(self, message):
        self.records.append(("error", str(message)))

    def __getattr__(self, name):
        # pipeline.main_pipeline may call logger.debug; provide no-op.
        def noop(*args, **kwargs):
            return None

        return noop


def _make_args(config_path, **overrides):
    base = dict(
        config=str(config_path),
        force=False,
        device=None,
        epochs=None,
        imgsz=None,
        batch=None,
        model=None,
        project=None,
        name=None,
        weights=None,
        infer_input=None,
        infer_output=None,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


@pytest.fixture()
def minimal_config(tmp_path):
    cfg = {
        "pipeline": {
            "log_file": str(tmp_path / "logs" / "pipeline.log"),
            "tasks": [
                {"name": "dataset_splitter", "enabled": True, "dependencies": []},
                {
                    "name": "yolo_train",
                    "enabled": True,
                    "dependencies": ["dataset_splitter"],
                },
            ],
            "task_groups": {"train": ["dataset_splitter", "yolo_train"]},
        },
        "train_test_split": {
            "input": {
                "image_dir": str(tmp_path / "images"),
                "label_dir": str(tmp_path / "labels"),
            },
            "output": {"output_dir": str(tmp_path / "split")},
        },
        "yolo_training": {
            "dataset_dir": str(tmp_path / "split"),
            "class_names": ["class"],
            "model": "dummy.pt",
            "epochs": 1,
            "imgsz": 32,
            "batch": 1,
            "device": "cpu",
            "project": str(tmp_path / "runs"),
            "name": "test",
        },
    }
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg, allow_unicode=True), encoding="utf-8")
    return cfg, cfg_path


def test_validate_dependencies_injects_missing(minimal_config):
    cfg, _ = minimal_config
    logger = DummyLogger()
    ordered = pipeline.validate_dependencies(["yolo_train"], cfg, logger)
    assert ordered == ["dataset_splitter", "yolo_train"]
    assert any(level == "info" for level, _ in logger.records)


def test_run_pipeline_executes_in_declared_order(monkeypatch, minimal_config):
    cfg, cfg_path = minimal_config
    args = _make_args(cfg_path)
    logger = DummyLogger()
    calls = []

    monkeypatch.setattr(
        pipeline, "load_config_if_updated", lambda path, config, lg: config
    )
    monkeypatch.setattr(pipeline, "_apply_cli_overrides", lambda config, args, lg: None)
    monkeypatch.setattr(pipeline, "_auto_device", lambda config, lg: None)
    monkeypatch.setattr(pipeline, "_should_skip", lambda task, config, args, lg: None)

    def handler_factory(name):
        def _handler(config, args):
            calls.append(name)

        return _handler

    monkeypatch.setitem(
        pipeline.TASK_HANDLERS, "dataset_splitter", handler_factory("dataset_splitter")
    )
    monkeypatch.setitem(
        pipeline.TASK_HANDLERS, "yolo_train", handler_factory("yolo_train")
    )

    pipeline.run_pipeline(["yolo_train"], cfg, logger, args)

    assert calls == ["dataset_splitter", "yolo_train"]


def test_run_pipeline_honours_stop_event(monkeypatch, minimal_config):
    cfg, cfg_path = minimal_config
    args = _make_args(cfg_path)
    logger = DummyLogger()
    calls = []

    class StopEvent:
        def __init__(self):
            self.count = 0

        def is_set(self):
            self.count += 1
            return self.count > 1

    stop_event = StopEvent()

    monkeypatch.setattr(
        pipeline, "load_config_if_updated", lambda path, config, lg: config
    )
    monkeypatch.setattr(pipeline, "_apply_cli_overrides", lambda config, args, lg: None)
    monkeypatch.setattr(pipeline, "_auto_device", lambda config, lg: None)
    monkeypatch.setattr(pipeline, "_should_skip", lambda task, config, args, lg: None)

    def dataset_handler(config, args):
        calls.append("dataset_splitter")

    def train_handler(config, args):
        pytest.fail("Pipeline should have stopped before yolo_train executed")

    monkeypatch.setitem(pipeline.TASK_HANDLERS, "dataset_splitter", dataset_handler)
    monkeypatch.setitem(pipeline.TASK_HANDLERS, "yolo_train", train_handler)

    pipeline.run_pipeline(["yolo_train"], cfg, logger, args, stop_event=stop_event)

    assert calls == ["dataset_splitter"]


def test_get_tasks_from_groups_union(minimal_config):
    cfg, _ = minimal_config
    tasks = pipeline.get_tasks_from_groups(["train", "nonexistent"], cfg)
    assert set(tasks) == {"dataset_splitter", "yolo_train"}


def test_train_yolo_requires_class_names(tmp_path):
    bad_config = {
        "yolo_training": {"dataset_dir": str(tmp_path / "dataset"), "class_names": []}
    }
    logger = logging.getLogger("test")
    with pytest.raises(ValueError):
        yolo_trainer.train_yolo(bad_config, logger=logger)

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
            # 'tasks' list is used for enabling/disabling, but not for defining dependencies in new arch
            "task_groups": {
                 "train": ["dataset_splitter", "yolo_train"]
            }
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
    
    # Mock task registry
    from picture_tool.pipeline.core import Task
    
    def splitter(c, a): calls.append("dataset_splitter")
    def trainer(c, a): calls.append("yolo_train")
    
    tasks = {
        "dataset_splitter": Task("dataset_splitter", splitter),
        "yolo_train": Task("yolo_train", trainer, dependencies=["dataset_splitter"])
    }
    
    monkeypatch.setattr(pipeline, "build_task_registry", lambda c: tasks)

    # Note: run_pipeline now expects 'validate_dependencies' to pass through user requests.
    # The 'auto-dependency resolving' happens in Pipeline.run, so we just ask for 'yolo_train'.
    
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

    def dataset_handler(config, args):
        calls.append("dataset_splitter")

    def train_handler(config, args):
        pytest.fail("Pipeline should have stopped before yolo_train executed")
                
    from picture_tool.pipeline.core import Task
    tasks = {
        "dataset_splitter": Task("dataset_splitter", dataset_handler),
        # yolo_train depends on splitter, so splitter runs first. StopEvent triggers after splitter?
        # StopEvent logic is checked inside the loop.
        # Loop order: splitter (checked Stop?), train (checked Stop?).
        "yolo_train": Task("yolo_train", train_handler, dependencies=["dataset_splitter"])
    }
    monkeypatch.setattr(pipeline, "build_task_registry", lambda c: tasks)

    pipeline.run_pipeline(["yolo_train"], cfg, logger, args, stop_event=stop_event)

    assert calls == ["dataset_splitter"]


def test_get_tasks_from_groups_union(minimal_config):
    cfg, _ = minimal_config
    # Need to populate task_groups in config for this test
    # (minimal_config fixture sets it up, but let's double check coverage of function)
    assert "train" in cfg["pipeline"]["task_groups"]
    tasks = pipeline.get_tasks_from_groups(["train", "nonexistent"], cfg)
    assert set(tasks) == {"dataset_splitter", "yolo_train"}


def test_train_yolo_requires_class_names(tmp_path):
    bad_config = {
        "yolo_training": {"dataset_dir": str(tmp_path / "dataset"), "class_names": []}
    }
    logger = logging.getLogger("test")
    with pytest.raises(ValueError):
        yolo_trainer.train_yolo(bad_config, logger=logger)

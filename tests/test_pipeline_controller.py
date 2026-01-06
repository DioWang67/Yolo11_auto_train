from types import SimpleNamespace
from typing import Any

import pytest

from picture_tool.gui import pipeline_controller


class DummyLineEdit:
    def __init__(self, value: str = ""):
        self._value = value

    def text(self) -> str:
        return self._value

    def setText(self, value: str) -> None:
        self._value = value


class DummyButton:
    def __init__(self):
        self.enabled = True

    def setEnabled(self, value: bool) -> None:
        self.enabled = value


class DummyProgressBar:
    def __init__(self):
        self.value = 0

    def setValue(self, value: int) -> None:
        self.value = value


class DummyLabel:
    def __init__(self):
        self.text = ""
        self.style = ""

    def setText(self, value: str) -> None:
        self.text = value

    def setStyleSheet(self, value: str) -> None:
        self.style = value


class DummyTextEdit:
    def __init__(self):
        self.text = ""

    def setPlainText(self, value: str) -> None:
        self.text = value


class DummyCheckbox:
    def __init__(self, checked: bool = False):
        self.checked = checked

    def setChecked(self, checked: bool) -> None:
        self.checked = checked

    def isChecked(self) -> bool:
        return self.checked


class DummyController(pipeline_controller.PipelineControllerMixin):
    POSITION_TASK_LABEL = "位置檢查"
    YOLO_TRAIN_LABEL = "YOLO訓練"

    def __init__(self, tmp_path):
        self.logs = []
        self._init_pipeline_controller()
        self.config_path_edit = DummyLineEdit()
        self.start_btn = DummyButton()
        self.stop_btn = DummyButton()
        self.progress_bar = DummyProgressBar()
        self.status_label = DummyLabel()
        self.config_text = DummyTextEdit()
        self.task_checkboxes = {}
        self._last_tasks_reset = None
        self._tmp_path = tmp_path

    # helpers required by mixin -----------------------------------------
    def log_message(self, message: str) -> None:
        self.logs.append(message)

    def _populate_position_widgets(self) -> None:
        pass

    def _apply_position_settings(self) -> None:
        pass

    def _load_preset_storage(self) -> None:
        pass

    def _update_preset_display(self) -> None:
        pass

    def reset_task_statuses(self, tasks):
        self._last_tasks_reset = list(tasks)

    def _validate_pipeline_configuration(self, tasks):
        return []

    def _set_task_status(self, task: str, message: str, color: Any = None) -> None:
        pass

    def refresh_metrics_dashboard(self):
        pass


@pytest.fixture()
def controller(tmp_path):
    return DummyController(tmp_path)


def test_load_default_config_reads_packaged_file(controller, tmp_path, monkeypatch):
    default_cfg = tmp_path / "config.yaml"
    default_cfg.write_text(
        "pipeline:\n  log_file: logs/pipeline.log\n  tasks:\n    - name: dataset_splitter\n      enabled: true\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        pipeline_controller.PipelineControllerMixin,
        "_default_config_path",
        lambda self: default_cfg,
    )

    controller.load_default_config()

    assert controller.config["pipeline"]["tasks"][0]["name"] == "dataset_splitter"
    assert controller.config_path_edit.text() == str(default_cfg.resolve())


def test_load_config_missing_file_falls_back(controller, tmp_path, monkeypatch):
    fallback = tmp_path / "fallback.yaml"
    fallback.write_text(
        "pipeline: {log_file: logs/pipeline.log, tasks: []}\n", encoding="utf-8"
    )
    monkeypatch.setattr(
        pipeline_controller.PipelineControllerMixin,
        "_default_config_path",
        lambda self: fallback,
    )

    controller.config_path_edit.setText(str(tmp_path / "missing.yaml"))
    controller.load_config()

    assert controller.config["pipeline"]["tasks"] == []
    assert any("Config file not found" in log for log in controller.logs)


def test_start_pipeline_creates_worker(monkeypatch, controller, tmp_path):
    config = {
        "pipeline": {
            "log_file": "logs/pipeline.log",
            "tasks": [
                {"name": "dataset_splitter", "enabled": True, "dependencies": []},
                {
                    "name": "yolo_train",
                    "enabled": True,
                    "dependencies": ["dataset_splitter"],
                },
            ],
        },
        "train_test_split": {
            "input": {"image_dir": "data/in/images", "label_dir": "data/in/labels"},
            "output": {"output_dir": "data/out"},
        },
        "yolo_training": {
            "dataset_dir": "data/out",
            "class_names": ["a"],
            "model": "model.pt",
            "epochs": 1,
            "imgsz": 32,
            "batch": 1,
            "device": "cpu",
            "project": str(tmp_path / "runs"),
            "name": "test",
        },
    }
    controller.config = config
    controller.config_path_edit.setText(str(tmp_path / "config.yaml"))
    controller.task_checkboxes = {
        "dataset_splitter": DummyCheckbox(True),
        "yolo_train": DummyCheckbox(True),
    }

    created = SimpleNamespace(start_called=False)

    class DummySignal:
        def __init__(self):
            self._callbacks = []

        def connect(self, callback):
            self._callbacks.append(callback)

        def emit(self, *args, **kwargs):
            for cb in self._callbacks:
                cb(*args, **kwargs)

    class DummyThread:
        def __init__(self, tasks, cfg, cfg_path):
            created.tasks = tasks
            created.cfg = cfg
            created.cfg_path = cfg_path
            # Define signals as instance attributes
            self.progress_updated = DummySignal()
            self.task_started = DummySignal()
            self.task_completed = DummySignal()
            self.log_message = DummySignal()
            self.finished_signal = DummySignal()
            self.error_occurred = DummySignal()

        def start(self):
            created.start_called = True

    def _factory(tasks, cfg, cfg_path, **kwargs):
        return DummyThread(tasks, cfg, cfg_path)

    monkeypatch.setattr(pipeline_controller, "WorkerThread", _factory)

    controller.start_pipeline()

    assert created.tasks == ["dataset_splitter", "yolo_train"]
    assert created.cfg_path == str(tmp_path / "config.yaml")
    assert created.cfg == controller.config
    assert controller.start_btn.enabled is False
    assert controller.stop_btn.enabled is True
    assert controller.progress_bar.value == 0

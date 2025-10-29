from types import SimpleNamespace

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

    def _set_task_status(self, task, message, color):
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
    assert any("不存在" in log for log in controller.logs)


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

    class DummyThread:
        def __init__(self, tasks, cfg, cfg_path):
            created.tasks = tasks
            created.cfg = cfg
            created.cfg_path = cfg_path

        def start(self):
            created.start_called = True

        def progress_updated(self):
            pass

        def task_started(self, *args, **kwargs):
            pass

        def task_completed(self, *args, **kwargs):
            pass

        def log_message(self, *args, **kwargs):
            pass

        def finished_signal(self, *args, **kwargs):
            pass

        def error_occurred(self, *args, **kwargs):
            pass

        # mimic Qt connect pattern
        def __getattr__(self, item):
            if item.endswith("connect"):
                return lambda callback: None
            return super().__getattribute__(item)

    def _factory(tasks, cfg, cfg_path):
        thread = DummyThread(tasks, cfg, cfg_path)

        # Provide connect callables
        thread.progress_updated = SimpleNamespace(connect=lambda cb: None)
        thread.task_started = SimpleNamespace(connect=lambda cb: None)
        thread.task_completed = SimpleNamespace(connect=lambda cb: None)
        thread.log_message = SimpleNamespace(connect=lambda cb: None)
        thread.finished_signal = SimpleNamespace(connect=lambda cb: None)
        thread.error_occurred = SimpleNamespace(connect=lambda cb: None)

        return thread

    monkeypatch.setattr(pipeline_controller, "WorkerThread", _factory)

    controller.start_pipeline()

    assert created.tasks == ["dataset_splitter", "yolo_train"]
    assert created.cfg_path == str(tmp_path / "config.yaml")
    assert created.cfg is controller.config
    assert controller.start_btn.enabled is False
    assert controller.stop_btn.enabled is True
    assert controller.progress_bar.value == 0

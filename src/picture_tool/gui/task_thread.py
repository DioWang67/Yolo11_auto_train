"""Background worker that executes pipeline tasks without blocking the UI."""

from __future__ import annotations

import copy
import logging
import threading
from pathlib import Path
from types import SimpleNamespace
from typing import Iterable, Optional

import yaml
from PyQt5.QtCore import QThread, pyqtSignal

from picture_tool import main_pipeline as pipeline
from picture_tool.exceptions import ConfigurationError, PipelineError
from picture_tool.pipeline.core import Pipeline, Task


class _SignalLoggingHandler(logging.Handler):
    """Logging handler that forwards records through a Qt signal."""

    def __init__(self, emit_callback):
        super().__init__()
        self._emit_callback = emit_callback

    def emit(
        self, record: logging.LogRecord
    ) -> None:  # pragma: no cover - thin wrapper
        try:
            message = self.format(record)
        except (ValueError, TypeError, AttributeError):
            message = record.getMessage()
        self._emit_callback(message)


class WorkerThread(QThread):
    """Run a subset of pipeline tasks on a background thread.

    Delegates execution to :class:`Pipeline.run` so that dependency
    resolution, skip logic, and task ordering are defined in exactly
    one place.  Progress is reported via Qt signals through the
    ``before_task`` and ``after_task`` hooks.
    """

    progress_updated = pyqtSignal(int)
    task_started = pyqtSignal(str)
    task_completed = pyqtSignal(str)
    log_message = pyqtSignal(str)
    finished_signal = pyqtSignal()
    error_occurred = pyqtSignal(str)

    def __init__(
        self,
        tasks: Iterable[str],
        config: dict,
        config_path: Optional[str] = None,
        product: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.tasks = list(tasks)
        self.config = copy.deepcopy(config)
        self.config_path = config_path
        self.product = product
        self._cancel_requested = False
        self.stop_event = threading.Event()

    def request_stop(self) -> None:
        self._cancel_requested = True
        self.stop_event.set()

    # ------------------------------------------------------------------
    # QThread implementation
    # ------------------------------------------------------------------
    def run(self) -> None:  # pragma: no cover - exercised via GUI, hard to unit test
        logger = logging.getLogger(f"picture_tool.gui.worker.{id(self)}")
        logger.setLevel(logging.INFO)
        logger.propagate = True

        base_logger = logging.getLogger("picture_tool")
        yolo_logger = logging.getLogger("ultralytics")

        handler = _SignalLoggingHandler(self.log_message.emit)
        handler.setFormatter(
            logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        )
        base_logger.addHandler(handler)
        yolo_logger.addHandler(handler)

        file_handler = None
        try:
            config = self._resolve_config()
            args = self._build_args()

            registry = pipeline.build_task_registry(config)

            log_file = config.get("pipeline", {}).get("log_file", "logs/pipeline.log")
            try:
                log_path = Path(log_file)
                log_path.parent.mkdir(parents=True, exist_ok=True)
                file_handler = logging.FileHandler(log_path, encoding="utf-8")
                file_handler.setFormatter(
                    logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
                )
                base_logger.addHandler(file_handler)
                yolo_logger.addHandler(file_handler)
            except OSError as e:
                logger.warning(f"Failed to setup file logging: {e}")

            # --- before_task hook: apply overrides + emit signal ---
            def _before_task(task_obj: Task, cfg: dict) -> dict:
                if self._cancel_requested:
                    self.stop_event.set()
                self.task_started.emit(task_obj.name)
                cfg = pipeline._apply_cli_overrides(cfg, args, logger)
                pipeline._auto_device(cfg, logger)
                return cfg

            # --- after_task hook: emit progress + completion signal ---
            def _after_task(task_obj: Task, index: int, total: int) -> None:
                self.task_completed.emit(task_obj.name)
                self._emit_progress(index + 1, max(total, 1))

            pipe = Pipeline(registry, logger=logger)
            pipe.run(
                self.tasks,
                config,
                args,
                before_task=_before_task,
                after_task=_after_task,
            )

            if not self._cancel_requested:
                logger.info("All tasks completed.")
            self.finished_signal.emit()
        except (PipelineError, RuntimeError, OSError) as exc:  # pragma: no cover
            logger.exception(f"Pipeline execution failed: {exc}")
            self.error_occurred.emit(str(exc))
        finally:
            base_logger.removeHandler(handler)
            yolo_logger.removeHandler(handler)
            if file_handler:
                base_logger.removeHandler(file_handler)
                yolo_logger.removeHandler(file_handler)
                file_handler.close()

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    def _resolve_config(self) -> dict:
        if self.config:
            return self.config
        if self.config_path:
            try:
                return pipeline.load_config(self.config_path)
            except (ConfigurationError, OSError, yaml.YAMLError):
                pass
        return pipeline.load_config()

    def _build_args(self) -> SimpleNamespace:
        return SimpleNamespace(
            config=self.config_path or "",
            tasks=self.tasks,
            exclude_tasks=None,
            task_groups=None,
            interactive=False,
            input_format=None,
            output_format=None,
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
            product=self.product,
            stop_event=self.stop_event,
        )

    def _emit_progress(self, completed: int, total: int) -> None:
        percent = int((completed / total) * 100)
        self.progress_updated.emit(percent)

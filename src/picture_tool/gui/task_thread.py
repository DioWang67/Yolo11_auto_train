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
    """Run a subset of pipeline tasks on a background thread."""

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
        # Local logger for thread-level messages
        logger = logging.getLogger(f"picture_tool.gui.worker.{id(self)}")
        logger.setLevel(logging.INFO)
        logger.propagate = True  # Ensure messages bubble up to 'picture_tool'

        # Attach signal handler to the root 'picture_tool' logger so we capture
        # logs emitted by submodules (like train.yolo_trainer, quality.dataset_linter)
        base_logger = logging.getLogger("picture_tool")
        yolo_logger = logging.getLogger("ultralytics")
        
        handler = _SignalLoggingHandler(self.log_message.emit)
        handler.setFormatter(
            logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        )
        base_logger.addHandler(handler)
        yolo_logger.addHandler(handler)

        try:
            config = self._resolve_config()
            args = self._build_args()

            # Using build_task_registry to get Task objects
            registry = pipeline.build_task_registry(config)

            # Ensure logging to file is set up for GUI runs too
            log_file = config.get("pipeline", {}).get("log_file", "logs/pipeline.log")
            
            # Attaching a FileHandler to the base_logger ensures persistence for all module logs
            file_handler = None
            try:
                log_path = Path(log_file)
                log_path.parent.mkdir(parents=True, exist_ok=True)
                file_handler = logging.FileHandler(log_path, encoding='utf-8')
                file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
                base_logger.addHandler(file_handler)
                yolo_logger.addHandler(file_handler)
            except OSError as e:
                logger.warning(f"Failed to setup file logging: {e}")

            # Using Pipeline core to resolve dependencies correctly
            pipe_core = pipeline.Pipeline(registry, logger)
            collected = pipe_core._collect(self.tasks)
            # The original logic expected `validate_dependencies` to return a list,
            # but now we should trust Pipeline topological sort if possible.
            # However, to maintain incremental signal emission, we iterate manually on the sorted tasks.
            ordered_tasks = pipe_core._toposort(collected)

            planned_tasks = [t.name for t in ordered_tasks]
            total = max(len(planned_tasks), 1)

            for index, task_obj in enumerate(ordered_tasks):
                task_name = task_obj.name
                if self._cancel_requested:
                    logger.info("Pipeline cancelled by user.")
                    break

                self.task_started.emit(task_name)

                # Apply overrides before each task (mimicking Pipeline logic)
                pipeline._apply_cli_overrides(config, args, logger)  # type: ignore[attr-defined]
                pipeline._auto_device(config, logger)  # type: ignore[attr-defined]

                skip_reason = None
                if task_obj.skip_fn:
                    try:
                        skip_reason = task_obj.skip_fn(config, args)
                    except (TypeError, AttributeError, RuntimeError) as exc:
                        logger.warning(f"Skip check for {task_name} failed: {exc}")

                if skip_reason:
                    logger.info(f"Skipping {task_name}: {skip_reason}")
                    self.task_completed.emit(task_name)
                    self._emit_progress(index + 1, total)
                    continue

                logger.info(f"Running task: {task_name}")
                task_obj.run(config, args)
                logger.info(f"Finished task {task_name}")

                self.task_completed.emit(task_name)
                self._emit_progress(index + 1, total)

            if not self._cancel_requested:
                logger.info("All tasks completed.")
            self.finished_signal.emit()
        except (PipelineError, RuntimeError, OSError) as exc:  # pragma: no cover - defensive guard
            logger.exception(f"Pipeline execution failed: {exc}")
            self.error_occurred.emit(str(exc))
        finally:
            base_logger.removeHandler(handler)
            yolo_logger.removeHandler(handler)
            if file_handler:
                base_logger.removeHandler(file_handler)
                yolo_logger.removeHandler(file_handler)
                file_handler.close()  # Fix: Ensure file handle is closed to avoid Windows locking

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

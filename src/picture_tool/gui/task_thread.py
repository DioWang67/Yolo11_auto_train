"""Background worker that executes pipeline tasks without blocking the UI."""

from __future__ import annotations

import copy
import logging
from types import SimpleNamespace
from typing import Iterable, Optional

from PyQt5.QtCore import QThread, pyqtSignal

from picture_tool import main_pipeline as pipeline


class _SignalLoggingHandler(logging.Handler):
    """Logging handler that forwards records through a Qt signal."""

    def __init__(self, emit_callback):
        super().__init__()
        self._emit_callback = emit_callback

    def emit(self, record: logging.LogRecord) -> None:  # pragma: no cover - thin wrapper
        try:
            message = self.format(record)
        except Exception:
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
    ) -> None:
        super().__init__()
        self.tasks = list(tasks)
        self.config = copy.deepcopy(config)
        self.config_path = config_path
        self._cancel_requested = False

    def request_stop(self) -> None:
        self._cancel_requested = True

    # ------------------------------------------------------------------
    # QThread implementation
    # ------------------------------------------------------------------
    def run(self) -> None:  # pragma: no cover - exercised via GUI, hard to unit test
        logger = logging.getLogger(f"picture_tool.gui.worker.{id(self)}")
        logger.setLevel(logging.INFO)
        handler = _SignalLoggingHandler(self.log_message.emit)
        handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
        logger.addHandler(handler)
        logger.propagate = False

        try:
            config = self._resolve_config()
            args = self._build_args()

            planned_tasks = pipeline.validate_dependencies(self.tasks, config, logger)
            total = max(len(planned_tasks), 1)

            for index, task in enumerate(planned_tasks):
                if self._cancel_requested:
                    logger.info("Pipeline cancelled by user.")
                    break

                self.task_started.emit(task)
                pipeline._apply_cli_overrides(config, args, logger)  # type: ignore[attr-defined]
                pipeline._auto_device(config, logger)  # type: ignore[attr-defined]
                skip_reason = pipeline._should_skip(  # type: ignore[attr-defined]
                    task, config, args, logger
                )
                if skip_reason:
                    logger.info(f"Skipping {task}: {skip_reason}")
                    self.task_completed.emit(task)
                    self._emit_progress(index + 1, total)
                    continue

                handler_fn = pipeline.TASK_HANDLERS.get(task)
                if handler_fn is None:
                    logger.warning(f"No handler registered for task {task}")
                else:
                    handler_fn(config, args)
                    logger.info(f"Finished task {task}")
                self.task_completed.emit(task)
                self._emit_progress(index + 1, total)

            if not self._cancel_requested:
                logger.info("All tasks completed.")
            self.finished_signal.emit()
        except Exception as exc:  # pragma: no cover - defensive guard
            logger.exception(f"Pipeline execution failed: {exc}")
            self.error_occurred.emit(str(exc))
        finally:
            logger.removeHandler(handler)

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    def _resolve_config(self) -> dict:
        if self.config:
            return self.config
        if self.config_path:
            try:
                return pipeline.load_config(self.config_path)
            except Exception:
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
        )

    def _emit_progress(self, completed: int, total: int) -> None:
        percent = int((completed / total) * 100)
        self.progress_updated.emit(percent)

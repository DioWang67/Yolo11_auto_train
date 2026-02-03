"""
Pipeline Manager module.

This module provides the `PipelineManager` class, which acts as the controller
for the Picture Tool's processing pipeline. It manages configuration,
worker thread execution, and status updates via Qt signals, decoupling
logic from the GUI presentation layer.
"""

from __future__ import annotations

import copy
import logging
from pathlib import Path
from typing import Any, List, Optional, Union

import yaml
from PyQt5.QtCore import QObject, pyqtSignal

from picture_tool.gui.task_thread import WorkerThread


class PipelineManager(QObject):
    """
    Manages the execution of the processing pipeline and configuration state.

    Signals:
        config_loaded (Path, dict): Emitted when a configuration is successfully loaded.
        config_saved (Path): Emitted when configuration is saved.
        task_started (str): Emitted when a specific task starts.
        task_completed (str): Emitted when a specific task completes.
        progress_updated (int): Emitted with progress percentage (0-100).
        log_message (str): Emitted for general logging messages.
        pipeline_finished: Emitted when the entire pipeline sequence finishes.
        error_occurred (str): Emitted when a critical error stops the pipeline.
        status_message (str, str): Emitted to update status bar/labels (message, context).
    """

    config_loaded = pyqtSignal(Path, dict)
    config_saved = pyqtSignal(Path)
    task_started = pyqtSignal(str)
    task_completed = pyqtSignal(str)
    progress_updated = pyqtSignal(int)
    log_message = pyqtSignal(str)
    pipeline_finished = pyqtSignal()
    error_occurred = pyqtSignal(str)
    status_message = pyqtSignal(str, str)  # msg, context (e.g., "info", "error")

    def __init__(self, parent: Optional[QObject] = None):
        super().__init__(parent)
        self.config: dict[str, Any] = {}
        self.current_config_path: Optional[Path] = None
        self.worker_thread: Optional[WorkerThread] = None
        self._logger = logging.getLogger("picture_tool.gui.manager")

    # ------------------------------------------------------------------
    # Configuration Management
    # ------------------------------------------------------------------
    def default_config_path(self) -> Path:
        """Return the best-effort default configuration path."""
        cwd = Path.cwd()
        candidates = [
            cwd / "configs" / "default_pipeline.yaml",
            cwd / "config.yaml",
            Path(__file__).resolve().parent.parent
            / "resources"
            / "default_pipeline.yaml",
            Path(__file__).resolve().parent.parent / "preset_config.yaml",
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate
        return candidates[-1]

    def load_config(self, path: Optional[Union[str, Path]] = None) -> bool:
        """
        Load configuration from the given path. If None, tries default paths.

        Args:
            path: Path to the config file (str or Path object).

        Returns:
            bool: True if loading was successful, False otherwise.
        """
        target: Optional[Path] = None
        if path:
            target = Path(path)
        else:
            target = self.default_config_path()

        try:
            cfg = self._load_config_file(target)
            if not cfg:
                self.log_message.emit(
                    f"Config file not found or invalid: {target}. Using fallback."
                )
                cfg = self._fallback_config()
                # Note: We don't update current_config_path to a missing file if we fallback
            else:
                self.current_config_path = target

            self.config = cfg
            # Emit signal so UI can update (e.g., text editor content, path display)
            self.config_loaded.emit(
                self.current_config_path or Path("fallback"), self.config
            )
            return True

        except (OSError, yaml.YAMLError) as e:
            self.error_occurred.emit(f"Failed to load config: {e}")
            return False

    def save_config(self, path: Optional[Union[str, Path]] = None) -> bool:
        """
        Save the current configuration to a file.

        Args:
            path: Target path. If None, uses self.current_config_path.
        """
        target = Path(path) if path else self.current_config_path
        if not target:
            self.error_occurred.emit("No save path specified.")
            return False

        try:
            # Ensure parent directory exists
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("w", encoding="utf-8") as handle:
                yaml.safe_dump(self.config, handle, allow_unicode=True, sort_keys=False)

            self.current_config_path = target
            self.log_message.emit(f"Configuration saved to {target}")
            self.config_saved.emit(target)
            return True

        except (OSError, yaml.YAMLError) as exc:
            self.error_occurred.emit(f"Could not save config: {exc}")
            return False

    def update_config(self, new_config: dict[str, Any]) -> None:
        """Update the internal config dictionary (e.g. from UI editor)."""
        self.config = new_config

    def _load_config_file(self, path: Path) -> Optional[dict[str, Any]]:
        if not path.exists():
            return None
        try:
            with path.open("r", encoding="utf-8") as handle:
                data = yaml.safe_load(handle) or {}
            if not isinstance(data, dict):
                self._logger.warning(f"Config {path} is not a valid mapping.")
                return None
            return data
        except (OSError, yaml.YAMLError) as exc:
            self._logger.error(f"Failed to load config {path}: {exc}")
            raise  # Re-raise to be handled by caller

    def _fallback_config(self) -> dict[str, Any]:
        return {
            "pipeline": {
                "log_file": "logs/pipeline.log",
                "tasks": [],
            }
        }

    # ------------------------------------------------------------------
    # Pipeline Execution
    # ------------------------------------------------------------------
    def is_running(self) -> bool:
        return self.worker_thread is not None and self.worker_thread.isRunning()

    def start_pipeline(
        self,
        tasks: List[str],
        config_override_path: Optional[str] = None,
        product_override: Optional[str] = None,
    ) -> None:
        """
        Start the processing pipeline with the selected tasks.

        Args:
            tasks: List of task names to execute.
            config_override_path: Optional path string to pass to worker.
            product_override: Optional product ID override.
        """
        if not tasks:
            self.status_message.emit("No tasks selected.", "warning")
            return

        if self.is_running():
            self.status_message.emit("Pipeline already running.", "warning")
            return

        # Deep copy config to prevent mutation during run
        config_copy = copy.deepcopy(self.config)

        try:
            self.worker_thread = WorkerThread(
                tasks,
                config_copy,
                config_override_path or (str(self.current_config_path) if self.current_config_path else None),
                product=product_override,
            )

            # Connect Signals
            self.worker_thread.task_started.connect(self.task_started.emit)
            self.worker_thread.task_completed.connect(self.task_completed.emit)
            self.worker_thread.progress_updated.connect(self.progress_updated.emit)
            self.worker_thread.log_message.connect(self.log_message.emit)
            self.worker_thread.finished_signal.connect(self._on_worker_finished)
            self.worker_thread.error_occurred.connect(self._on_worker_error)

            self.worker_thread.start()
            self.log_message.emit(f"Starting pipeline: {', '.join(tasks)}")

        except (RuntimeError, OSError) as e:
            self.error_occurred.emit(f"Failed to start pipeline thread: {e}")

    def stop_pipeline(self) -> None:
        """Request the pipeline to stop."""
        if self.is_running() and self.worker_thread:
            self.worker_thread.request_stop()
            self.log_message.emit("Stop requested; waiting for current task...")
        else:
            self.status_message.emit("Pipeline is not running.", "info")

    def _on_worker_finished(self) -> None:
        """Handler for worker thread completion."""
        self.pipeline_finished.emit()
        if self.worker_thread:
            self.worker_thread.deleteLater()
            self.worker_thread = None

    def _on_worker_error(self, message: str) -> None:
        """Handler for worker thread errors."""
        self.error_occurred.emit(message)
        # Worker thread usually emits finished_signal after error, but we clean up to be safe
        if self.worker_thread:
            # We don't force kill, just ensure reference is ready to be cleared
            pass

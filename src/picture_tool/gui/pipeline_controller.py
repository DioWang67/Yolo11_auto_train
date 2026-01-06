"""Reusable controller mixin that wires the GUI to the worker thread."""

from __future__ import annotations

import copy
import logging
from pathlib import Path
from typing import Any, Iterable, List, Sequence

import yaml
from PyQt5.QtWidgets import QMessageBox, QWidget

from picture_tool.gui.task_thread import WorkerThread


class PipelineControllerMixin:
    """Lightweight orchestration helpers shared by Picture Tool GUIs."""

    worker_thread: WorkerThread | None
    config: dict[str, Any]

    # ------------------------------------------------------------------
    # lifecycle helpers
    # ------------------------------------------------------------------
    def _init_pipeline_controller(self) -> None:
        self.config = {}
        self.worker_thread = None
        self._controller_logger = logging.getLogger("picture_tool.gui.controller")

    # ------------------------------------------------------------------
    # configuration loading
    # ------------------------------------------------------------------
    def _default_config_path(self) -> Path:
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

    def _show_warning(self, title: str, message: str) -> None:
        try:  # pragma: no cover - UI specific path
            parent: QWidget | None = self if isinstance(self, QWidget) else None
            QMessageBox.warning(parent, title, message)
        except Exception:
            self.log_message(f"[warning] {title}: {message}")

    def _show_error(self, title: str, message: str) -> None:
        try:  # pragma: no cover - UI specific path
            parent: QWidget | None = self if isinstance(self, QWidget) else None
            QMessageBox.critical(parent, title, message)
        except Exception:
            self.log_message(f"[error] {title}: {message}")

    def _show_info(self, title: str, message: str) -> None:
        try:  # pragma: no cover - UI specific path
            parent: QWidget | None = self if isinstance(self, QWidget) else None
            QMessageBox.information(parent, title, message)
        except Exception:
            self.log_message(f"[info] {title}: {message}")

    def load_default_config(self) -> None:
        path = self._default_config_path()
        config = self._load_config_file(path)
        self.config = config or self._fallback_config()
        if hasattr(self, "config_path_edit"):
            try:
                self.config_path_edit.setText(str(path.resolve()))
            except Exception:  # pragma: no cover - defensive UI update
                self.config_path_edit.setText(str(path))
        self._after_config_loaded()

    def load_config(self) -> None:
        target: Path | None = None
        if hasattr(self, "config_path_edit"):
            try:
                text = self.config_path_edit.text().strip()
                if text:
                    target = Path(text)
            except Exception:  # pragma: no cover - defensive UI update
                target = None
        if target is None:
            target = self._default_config_path()

        cfg = self._load_config_file(target)
        if not cfg:
            self.log_message(f"Config file not found or invalid: {target}")
            cfg = self._fallback_config()
        self.config = cfg
        self._after_config_loaded()

    def _after_config_loaded(self) -> None:
        if hasattr(self, "_populate_position_widgets"):
            try:
                self._populate_position_widgets()
            except Exception:  # pragma: no cover - optional hook
                pass
        if hasattr(self, "_load_preset_storage"):
            try:
                self._load_preset_storage()
            except Exception:  # pragma: no cover - optional hook
                pass
        if hasattr(self, "_update_preset_display"):
            try:
                self._update_preset_display()
            except Exception:  # pragma: no cover - optional hook
                pass
        if hasattr(self, "config_text"):
            try:
                dumped = yaml.safe_dump(
                    self.config, allow_unicode=True, sort_keys=False
                )
                self.config_text.setPlainText(dumped)
            except Exception:
                self.config_text.setPlainText("")
        if hasattr(self, "refresh_metrics_dashboard"):
            try:
                self.refresh_metrics_dashboard()
            except Exception:  # pragma: no cover - optional hook
                pass

    def _load_config_file(self, path: Path) -> dict[str, Any] | None:
        if not path.exists():
            return None
        try:
            with path.open("r", encoding="utf-8") as handle:
                data = yaml.safe_load(handle) or {}
        except Exception as exc:
            self.log_message(f"Failed to load config {path}: {exc}")
            return None
        if not isinstance(data, dict):
            self.log_message(f"Config {path} is not a mapping; ignoring.")
            return None
        return data

    @staticmethod
    def _fallback_config() -> dict[str, Any]:
        return {
            "pipeline": {
                "log_file": "logs/pipeline.log",
                "tasks": [],
            }
        }

    def save_config(self) -> None:
        """Save the current configuration to the file specified in the UI."""
        target: Path | None = None
        if hasattr(self, "config_path_edit"):
            try:
                text = self.config_path_edit.text().strip()
                if text:
                    target = Path(text)
            except Exception:
                target = None

        if not target:
            self._show_warning(
                "No config file", "Please select or enter a path to save the config."
            )
            return

        try:
            with target.open("w", encoding="utf-8") as handle:
                yaml.safe_dump(self.config, handle, allow_unicode=True, sort_keys=False)
            self.log_message(f"Configuration saved to {target}")
            self._show_info("Success", f"Config saved to {target}")
        except Exception as exc:
            self._show_error("Save Failed", f"Could not save config: {exc}")

    # ------------------------------------------------------------------
    # task selection helpers
    # ------------------------------------------------------------------
    def get_selected_tasks(self) -> List[str]:
        tasks: List[str] = []
        for name, checkbox in getattr(self, "task_checkboxes", {}).items():
            try:
                if checkbox.isChecked():
                    tasks.append(name)
            except Exception:
                continue
        return tasks

    def reset_task_statuses(self, tasks: Sequence[str]) -> None:
        if hasattr(self, "task_status_items"):
            for task in tasks:
                item = self.task_status_items.get(task)
                if item is not None:
                    item.setText(f"{task} - pending")

    def _set_task_status(
        self, task: str, message: str, color: Any | None = None
    ) -> None:
        if hasattr(self, "task_status_items"):
            item = self.task_status_items.get(task)
            if item is not None:
                item.setText(f"{task} - {message}")

    # ------------------------------------------------------------------
    # pipeline execution
    # ------------------------------------------------------------------
    def start_pipeline(self) -> None:
        tasks = self.get_selected_tasks()
        if not tasks:
            self._show_warning("No tasks selected", "Choose at least one task to run.")
            return

        if self.worker_thread is not None and self.worker_thread.isRunning():
            self._show_warning(
                "Pipeline running", "A pipeline run is already in progress."
            )
            return

        validation_errors: Iterable[str] | None = None
        if hasattr(self, "_validate_pipeline_configuration"):
            try:
                validation_errors = self._validate_pipeline_configuration(tasks)
            except Exception:  # pragma: no cover - optional hook
                validation_errors = None
        if validation_errors:
            self._show_warning(
                "Configuration check failed", "\n".join(validation_errors)
            )
            return

        config_copy = copy.deepcopy(self.config)
        config_path = ""
        if hasattr(self, "config_path_edit"):
            try:
                config_path = self.config_path_edit.text().strip()
            except Exception:
                config_path = ""

        # [NEW] Capture Product Override
        product_text = None
        if hasattr(self, "product_override_edit"):
            try:
                txt = self.product_override_edit.text().strip()
                if txt:
                    product_text = txt
            except Exception:
                pass

        self.worker_thread = WorkerThread(
            tasks, config_copy, config_path or None, product=product_text
        )
        self.worker_thread.task_started.connect(self.on_task_started)
        self.worker_thread.task_completed.connect(self.on_task_completed)
        self.worker_thread.progress_updated.connect(self.update_progress)
        self.worker_thread.log_message.connect(self.log_message)
        self.worker_thread.finished_signal.connect(self.on_pipeline_finished)
        self.worker_thread.error_occurred.connect(self.on_error_occurred)

        self.reset_task_statuses(tasks)
        self._set_running_state(is_running=True)
        self.worker_thread.start()
        self.log_message(f"Starting pipeline: {', '.join(tasks)}")

    def stop_pipeline(self) -> None:
        if self.worker_thread and self.worker_thread.isRunning():
            self.worker_thread.request_stop()
            self.log_message("Stop requested; waiting for the current task to finish.")

    # ------------------------------------------------------------------
    # worker callbacks
    # ------------------------------------------------------------------
    def update_progress(self, value: int) -> None:
        if hasattr(self, "progress_bar"):
            try:
                self.progress_bar.setValue(value)
            except Exception:  # pragma: no cover - defensive UI update
                pass

    def on_task_started(self, task_name: str) -> None:
        self._set_task_status(task_name, "running")
        if hasattr(self, "status_label"):
            try:
                self.status_label.setText(f"Running: {task_name}")
            except Exception:
                pass

    def on_task_completed(self, task_name: str) -> None:
        self._set_task_status(task_name, "completed")

    def on_pipeline_finished(self) -> None:
        self._set_running_state(is_running=False)
        self.log_message("Pipeline finished.")
        if self.worker_thread:
            self.worker_thread.deleteLater()
            self.worker_thread = None

    def on_error_occurred(self, message: str) -> None:
        self._set_running_state(is_running=False)
        self._show_error("Pipeline error", message)
        self.log_message(f"[error] {message}")

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    def _set_running_state(self, *, is_running: bool) -> None:
        start_enabled = not is_running
        stop_enabled = is_running
        if hasattr(self, "start_btn"):
            try:
                self.start_btn.setEnabled(start_enabled)
            except Exception:
                pass
        if hasattr(self, "stop_btn"):
            try:
                self.stop_btn.setEnabled(stop_enabled)
            except Exception:
                pass
        if hasattr(self, "status_label"):
            try:
                self.status_label.setText("Running..." if is_running else "Idle")
            except Exception:
                pass
        if hasattr(self, "progress_bar") and not is_running:
            try:
                self.progress_bar.setValue(0)
            except Exception:
                pass

    # ------------------------------------------------------------------
    # abstract hooks implemented by concrete GUI classes
    # ------------------------------------------------------------------
    def log_message(
        self, message: str
    ) -> None:  # pragma: no cover - provided by subclasses
        raise NotImplementedError

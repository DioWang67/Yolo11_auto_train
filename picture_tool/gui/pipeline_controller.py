"""Shared pipeline management logic for Picture Tool frontends.

This mixin encapsulates configuration IO and worker-thread lifecycle so that
different UI layers (Qt, web, etc.) can reuse the same behaviour.

Expected attributes provided by subclasses
-----------------------------------------
- ``config_path_edit``: widget exposing ``text()`` / ``setText(str)``
- ``task_checkboxes``: ``dict[str, Any]`` where values expose ``setChecked(bool)`` /
  ``isChecked()``
- ``start_btn`` / ``stop_btn``: widgets exposing ``setEnabled(bool)``
- ``progress_bar``: widget exposing ``setValue(int)``
- ``status_label``: widget exposing ``setText(str)`` / ``setStyleSheet(str)``
- ``config_text``: widget exposing ``setPlainText(str)``
- ``task_status_items`` (optional): dict of task label -> ``QListWidgetItem``

Expected helper methods (provided by other mixins)
--------------------------------------------------
- ``log_message(str)``
- ``_populate_position_widgets()``
- ``_apply_position_settings()``
- ``_load_preset_storage()`` / ``_update_preset_display()``
- ``reset_task_statuses(tasks: list[str])``
- ``_validate_pipeline_configuration(tasks: list[str]) -> list[str]``
- ``_set_task_status(task: str, message: str, color: QColor)``
- ``refresh_metrics_dashboard()``
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, List

import yaml
from PyQt5.QtGui import QColor
from PyQt5.QtWidgets import QMessageBox

from picture_tool.gui.task_thread import WorkerThread


class PipelineControllerMixin:
    """Pipeline orchestration helpers reusable across UI frontends."""

    worker_thread: WorkerThread | None
    config: dict

    def _init_pipeline_controller(self) -> None:
        self.config = {}
        self.worker_thread: WorkerThread | None = None

    # ------------------------------------------------------------------
    # configuration helpers
    # ------------------------------------------------------------------
    def _default_config_path(self) -> Path:
        candidates = [
            Path(__file__).resolve().parent.parent / "config.yaml",
            Path(__file__).resolve().parent / "config.yaml",
            Path.cwd() / "picture_tool" / "config.yaml",
            Path.cwd() / "config.yaml",
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate
        return candidates[-1]

    def _show_warning(
        self, title: str, message: str
    ) -> None:  # pragma: no cover - UI helper
        try:
            QMessageBox.warning(self, title, message)
        except Exception:
            self.log_message(f"[警告] {title}: {message}")

    def _show_error(
        self, title: str, message: str
    ) -> None:  # pragma: no cover - UI helper
        try:
            QMessageBox.critical(self, title, message)
        except Exception:
            self.log_message(f"[錯誤] {title}: {message}")

    def load_default_config(self) -> None:
        default_path = self._default_config_path()
        if default_path.exists():
            try:
                with default_path.open("r", encoding="utf-8") as handle:
                    loaded = yaml.safe_load(handle) or {}
                if isinstance(loaded, dict):
                    self.config = loaded
                    self.log_message(f" 已載入預設設定檔: {default_path}")
                    if hasattr(self, "config_path_edit"):
                        try:
                            self.config_path_edit.setText(str(default_path.resolve()))
                        except Exception:
                            self.config_path_edit.setText(str(default_path))
                    self._populate_position_widgets()
                    self._load_preset_storage()
                    self.update_config_display()
                    self.refresh_metrics_dashboard()
                    return
            except (
                Exception
            ) as exc:  # pragma: no cover - fallback when bundled config invalid
                self.log_message(f" 載入預設設定檔失敗: {exc}")

        fallback = {
            "pipeline": {
                "log_file": "pipeline.log",
                "tasks": [{"name": "yolo_train", "enabled": True, "dependencies": []}],
            }
        }
        self.config = fallback
        self._populate_position_widgets()
        self._load_preset_storage()
        self.update_config_display()
        self.refresh_metrics_dashboard()

    def load_config(self) -> None:
        raw_path = ""
        if hasattr(self, "config_path_edit"):
            raw_path = (self.config_path_edit.text() or "").strip()

        if not raw_path:
            self.log_message(" 未提供設定檔路徑，改用預設設定。")
            self.load_default_config()
            return

        try:
            cfg_path = Path(raw_path).expanduser()
        except Exception:
            cfg_path = Path(raw_path)

        if not cfg_path.exists():
            self.log_message(f" 設定檔不存在: {cfg_path}，改用預設設定。")
            self.load_default_config()
            return

        try:
            with cfg_path.open("r", encoding="utf-8") as handle:
                loaded_config = yaml.safe_load(handle) or {}

            if not isinstance(loaded_config, dict):
                raise ValueError("設定檔格式錯誤：內容需為映射 (mapping)。")

            self.config = loaded_config
            self.log_message(f" 成功載入設定檔: {cfg_path}")
            if hasattr(self, "config_path_edit"):
                try:
                    self.config_path_edit.setText(str(cfg_path.resolve()))
                except Exception:
                    self.config_path_edit.setText(str(cfg_path))
            self._populate_position_widgets()
            self._load_preset_storage()
            self.update_config_display()
            self.refresh_metrics_dashboard()

        except yaml.YAMLError as exc:
            self.log_message(f" YAML 解析失敗: {exc}")
            self._show_error("設定錯誤", f"設定檔解析失敗:\n{exc}\n\n改用預設設定。")
            self.load_default_config()
        except Exception as exc:  # pragma: no cover - guard around file IO
            self.log_message(f" 載入設定發生例外: {exc}")
            self._show_error("載入失敗", f"無法讀取設定檔:\n{exc}\n\n改用預設設定。")
            self.load_default_config()

    def update_config_display(self) -> None:
        self._apply_position_settings()
        config_text = yaml.dump(
            self.config, default_flow_style=False, allow_unicode=True, indent=2
        )
        self.config_text.setPlainText(config_text)
        self._update_preset_display()

    # ------------------------------------------------------------------
    # task selection
    # ------------------------------------------------------------------
    def select_all_tasks(self) -> None:
        for checkbox in self.task_checkboxes.values():
            checkbox.setChecked(True)
        self.log_message(" 已勾選所有任務")

    def deselect_all_tasks(self) -> None:
        for checkbox in self.task_checkboxes.values():
            checkbox.setChecked(False)
        self.log_message(" 已取消所有任務")

    def get_selected_tasks(self) -> List[str]:
        selected: List[str] = []
        for task_name, checkbox in self.task_checkboxes.items():
            if checkbox.isChecked():
                selected.append(task_name)
        return selected

    # ------------------------------------------------------------------
    # pipeline execution
    # ------------------------------------------------------------------
    def start_pipeline(self) -> None:
        selected_tasks = self.get_selected_tasks()
        if not selected_tasks:
            self._show_warning("提醒", "請至少選擇一個任務。")
            return

        if self.worker_thread is not None:
            if self.worker_thread.isRunning():
                self._show_warning("提醒", "任務仍在執行中。")
                return
            self.worker_thread.progress_updated.disconnect()
            self.worker_thread.task_started.disconnect()
            self.worker_thread.task_completed.disconnect()
            self.worker_thread.log_message.disconnect()
            self.worker_thread.finished_signal.disconnect()
            self.worker_thread.error_occurred.disconnect()
            self.worker_thread.deleteLater()
            self.worker_thread = None

        self.reset_task_statuses(selected_tasks)
        self._apply_position_settings()
        validation_errors = self._validate_pipeline_configuration(selected_tasks)
        if validation_errors:
            self._show_warning("預檢結果", "\n".join(validation_errors))
            return

        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.progress_bar.setValue(0)
        self.status_label.setText(" 執行中...")
        self.status_label.setStyleSheet(
            """
            QLabel {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #fff3cd, stop:1 #ffeaa7);
                color: #856404;
                padding: 10px;
                border: 2px solid #ffeaa7;
                border-radius: 6px;
                font-weight: bold;
                font-size: 10pt;
            }
        """
        )

        try:
            if (
                hasattr(self, "apply_overrides_cb")
                and self.apply_overrides_cb.isChecked()
            ):
                yt = self.config.get("yolo_training", {}) or {}
                dev = self.override_device_edit.text().strip()
                ep = self.override_epochs_edit.text().strip()
                im = self.override_imgsz_edit.text().strip()
                bt = self.override_batch_edit.text().strip()
                if dev:
                    yt["device"] = dev
                if ep.isdigit():
                    yt["epochs"] = int(ep)
                if im.isdigit():
                    yt["imgsz"] = int(im)
                if bt.isdigit():
                    yt["batch"] = int(bt)
                self.config["yolo_training"] = yt

                ye = self.config.get("yolo_evaluation", {}) or {}
                if yt.get("device"):
                    ye["device"] = yt["device"]
                self.config["yolo_evaluation"] = ye

                bi = self.config.get("batch_inference", {}) or {}
                if yt.get("device"):
                    bi["device"] = yt["device"]
                self.config["batch_inference"] = bi

            pos_cfg: dict[str, Any] = {}
            if isinstance(self.config, dict):
                ycfg = self.config.get("yolo_training", {})
                if isinstance(ycfg, dict):
                    pos_cfg = ycfg.get("position_validation", {}) or {}

            want_position_validation = (
                getattr(self, "POSITION_TASK_LABEL", "") in selected_tasks
            )
            train_selected = getattr(self, "YOLO_TRAIN_LABEL", "") in selected_tasks

            if want_position_validation and not pos_cfg.get("enabled"):
                self._show_warning(
                    "提醒", "已勾選位置檢查，請先啟用位置設定或取消勾選。"
                )
                self.on_pipeline_finished()
                return

            if pos_cfg.get("enabled") and (want_position_validation or train_selected):
                missing_fields = []
                if not pos_cfg.get("product"):
                    missing_fields.append("產品")
                if not pos_cfg.get("area"):
                    missing_fields.append("區域")
                if not (pos_cfg.get("config_path") or pos_cfg.get("config")):
                    missing_fields.append("位置設定檔")
                if missing_fields:
                    self._show_warning(
                        "提醒", "位置檢查需求未滿足：" + "、".join(missing_fields)
                    )
                    self.on_pipeline_finished()
                    return

            if hasattr(self, "force_cb") and self.force_cb.isChecked():
                pl = self.config.get("pipeline", {}) or {}
                pl["force"] = True
                self.config["pipeline"] = pl
        except Exception:  # pragma: no cover - defensive around UI extras
            pass

        config_path_value = ""
        if hasattr(self, "config_path_edit"):
            try:
                config_path_value = self.config_path_edit.text()
            except Exception:
                config_path_value = ""

        self.worker_thread = WorkerThread(
            selected_tasks,
            self.config,
            config_path_value,
        )
        self.worker_thread.progress_updated.connect(self.update_progress)
        self.worker_thread.task_started.connect(self.on_task_started)
        self.worker_thread.task_completed.connect(self.on_task_completed)
        self.worker_thread.log_message.connect(self.log_message)
        self.worker_thread.finished_signal.connect(self.on_pipeline_finished)
        self.worker_thread.error_occurred.connect(self.on_error_occurred)

        self.worker_thread.start()
        self.log_message(f" 開始執行任務: {', '.join(selected_tasks)}")

    def stop_pipeline(self) -> None:
        if self.worker_thread:
            self.worker_thread.cancel()
            if hasattr(self, "task_status_items"):
                for label, item in self.task_status_items.items():
                    if "進行中" in item.text():
                        self._set_task_status(label, "已取消", QColor("#dc3545"))
            self.log_message(" 正在停止任務...")

    # ------------------------------------------------------------------
    # worker callbacks
    # ------------------------------------------------------------------
    def update_progress(self, value: int) -> None:
        self.progress_bar.setValue(value)

    def on_task_started(self, task_name: str) -> None:
        self.log_message(f" 開始任務: {task_name}")
        if hasattr(self, "task_status_items"):
            self._set_task_status(task_name, "進行中", QColor("#17a2b8"))

    def on_task_completed(self, task_name: str) -> None:
        self.log_message(f" 任務完成: {task_name}")
        if hasattr(self, "task_status_items"):
            self._set_task_status(task_name, "完成", QColor("#28a745"))

    def on_pipeline_finished(self) -> None:
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.status_label.setText(" 待命")
        self.status_label.setStyleSheet(
            """
            QLabel {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #d4edda, stop:1 #c3e6cb);
                color: #155724;
                padding: 10px;
                border: 2px solid #c3e6cb;
                border-radius: 6px;
                font-weight: bold;
                font-size: 10pt;
            }
        """
        )
        if hasattr(self, "task_status_items"):
            for label, item in self.task_status_items.items():
                if "進行中" in item.text():
                    self._set_task_status(label, "完成", QColor("#28a745"))
        self.refresh_metrics_dashboard()
        self.log_message(" 所有任務已完成。")

    def on_error_occurred(self, error_message: str) -> None:
        self.log_message(f" 任務錯誤: {error_message}")
        detail = f"執行任務時發生錯誤：\n{error_message}"
        self._show_error("錯誤", detail)
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.status_label.setText(" 發生錯誤")
        self.status_label.setStyleSheet(
            """
            QLabel {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #f8d7da, stop:1 #f1b0b7);
                color: #721c24;
                padding: 10px;
                border: 2px solid #f5c6cb;
                border-radius: 6px;
                font-weight: bold;
                font-size: 10pt;
            }
        """
        )
        if hasattr(self, "task_status_items"):
            error_color = QColor("#e83e8c")
            for label, item in self.task_status_items.items():
                if "進行中" in item.text():
                    self._set_task_status(label, "錯誤", error_color)
        if self.worker_thread:
            try:
                self.worker_thread.wait(100)
            except Exception:
                pass
            self.worker_thread = None

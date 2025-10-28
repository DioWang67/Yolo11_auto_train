"""Preset management helpers for the auto-train GUI."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from PyQt5.QtWidgets import QFileDialog, QInputDialog, QMessageBox

DEFAULT_PRESET_CONFIG_PATH = (Path(__file__).resolve().parent.parent / "preset_config.yaml").resolve()


class PresetManagerMixin:
    preset_storage: dict[str, Any]
    task_presets: dict[str, list[str]]

    def bind_preset_controls(
        self,
        *,
        text: Any | None = None,
        combo: Any | None = None,
        apply_button: Any | None = None,
        delete_button: Any | None = None,
    ) -> None:
        if text is not None:
            self._preset_text_widget = text
        if combo is not None:
            self._preset_combo_widget = combo
        if apply_button is not None:
            self._preset_apply_button = apply_button
        if delete_button is not None:
            self._preset_delete_button = delete_button

    def _get_preset_text_widget(self) -> Any | None:
        return getattr(self, "_preset_text_widget", None) or getattr(self, "preset_text", None)

    def _get_preset_combo(self) -> Any | None:
        return getattr(self, "_preset_combo_widget", None) or getattr(self, "preset_combo", None)

    def _get_preset_apply_button(self) -> Any | None:
        return getattr(self, "_preset_apply_button", None) or getattr(self, "apply_preset_btn", None)

    def _get_preset_delete_button(self) -> Any | None:
        return getattr(self, "_preset_delete_button", None) or getattr(self, "delete_preset_btn", None)

    def _load_preset_storage(self) -> None:
        storage: dict[str, Any] = {"presets": {}}
        config_path = getattr(self, "preset_config_path", DEFAULT_PRESET_CONFIG_PATH)
        try:
            if config_path.exists():
                with config_path.open("r", encoding="utf-8") as handle:
                    loaded = yaml.safe_load(handle) or {}
                presets = loaded.get("presets") if isinstance(loaded, dict) else None
                if isinstance(presets, dict):
                    storage["presets"] = presets
            else:
                config_path.parent.mkdir(parents=True, exist_ok=True)
        except Exception as exc:  # pragma: no cover - UI feedback only
            self.log_message(f"載入流程設定失敗：{exc}")
        self.preset_storage = storage
        self._rebuild_task_presets()
        self._update_preset_display()

    def _save_preset_storage(self) -> None:
        config_path = getattr(self, "preset_config_path", DEFAULT_PRESET_CONFIG_PATH)
        data = {"presets": self.preset_storage.get("presets", {})}
        try:
            config_path.parent.mkdir(parents=True, exist_ok=True)
            with config_path.open("w", encoding="utf-8") as handle:
                yaml.safe_dump(data, handle, allow_unicode=True, sort_keys=True)
        except Exception as exc:  # pragma: no cover - UI feedback only
            self.log_message(f"寫入流程設定失敗：{exc}")

    def _update_preset_display(self) -> None:
        preset_text = self._get_preset_text_widget()
        if preset_text is None:
            return
        try:
            display = yaml.safe_dump(
                {"presets": self.preset_storage.get("presets", {})},
                allow_unicode=True,
                sort_keys=True,
            )
        except Exception:  # pragma: no cover - UI feedback only
            display = "無法顯示流程設定"
        preset_text.setPlainText(display or "目前尚無流程")

    def _rebuild_task_presets(self) -> None:
        preset_combo = self._get_preset_combo()
        if preset_combo is None:
            return
        apply_button = self._get_preset_apply_button()
        delete_button = self._get_preset_delete_button()
        self.task_presets = {}
        preset_combo.blockSignals(True)
        preset_combo.clear()
        presets = self.preset_storage.get("presets") if isinstance(self.preset_storage, dict) else {}
        if not isinstance(presets, dict):
            presets = {}
        has_presets = False
        for name in sorted(presets.keys()):
            labels = [label for label in presets.get(name, []) if label in self.task_checkboxes]
            if not labels:
                continue
            key = f"custom::{name}"
            self.task_presets[key] = labels
            preset_combo.addItem(f"{name} ({len(labels)})", key)
            has_presets = True
        if not has_presets:
            preset_combo.addItem("目前無流程", None)
            preset_combo.setEnabled(False)
            if apply_button:
                apply_button.setEnabled(False)
            if delete_button:
                delete_button.setEnabled(False)
            preset_combo.setCurrentIndex(0)
        else:
            preset_combo.setEnabled(True)
            if apply_button:
                apply_button.setEnabled(True)
            if delete_button:
                delete_button.setEnabled(False)
            preset_combo.setCurrentIndex(0)
        preset_combo.blockSignals(False)
        self._on_preset_selection_changed()

    def _on_preset_selection_changed(self) -> None:
        preset_combo = self._get_preset_combo()
        delete_button = self._get_preset_delete_button()
        if preset_combo is None or delete_button is None:
            return
        key = preset_combo.currentData()
        can_delete = isinstance(key, str) and key.startswith("custom::")
        delete_button.setEnabled(bool(can_delete))

    def apply_selected_preset(self) -> None:
        preset_combo = self._get_preset_combo()
        if preset_combo is None:
            return
        key = preset_combo.currentData()
        labels = self.task_presets.get(key, [])
        if not labels:
            if key is not None:
                self.log_message("流程內容為空，無法套用。")
            return
        for label, checkbox in self.task_checkboxes.items():
            checkbox.setChecked(label in labels)
        name = key.split("::", 1)[1] if isinstance(key, str) and "::" in key else str(key)
        self.log_message(f"✅ 已套用流程：{name}")

    def save_selected_as_preset(self) -> None:
        if not hasattr(self, "task_checkboxes"):
            return
        selected = [label for label, checkbox in self.task_checkboxes.items() if checkbox.isChecked()]
        if not selected:
            QMessageBox.warning(self, "警示", "請先選擇至少一個任務後再儲存流程")
            return
        name, ok = QInputDialog.getText(self, "儲存流程", "流程名稱：")
        if not ok:
            return
        name = (name or "").strip()
        if not name:
            QMessageBox.warning(self, "警示", "流程名稱不可為空")
            return
        presets = self.preset_storage.setdefault("presets", {})
        presets[name] = selected
        self._save_preset_storage()
        self._rebuild_task_presets()
        self._update_preset_display()
        preset_combo = self._get_preset_combo()
        if preset_combo:
            for index in range(preset_combo.count()):
                if preset_combo.itemData(index) == f"custom::{name}":
                    preset_combo.setCurrentIndex(index)
                    break
        self.log_message(f"💾 已儲存流程：{name}")
        self.log_message(f"已儲存流程：{name}")

    def delete_selected_preset(self) -> None:
        preset_combo = self._get_preset_combo()
        if preset_combo is None:
            return
        key = preset_combo.currentData()
        if not isinstance(key, str) or not key.startswith("custom::"):
            QMessageBox.information(self, "提示", "請先選擇自訂流程後再刪除")
            return
        name = key.split("::", 1)[1]
        presets = self.preset_storage.get("presets") if isinstance(self.preset_storage, dict) else None
        if not isinstance(presets, dict) or name not in presets:
            QMessageBox.information(self, "提示", "流程已不存在或無法刪除")
            return
        reply = QMessageBox.question(
            self,
            "刪除流程",
            f"確定要刪除流程「{name}」嗎？",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return
        presets.pop(name, None)
        self._save_preset_storage()
        self._rebuild_task_presets()
        self._update_preset_display()
        self.update_config_display()
        self.log_message(f"已刪除流程：{name}")

    def export_presets(self) -> None:
        if not isinstance(self.preset_storage, dict) or not self.preset_storage.get("presets"):
            QMessageBox.information(self, "提示", "目前沒有可匯出的流程設定")
            return
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "匯出流程設定",
            str(Path.cwd() / "preset_export.yaml"),
            "YAML Files (*.yaml *.yml)",
        )
        if not file_path:
            return
        try:
            with open(file_path, "w", encoding="utf-8") as fh:
                yaml.safe_dump({"presets": self.preset_storage.get("presets", {})}, fh, allow_unicode=True, sort_keys=True)
            QMessageBox.information(self, "匯出成功", f"流程設定已匯出至：\n{file_path}")
        except Exception as exc:  # pragma: no cover - UI feedback only
            QMessageBox.critical(self, "匯出失敗", f"寫入檔案時發生錯誤：\n{exc}")

    def import_presets(self) -> None:
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "匯入流程設定",
            "",
            "YAML Files (*.yaml *.yml)",
        )
        if not file_path:
            return
        try:
            with open(file_path, "r", encoding="utf-8") as fh:
                data = yaml.safe_load(fh) or {}
            presets = data.get("presets") if isinstance(data, dict) else None
            if not isinstance(presets, dict):
                QMessageBox.warning(self, "匯入失敗", "檔案格式不符合預期")
                return
            self.preset_storage.setdefault("presets", {}).update(presets)
            self._save_preset_storage()
            self._rebuild_task_presets()
            self._update_preset_display()
            QMessageBox.information(self, "匯入完成", "流程設定已更新")
        except Exception as exc:  # pragma: no cover - UI feedback only
            QMessageBox.critical(self, "匯入失敗", f"讀取檔案時發生錯誤：\n{exc}")

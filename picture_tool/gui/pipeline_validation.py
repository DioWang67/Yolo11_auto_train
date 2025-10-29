"""Pipeline validation helpers for the auto-train GUI."""

from __future__ import annotations

from pathlib import Path
from typing import List

from PyQt5.QtWidgets import QMessageBox


class PipelineValidationMixin:
    def _validate_pipeline_configuration(self, selected_tasks: List[str]) -> List[str]:
        issues: List[str] = []

        config = self.config if isinstance(self.config, dict) else {}
        yolo_cfg = config.get("yolo_training") if isinstance(config, dict) else None
        if not isinstance(yolo_cfg, dict):
            yolo_cfg = {}
        pos_cfg = (
            yolo_cfg.get("position_validation") if isinstance(yolo_cfg, dict) else None
        )
        if not isinstance(pos_cfg, dict):
            pos_cfg = {}

        position_label = getattr(self, "POSITION_TASK_LABEL", "位置檢查")
        train_label = getattr(self, "YOLO_TRAIN_LABEL", "YOLO訓練")

        want_position_validation = position_label in selected_tasks
        train_selected = train_label in selected_tasks

        if want_position_validation and not pos_cfg.get("enabled"):
            issues.append(
                "已選擇位置檢查任務但未啟用位置檢查設定，請在左側啟用並填寫必填欄位"
            )

        if pos_cfg.get("enabled") and (want_position_validation or train_selected):
            missing_fields: List[str] = []
            if not pos_cfg.get("product"):
                missing_fields.append("產品")
            if not pos_cfg.get("area"):
                missing_fields.append("作業區")
            if not (pos_cfg.get("config_path") or pos_cfg.get("config")):
                missing_fields.append("位置設定檔")
            if missing_fields:
                issues.append("定位檢查缺少必填欄位：" + "、".join(missing_fields))

            config_path = pos_cfg.get("config_path")
            if (
                isinstance(config_path, str)
                and config_path
                and not Path(config_path).exists()
            ):
                issues.append(f"位置設定檔不存在：{config_path}")

            sample_dir = pos_cfg.get("sample_dir")
            if (
                isinstance(sample_dir, str)
                and sample_dir
                and not Path(sample_dir).exists()
            ):
                issues.append(f"定位檢查樣本資料夾不存在：{sample_dir}")

        if train_selected:
            dataset_dir = (
                yolo_cfg.get("dataset_dir") if isinstance(yolo_cfg, dict) else None
            )
            if (
                isinstance(dataset_dir, str)
                and dataset_dir
                and not Path(dataset_dir).exists()
            ):
                issues.append(f"YOLO 訓練資料夾不存在：{dataset_dir}")

        return issues

    def run_preflight_check(self) -> None:
        selected_tasks = self.get_selected_tasks()
        if not selected_tasks:
            QMessageBox.information(self, "提示", "請先選擇至少一個任務後再進行檢查")
            return
        self._apply_position_settings()
        issues = self._validate_pipeline_configuration(selected_tasks)
        if issues:
            QMessageBox.warning(
                self,
                "流程檢查",
                "\n".join(issues),
            )
        else:
            QMessageBox.information(
                self,
                "流程檢查",
                "所有檢查皆通過，流程準備就緒",
            )

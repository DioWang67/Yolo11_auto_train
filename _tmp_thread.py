from __future__ import annotations

import copy
import time
from pathlib import Path
from typing import List, Optional

import yaml  # type: ignore[import]
from PyQt5.QtCore import QThread, pyqtSignal

try:
    from picture_tool.format import convert_format
    from picture_tool.anomaly import process_anomaly_detection
    from picture_tool.augment import YoloDataAugmentor, ImageAugmentor
    from picture_tool.split import split_dataset
    from picture_tool.train.yolo_trainer import train_yolo
    from picture_tool.eval.yolo_evaluator import evaluate_yolo
    from picture_tool.report.report_generator import generate_report
    from picture_tool.quality.dataset_linter import lint_dataset, preview_dataset
    from picture_tool.infer.batch_infer import run_batch_inference
    from picture_tool.position.yolo_position_validator import run_position_validation
except ImportError as exc:  # pragma: no cover - surface the actual dependency issue
    import sys
    import traceback

    detailed = ''.join(traceback.format_exception(exc.__class__, exc, exc.__traceback__))
    message = (
        "Failed to import required picture_tool modules. "
        "Install project dependencies in the current Python environment.\n"
        f"{detailed}"
    )
    print(message, file=sys.stderr)

    def run_position_validation(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("position validation dependencies are missing")

    raise



from picture_tool.gui.led_qc_manager import LedQcManager


class WorkerThread(QThread):
    """背景執行工作任務的 QThread。"""

    progress_updated = pyqtSignal(int)
    task_started = pyqtSignal(str)
    task_completed = pyqtSignal(str)
    log_message = pyqtSignal(str)
    finished_signal = pyqtSignal()
    error_occurred = pyqtSignal(str)

    def __init__(
        self, tasks: List[str], config: dict, config_path: Optional[str] = None
    ) -> None:
        super().__init__()
        self.tasks = tasks
        self.config = config
        self.config_path = config_path
        self.is_cancelled = False
        self._last_mtime_ns = 0
        self.led_manager = LedQcManager(self.log_message.emit)

        # 任務執行表
        basic_handlers = {
            "格式轉換": self.run_format_conversion,
            "異常檢測": self.run_anomaly_detection,
            "YOLO數據增強": self.run_yolo_augmentation,
            "影像增強": self.run_image_augmentation,
            "數據分割": self.run_dataset_splitter,
            YOLO_TRAIN_LABEL: self.run_yolo_train,
            "YOLO評估": self.run_yolo_evaluation,
            "生成報告": self.run_generate_report,
            "數據檢查": self.run_dataset_lint,
            "增強預覽": self.run_aug_preview,
            "批次推論": self.run_batch_inference,
            POSITION_TASK_LABEL: self.run_position_validation_task,
        }


        alt_labels = {
            "格式??": self.run_format_conversion,
            "异常??": self.run_anomaly_detection,
            "YOLO?据增?": self.run_yolo_augmentation,
            "影像增?": self.run_image_augmentation,
            "?据分割": self.run_dataset_splitter,
            "YOLO??": self.run_yolo_train,
            "YOLO?估": self.run_yolo_evaluation,
            "生成?告": self.run_generate_report,
            "?据?查": self.run_dataset_lint,
            "增???": self.run_aug_preview,
            "批次推論": self.run_batch_inference,
            "批次推?": self.run_batch_inference,
            POSITION_TASK_LABEL: self.run_position_validation_task,
            "位置?查": self.run_position_validation_task,
        }


        led_handlers = {
            "LED QC 建模": lambda: self.led_manager.build(self.config),
            "LED QC 單張檢測": lambda: self.led_manager.detect_single(self.config),
            "LED QC 批次檢測": lambda: self.led_manager.detect_dir(self.config),
            "LED QC 分析": lambda: self.led_manager.analyze(self.config),
        }
        self.task_handlers = {**basic_handlers, **alt_labels, **led_handlers}

    # ------------------------------------------------------------------
    # lifecycle helpers
    # ------------------------------------------------------------------
    def cancel(self) -> None:
        self.is_cancelled = True

    def _reload_config_if_changed(self) -> None:
        if not self.config_path:
            return
        try:
            cfg_path = Path(self.config_path)
            if not cfg_path.exists():
                return
            mtime_ns = cfg_path.stat().st_mtime_ns
            if self._last_mtime_ns == 0:
                self._last_mtime_ns = mtime_ns
                return
            if mtime_ns > self._last_mtime_ns:
                with cfg_path.open("r", encoding="utf-8") as f:
                    new_cfg = yaml.safe_load(f)
                if isinstance(new_cfg, dict):
                    self.config = new_cfg
                    self._last_mtime_ns = mtime_ns
                    self.log_message.emit(f"偵測到設定檔更新，已重新載入：{cfg_path}")
        except Exception as exc:  # pragma: no cover - UI log only
            self.log_message.emit(f"重新載入設定檔失敗：{exc}")

    # ------------------------------------------------------------------
    # QThread interface
    # ------------------------------------------------------------------
    def run(self) -> None:
        try:
            total_tasks = len(self.tasks)
            for index, task in enumerate(self.tasks):
                if self.is_cancelled:
                    break

                self._reload_config_if_changed()
                self.task_started.emit(task)
                self.log_message.emit(f"開始執行任務：{task}")

                handler = self.task_handlers.get(task)
                if handler:
                    handler()
                    self.log_message.emit(f"完成 {task} 任務")
                    self.task_completed.emit(task)
                else:
                    self.log_message.emit(f"未知任務：{task}")

                progress = int((index + 1) / total_tasks * 100)
                self.progress_updated.emit(progress)
                time.sleep(1)

            if not self.is_cancelled:
                self.log_message.emit("全部任務已完成")
            else:
                self.log_message.emit("任務已取消")
            self.finished_signal.emit()
        except Exception as exc:
            self.error_occurred.emit(str(exc))

    # ------------------------------------------------------------------
    # task handlers
    # ------------------------------------------------------------------
    def run_format_conversion(self) -> None:
        if "format_conversion" in self.config:
            convert_format(copy.deepcopy(self.config["format_conversion"]))

    def run_anomaly_detection(self) -> None:
        if "anomaly_detection" in self.config:
            process_anomaly_detection(copy.deepcopy(self.config))

    def run_yolo_augmentation(self) -> None:
        augmentor = YoloDataAugmentor()
        if hasattr(augmentor, "config"):
            augmentor.config = copy.deepcopy(self.config.get("yolo_augmentation", {}))
        augmentor.process_dataset()

    def run_image_augmentation(self) -> None:
        augmentor = ImageAugmentor()
        if hasattr(augmentor, "config"):
            augmentor.config = copy.deepcopy(self.config.get("image_augmentation", {}))
        augmentor.process_dataset()

    def run_dataset_splitter(self) -> None:
        split_dataset(copy.deepcopy(self.config))

    def run_yolo_train(self) -> None:
        train_yolo(copy.deepcopy(self.config))

    def run_yolo_evaluation(self) -> None:
        evaluate_yolo(copy.deepcopy(self.config))

    def run_generate_report(self) -> None:
        generate_report(copy.deepcopy(self.config))

    def run_dataset_lint(self) -> None:
        lint_dataset(copy.deepcopy(self.config))

    def run_aug_preview(self) -> None:
        preview_dataset(copy.deepcopy(self.config))

    def run_position_validation_task(self) -> None:
        cfg = copy.deepcopy(self.config)
        ycfg = cfg.get("yolo_training", {}) if isinstance(cfg, dict) else {}
        if not isinstance(ycfg, dict):
            raise RuntimeError("需要有效的 yolo_training 設定才能執行位置檢查。")
        pos_cfg = ycfg.get("position_validation", {}) if isinstance(ycfg, dict) else {}
        if not pos_cfg.get("enabled"):
            raise RuntimeError("已選擇位置檢查任務，但尚未啟用或設定位置檢查。請在左側設定面板啟用並填寫必要欄位。")
        project = Path(str(ycfg.get("project", "./runs/detect")))
        name = str(ycfg.get("name", "train"))
        run_dir = project / name
        if not run_dir.exists():
            run_dir.mkdir(parents=True, exist_ok=True)
            self.log_message.emit(f"[位置檢查] 建立輸出目錄：{run_dir}")
        run_position_validation(cfg, run_dir, logger=None)

    def run_batch_inference(self) -> None:
        run_batch_inference(copy.deepcopy(self.config))

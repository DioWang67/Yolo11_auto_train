import sys
import os
import yaml
import threading
import time
import copy
import json
import csv
import hashlib
from pathlib import Path
from dataclasses import asdict
from types import SimpleNamespace
from PyQt5.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, 
                             QWidget, QPushButton, QLabel, QTextEdit, QProgressBar,
                             QGroupBox, QCheckBox, QComboBox, QLineEdit, QFileDialog,
                             QSplitter, QTabWidget, QScrollArea, QFrame, QGridLayout,
                             QMessageBox, QListWidget, QListWidgetItem, QSpacerItem,
                             QSizePolicy, QFormLayout, QToolButton)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt5.QtGui import QFont, QIcon, QPalette, QColor, QPixmap

# 假設這些是您原始代碼中的模組
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
    from picture_tool.color import led_qc_enhanced
except ImportError:
    # 如果無法導入，創建模擬函數
    def convert_format(config): print("執行格式轉換")
    def process_anomaly_detection(config): print("執行異常檢測")
    def split_dataset(config): print("執行數據集分割")
    def train_yolo(config): print("執行YOLO訓練")
    def evaluate_yolo(config): print("執行YOLO評估")
    def generate_report(config): print("生成報告")
    def lint_dataset(config): print("執行數據集檢查")
    def preview_dataset(config): print("預覽數據集")
    
    class YoloDataAugmentor:
        def __init__(self): pass
        def process_dataset(self): print("YOLO數據增強")
    
    class ImageAugmentor:
        def __init__(self): pass
        def process_dataset(self): print("圖像數據增強")

    led_qc_enhanced = None

class WorkerThread(QThread):
    """工作線程用於執行任務"""
    progress_updated = pyqtSignal(int)
    task_started = pyqtSignal(str)
    task_completed = pyqtSignal(str)
    log_message = pyqtSignal(str)
    finished_signal = pyqtSignal()
    error_occurred = pyqtSignal(str)

    def __init__(self, tasks, config, config_path=None):
        super().__init__()
        self.tasks = tasks
        self.config = config
        self.is_cancelled = False
        self.config_path = config_path
        self._last_mtime_ns = 0

        # 任務處理器字典
        self.task_handlers = {
            "格式轉換": self.run_format_conversion,
            "異常檢測": self.run_anomaly_detection,
            "YOLO數據增強": self.run_yolo_augmentation,
            "圖像數據增強": self.run_image_augmentation,
            "數據集分割": self.run_dataset_splitter,
            "YOLO訓練": self.run_yolo_train,
            "YOLO評估": self.run_yolo_evaluation,
            "生成報告": self.run_generate_report,
            "數據集檢查": self.run_dataset_lint,
            "增強預覽": self.run_aug_preview,
            "批次推論": self.run_batch_inference,
            "LED QC 建模": self.run_led_qc_build,
            "LED QC 單張檢測": self.run_led_qc_detect_single,
            "LED QC 批次檢測": self.run_led_qc_detect_dir,
            "LED QC 分析": self.run_led_qc_analyze,
            # -- readable Chinese labels --
            "YOLO數據增強": self.run_yolo_augmentation,
            "數據分割": self.run_dataset_splitter,
            "YOLO訓練": self.run_yolo_train,
            "YOLO評估": self.run_yolo_evaluation,
            "生成報告": self.run_generate_report,
            "數據檢查": self.run_dataset_lint,
            "增強預覽": self.run_aug_preview,
            "批次推論": self.run_batch_inference,
            "LED QC 建模": self.run_led_qc_build,
            "LED QC 單張檢測": self.run_led_qc_detect_single,
            "LED QC 批次檢測": self.run_led_qc_detect_dir,
            "LED QC 分析": self.run_led_qc_analyze,
        }

    def cancel(self):
        self.is_cancelled = True

    def _reload_config_if_changed(self):
        try:
            if not self.config_path:
                return
            p = Path(self.config_path)
            if not p.exists():
                return
            mtime_ns = p.stat().st_mtime_ns
            if self._last_mtime_ns == 0:
                self._last_mtime_ns = mtime_ns
                return
            if mtime_ns > self._last_mtime_ns:
                with open(p, 'r', encoding='utf-8') as f:
                    new_cfg = yaml.safe_load(f)
                if isinstance(new_cfg, dict):
                    self.config = new_cfg
                    self._last_mtime_ns = mtime_ns
                    self.log_message.emit(f"偵測到設定檔更新，已重新載入: {p}")
        except Exception as e:
            self.log_message.emit(f"重新載入設定檔失敗: {e}")

    def run(self):
        try:
            total_tasks = len(self.tasks)
            for i, task in enumerate(self.tasks):
                if self.is_cancelled:
                    break
                # auto-reload config if file changed
                self._reload_config_if_changed()

                self.task_started.emit(task)
                self.log_message.emit(f"開始執行任務: {task}")
                
                # 執行任務
                handler = self.task_handlers.get(task)
                if handler:
                    handler()
                    self.log_message.emit(f"任務 {task} 完成")
                    self.task_completed.emit(task)
                else:
                    self.log_message.emit(f"未知任務: {task}")
                
                # 更新進度
                progress = int((i + 1) / total_tasks * 100)
                self.progress_updated.emit(progress)
                
                # 模擬任務執行時間
                time.sleep(1)
            
            if not self.is_cancelled:
                self.log_message.emit("所有任務執行完成！")
            else:
                self.log_message.emit("任務執行已取消")
            
            self.finished_signal.emit()
        except Exception as e:
            self.error_occurred.emit(str(e))

    # 任務處理方法
    def run_format_conversion(self):
        if 'format_conversion' in self.config:
            convert_format(copy.deepcopy(self.config['format_conversion']))

    def run_anomaly_detection(self):
        if 'anomaly_detection' in self.config:
            process_anomaly_detection(copy.deepcopy(self.config))

    def run_yolo_augmentation(self):
        augmentor = YoloDataAugmentor()
        if hasattr(augmentor, 'config'):
            augmentor.config = copy.deepcopy(self.config.get('yolo_augmentation', {}))
        augmentor.process_dataset()

    def run_image_augmentation(self):
        augmentor = ImageAugmentor()
        if hasattr(augmentor, 'config'):
            augmentor.config = copy.deepcopy(self.config.get('image_augmentation', {}))
        augmentor.process_dataset()

    def run_dataset_splitter(self):
        split_dataset(copy.deepcopy(self.config))

    def run_yolo_train(self):
        train_yolo(copy.deepcopy(self.config))

    def run_yolo_evaluation(self):
        evaluate_yolo(copy.deepcopy(self.config))

    def run_generate_report(self):
        generate_report(copy.deepcopy(self.config))

    def run_dataset_lint(self):
        lint_dataset(copy.deepcopy(self.config))

    def run_aug_preview(self):
        preview_dataset(copy.deepcopy(self.config))

    def run_batch_inference(self):
        run_batch_inference(copy.deepcopy(self.config))

    def _ensure_led_module(self):
        if led_qc_enhanced is None:
            raise RuntimeError("LED QC 模組未可用，請確認相依套件已正確安裝。")

    def _as_bool(self, value):
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "y", "on"}
        return bool(value)

    def _collect_led_overrides(self, sub_cfg, led_section):
        overrides = {}
        if isinstance(led_section, dict):
            for key in ("config_overrides", "overrides"):
                source = led_section.get(key)
                if isinstance(source, dict):
                    overrides.update(source)
        if isinstance(sub_cfg, dict):
            for key in ("config_overrides", "overrides"):
                source = sub_cfg.get(key)
                if isinstance(source, dict):
                    overrides.update(source)
        return overrides

    def _prepare_led_config(self, sub_cfg, led_section):
        cfg_path = None
        if isinstance(sub_cfg, dict):
            cfg_path = sub_cfg.get("config_path")
        if not cfg_path and isinstance(led_section, dict):
            cfg_path = led_section.get("config_path")
        if cfg_path:
            cfg_file = Path(cfg_path).expanduser()
            if not cfg_file.exists():
                raise FileNotFoundError(f"LED QC 設定檔不存在：{cfg_file}")
            cfg = led_qc_enhanced.load_config(cfg_file)
        else:
            base_cfg = copy.deepcopy(getattr(led_qc_enhanced, "DEFAULT_CONFIG", {}))
            cfg = led_qc_enhanced.apply_high_conf_preset(base_cfg)
        overrides = self._collect_led_overrides(sub_cfg if isinstance(sub_cfg, dict) else {}, led_section if isinstance(led_section, dict) else {})
        if overrides:
            cfg.update(overrides)
        if hasattr(led_qc_enhanced, "set_active_colors"):
            led_qc_enhanced.set_active_colors(cfg.get("colors"), cfg.get("color_aliases"))
        return cfg

    def run_led_qc_build(self):
        self._ensure_led_module()
        led_section = self.config.get("led_qc_enhanced", {}) or {}
        build_cfg = led_section.get("build", {}) or {}
        ref_dir = build_cfg.get("ref_dir") or led_section.get("ref_dir")
        model_out = build_cfg.get("model_out")
        if not ref_dir or not model_out:
            raise ValueError("請在 config.yaml 設定 led_qc_enhanced.build.ref_dir 與 model_out。")
        ref_path = Path(ref_dir).expanduser()
        if not ref_path.exists():
            raise FileNotFoundError(f"LED QC 參考資料夾不存在：{ref_path}")
        cfg = self._prepare_led_config(build_cfg, led_section)
        model = led_qc_enhanced.build_enhanced_reference(ref_path, cfg)
        out_path = Path(model_out).expanduser()
        led_qc_enhanced.ensure_dir(out_path.parent)
        model.to_json(out_path)
        save_cfg_path = build_cfg.get("save_config")
        if save_cfg_path:
            cfg_path = Path(save_cfg_path).expanduser()
            led_qc_enhanced.ensure_dir(cfg_path.parent)
            saver = getattr(led_qc_enhanced, "save_default_config", None)
            if callable(saver):
                saver(cfg_path)
                self.log_message.emit(f"LED QC 預設設定已輸出：{cfg_path}")
        self.log_message.emit(f"LED QC 模型已建立：{out_path}，樣本數 {model.total_samples}")

    def run_led_qc_detect_single(self):
        self._ensure_led_module()
        led_section = self.config.get("led_qc_enhanced", {}) or {}
        detect_cfg = led_section.get("detect", {}) or {}
        model_path = detect_cfg.get("model") or led_section.get("model")
        image_path = detect_cfg.get("image")
        if not model_path or not image_path:
            raise ValueError("請提供 led_qc_enhanced.detect.model 與 image。")
        model_file = Path(model_path).expanduser()
        if not model_file.exists():
            raise FileNotFoundError(f"LED QC 模型不存在：{model_file}")
        model = led_qc_enhanced.EnhancedReferenceModel.from_json(model_file)
        overrides = self._collect_led_overrides(detect_cfg, led_section)
        if overrides:
            model.config.update(overrides)
        img_path = Path(image_path).expanduser()
        if not img_path.exists():
            raise FileNotFoundError(f"LED QC 檢測影像不存在：{img_path}")
        img = led_qc_enhanced.robust_imread(str(img_path))
        if img is None:
            raise RuntimeError(f"無法讀取檢測影像：{img_path}")
        label = detect_cfg.get("label") or None
        if isinstance(label, str) and not label.strip():
            label = None
        sensitivity = detect_cfg.get("sensitivity", 1.0)
        try:
            sensitivity = float(sensitivity)
        except Exception:
            sensitivity = 1.0
        result = led_qc_enhanced.enhanced_detect_one(img, model, label, sensitivity)
        status = "正常" if not result.is_anomaly else "異常"
        message = (
            f"LED QC 單張檢測 {img_path.name}: {status} "
            f"conf={result.confidence:.2f} color={result.color_used} "
            f"color_conf={result.color_confidence:.2f} 嚴重度={result.severity_score:.2f}"
        )
        self.log_message.emit(message)
        if result.reasons:
            self.log_message.emit("原因：" + "；".join(result.reasons))
        out_dir = detect_cfg.get("out_dir")
        if out_dir:
            out_path = Path(out_dir).expanduser()
            led_qc_enhanced.ensure_dir(out_path)
            if self._as_bool(detect_cfg.get("save_annotated", True)):
                annotated = led_qc_enhanced.enhanced_annotate(img, result)
                ann_path = out_path / f"{img_path.stem}_annotated.png"
                led_qc_enhanced.cv2.imwrite(str(ann_path), annotated)
                self.log_message.emit(f"已輸出標註影像：{ann_path}")
            if self._as_bool(detect_cfg.get("save_json", True)):
                payload = {
                    "file": str(img_path),
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "result": {
                        "is_anomaly": result.is_anomaly,
                        "severity_score": result.severity_score,
                        "reasons": result.reasons,
                        "recommendations": result.recommendations,
                        "color_used": result.color_used,
                        "color_confidence": result.color_confidence,
                        "feature_anomalies": result.feature_anomalies,
                        "processing_time": result.processing_time,
                        "anomaly_regions": result.anomaly_regions,
                    },
                    "features": asdict(result.features),
                }
                json_path = out_path / f"{img_path.stem}_result.json"
                with open(json_path, "w", encoding="utf-8") as f:
                    json.dump(payload, f, ensure_ascii=False, indent=2, default=led_qc_enhanced._json_default)
                self.log_message.emit(f"已輸出檢測結果 JSON：{json_path}")

    def run_led_qc_detect_dir(self):
        self._ensure_led_module()
        led_section = self.config.get("led_qc_enhanced", {}) or {}
        dir_cfg = led_section.get("detect_dir", {}) or {}
        model_path = dir_cfg.get("model") or led_section.get("model")
        input_dir = dir_cfg.get("dir") or dir_cfg.get("input_dir")
        out_dir = dir_cfg.get("out_dir") or dir_cfg.get("output_dir")
        if not model_path or not input_dir or not out_dir:
            raise ValueError("請提供 led_qc_enhanced.detect_dir 的 model、dir 與 out_dir。")

        model_file = Path(model_path).expanduser()
        if not model_file.exists():
            raise FileNotFoundError(f"LED QC 模型不存在：{model_file}")
        images_dir = Path(input_dir).expanduser()
        if not images_dir.exists():
            raise FileNotFoundError(f"LED QC 檢測資料夾不存在：{images_dir}")
        output_dir = Path(out_dir).expanduser()
        led_qc_enhanced.ensure_dir(output_dir)

        model = led_qc_enhanced.EnhancedReferenceModel.from_json(model_file)
        overrides = self._collect_led_overrides(dir_cfg, led_section)
        if overrides:
            model.config.update(overrides)

        try:
            sensitivity = float(dir_cfg.get("sensitivity", 0.85))
        except Exception:
            sensitivity = 0.85

        valid_suffixes = {suffix.lower() for suffix in getattr(led_qc_enhanced, "SUPPORTED_FORMATS", [])}
        raw_paths = [
            p for p in images_dir.rglob("*")
            if p.is_file() and (not valid_suffixes or p.suffix.lower() in valid_suffixes)
        ]
        dedup = {}
        for path in raw_paths:
            key = os.path.normcase(os.path.abspath(str(path)))
            dedup.setdefault(key, path)
        paths = sorted(dedup.values())

        if not paths:
            self.log_message.emit(f"LED QC 批次檢測找不到影像：{images_dir}")
            return

        save_annotated = self._as_bool(dir_cfg.get("save_annotated", True))
        save_json = self._as_bool(dir_cfg.get("save_json", False))
        csv_name = dir_cfg.get("csv_name") or "enhanced_summary.csv"

        results = []
        errors = 0
        anomalies = 0
        start_time = time.time()

        for idx, path in enumerate(paths, 1):
            img = led_qc_enhanced.robust_imread(str(path))
            if img is None:
                results.append({"file": str(path), "error": "read_fail"})
                errors += 1
                continue

            det = led_qc_enhanced.enhanced_detect_one(img, model, None, sensitivity)
            if det.is_anomaly:
                anomalies += 1

            if save_annotated:
                digest = hashlib.md5(os.path.normcase(os.path.abspath(str(path))).encode("utf-8")).hexdigest()[:8]
                ann_name = f"{path.stem}_{digest}_annotated.png"
                annotated = led_qc_enhanced.enhanced_annotate(img, det)
                led_qc_enhanced.cv2.imwrite(str(output_dir / ann_name), annotated)

            if save_json:
                payload = {
                    "file": str(path),
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "result": {
                        "is_anomaly": det.is_anomaly,
                        "severity_score": det.severity_score,
                        "reasons": det.reasons,
                        "recommendations": det.recommendations,
                        "color_used": det.color_used,
                        "color_confidence": det.color_confidence,
                        "feature_anomalies": det.feature_anomalies,
                        "processing_time": det.processing_time,
                        "anomaly_regions": det.anomaly_regions,
                    },
                    "features": asdict(det.features),
                }
                json_file = output_dir / f"{path.stem}_result.json"
                with open(json_file, "w", encoding="utf-8") as f:
                    json.dump(payload, f, ensure_ascii=False, indent=2, default=led_qc_enhanced._json_default)

            results.append({
                "file": str(path),
                "is_anomaly": int(det.is_anomaly),
                "severity_score": float(det.severity_score),
                "color": det.color_used,
                "color_conf": float(det.color_confidence),
                "conf": float(det.confidence),
                "time_ms": float(det.processing_time * 1000.0),
                "anomaly_boxes": ";".join(f"{x},{y},{w},{h}" for x, y, w, h in det.anomaly_regions),
                "reasons": "; ".join(det.reasons),
                "error": "",
                "area_ratio": float(det.features.area_ratio),
                "valid_mask": int(det.features.valid_mask),
                "mean_v": float(det.features.mean_v),
                "uniformity": float(det.features.uniformity),
                "hole_ratio": float(det.features.hole_ratio),
            })

            if idx % 10 == 0 or idx == len(paths):
                self.log_message.emit(f"LED QC 批次檢測進度：{idx}/{len(paths)}")

        elapsed = time.time() - start_time
        csv_path = output_dir / csv_name
        fieldnames = [
            "file", "is_anomaly", "severity_score", "color", "color_conf", "conf", "time_ms",
            "anomaly_boxes", "reasons", "error", "area_ratio", "valid_mask", "mean_v", "uniformity", "hole_ratio"
        ]
        with open(csv_path, "w", newline="", encoding="utf-8-sig") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(results)

        total = len(results)
        per_image = (elapsed / total * 1000.0) if total else 0.0
        summary = (
            f"LED QC 批次檢測完成：{total} 張，異常 {anomalies}，錯誤 {errors}，"
            f"耗時 {elapsed:.1f}s (~{per_image:.1f}ms/張)，CSV：{csv_path}"
        )
        self.log_message.emit(summary)

    def run_led_qc_analyze(self):
        self._ensure_led_module()
        led_section = self.config.get('led_qc_enhanced', {}) or {}
        analyze_cfg = led_section.get('analyze', {}) or {}
        model_path = analyze_cfg.get('model') or led_section.get('model')
        image_path = analyze_cfg.get('image')
        if not model_path or not image_path:
            raise ValueError("請在 config 設定 led_qc_enhanced.analyze.model 與 image。")
        model_file = Path(model_path).expanduser()
        if not model_file.exists():
            raise FileNotFoundError(f"LED QC 模型檔案不存在：{model_file}")
        img_path = Path(image_path).expanduser()
        if not img_path.exists():
            raise FileNotFoundError(f"LED QC 圖片檔案不存在：{img_path}")
        overrides = self._collect_led_overrides(analyze_cfg, led_section)
        out_dir_value = analyze_cfg.get('out_dir')
        output_dir = None
        if out_dir_value:
            output_dir = Path(out_dir_value).expanduser()
            output_dir.mkdir(parents=True, exist_ok=True)
        tmp_model_path = None
        model_path_to_use = model_file
        if overrides:
            temp_model = led_qc_enhanced.EnhancedReferenceModel.from_json(model_file)
            temp_model.config.update(overrides)
            if hasattr(led_qc_enhanced, "set_active_colors"):
                led_qc_enhanced.set_active_colors(temp_model.config.get("colors"), temp_model.config.get("color_aliases"))
            target_dir = output_dir if output_dir else model_file.parent
            target_dir.mkdir(parents=True, exist_ok=True)
            tmp_model_path = target_dir / f"{model_file.stem}_analyze_tmp.json"
            temp_model.to_json(tmp_model_path)
            model_path_to_use = tmp_model_path
        else:
            if hasattr(led_qc_enhanced, "set_active_colors"):
                led_qc_enhanced.set_active_colors(led_section.get("colors"), led_section.get("color_aliases"))
        analyze_args = SimpleNamespace(
            model=str(model_path_to_use),
            image=str(img_path),
            visualize=self._as_bool(analyze_cfg.get('visualize', True)),
            stability=self._as_bool(analyze_cfg.get('stability', False)),
            out_dir=str(output_dir) if output_dir else None,
        )
        try:
            led_qc_enhanced.cmd_analyze(analyze_args)
        finally:
            if tmp_model_path and tmp_model_path.exists():
                try:
                    tmp_model_path.unlink()
                except OSError:
                    pass
        message = f"LED QC 分析圖片 {img_path}"
        if output_dir:
            message += f"結果路徑 {output_dir}"
        else:
            if analyze_args.visualize:
                message += "（結果已於視窗顯示）"
        self.log_message.emit(message)

class CompactCheckBox(QCheckBox):
    def __init__(self, text, parent=None):
        super().__init__(text, parent)
        self.setStyleSheet("""
            QCheckBox {
                font-size: 9pt;
                color: #2c3e50;
                spacing: 5px;
                padding: 2px;
                margin: 1px;
            }
            QCheckBox::indicator {
                width: 14px;
                height: 14px;
                border: 1px solid #ced4da;
                border-radius: 3px;
                background-color: white;
            }
            QCheckBox::indicator:checked {
                background-color: #007bff;
                border-color: #0056b3;
            }
            QCheckBox::indicator:hover {
                border-color: #80bdff;
            }
        """)

class CompactButton(QPushButton):
    """緊湊型按鈕"""
    def __init__(self, text, color_theme="primary", parent=None):
        super().__init__(text, parent)
        self.color_theme = color_theme
        self._setup_style()

    def _setup_style(self):
        base_style = """
            QPushButton {
                border: none;
                padding: 6px 12px;
                border-radius: 4px;
                font-size: 9pt;
                font-weight: bold;
                min-height: 24px;
                text-align: center;
            }
            QPushButton:disabled {
                background-color: #6c757d;
                color: #dee2e6;
            }
        """
        
        if self.color_theme == "primary":
            color_style = """
                QPushButton {
                    background-color: #007bff;
                    color: white;
                }
                QPushButton:hover {
                    background-color: #0056b3;
                }
                QPushButton:pressed {
                    background-color: #004085;
                }
            """
        elif self.color_theme == "success":
            color_style = """
                QPushButton {
                    background-color: #28a745;
                    color: white;
                }
                QPushButton:hover {
                    background-color: #218838;
                }
                QPushButton:pressed {
                    background-color: #1e7e34;
                }
            """
        elif self.color_theme == "danger":
            color_style = """
                QPushButton {
                    background-color: #dc3545;
                    color: white;
                }
                QPushButton:hover {
                    background-color: #c82333;
                }
                QPushButton:pressed {
                    background-color: #bd2130;
                }
            """
        else:
            color_style = base_style
            
        self.setStyleSheet(base_style + color_style)

class PictureToolGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.config = {}
        self.worker_thread = None
        self.init_ui()
        self.load_config()

    def init_ui(self):
        self.setWindowTitle('圖像處理工具 - 響應式GUI')
        self.setGeometry(100, 100, 1200, 800)
        self.setMinimumSize(800, 600)
        
        # 設置全局樣式
        self.setStyleSheet("""
            QMainWindow {
                background-color: #f8f9fa;
            }
            QGroupBox {
                font-size: 10pt;
                font-weight: bold;
                color: #2c3e50;
                border: 2px solid #dee2e6;
                border-radius: 6px;
                margin: 8px 2px 2px 2px;
                padding-top: 8px;
                background-color: white;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 8px;
                padding: 0 5px 0 5px;
            }
            QTextEdit {
                border: 1px solid #dee2e6;
                border-radius: 4px;
                background-color: #ffffff;
                font-family: 'Consolas', monospace;
                font-size: 9pt;
                padding: 8px;
            }
            QProgressBar {
                border: 1px solid #dee2e6;
                border-radius: 4px;
                background-color: #f8f9fa;
                text-align: center;
                font-size: 9pt;
                font-weight: bold;
                color: #2c3e50;
                height: 20px;
            }
            QProgressBar::chunk {
                background-color: #28a745;
                border-radius: 3px;
            }
            QComboBox, QLineEdit {
                padding: 4px 8px;
                border: 1px solid #ced4da;
                border-radius: 4px;
                background-color: white;
                font-size: 9pt;
                min-height: 16px;
            }
            QComboBox:hover, QLineEdit:hover {
                border-color: #80bdff;
            }
            QComboBox:focus, QLineEdit:focus {
                border-color: #007bff;
            }
            QLabel {
                font-size: 9pt;
                color: #495057;
            }
        """)

        # 中央部件
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # 使用 QSplitter 來創建可調整的佈局
        main_splitter = QSplitter(Qt.Horizontal)
        central_layout = QVBoxLayout(central_widget)
        central_layout.setContentsMargins(5, 5, 5, 5)
        central_layout.addWidget(main_splitter)

        # 左側面板（使用滾動區域）
        left_scroll = QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        left_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        left_scroll.setMinimumWidth(280)
        left_scroll.setMaximumWidth(400)
        
        left_panel = self.create_left_panel()
        left_scroll.setWidget(left_panel)

        # 右側面板
        right_panel = self.create_right_panel()

        # 添加到分割器
        main_splitter.addWidget(left_scroll)
        main_splitter.addWidget(right_panel)
        
        # 設置初始比例
        main_splitter.setSizes([300, 700])

    def create_left_panel(self):
        """創建左側控制面板"""
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setSpacing(8)
        left_layout.setContentsMargins(5, 5, 5, 5)

        # 配置文件選擇 - 緊湊版
        config_group = QGroupBox("配置設定")
        config_layout = QVBoxLayout(config_group)
        config_layout.setSpacing(6)
        
        # 配置文件路徑
        try:
            default_cfg_path = str((Path(__file__).parent / "config.yaml").resolve())
        except Exception:
            default_cfg_path = os.path.join("picture_tool", "config.yaml")
        
        self.config_path_edit = QLineEdit(default_cfg_path)
        self.config_path_edit.setToolTip("配置文件路徑")
        
        config_btn_layout = QHBoxLayout()
        config_btn_layout.setSpacing(4)
        config_browse_btn = CompactButton("瀏覽", "primary")
        config_browse_btn.clicked.connect(self.browse_config_file)
        reload_config_btn = CompactButton("重載", "primary")
        reload_config_btn.clicked.connect(self.load_config)
        
        config_btn_layout.addWidget(config_browse_btn)
        config_btn_layout.addWidget(reload_config_btn)
        
        config_layout.addWidget(QLabel("配置文件:"))
        config_layout.addWidget(self.config_path_edit)
        config_layout.addLayout(config_btn_layout)

        # 任務選擇 - 緊湊版
        task_group = QGroupBox("任務選擇")
        task_layout = QVBoxLayout(task_group)
        task_layout.setSpacing(2)
        # 收合切換列（任務選擇）
        all_toggle_bar = QHBoxLayout()
        all_toggle_bar.addStretch()
        all_toggle_btn = QToolButton()
        all_toggle_btn.setCheckable(True)
        all_toggle_btn.setChecked(True)
        all_toggle_btn.setArrowType(Qt.DownArrow)
        all_toggle_btn.setToolTip("展開/收起")
        all_toggle_bar.addWidget(all_toggle_btn)
        task_layout.addLayout(all_toggle_bar)

        # 可收合容器
        all_tasks_content = QWidget()
        all_tasks_content_layout = QVBoxLayout(all_tasks_content)
        all_tasks_content_layout.setSpacing(2)
        
        # 任務複選框 - 使用網格佈局節省空間
        task_grid = QGridLayout()
        task_grid.setSpacing(2)
        task_grid.setContentsMargins(2, 2, 2, 2)
        
        self.task_checkboxes = {}
        tasks = [
            "格式轉換", "異常檢測", "YOLO數據增強", 
            "圖像數據增強", "數據集分割", "YOLO訓練",
            "YOLO評估", "生成報告", "數據集檢查", 
            "增強預覽", "批次推論", "LED QC 建模", 
            "LED QC 單張檢測", "LED QC 批次檢測", "LED QC 分析"
        ]
        
        # 2列佈局
        for i, task in enumerate(tasks):
            checkbox = CompactCheckBox(task)
            self.task_checkboxes[task] = checkbox
            row = i // 2
            col = i % 2
            task_grid.addWidget(checkbox, row, col)
        
        all_tasks_content_layout.addLayout(task_grid)

        # 全選/取消全選按鈕
        select_layout = QHBoxLayout()
        select_layout.setSpacing(4)
        select_all_btn = CompactButton("全選", "success")
        deselect_all_btn = CompactButton("取消", "danger")
        select_all_btn.clicked.connect(self.select_all_tasks)
        deselect_all_btn.clicked.connect(self.deselect_all_tasks)
        select_layout.addWidget(select_all_btn)
        select_layout.addWidget(deselect_all_btn)
        all_tasks_content_layout.addLayout(select_layout)

        task_layout.addWidget(all_tasks_content)

        def _toggle_all(checked):
            all_tasks_content.setVisible(checked)
            all_toggle_btn.setArrowType(Qt.DownArrow if checked else Qt.RightArrow)
        all_toggle_btn.toggled.connect(_toggle_all)

        # YOLO 訓練群組（可收合）
        yolo_group = QGroupBox("YOLO 訓練")
        yolo_layout = QVBoxLayout(yolo_group)
        yolo_layout.setSpacing(2)

        yolo_toggle_bar = QHBoxLayout()
        yolo_toggle_bar.addStretch()
        yolo_toggle_btn = QToolButton()
        yolo_toggle_btn.setCheckable(True)
        yolo_toggle_btn.setChecked(True)
        yolo_toggle_btn.setArrowType(Qt.DownArrow)
        yolo_toggle_btn.setToolTip("展開/收起")
        yolo_toggle_bar.addWidget(yolo_toggle_btn)
        yolo_layout.addLayout(yolo_toggle_bar)

        yolo_content = QWidget()
        yolo_content_layout = QVBoxLayout(yolo_content)
        yolo_content_layout.setSpacing(2)
        yolo_grid = QGridLayout()
        yolo_grid.setSpacing(2)
        yolo_grid.setContentsMargins(2, 2, 2, 2)
        yolo_entries = [
            "YOLO數據增強",
            "數據分割",
            "YOLO訓練",
            "YOLO評估",
            "生成報告",
            "數據檢查",
            "增強預覽",
            "批次推論",
        ]
        for i, label_text in enumerate(yolo_entries):
            cb = CompactCheckBox(label_text)
            self.task_checkboxes[label_text] = cb
            row = i // 2
            col = i % 2
            yolo_grid.addWidget(cb, row, col)
        yolo_content_layout.addLayout(yolo_grid)

        yolo_select_layout = QHBoxLayout()
        yolo_select_layout.setSpacing(4)
        yolo_select_all = CompactButton("全選", "success")
        yolo_deselect_all = CompactButton("全不選", "danger")
        yolo_select_all.clicked.connect(self.select_all_tasks)
        yolo_deselect_all.clicked.connect(self.deselect_all_tasks)
        yolo_select_layout.addWidget(yolo_select_all)
        yolo_select_layout.addWidget(yolo_deselect_all)
        yolo_content_layout.addLayout(yolo_select_layout)
        yolo_layout.addWidget(yolo_content)

        def _toggle_yolo(checked):
            yolo_content.setVisible(checked)
            yolo_toggle_btn.setArrowType(Qt.DownArrow if checked else Qt.RightArrow)
        yolo_toggle_btn.toggled.connect(_toggle_yolo)

        # 訓練參數覆蓋 - 摺疊式設計
        options_group = QGroupBox("訓練參數")
        options_layout = QVBoxLayout(options_group)
        options_layout.setSpacing(4)
        # 收合切換列（訓練參數）
        opts_toggle_bar = QHBoxLayout()
        opts_toggle_bar.addStretch()
        opts_toggle_btn = QToolButton()
        opts_toggle_btn.setCheckable(True)
        opts_toggle_btn.setChecked(True)
        opts_toggle_btn.setArrowType(Qt.DownArrow)
        opts_toggle_btn.setToolTip("展開/收起")
        opts_toggle_bar.addWidget(opts_toggle_btn)
        options_layout.addLayout(opts_toggle_bar)
        
        self.apply_overrides_cb = CompactCheckBox("啟用參數覆蓋")
        options_layout.addWidget(self.apply_overrides_cb)
        
        # 參數輸入 - 2x2 網格
        param_grid = QGridLayout()
        param_grid.setSpacing(4)
        
        self.override_device_edit = QLineEdit()
        self.override_device_edit.setPlaceholderText("設備 (0/cpu)")
        self.override_epochs_edit = QLineEdit()
        self.override_epochs_edit.setPlaceholderText("輪數")
        self.override_imgsz_edit = QLineEdit()
        self.override_imgsz_edit.setPlaceholderText("圖片尺寸")
        self.override_batch_edit = QLineEdit()
        self.override_batch_edit.setPlaceholderText("批次大小")
        
        param_grid.addWidget(QLabel("Device:"), 0, 0)
        param_grid.addWidget(self.override_device_edit, 0, 1)
        param_grid.addWidget(QLabel("Epochs:"), 1, 0)
        param_grid.addWidget(self.override_epochs_edit, 1, 1)
        param_grid.addWidget(QLabel("ImgSz:"), 2, 0)
        param_grid.addWidget(self.override_imgsz_edit, 2, 1)
        param_grid.addWidget(QLabel("Batch:"), 3, 0)
        param_grid.addWidget(self.override_batch_edit, 3, 1)
        
        options_layout.addLayout(param_grid)
        
        # GPU偵測和強制選項
        option_buttons = QHBoxLayout()
        option_buttons.setSpacing(4)
        detect_btn = CompactButton("偵測GPU", "primary")
        def _detect_gpu():
            try:
                import torch
                self.override_device_edit.setText('0' if torch.cuda.is_available() else 'cpu')
            except Exception:
                self.override_device_edit.setText('cpu')
        detect_btn.clicked.connect(_detect_gpu)
        option_buttons.addWidget(detect_btn)
        
        self.force_cb = CompactCheckBox("強制重跑")
        option_buttons.addWidget(self.force_cb)
        options_layout.addLayout(option_buttons)
        # 收合切換：延後綁定，隱藏/顯示參數元件
        def _toggle_opts(checked):
            self.apply_overrides_cb.setVisible(checked)
            for w in [self.override_device_edit, self.override_epochs_edit, self.override_imgsz_edit, self.override_batch_edit]:
                w.setVisible(checked)
            for i in range(option_buttons.count()):
                item = option_buttons.itemAt(i)
                wid = item.widget()
                if wid:
                    wid.setVisible(checked)
            opts_toggle_btn.setArrowType(Qt.DownArrow if checked else Qt.RightArrow)
        opts_toggle_btn.toggled.connect(_toggle_opts)

        # 執行控制
        control_group = QGroupBox("執行控制")
        control_layout = QVBoxLayout(control_group)
        control_layout.setSpacing(6)
        
        # 執行按鈕
        control_btn_layout = QHBoxLayout()
        control_btn_layout.setSpacing(6)
        self.start_btn = CompactButton("▶ 開始", "success")
        self.stop_btn = CompactButton("⏹ 停止", "danger")
        self.stop_btn.setEnabled(False)
        
        self.start_btn.clicked.connect(self.start_pipeline)
        self.stop_btn.clicked.connect(self.stop_pipeline)
        
        control_btn_layout.addWidget(self.start_btn)
        control_btn_layout.addWidget(self.stop_btn)
        control_layout.addLayout(control_btn_layout)

        # 進度條
        self.progress_bar = QProgressBar()
        self.progress_bar.setTextVisible(True)
        control_layout.addWidget(self.progress_bar)

        # 狀態標籤
        self.status_label = QLabel("就緒")
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setStyleSheet("""
            QLabel {
                background-color: #d4edda;
                color: #155724;
                padding: 6px;
                border: 1px solid #c3e6cb;
                border-radius: 4px;
                font-weight: bold;
            }
        """)
        control_layout.addWidget(self.status_label)

        # 添加到主佈局
        left_layout.addWidget(config_group)
        # 插入 YOLO 訓練群組於任務選擇之前
        left_layout.addWidget(yolo_group)
        left_layout.addWidget(task_group)
        left_layout.addWidget(options_group)
        left_layout.addWidget(control_group)
        left_layout.addStretch()

        return left_widget

    def create_right_panel(self):
        """創建右側內容面板"""
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setSpacing(8)
        right_layout.setContentsMargins(5, 5, 5, 5)

        # 標題
        title_label = QLabel("圖像處理工具管理")
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet("""
            QLabel {
                font-size: 14pt;
                font-weight: bold;
                color: #2c3e50;
                padding: 10px;
                background-color: #e9ecef;
                border: 1px solid #dee2e6;
                border-radius: 6px;
            }
        """)

        # Tab Widget - 響應式設計
        self.tab_widget = QTabWidget()
        self.tab_widget.setStyleSheet("""
            QTabWidget::pane {
                border: 1px solid #dee2e6;
                border-radius: 4px;
                background-color: white;
                padding: 5px;
            }
            QTabBar::tab {
                background-color: #f8f9fa;
                color: #495057;
                padding: 8px 16px;
                margin-right: 2px;
                border: 1px solid #dee2e6;
                border-top-left-radius: 4px;
                border-top-right-radius: 4px;
                font-size: 9pt;
                font-weight: bold;
            }
            QTabBar::tab:selected {
                background-color: #007bff;
                color: white;
                border-bottom: none;
            }
            QTabBar::tab:hover:!selected {
                background-color: #e9ecef;
            }
        """)

        # 日誌標籤頁
        log_widget = QWidget()
        log_layout = QVBoxLayout(log_widget)
        log_layout.setSpacing(5)
        
        self.log_text = QTextEdit()
        self.log_text.setPlainText("歡迎使用圖像處理工具！\n請選擇任務並點擊開始執行。\n")
        self.log_text.setStyleSheet("""
            QTextEdit {
                background-color: #f8f9fa;
                color: #2c3e50;
                border: 1px solid #dee2e6;
                border-radius: 4px;
                font-family: 'Consolas', monospace;
                font-size: 9pt;
                padding: 8px;
            }
        """)
        
        clear_log_btn = CompactButton("清空日誌", "danger")
        clear_log_btn.clicked.connect(self.clear_log)
        
        log_layout.addWidget(self.log_text)
        
        # 按鈕佈局
        log_btn_layout = QHBoxLayout()
        log_btn_layout.addStretch()
        log_btn_layout.addWidget(clear_log_btn)
        log_layout.addLayout(log_btn_layout)
        
        # 配置預覽標籤頁
        config_widget = QWidget()
        config_layout = QVBoxLayout(config_widget)
        config_layout.setSpacing(5)
        
        self.config_text = QTextEdit()
        self.config_text.setPlainText("請載入配置文件...")
        self.config_text.setStyleSheet("""
            QTextEdit {
                background-color: #ffffff;
                border: 1px solid #dee2e6;
                border-radius: 4px;
                font-family: 'Consolas', monospace;
                font-size: 9pt;
                padding: 8px;
            }
        """)
        
        config_layout.addWidget(self.config_text)
        
        # 添加標籤頁
        self.tab_widget.addTab(log_widget, "📋 執行日誌")
        self.tab_widget.addTab(config_widget, "⚙️ 配置預覽")

        # 添加到右側佈局
        right_layout.addWidget(title_label)
        right_layout.addWidget(self.tab_widget)

        return right_widget

    def browse_config_file(self):
        """瀏覽配置文件"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "選擇配置文件", "", "YAML files (*.yaml *.yml)"
        )
        if file_path:
            self.config_path_edit.setText(file_path)
            self.load_config()

    def load_default_config(self):
        """載入默認配置"""
        default_config = {
            "pipeline": {
                "log_file": "pipeline.log",
                "tasks": [
                    {"name": "format_conversion", "enabled": True},
                    {"name": "anomaly_detection", "enabled": True},
                    {"name": "yolo_augmentation", "enabled": True},
                    {"name": "image_augmentation", "enabled": True},
                    {"name": "dataset_splitter", "enabled": True},
                    {"name": "yolo_train", "enabled": False},
                    {"name": "yolo_evaluation", "enabled": False},
                    {"name": "generate_report", "enabled": True},
                    {"name": "dataset_lint", "enabled": True},
                    {"name": "aug_preview", "enabled": True},
                ]
            },
            "format_conversion": {
                "input_formats": [".bmp", ".tiff"],
                "output_format": ".png",
                "input_dir": "input/",
                "output_dir": "output/"
            }
        }
        self.config = default_config
        self.update_config_display()

    def load_config(self):
        """載入配置文件"""
        config_path = self.config_path_edit.text()
        try:
            if os.path.exists(config_path):
                with open(config_path, 'r', encoding='utf-8') as f:
                    self.config = yaml.safe_load(f)
                self.log_message(f"成功載入配置文件: {config_path}")
                self.update_config_display()
            else:
                self.log_message(f"配置文件不存在: {config_path}")
                self.load_default_config()
        except Exception as e:
            self.log_message(f"載入配置文件失敗: {str(e)}")
            self.load_default_config()

    def update_config_display(self):
        """更新配置顯示"""
        config_text = yaml.dump(self.config, default_flow_style=False, 
                               allow_unicode=True, indent=2)
        self.config_text.setPlainText(config_text)

    def select_all_tasks(self):
        """全選任務"""
        for checkbox in self.task_checkboxes.values():
            checkbox.setChecked(True)

    def deselect_all_tasks(self):
        """取消全選任務"""
        for checkbox in self.task_checkboxes.values():
            checkbox.setChecked(False)

    def get_selected_tasks(self):
        """獲取選中的任務"""
        selected_tasks = []
        for task_name, checkbox in self.task_checkboxes.items():
            if checkbox.isChecked():
                selected_tasks.append(task_name)
        return selected_tasks

    def start_pipeline(self):
        """開始執行管道"""
        selected_tasks = self.get_selected_tasks()
        if not selected_tasks:
            QMessageBox.warning(self, "警告", "請至少選擇一個任務！")
            return

        # 更新UI狀態
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.progress_bar.setValue(0)
        self.status_label.setText("執行中...")
        self.status_label.setStyleSheet("""
            QLabel {
                background-color: #fff3cd;
                color: #856404;
                padding: 6px;
                border: 1px solid #ffeaa7;
                border-radius: 4px;
                font-weight: bold;
            }
        """)

        # 應用覆蓋與強制選項
        try:
            if hasattr(self, 'apply_overrides_cb') and self.apply_overrides_cb.isChecked():
                yt = self.config.get('yolo_training', {}) or {}
                dev = self.override_device_edit.text().strip()
                ep = self.override_epochs_edit.text().strip()
                im = self.override_imgsz_edit.text().strip()
                bt = self.override_batch_edit.text().strip()
                if dev:
                    yt['device'] = dev
                if ep.isdigit():
                    yt['epochs'] = int(ep)
                if im.isdigit():
                    yt['imgsz'] = int(im)
                if bt.isdigit():
                    yt['batch'] = int(bt)
                self.config['yolo_training'] = yt
                ye = self.config.get('yolo_evaluation', {}) or {}
                if yt.get('device'):
                    ye['device'] = yt['device']
                self.config['yolo_evaluation'] = ye
                bi = self.config.get('batch_inference', {}) or {}
                if yt.get('device'):
                    bi['device'] = yt['device']
                self.config['batch_inference'] = bi
            if hasattr(self, 'force_cb') and self.force_cb.isChecked():
                pl = self.config.get('pipeline', {}) or {}
                pl['force'] = True
                self.config['pipeline'] = pl
        except Exception:
            pass

        self.worker_thread = WorkerThread(selected_tasks, self.config, self.config_path_edit.text())
        self.worker_thread.progress_updated.connect(self.update_progress)
        self.worker_thread.task_started.connect(self.on_task_started)
        self.worker_thread.task_completed.connect(self.on_task_completed)
        self.worker_thread.log_message.connect(self.log_message)
        self.worker_thread.finished_signal.connect(self.on_pipeline_finished)
        self.worker_thread.error_occurred.connect(self.on_error_occurred)
        
        self.worker_thread.start()
        self.log_message(f"開始執行管道，選中的任務: {', '.join(selected_tasks)}")

    def stop_pipeline(self):
        """停止執行管道"""
        if self.worker_thread:
            self.worker_thread.cancel()
            self.log_message("正在停止執行...")

    def update_progress(self, value):
        """更新進度條"""
        self.progress_bar.setValue(value)

    def on_task_started(self, task_name):
        """任務開始時的回調"""
        self.log_message(f"⚡ 開始執行任務: {task_name}")

    def on_task_completed(self, task_name):
        """任務完成時的回調"""
        self.log_message(f"✅ 任務完成: {task_name}")

    def on_pipeline_finished(self):
        """管道執行完成時的回調"""
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.status_label.setText("完成")
        self.status_label.setStyleSheet("""
            QLabel {
                background-color: #d4edda;
                color: #155724;
                padding: 6px;
                border: 1px solid #c3e6cb;
                border-radius: 4px;
                font-weight: bold;
            }
        """)
        self.log_message("🎉 所有任務執行完成！")

    def on_error_occurred(self, error_message):
        """錯誤發生時的回調"""
        self.log_message(f"❌ 錯誤: {error_message}")
        QMessageBox.critical(self, "錯誤", f"執行過程中發生錯誤:\n{error_message}")
        self.on_pipeline_finished()

    def log_message(self, message):
        """添加日誌訊息"""
        from datetime import datetime
        timestamp = datetime.now().strftime("%H:%M:%S")
        formatted_message = f"[{timestamp}] {message}"
        self.log_text.append(formatted_message)
        # 自動滾動到底部
        scrollbar = self.log_text.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def clear_log(self):
        """清空日誌"""
        self.log_text.clear()
        self.log_message("日誌已清空")

    def closeEvent(self, event):
        """窗口關閉事件"""
        if self.worker_thread and self.worker_thread.isRunning():
            reply = QMessageBox.question(
                self, "確認退出", 
                "任務正在執行中，確定要退出嗎？",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No
            )
            if reply == QMessageBox.Yes:
                self.worker_thread.cancel()
                self.worker_thread.wait()
                event.accept()
            else:
                event.ignore()
        else:
            event.accept()

def main():
    """主函數"""
    app = QApplication(sys.argv)
    
    # 設置應用程式
    app.setApplicationName("圖像處理工具")
    app.setApplicationVersion("1.0")
    app.setOrganizationName("ImageTool Corp")
    
    # 設置全局字體 - 使用系統預設字體
    font = app.font()
    font.setPointSize(9)
    app.setFont(font)
    
    # 創建主窗口
    window = PictureToolGUI()
    window.show()
    
    # 運行應用程式
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()

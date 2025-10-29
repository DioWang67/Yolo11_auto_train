from __future__ import annotations

import copy
import csv
import hashlib
import json
import os
import time
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from typing import Callable, Dict


import importlib.util
import sys


def _load_led_module_fallback():
    """Load led_qc_enhanced without importing heavy package init."""
    base_dir = Path(__file__).resolve().parent.parent
    candidates = [
        base_dir / "picture_tool" / "color" / "led_qc_enhanced.py",
    ]
    for candidate in candidates:
        if candidate.exists():
            spec = importlib.util.spec_from_file_location(
                "picture_tool_fallback_led_qc", candidate
            )
            module = importlib.util.module_from_spec(spec)
            loader = spec.loader
            if loader is None:
                continue
            sys.modules.setdefault(spec.name, module)
            loader.exec_module(module)
            return module
    return None


try:
    from picture_tool.picture_tool.color import led_qc_enhanced  # type: ignore
except Exception:
    try:
        from picture_tool.color import led_qc_enhanced  # type: ignore
    except Exception:
        led_qc_enhanced = _load_led_module_fallback()


class LedQcManager:
    """Encapsulates LED QC configuration loading and task execution for the GUI."""

    def __init__(self, log_callback: Callable[[str], None]):
        self._log = log_callback

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    def _ensure_led_module(self) -> None:
        if led_qc_enhanced is None:
            raise RuntimeError("LED QC 模組未可用，請確認相依套件已正確安裝。")

    @staticmethod
    def _as_bool(value) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "y", "on"}
        return bool(value)

    def _collect_led_overrides(self, sub_cfg: Dict, led_section: Dict) -> Dict:
        overrides: Dict = {}
        for source_section in (led_section, sub_cfg):
            if not isinstance(source_section, dict):
                continue
            for key in ("config_overrides", "overrides"):
                candidate = source_section.get(key)
                if isinstance(candidate, dict):
                    overrides.update(candidate)
        return overrides

    def _prepare_led_config(self, sub_cfg: Dict, led_section: Dict) -> Dict:
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

        overrides = self._collect_led_overrides(
            sub_cfg if isinstance(sub_cfg, dict) else {},
            led_section if isinstance(led_section, dict) else {},
        )
        if overrides:
            cfg.update(overrides)

        simple_keys = (
            "colors",
            "color_aliases",
            "color_hue_ranges",
            "color_conf_min_per_color",
        )
        for source in (led_section, sub_cfg):
            if not isinstance(source, dict):
                continue
            for key in simple_keys:
                if key in source and source[key] is not None:
                    cfg[key] = copy.deepcopy(source[key])

        if hasattr(led_qc_enhanced, "set_active_colors"):
            led_qc_enhanced.set_active_colors(
                cfg.get("colors"),
                cfg.get("color_aliases"),
                cfg.get("color_hue_ranges"),
            )
        return cfg

    # ------------------------------------------------------------------
    # public api
    # ------------------------------------------------------------------
    def build(self, root_config: Dict) -> None:
        self._ensure_led_module()
        led_section = root_config.get("led_qc_enhanced", {}) or {}
        build_cfg = led_section.get("build", {}) or {}

        ref_dir = build_cfg.get("ref_dir") or led_section.get("ref_dir")
        model_out = build_cfg.get("model_out")
        if not ref_dir or not model_out:
            raise ValueError(
                "請在 config.yaml 設定 led_qc_enhanced.build.ref_dir 與 model_out。"
            )

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
                self._log(f"LED QC 設定檔已輸出：{cfg_path}")

        self._log(f"LED QC 模型建立完成：{out_path}，收錄樣本 {model.total_samples}")

    def detect_single(self, root_config: Dict) -> None:
        self._ensure_led_module()
        led_section = root_config.get("led_qc_enhanced", {}) or {}
        detect_cfg = led_section.get("detect", {}) or {}

        model_path = detect_cfg.get("model") or led_section.get("model")
        image_path = detect_cfg.get("image")
        if not model_path or not image_path:
            raise ValueError("請提供 led_qc_enhanced.detect.model 與 image。")

        model_file = Path(model_path).expanduser()
        if not model_file.exists():
            raise FileNotFoundError(f"LED QC 模型檔案不存在：{model_file}")
        model = led_qc_enhanced.EnhancedReferenceModel.from_json(model_file)

        overrides = self._collect_led_overrides(detect_cfg, led_section)
        if overrides:
            model.config.update(overrides)

        img_path = Path(image_path).expanduser()
        if not img_path.exists():
            raise FileNotFoundError(f"LED QC 圖片檔案不存在：{img_path}")
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
        self._log(
            f"LED QC 單張檢測 {img_path.name}: {status} "
            f"conf={result.confidence:.2f} color={result.color_used} "
            f"color_conf={result.color_confidence:.2f} "
            f"hue_cov={result.color_hue_coverage.get(result.color_used, 0.0):.2f} "
            f"嚴重度={result.severity_score:.2f}"
        )
        if result.reasons:
            self._log("原因：" + "；".join(result.reasons))

        out_dir = detect_cfg.get("out_dir")
        if out_dir:
            out_path = Path(out_dir).expanduser()
            led_qc_enhanced.ensure_dir(out_path)

            if self._as_bool(detect_cfg.get("save_annotated", True)):
                annotated = led_qc_enhanced.enhanced_annotate(img, result)
                ann_path = out_path / f"{img_path.stem}_annotated.png"
                led_qc_enhanced.cv2.imwrite(str(ann_path), annotated)
                self._log(f"已輸出標註影像：{ann_path}")

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
                        "color_diagnostics": {
                            "distances": result.color_distances,
                            "hue_coverage": result.color_hue_coverage,
                        },
                    },
                    "features": asdict(result.features),
                }
                json_path = out_path / f"{img_path.stem}_result.json"
                with open(json_path, "w", encoding="utf-8") as f:
                    json.dump(
                        payload,
                        f,
                        ensure_ascii=False,
                        indent=2,
                        default=led_qc_enhanced._json_default,
                    )
                self._log(f"已輸出檢測結果 JSON：{json_path}")

    def detect_dir(self, root_config: Dict) -> None:
        self._ensure_led_module()
        led_section = root_config.get("led_qc_enhanced", {}) or {}
        dir_cfg = led_section.get("detect_dir", {}) or {}

        model_path = dir_cfg.get("model") or led_section.get("model")
        input_dir = dir_cfg.get("dir") or dir_cfg.get("input_dir")
        out_dir = dir_cfg.get("out_dir") or dir_cfg.get("output_dir")
        if not model_path or not input_dir or not out_dir:
            raise ValueError(
                "請提供 led_qc_enhanced.detect_dir 的 model、dir 與 out_dir。"
            )

        model_file = Path(model_path).expanduser()
        if not model_file.exists():
            raise FileNotFoundError(f"LED QC 模型檔案不存在：{model_file}")
        model = led_qc_enhanced.EnhancedReferenceModel.from_json(model_file)

        overrides = self._collect_led_overrides(dir_cfg, led_section)
        if overrides:
            model.config.update(overrides)

        images_dir = Path(input_dir).expanduser()
        if not images_dir.exists():
            raise FileNotFoundError(f"LED QC 檢測資料夾不存在：{images_dir}")

        output_dir = Path(out_dir).expanduser()
        led_qc_enhanced.ensure_dir(output_dir)

        try:
            sensitivity = float(dir_cfg.get("sensitivity", 0.85))
        except Exception:
            sensitivity = 0.85

        valid_suffixes = {
            suffix.lower()
            for suffix in getattr(led_qc_enhanced, "SUPPORTED_FORMATS", [])
        }
        raw_paths = [
            p
            for p in images_dir.rglob("*")
            if p.is_file()
            and (not valid_suffixes or p.suffix.lower() in valid_suffixes)
        ]
        dedup = {}
        for path in raw_paths:
            key = os.path.normcase(os.path.abspath(str(path)))
            dedup.setdefault(key, path)
        paths = sorted(dedup.values())

        if not paths:
            self._log(f"LED QC 批次檢測未找到影像：{images_dir}")
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
                digest = hashlib.md5(
                    os.path.normcase(os.path.abspath(str(path))).encode("utf-8")
                ).hexdigest()[:8]
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
                        "color_diagnostics": {
                            "distances": det.color_distances,
                            "hue_coverage": det.color_hue_coverage,
                        },
                    },
                    "features": asdict(det.features),
                }
                json_file = output_dir / f"{path.stem}_result.json"
                with open(json_file, "w", encoding="utf-8") as f:
                    json.dump(
                        payload,
                        f,
                        ensure_ascii=False,
                        indent=2,
                        default=led_qc_enhanced._json_default,
                    )

            results.append(
                {
                    "file": str(path),
                    "is_anomaly": int(det.is_anomaly),
                    "severity_score": float(det.severity_score),
                    "color": det.color_used,
                    "color_conf": float(det.color_confidence),
                    "conf": float(det.confidence),
                    "time_ms": float(det.processing_time * 1000.0),
                    "anomaly_boxes": ";".join(
                        f"{x},{y},{w},{h}" for x, y, w, h in det.anomaly_regions
                    ),
                    "reasons": "; ".join(det.reasons),
                    "error": "",
                    "area_ratio": float(det.features.area_ratio),
                    "valid_mask": int(det.features.valid_mask),
                    "mean_v": float(det.features.mean_v),
                    "uniformity": float(det.features.uniformity),
                    "hole_ratio": float(det.features.hole_ratio),
                    "color_dist": float(det.color_distances.get(det.color_used, 0.0)),
                    "color_hue_cov": float(
                        det.color_hue_coverage.get(det.color_used, 0.0)
                    ),
                }
            )

            if idx % 10 == 0 or idx == len(paths):
                self._log(f"LED QC 批次檢測進度：{idx}/{len(paths)}")

        elapsed = time.time() - start_time
        csv_path = output_dir / csv_name
        fieldnames = [
            "file",
            "is_anomaly",
            "severity_score",
            "color",
            "color_conf",
            "conf",
            "time_ms",
            "anomaly_boxes",
            "reasons",
            "error",
            "area_ratio",
            "valid_mask",
            "mean_v",
            "uniformity",
            "hole_ratio",
            "color_dist",
            "color_hue_cov",
        ]
        with open(csv_path, "w", newline="", encoding="utf-8-sig") as csvfile:
            writer = csv.DictWriter(
                csvfile, fieldnames=fieldnames, extrasaction="ignore"
            )
            writer.writeheader()
            writer.writerows(results)

        total = len(results)
        per_image = (elapsed / total * 1000.0) if total else 0.0
        summary = (
            f"LED QC 批次檢測完成：{total} 張，異常 {anomalies}，錯誤 {errors}，"
            f"耗時 {elapsed:.1f}s (~{per_image:.1f}ms/張)，CSV：{csv_path}"
        )
        self._log(summary)

    def analyze(self, root_config: Dict) -> None:
        self._ensure_led_module()
        led_section = root_config.get("led_qc_enhanced", {}) or {}
        analyze_cfg = led_section.get("analyze", {}) or {}

        model_path = analyze_cfg.get("model") or led_section.get("model")
        image_path = analyze_cfg.get("image")
        if not model_path or not image_path:
            raise ValueError(
                "請在 config 設定 led_qc_enhanced.analyze.model 與 image。"
            )

        model_file = Path(model_path).expanduser()
        if not model_file.exists():
            raise FileNotFoundError(f"LED QC 模型檔案不存在：{model_file}")
        img_path = Path(image_path).expanduser()
        if not img_path.exists():
            raise FileNotFoundError(f"LED QC 圖片檔案不存在：{img_path}")

        overrides = self._collect_led_overrides(analyze_cfg, led_section)
        out_dir_value = analyze_cfg.get("out_dir")
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
                led_qc_enhanced.set_active_colors(
                    temp_model.config.get("colors"),
                    temp_model.config.get("color_aliases"),
                    temp_model.config.get("color_hue_ranges"),
                )
            target_dir = output_dir if output_dir else model_file.parent
            target_dir.mkdir(parents=True, exist_ok=True)
            tmp_model_path = target_dir / f"{model_file.stem}_analyze_tmp.json"
            temp_model.to_json(tmp_model_path)
            model_path_to_use = tmp_model_path
        else:
            if hasattr(led_qc_enhanced, "set_active_colors"):
                led_qc_enhanced.set_active_colors(
                    led_section.get("colors"),
                    led_section.get("color_aliases"),
                    led_section.get("color_hue_ranges"),
                )

        analyze_args = SimpleNamespace(
            model=str(model_path_to_use),
            image=str(img_path),
            visualize=self._as_bool(analyze_cfg.get("visualize", True)),
            stability=self._as_bool(analyze_cfg.get("stability", False)),
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

        message = f"LED QC 分析完成：{img_path}"
        if output_dir:
            message += f"，結果輸出於 {output_dir}"
        else:
            if analyze_args.visualize:
                message += "（結果已於視窗顯示）"
        self._log(message)

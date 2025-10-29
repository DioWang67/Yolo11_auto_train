"""Metrics dashboard helpers for the auto-train GUI."""

from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path
from typing import Any, List


class MetricsDashboardMixin:
    def bind_metrics_display(self, widget: Any) -> None:
        self._metrics_text_widget = widget

    def _get_metrics_text_widget(self) -> Any | None:
        return getattr(self, "_metrics_text_widget", None) or getattr(
            self, "metrics_text", None
        )

    def refresh_metrics_dashboard(self) -> None:
        metrics_text = self._get_metrics_text_widget()
        if metrics_text is None:
            return
        summary_lines: List[str] = []
        summary_lines.append("📊 YOLO 訓練摘要")
        yolo_stats = self._load_latest_yolo_metrics()
        if yolo_stats:
            summary_lines.extend(yolo_stats)
        else:
            summary_lines.append("尚未找到訓練結案 (runs/detect)")
        summary_lines.append("")
        summary_lines.append("💡 LED QC 統計")
        led_stats = self._load_latest_led_metrics()
        if led_stats:
            summary_lines.extend(led_stats)
        else:
            summary_lines.append("尚未找到 LED QC 報表 (reports/led_qc)")
        metrics_text.setPlainText("\n".join(summary_lines))

    def _load_latest_yolo_metrics(self) -> List[str]:
        search_roots: List[Path] = []
        if isinstance(self.config, dict):
            project = self.config.get("yolo_training", {}).get("project")
            if isinstance(project, str) and project:
                search_roots.append(Path(project))
        for fallback in ("./runs/train", "./runs/detect"):
            candidate = Path(fallback)
            if candidate not in search_roots:
                search_roots.append(candidate)

        candidates: List[Path] = []
        for root in search_roots:
            try:
                if root.exists():
                    candidates.extend(root.glob("**/results.csv"))
            except Exception:  # pragma: no cover - best effort file search
                continue
        if not candidates:
            return []

        latest = max(candidates, key=lambda p: p.stat().st_mtime)
        try:
            with latest.open("r", encoding="utf-8") as fh:
                rows = list(csv.DictReader(fh))
        except Exception:  # pragma: no cover - best effort file read
            return []
        if not rows:
            return []

        last = rows[-1]

        def pick(*keys: str) -> str:
            for key in keys:
                value = last.get(key)
                if value not in (None, "", "nan"):
                    return str(value)
            return "N/A"

        epoch = pick("epoch", "Epoch", "epochs")
        map50 = pick("metrics/mAP50", "metrics/mAP_50", "mAP50", "map50")
        map50_95 = pick("metrics/mAP50-95", "metrics/mAP_50_95", "mAP50-95", "map50-95")
        precision = pick("metrics/precision(B)", "precision", "metrics/precision")
        recall = pick("metrics/recall(B)", "recall", "metrics/recall")
        box_loss = pick("train/box_loss", "box_loss", "loss/box")
        cls_loss = pick("train/cls_loss", "cls_loss", "loss/cls")

        last_update = datetime.fromtimestamp(latest.stat().st_mtime).strftime(
            "%Y-%m-%d %H:%M"
        )
        metrics = [
            f"結案資料夾：{self._format_relative_path(latest.parent)}",
            f"最後更新：{last_update}",
            f"最新 Epoch：{epoch}",
            f"mAP50：{map50}",
            f"mAP50-95：{map50_95}",
            f"Precision：{precision}",
            f"Recall：{recall}",
        ]
        if box_loss != "N/A" or cls_loss != "N/A":
            metrics.append(f"Box/Cls Loss：{box_loss} / {cls_loss}")
        return metrics

    def _load_latest_led_metrics(self) -> List[str]:
        led_config = (
            self.config.get("led_qc_enhanced", {})
            if isinstance(self.config, dict)
            else {}
        )
        detect_dir_cfg = (
            led_config.get("detect_dir", {}) if isinstance(led_config, dict) else {}
        )
        led_dir = Path(detect_dir_cfg.get("out_dir") or "./reports/led_qc/batch")
        fallback_dirs = [Path("./reports/led_qc"), Path("./reports/led_qc/batch")]

        search_roots: List[Path] = []
        for root in [led_dir, *fallback_dirs]:
            if root not in search_roots:
                search_roots.append(root)

        candidates: List[Path] = []
        for root in search_roots:
            if root and root.exists():
                candidates.extend(root.glob("**/*.csv"))
        if not candidates:
            return []

        latest = max(candidates, key=lambda p: p.stat().st_mtime)
        try:
            with latest.open("r", encoding="utf-8") as fh:
                rows = list(csv.DictReader(fh))
        except Exception:  # pragma: no cover - best effort file read
            return []
        if not rows:
            return []

        total = len(rows)
        pass_markers = {"PASS", "OK", "合格", "正常", "通過"}
        fail_markers = {"FAIL", "NG", "異常", "不良", "FAILURE"}

        anomalies = 0
        for row in rows:
            raw_status = None
            for key in ("color_status", "status", "result", "decision"):
                if row.get(key):
                    raw_status = row.get(key)
                    break
            status = (raw_status or "").strip()
            normalized = status.upper()
            if not status:
                continue
            if normalized in pass_markers or status in pass_markers:
                continue
            if normalized in fail_markers or status in fail_markers or status:
                anomalies += 1

        pass_count = total - anomalies
        pass_rate = (pass_count / total * 100) if total else 0.0
        anomaly_rate = (anomalies / total * 100) if total else 0.0
        last_update = datetime.fromtimestamp(latest.stat().st_mtime).strftime(
            "%Y-%m-%d %H:%M"
        )

        metrics = [
            f"統計檔案：{self._format_relative_path(latest)}",
            f"最後更新：{last_update}",
            f"檢測總數：{total}",
        ]
        if total:
            metrics.append(f"疑似異常：{anomalies} ({anomaly_rate:.1f}%)")
            metrics.append(f"判定通過：{pass_count}")
            metrics.append(f"通過率：{pass_rate:.1f}%")
        return metrics

    @staticmethod
    def _format_relative_path(path: Path) -> str:
        try:
            return str(path.resolve().relative_to(Path.cwd()))
        except Exception:  # pragma: no cover - path resolution best effort
            return str(path)

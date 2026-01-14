from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional


def _load_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None


def _count_csv_rows(path: Path) -> int:
    try:
        with path.open("r", encoding="utf-8", newline="") as fh:
            reader = csv.reader(fh)
            # Skip header if present
            rows = list(reader)
            return max(len(rows) - 1, 0) if rows else 0
    except Exception:
        return 0


def generate_qc_summary(
    config: dict,
    output_path: Optional[Path] = None,
    logger: Optional[logging.Logger] = None,
) -> Path:
    """Aggregate QC artifacts (color, position, detection) into one report."""
    logger = logger or logging.getLogger(__name__)
    out_dir = Path("reports/qc_summary")
    out_path = Path(output_path) if output_path else out_dir / "qc_summary.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    color_cfg = config.get("color_verification", {}) or {}
    color_json = Path(color_cfg.get("output_json") or "./reports/led_qc/verify.json")
    position_cfg = (config.get("yolo_training", {}) or {}).get(
        "position_validation", {}
    ) or {}
    position_dir = Path(
        position_cfg.get("output_dir") or "./reports/position_validation"
    )
    position_json = position_dir / "position_validation.json"
    infer_cfg = config.get("batch_inference", {}) or {}
    infer_csv = Path(infer_cfg.get("output_dir", "./reports/infer")) / "predictions.csv"

    color_data = _load_json(color_json) if color_json.exists() else None
    position_data = _load_json(position_json) if position_json.exists() else None

    summary: Dict[str, Any] = {
        "color_verification": {
            "path": str(color_json),
            "exists": color_json.exists(),
            "count": len(color_data.get("records", []))
            if color_data and isinstance(color_data, dict)
            else None,
        },
        "position_validation": {
            "path": str(position_json),
            "exists": position_json.exists(),
            "status_counts": position_data.get("summary", {}).get("status_counts")
            if position_data
            else None,
            "samples": position_data.get("summary", {}).get("samples")
            if position_data
            else None,
        },
        "detection": {
            "path": str(infer_csv),
            "exists": infer_csv.exists(),
            "predictions": _count_csv_rows(infer_csv) if infer_csv.exists() else None,
        },
    }

    out_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    logger.info("QC summary written to %s", out_path)
    return out_path

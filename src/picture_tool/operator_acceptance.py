"""Read-only acceptance summary for a completed operator model deployment."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml


class OperatorAcceptanceError(RuntimeError):
    """Deployment evidence is missing or cannot be trusted."""


@dataclass(frozen=True)
class OperatorAcceptanceSummary:
    """Validated evidence displayed after an operator training job."""

    product: str
    area: str
    deployed_version: str
    gate_passed: bool
    precision: float
    recall: float
    map50: float
    map50_95: float
    color_model_source: str
    manifest_path: Path

    def to_operator_text(self) -> str:
        color_note = (
            "沿用原站點顏色設定，這次沒有重新校正"
            if self.color_model_source == "existing_station"
            else f"來源：{self.color_model_source or '未記錄'}"
        )
        gate_text = "通過" if self.gate_passed else "未通過"
        return (
            f"{self.product}/{self.area} 已部署版本：{self.deployed_version}\n"
            f"離線 Evaluation Gate：{gate_text}\n"
            f"Precision {self.precision:.3f}  |  Recall {self.recall:.3f}\n"
            f"mAP50 {self.map50:.3f}  |  mAP50-95 {self.map50_95:.3f}\n"
            f"顏色校正：{color_note}\n\n"
            "注意：離線 Gate 通過不等於產線驗收完成。請用未參與訓練的 "
            "OK/NG 圖片，並在相同機台、相機與光源下試跑後再確認。"
        )


def load_operator_acceptance_summary(
    inference_models_dir: str | Path,
    *,
    product: str,
    area: str,
) -> OperatorAcceptanceSummary:
    """Load and validate the deployed station's evaluation evidence."""
    station_dir = Path(inference_models_dir).resolve() / product / area / "yolo"
    manifest_path = station_dir / "deployment_manifest.yaml"
    try:
        payload = yaml.safe_load(manifest_path.read_text(encoding="utf-8")) or {}
    except FileNotFoundError as exc:
        raise OperatorAcceptanceError(
            f"Deployment acceptance manifest is missing: {manifest_path}"
        ) from exc
    except (OSError, UnicodeDecodeError, yaml.YAMLError) as exc:
        raise OperatorAcceptanceError(
            f"Deployment acceptance manifest cannot be read: {exc}"
        ) from exc
    if not isinstance(payload, Mapping):
        raise OperatorAcceptanceError("Deployment acceptance manifest is invalid.")
    if payload.get("evaluation_gate_passed") is not True:
        raise OperatorAcceptanceError(
            "The deployed model has no passing evaluation gate evidence."
        )
    metrics = payload.get("evaluation_metrics")
    if not isinstance(metrics, Mapping):
        raise OperatorAcceptanceError("Deployment evaluation metrics are missing.")
    validated_metrics = {
        name: _validated_metric(metrics, name)
        for name in ("precision", "recall", "map50", "map50_95")
    }
    version = str(payload.get("deployed_version") or "").strip()
    if not version:
        raise OperatorAcceptanceError("Deployed model version is missing.")
    return OperatorAcceptanceSummary(
        product=product,
        area=area,
        deployed_version=version,
        gate_passed=True,
        precision=validated_metrics["precision"],
        recall=validated_metrics["recall"],
        map50=validated_metrics["map50"],
        map50_95=validated_metrics["map50_95"],
        color_model_source=str(payload.get("color_model_source") or "").strip(),
        manifest_path=manifest_path,
    )


def _validated_metric(metrics: Mapping[str, Any], name: str) -> float:
    value = metrics.get(name)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise OperatorAcceptanceError(f"Evaluation metric {name} is missing or invalid.")
    metric = float(value)
    if not math.isfinite(metric) or not 0.0 <= metric <= 1.0:
        raise OperatorAcceptanceError(
            f"Evaluation metric {name} must be between 0 and 1."
        )
    return metric

"""Read-only acceptance summary for a completed operator model deployment."""

from __future__ import annotations

import hashlib
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
    position_runtime_enabled: bool
    position_gate_required: bool
    position_gate_passed: bool | None
    position_ok_false_reject_rate: float | None
    position_ng_recall: float | None
    model_acceptance_gate_passed: bool | None
    model_acceptance_confirmed: int | None
    model_acceptance_accuracy: float | None
    model_acceptance_false_positives: int | None
    model_acceptance_false_negatives: int | None
    model_acceptance_overkill_rate: float | None
    manifest_path: Path

    def to_operator_text(self) -> str:
        color_note = (
            "沿用原站點顏色設定，這次沒有重新校正"
            if self.color_model_source == "existing_station"
            else f"來源：{self.color_model_source or '未記錄'}"
        )
        gate_text = "通過" if self.gate_passed else "未通過"
        if self.position_gate_required:
            position_gate_text = (
                "通過" if self.position_gate_passed is True else "未通過"
            )
        else:
            position_gate_text = "未要求"
        position_runtime_text = (
            "已啟用" if self.position_runtime_enabled else "維持停用"
        )
        position_metric_text = (
            f"位置 OK 誤殺率 {self.position_ok_false_reject_rate:.3%}"
            if self.position_ok_false_reject_rate is not None
            else "位置 OK 誤殺率未記錄"
        )
        if self.position_ng_recall is not None:
            position_metric_text += (
                f"  |  位置 NG Recall {self.position_ng_recall:.3%}"
            )
        if self.model_acceptance_gate_passed is True:
            model_acceptance_text = (
                "固定驗收集：通過"
                f"  |  {self.model_acceptance_confirmed or 0} 張"
                f"  |  Accuracy {self.model_acceptance_accuracy or 0.0:.2%}"
                f"  |  FP {self.model_acceptance_false_positives or 0}"
                f"  |  FN {self.model_acceptance_false_negatives or 0}"
                f"  |  誤殺率 {self.model_acceptance_overkill_rate or 0.0:.2%}"
            )
        else:
            model_acceptance_text = "固定驗收集：未啟用"
        return (
            f"{self.product}/{self.area} 已部署版本：{self.deployed_version}\n"
            f"離線 Evaluation Gate：{gate_text}\n"
            f"Precision {self.precision:.3f}  |  Recall {self.recall:.3f}\n"
            f"mAP50 {self.map50:.3f}  |  mAP50-95 {self.map50_95:.3f}\n"
            f"{model_acceptance_text}\n"
            f"Position Gate：{position_gate_text}｜產線位置檢查："
            f"{position_runtime_text}\n"
            f"{position_metric_text}\n"
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
    acceptance_gate_value = payload.get("model_acceptance_gate_passed")
    acceptance_gate_passed = (
        acceptance_gate_value
        if isinstance(acceptance_gate_value, bool)
        else None
    )
    if acceptance_gate_passed is False:
        raise OperatorAcceptanceError(
            "The deployed model has a failing fixed-set acceptance gate."
        )
    acceptance_confirmed: int | None = None
    acceptance_accuracy: float | None = None
    acceptance_false_positives: int | None = None
    acceptance_false_negatives: int | None = None
    acceptance_overkill_rate: float | None = None
    if acceptance_gate_passed is True:
        _verify_relative_evidence(
            station_dir,
            payload,
            path_field="model_acceptance_report",
            hash_field="model_acceptance_report_sha256",
            label="model acceptance",
        )
        acceptance_metrics = payload.get("model_acceptance_metrics")
        if not isinstance(acceptance_metrics, Mapping):
            raise OperatorAcceptanceError(
                "Deployment model acceptance metrics are missing."
            )
        acceptance_confirmed = _validated_count(
            acceptance_metrics, "confirmed"
        )
        acceptance_false_positives = _validated_count(
            acceptance_metrics, "fp"
        )
        acceptance_false_negatives = _validated_count(
            acceptance_metrics, "fn"
        )
        acceptance_accuracy = _validated_metric(
            acceptance_metrics, "accuracy"
        )
        acceptance_overkill_rate = _validated_metric(
            acceptance_metrics, "overkill_rate"
        )
    position_gate_required = bool(payload.get("position_gate_required", False))
    position_gate_passed_value = payload.get("position_gate_passed")
    position_gate_passed = (
        bool(position_gate_passed_value)
        if isinstance(position_gate_passed_value, bool)
        else None
    )
    if position_gate_required and position_gate_passed is not True:
        raise OperatorAcceptanceError(
            "The deployed position configuration has no passing position gate evidence."
        )
    if position_gate_required:
        _verify_relative_evidence(
            station_dir,
            payload,
            path_field="position_gate_report",
            hash_field="position_gate_sha256",
            label="position gate",
        )
        _verify_relative_evidence(
            station_dir,
            payload,
            path_field="position_validation_report",
            hash_field="position_validation_sha256",
            label="position validation",
        )
    for path_field, hash_field, label in (
        (
            "position_baseline_report",
            "position_baseline_sha256",
            "position baseline",
        ),
        (
            "position_calibration_manifest",
            "position_calibration_sha256",
            "position calibration",
        ),
    ):
        if payload.get(path_field) or payload.get(hash_field):
            _verify_relative_evidence(
                station_dir,
                payload,
                path_field=path_field,
                hash_field=hash_field,
                label=label,
            )
    position_metrics = payload.get("position_metrics")
    if position_metrics is not None and not isinstance(position_metrics, Mapping):
        raise OperatorAcceptanceError("Deployment position metrics are invalid.")
    position_metrics = (
        position_metrics if isinstance(position_metrics, Mapping) else {}
    )
    position_false_reject_rate = _optional_validated_metric(
        position_metrics,
        "ok_false_reject_rate",
    )
    position_ng_recall = _optional_validated_metric(
        position_metrics,
        "ng_recall",
    )
    if position_gate_required and position_false_reject_rate is None:
        raise OperatorAcceptanceError(
            "Deployment position false-reject evidence is missing."
        )
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
        position_runtime_enabled=bool(
            payload.get("position_runtime_enabled", False)
        ),
        position_gate_required=position_gate_required,
        position_gate_passed=position_gate_passed,
        position_ok_false_reject_rate=position_false_reject_rate,
        position_ng_recall=position_ng_recall,
        model_acceptance_gate_passed=acceptance_gate_passed,
        model_acceptance_confirmed=acceptance_confirmed,
        model_acceptance_accuracy=acceptance_accuracy,
        model_acceptance_false_positives=acceptance_false_positives,
        model_acceptance_false_negatives=acceptance_false_negatives,
        model_acceptance_overkill_rate=acceptance_overkill_rate,
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


def _optional_validated_metric(
    metrics: Mapping[str, Any],
    name: str,
) -> float | None:
    value = metrics.get(name)
    if value is None:
        return None
    return _validated_metric(metrics, name)


def _validated_count(metrics: Mapping[str, Any], name: str) -> int:
    value = metrics.get(name)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise OperatorAcceptanceError(
            f"Model acceptance count {name} is missing or invalid."
        )
    return value


def _verify_relative_evidence(
    station_dir: Path,
    payload: Mapping[str, Any],
    *,
    path_field: str,
    hash_field: str,
    label: str,
) -> None:
    relative_value = str(payload.get(path_field) or "").strip()
    expected_hash = str(payload.get(hash_field) or "").strip().lower()
    if not relative_value or len(expected_hash) != 64:
        raise OperatorAcceptanceError(
            f"Deployment {label} evidence path or checksum is missing."
        )
    evidence_path = (station_dir / relative_value).resolve()
    if not evidence_path.is_relative_to(station_dir):
        raise OperatorAcceptanceError(
            f"Deployment {label} evidence path is outside the station."
        )
    if not evidence_path.is_file():
        raise OperatorAcceptanceError(
            f"Deployment {label} evidence is missing: {evidence_path}"
        )
    if _sha256_file(evidence_path) != expected_hash:
        raise OperatorAcceptanceError(
            f"Deployment {label} evidence checksum does not match."
        )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

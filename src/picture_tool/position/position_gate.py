"""Fail-closed promotion gate for offline position validation.

The gate consumes immutable JSON-compatible reports and returns a decision.
File I/O stays in thin boundary helpers so comparison rules remain testable
without an ML runtime.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping


class PositionGateError(RuntimeError):
    """Raised when required position evidence is missing or malformed."""


@dataclass(frozen=True)
class PositionGatePolicy:
    """Validated thresholds used to promote one position configuration."""

    min_ok_samples: int = 1
    max_ok_false_reject_rate: float = 0.0
    min_ng_samples: int = 0
    min_ng_recall: float = 0.0
    require_baseline: bool = False
    max_ok_false_reject_regression: float = 0.0
    max_ng_recall_regression: float = 0.0
    require_disjoint_calibration: bool = True

    def __post_init__(self) -> None:
        for name in ("min_ok_samples", "min_ng_samples"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(
                    f"{name} must be a non-negative integer."
                )
        for name in (
            "max_ok_false_reject_rate",
            "min_ng_recall",
            "max_ok_false_reject_regression",
            "max_ng_recall_regression",
        ):
            raw_value = getattr(self, name)
            if isinstance(raw_value, bool):
                raise ValueError(f"{name} must be between 0 and 1.")
            value = float(raw_value)
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be between 0 and 1.")
        for name in ("require_baseline", "require_disjoint_calibration"):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"{name} must be a boolean.")

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any] | None) -> "PositionGatePolicy":
        data = dict(raw or {})
        allowed = set(cls.__dataclass_fields__)
        return cls(**{key: value for key, value in data.items() if key in allowed})


@dataclass(frozen=True)
class PositionMetrics:
    """Normalized metrics extracted from one validation report."""

    ok_samples: int
    ok_false_rejects: int
    ok_false_reject_rate: float
    ng_samples: int
    ng_detected: int
    ng_recall: float | None


@dataclass(frozen=True)
class PositionGateDecision:
    """Immutable position promotion result."""

    passed: bool
    failures: tuple[str, ...]
    warnings: tuple[str, ...]
    metrics: PositionMetrics
    baseline_metrics: PositionMetrics | None
    calibration_overlap_count: int
    policy: PositionGatePolicy

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["schema_version"] = 1
        payload["failures"] = list(self.failures)
        payload["warnings"] = list(self.warnings)
        return payload


def evaluate_position_gate(
    candidate_report: Mapping[str, Any],
    *,
    policy: PositionGatePolicy,
    baseline_report: Mapping[str, Any] | None = None,
    calibration_manifest: Mapping[str, Any] | None = None,
) -> PositionGateDecision:
    """Evaluate absolute, regression, and data-leakage requirements."""

    candidate = position_metrics_from_report(candidate_report)
    _validate_report_sample_identities(candidate_report, "candidate")
    baseline = (
        position_metrics_from_report(baseline_report)
        if baseline_report is not None
        else None
    )
    if baseline_report is not None:
        _validate_report_sample_identities(baseline_report, "baseline")
    failures: list[str] = []
    warnings: list[str] = []

    if candidate.ok_samples < policy.min_ok_samples:
        failures.append(
            "insufficient_ok_samples: "
            f"required={policy.min_ok_samples}, actual={candidate.ok_samples}"
        )
    if candidate.ok_false_reject_rate > policy.max_ok_false_reject_rate:
        failures.append(
            "ok_false_reject_rate_exceeded: "
            f"maximum={policy.max_ok_false_reject_rate:.6f}, "
            f"actual={candidate.ok_false_reject_rate:.6f}"
        )
    if candidate.ng_samples < policy.min_ng_samples:
        failures.append(
            "insufficient_ng_samples: "
            f"required={policy.min_ng_samples}, actual={candidate.ng_samples}"
        )
    if policy.min_ng_samples > 0:
        if candidate.ng_recall is None:
            failures.append("ng_recall_missing")
        elif candidate.ng_recall < policy.min_ng_recall:
            failures.append(
                "ng_recall_below_minimum: "
                f"minimum={policy.min_ng_recall:.6f}, "
                f"actual={candidate.ng_recall:.6f}"
            )
    elif candidate.ng_samples == 0:
        warnings.append("ng_golden_set_not_configured")

    if policy.require_baseline and baseline is None:
        failures.append("baseline_position_report_missing")
    if baseline is not None:
        assert baseline_report is not None
        candidate_hashes = _sample_hashes(candidate_report.get("records"))
        baseline_hashes = _sample_hashes(baseline_report.get("records"))
        if candidate_hashes and baseline_hashes:
            if candidate_hashes != baseline_hashes:
                failures.append(
                    "baseline_golden_set_mismatch: "
                    f"candidate={len(candidate_hashes)}, "
                    f"baseline={len(baseline_hashes)}"
                )
        elif policy.require_baseline:
            failures.append("baseline_sample_identity_missing")
        else:
            warnings.append("baseline_sample_identity_not_verified")
        if (
            candidate.ok_false_reject_rate - baseline.ok_false_reject_rate
            > policy.max_ok_false_reject_regression
        ):
            failures.append(
                "ok_false_reject_regression: "
                f"allowed={policy.max_ok_false_reject_regression:.6f}, "
                f"baseline={baseline.ok_false_reject_rate:.6f}, "
                f"candidate={candidate.ok_false_reject_rate:.6f}"
            )
        if baseline.ng_recall is not None and candidate.ng_recall is not None:
            if (
                baseline.ng_recall - candidate.ng_recall
                > policy.max_ng_recall_regression
            ):
                failures.append(
                    "ng_recall_regression: "
                    f"allowed={policy.max_ng_recall_regression:.6f}, "
                    f"baseline={baseline.ng_recall:.6f}, "
                    f"candidate={candidate.ng_recall:.6f}"
                )

    if policy.require_disjoint_calibration and calibration_manifest is None:
        failures.append("calibration_manifest_missing")
    elif policy.require_disjoint_calibration:
        assert calibration_manifest is not None
        _validate_calibration_manifest(calibration_manifest)
    overlap_count = _calibration_overlap_count(candidate_report, calibration_manifest)
    if overlap_count:
        message = f"calibration_golden_overlap: count={overlap_count}"
        if policy.require_disjoint_calibration:
            failures.append(message)
        else:
            warnings.append(message)

    return PositionGateDecision(
        passed=not failures,
        failures=tuple(failures),
        warnings=tuple(warnings),
        metrics=candidate,
        baseline_metrics=baseline,
        calibration_overlap_count=overlap_count,
        policy=policy,
    )


def position_metrics_from_report(report: Mapping[str, Any]) -> PositionMetrics:
    """Extract metrics from schema-v1 or legacy position reports."""

    summary = report.get("summary")
    if not isinstance(summary, Mapping):
        raise PositionGateError("Position validation report has no summary object.")
    metrics = summary.get("metrics")
    if isinstance(metrics, Mapping):
        normalized = PositionMetrics(
            ok_samples=_non_negative_int(metrics.get("ok_samples"), "ok_samples"),
            ok_false_rejects=_non_negative_int(
                metrics.get("ok_false_rejects"),
                "ok_false_rejects",
            ),
            ok_false_reject_rate=_rate(
                metrics.get("ok_false_reject_rate"),
                "ok_false_reject_rate",
            ),
            ng_samples=_non_negative_int(metrics.get("ng_samples"), "ng_samples"),
            ng_detected=_non_negative_int(metrics.get("ng_detected"), "ng_detected"),
            ng_recall=_optional_rate(metrics.get("ng_recall"), "ng_recall"),
        )
        _validate_metric_consistency(normalized)
        return normalized

    records = report.get("records")
    if not isinstance(records, list):
        raise PositionGateError("Position validation report has no records array.")
    ok_samples = 0
    ok_false_rejects = 0
    ng_samples = 0
    ng_detected = 0
    for raw_record in records:
        if not isinstance(raw_record, Mapping):
            raise PositionGateError("Position validation record must be an object.")
        expected_status = str(raw_record.get("expected_status") or "PASS").upper()
        validation = raw_record.get("validation")
        if not isinstance(validation, Mapping):
            raise PositionGateError(
                "Position validation record has no validation object."
            )
        actual_status = str(validation.get("status") or "").upper()
        if actual_status not in {"PASS", "FAIL", "SKIPPED"}:
            raise PositionGateError(
                f"Unsupported position validation status: {actual_status or '<empty>'}"
            )
        if expected_status == "PASS":
            ok_samples += 1
            if actual_status != "PASS":
                ok_false_rejects += 1
        elif expected_status == "FAIL":
            ng_samples += 1
            if actual_status == "FAIL":
                ng_detected += 1
        else:
            raise PositionGateError(
                f"Unsupported expected position status: {expected_status}"
            )
    return PositionMetrics(
        ok_samples=ok_samples,
        ok_false_rejects=ok_false_rejects,
        ok_false_reject_rate=_safe_ratio(ok_false_rejects, ok_samples),
        ng_samples=ng_samples,
        ng_detected=ng_detected,
        ng_recall=(
            _safe_ratio(ng_detected, ng_samples) if ng_samples > 0 else None
        ),
    )


def load_json_mapping(path: Path, label: str) -> dict[str, Any]:
    """Load one required JSON object with a domain-specific error."""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise PositionGateError(f"{label} was not found: {path}") from exc
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PositionGateError(f"Unable to read {label}: {exc}") from exc
    if not isinstance(payload, dict):
        raise PositionGateError(f"{label} must contain a JSON object.")
    return payload


def write_position_gate_report(
    path: Path,
    decision: PositionGateDecision,
    *,
    product: str,
    area: str,
    candidate_report_path: Path,
    baseline_report_path: Path | None,
    calibration_manifest_path: Path | None,
) -> Path:
    """Atomically publish the position promotion decision."""

    payload = decision.as_dict()
    payload.update(
        {
            "product": product,
            "area": area,
            "candidate_report": str(candidate_report_path.resolve()),
            "candidate_report_sha256": _sha256_file(candidate_report_path),
            "baseline_report": (
                str(baseline_report_path.resolve())
                if baseline_report_path is not None
                else None
            ),
            "baseline_report_sha256": (
                _sha256_file(baseline_report_path)
                if baseline_report_path is not None
                else None
            ),
            "calibration_manifest": (
                str(calibration_manifest_path.resolve())
                if calibration_manifest_path is not None
                else None
            ),
            "calibration_manifest_sha256": (
                _sha256_file(calibration_manifest_path)
                if calibration_manifest_path is not None
                else None
            ),
        }
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary.replace(path)
    return path


def _calibration_overlap_count(
    candidate_report: Mapping[str, Any],
    calibration_manifest: Mapping[str, Any] | None,
) -> int:
    if calibration_manifest is None:
        return 0
    calibration_hashes = _sample_hashes(calibration_manifest.get("samples"))
    validation_hashes = _sample_hashes(candidate_report.get("records"))
    return len(calibration_hashes & validation_hashes)


def _sample_hashes(raw_samples: Any) -> set[str]:
    if not isinstance(raw_samples, list):
        return set()
    hashes: set[str] = set()
    for sample in raw_samples:
        if not isinstance(sample, Mapping):
            continue
        value = str(sample.get("image_sha256") or "").lower()
        if len(value) == 64 and all(char in "0123456789abcdef" for char in value):
            hashes.add(value)
    return hashes


def _validate_report_sample_identities(
    report: Mapping[str, Any],
    label: str,
) -> None:
    schema_version = report.get("schema_version")
    requires_hashes = isinstance(schema_version, int) and schema_version >= 2
    records = report.get("records")
    if not isinstance(records, list):
        if requires_hashes:
            raise PositionGateError(
                f"{label} position report has no records array."
            )
        return
    seen: set[str] = set()
    for index, record in enumerate(records):
        if not isinstance(record, Mapping):
            if requires_hashes:
                raise PositionGateError(
                    f"{label} position record {index} must be an object."
                )
            continue
        image_hash = str(record.get("image_sha256") or "").strip().lower()
        valid_hash = len(image_hash) == 64 and all(
            character in "0123456789abcdef" for character in image_hash
        )
        if not valid_hash:
            if requires_hashes:
                raise PositionGateError(
                    f"{label} position record {index} has no valid image SHA-256."
                )
            continue
        if image_hash in seen:
            raise PositionGateError(
                f"{label} position report contains duplicate image content."
            )
        seen.add(image_hash)


def _validate_calibration_manifest(
    manifest: Mapping[str, Any],
) -> None:
    samples = manifest.get("samples")
    if not isinstance(samples, list) or not samples:
        raise PositionGateError(
            "Position calibration manifest has no sample records."
        )
    seen: set[str] = set()
    for index, sample in enumerate(samples):
        if not isinstance(sample, Mapping):
            raise PositionGateError(
                f"Position calibration sample {index} must be an object."
            )
        image_hash = str(sample.get("image_sha256") or "").strip().lower()
        if len(image_hash) != 64 or any(
            character not in "0123456789abcdef" for character in image_hash
        ):
            raise PositionGateError(
                f"Position calibration sample {index} has no valid image SHA-256."
            )
        if image_hash in seen:
            raise PositionGateError(
                "Position calibration manifest contains duplicate image content."
            )
        seen.add(image_hash)


def _non_negative_int(value: Any, field_name: str) -> int:
    if isinstance(value, bool):
        raise PositionGateError(f"{field_name} must be a non-negative integer.")
    if isinstance(value, float) and not value.is_integer():
        raise PositionGateError(f"{field_name} must be a non-negative integer.")
    try:
        normalized = int(value)
    except (TypeError, ValueError) as exc:
        raise PositionGateError(
            f"{field_name} must be a non-negative integer."
        ) from exc
    if normalized < 0:
        raise PositionGateError(f"{field_name} must be a non-negative integer.")
    return normalized


def _rate(value: Any, field_name: str) -> float:
    try:
        normalized = float(value)
    except (TypeError, ValueError) as exc:
        raise PositionGateError(f"{field_name} must be a rate.") from exc
    if not math.isfinite(normalized) or not 0.0 <= normalized <= 1.0:
        raise PositionGateError(f"{field_name} must be between 0 and 1.")
    return normalized


def _optional_rate(value: Any, field_name: str) -> float | None:
    return None if value is None else _rate(value, field_name)


def _safe_ratio(numerator: int, denominator: int) -> float:
    return float(numerator) / float(denominator) if denominator else 0.0


def _validate_metric_consistency(metrics: PositionMetrics) -> None:
    if metrics.ok_false_rejects > metrics.ok_samples:
        raise PositionGateError("ok_false_rejects must not exceed ok_samples.")
    expected_false_reject_rate = _safe_ratio(
        metrics.ok_false_rejects,
        metrics.ok_samples,
    )
    if not math.isclose(
        metrics.ok_false_reject_rate,
        expected_false_reject_rate,
        abs_tol=1e-9,
    ):
        raise PositionGateError(
            "ok_false_reject_rate does not match the supplied sample counts."
        )
    if metrics.ng_detected > metrics.ng_samples:
        raise PositionGateError("ng_detected must not exceed ng_samples.")
    expected_ng_recall = (
        _safe_ratio(metrics.ng_detected, metrics.ng_samples)
        if metrics.ng_samples
        else None
    )
    if metrics.ng_recall != expected_ng_recall and not (
        metrics.ng_recall is not None
        and expected_ng_recall is not None
        and math.isclose(metrics.ng_recall, expected_ng_recall, abs_tol=1e-9)
    ):
        raise PositionGateError(
            "ng_recall does not match the supplied sample counts."
        )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

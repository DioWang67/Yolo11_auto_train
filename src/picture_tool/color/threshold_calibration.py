"""Offline color-threshold recommendation and explicitly approved deployment."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import time
from collections.abc import Iterable, Iterator
from contextlib import ExitStack, contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class CalibrationPolicy:
    """Safety and evidence requirements for one threshold recommendation."""

    minimum_total: int = 30
    minimum_ok: int = 5
    minimum_ng: int = 5
    false_accept_cost: float = 10.0
    false_reject_cost: float = 1.0
    maximum_false_accept_rate: float = 0.0

    def __post_init__(self) -> None:
        if min(self.minimum_total, self.minimum_ok, self.minimum_ng) < 1:
            raise ValueError("Calibration sample minimums must be positive")
        if self.false_accept_cost <= 0 or self.false_reject_cost <= 0:
            raise ValueError("Calibration costs must be positive")
        if not 0.0 <= self.maximum_false_accept_rate <= 1.0:
            raise ValueError("maximum_false_accept_rate must be between 0 and 1")


@dataclass(frozen=True)
class ColorFeedbackSample:
    """Validated item-level feedback exported by the inference review UI."""

    product: str
    area: str
    model_type: str
    checker_type: str
    threshold_key: str
    failure_kind: str
    sample_id: str
    item_index: str
    diff: float
    runtime_threshold: float
    actual_is_ok: bool


def load_feedback(paths: Iterable[str | Path]) -> list[ColorFeedbackSample]:
    """Load and deduplicate feedback manifests; the last review wins."""
    latest: dict[tuple[str, str, str, str], ColorFeedbackSample] = {}
    for path in _feedback_paths(paths):
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            for row_number, row in enumerate(csv.DictReader(handle), start=2):
                try:
                    sample = _parse_feedback_row(row)
                except ValueError as exc:
                    raise ValueError(f"{path}:{row_number}: {exc}") from exc
                key = (
                    sample.product,
                    sample.area,
                    sample.sample_id,
                    sample.item_index,
                )
                latest[key] = sample
    return list(latest.values())


def build_recommendation(
    feedback_paths: Iterable[str | Path],
    *,
    policy: CalibrationPolicy | None = None,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    """Build a shadow-mode report; this function never changes model config."""
    active_policy = policy or CalibrationPolicy()
    samples = load_feedback(feedback_paths)
    threshold_samples = [
        sample for sample in samples if sample.failure_kind == "threshold"
    ]
    recommendations = recommend_color_thresholds(
        threshold_samples,
        policy=active_policy,
    )
    payload: dict[str, Any] = {
        "schema_version": 1,
        "mode": "shadow",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "policy": asdict(active_policy),
        "feedback_item_count": len(samples),
        "threshold_feedback_item_count": len(threshold_samples),
        "rule_feedback_item_count": len(samples) - len(threshold_samples),
        "recommendations": recommendations,
    }
    payload["report_id"] = _report_id(payload)
    if output_path is not None:
        _write_json_atomic(Path(output_path), payload)
    return payload


def recommend_color_thresholds(
    samples: Iterable[ColorFeedbackSample],
    *,
    policy: CalibrationPolicy | None = None,
) -> list[dict[str, Any]]:
    """Return deterministic, side-effect-free recommendations for validated samples.

    This is the reusable application boundary used by the inference processing
    pipeline.  It intentionally excludes timestamps, report IDs, file writes,
    CLI output, and active-config mutation.
    """
    active_policy = policy or CalibrationPolicy()
    grouped: dict[tuple[str, str, str, str, str], list[ColorFeedbackSample]] = {}
    for sample in samples:
        if sample.failure_kind != "threshold":
            continue
        key = (
            sample.product,
            sample.area,
            sample.model_type,
            sample.checker_type,
            sample.threshold_key,
        )
        grouped.setdefault(key, []).append(sample)
    return [
        _recommend_group(key, group, active_policy)
        for key, group in sorted(grouped.items())
    ]


def apply_recommendations(
    report_path: str | Path,
    *,
    models_root: str | Path,
    approver: str,
) -> dict[str, Any]:
    """Apply ready recommendations after explicit named human approval.

    Every config is validated before any replacement. Config replacements are
    atomic, and a failure rolls already-replaced files back to their exact
    original bytes.
    """
    approval_name = approver.strip()
    if not approval_name:
        raise ValueError("approver is required; automatic approval is forbidden")
    source_report = Path(report_path).expanduser().resolve()
    report = _load_json_mapping(source_report)
    if int(report.get("schema_version") or 0) != 1:
        raise ValueError("Unsupported color calibration report schema")
    expected_report_id = str(report.get("report_id") or "")
    unsigned = {key: value for key, value in report.items() if key != "report_id"}
    if not expected_report_id or expected_report_id != _report_id(unsigned):
        raise ValueError("Calibration report checksum mismatch")
    ready = [
        item
        for item in report.get("recommendations", [])
        if isinstance(item, dict) and item.get("status") == "ready"
    ]
    if not ready:
        raise ValueError("Calibration report has no ready recommendation")

    root = Path(models_root).expanduser().resolve()
    updates_by_config: dict[Path, list[dict[str, Any]]] = {}
    for recommendation in ready:
        config_path = _target_config_path(root, recommendation)
        updates_by_config.setdefault(config_path, []).append(recommendation)

    originals: dict[Path, bytes] = {}
    next_configs: dict[Path, dict[str, Any]] = {}
    locks = sorted(updates_by_config, key=str)
    with ExitStack() as stack:
        for config_path in locks:
            stack.enter_context(_config_lock(config_path))
        for config_path, updates in updates_by_config.items():
            originals[config_path] = config_path.read_bytes()
            config = _load_yaml_mapping(config_path)
            for recommendation in updates:
                _validate_config_has_not_drifted(config, recommendation)
                _set_config_threshold(config, recommendation)
            next_configs[config_path] = config

        timestamp = datetime.now(timezone.utc)
        backup_paths: dict[Path, Path] = {}
        for config_path, original in originals.items():
            backup_dir = config_path.parent / "color_threshold_backups"
            backup_dir.mkdir(parents=True, exist_ok=True)
            backup_path = backup_dir / (
                f"{timestamp.strftime('%Y%m%dT%H%M%S%fZ')}_config.yaml"
            )
            _write_bytes_atomic(backup_path, original)
            backup_paths[config_path] = backup_path

        receipt = {
            "schema_version": 1,
            "report_id": expected_report_id,
            "report_path": str(source_report),
            "approved_by": approval_name,
            "applied_at": timestamp.isoformat(),
            "updates": [
                {
                    "config_path": str(config_path),
                    "backup_path": str(backup_paths[config_path]),
                    "recommendations": updates_by_config[config_path],
                }
                for config_path in locks
            ],
        }
        history_paths = [
            config_path.parent / "color_threshold_history.json"
            for config_path in locks
        ]
        history_originals = {
            path: path.read_bytes() if path.is_file() else None
            for path in history_paths
        }
        replaced: list[Path] = []
        written_histories: list[Path] = []
        try:
            for config_path in locks:
                _write_yaml_atomic(config_path, next_configs[config_path])
                replaced.append(config_path)
            for history_path in history_paths:
                _append_history_atomic(history_path, receipt)
                written_histories.append(history_path)
        except (OSError, json.JSONDecodeError, ValueError):
            for config_path in reversed(replaced):
                _write_bytes_atomic(config_path, originals[config_path])
            for history_path in reversed(written_histories):
                original = history_originals[history_path]
                if original is None:
                    history_path.unlink(missing_ok=True)
                else:
                    _write_bytes_atomic(history_path, original)
            raise
    return receipt


def _recommend_group(
    key: tuple[str, str, str, str, str],
    samples: list[ColorFeedbackSample],
    policy: CalibrationPolicy,
) -> dict[str, Any]:
    product, area, model_type, checker_type, threshold_key = key
    ok_count = sum(sample.actual_is_ok for sample in samples)
    ng_count = len(samples) - ok_count
    current_threshold = _mode_float(
        sample.runtime_threshold for sample in samples
    )
    current_metrics = _metrics(samples, current_threshold, policy)
    result: dict[str, Any] = {
        "product": product,
        "area": area,
        "model_type": model_type,
        "checker_type": checker_type,
        "threshold_key": threshold_key,
        "sample_count": len(samples),
        "ok_count": ok_count,
        "ng_count": ng_count,
        "current_public_threshold": current_threshold,
        "current_config_value": _public_to_config(
            checker_type, current_threshold
        ),
        "current_metrics": current_metrics,
    }
    evidence_reasons = []
    if len(samples) < policy.minimum_total:
        evidence_reasons.append(
            f"need at least {policy.minimum_total} total samples"
        )
    if ok_count < policy.minimum_ok:
        evidence_reasons.append(f"need at least {policy.minimum_ok} actual-OK samples")
    if ng_count < policy.minimum_ng:
        evidence_reasons.append(f"need at least {policy.minimum_ng} actual-NG samples")
    if evidence_reasons:
        result.update({"status": "insufficient_data", "reasons": evidence_reasons})
        return result

    candidates = sorted(
        {current_threshold, *(sample.diff for sample in samples)}
    )
    evaluated = [
        (candidate, _metrics(samples, candidate, policy))
        for candidate in candidates
    ]
    safe = [
        item
        for item in evaluated
        if item[1]["false_accept_rate"]
        <= policy.maximum_false_accept_rate + 1e-12
    ]
    if not safe:
        result.update(
            {
                "status": "blocked_by_safety_policy",
                "reasons": ["no candidate satisfies maximum false-accept rate"],
            }
        )
        return result
    best_threshold, best_metrics = min(
        safe,
        key=lambda item: (
            item[1]["weighted_cost"],
            abs(item[0] - current_threshold),
            item[0],
        ),
    )
    if (
        abs(best_threshold - current_threshold) <= 1e-12
        or best_metrics["weighted_cost"] >= current_metrics["weighted_cost"]
    ):
        result.update(
            {
                "status": "no_change",
                "reasons": ["no safer candidate improves weighted error cost"],
                "suggested_public_threshold": current_threshold,
                "suggested_config_value": _public_to_config(
                    checker_type, current_threshold
                ),
                "suggested_metrics": current_metrics,
            }
        )
        return result
    result.update(
        {
            "status": "ready",
            "reasons": [],
            "suggested_public_threshold": best_threshold,
            "suggested_config_value": _public_to_config(
                checker_type, best_threshold
            ),
            "suggested_metrics": best_metrics,
        }
    )
    return result


def _metrics(
    samples: list[ColorFeedbackSample],
    threshold: float,
    policy: CalibrationPolicy,
) -> dict[str, Any]:
    false_accept = sum(
        not sample.actual_is_ok and sample.diff <= threshold for sample in samples
    )
    false_reject = sum(
        sample.actual_is_ok and sample.diff > threshold for sample in samples
    )
    ok_count = sum(sample.actual_is_ok for sample in samples)
    ng_count = len(samples) - ok_count
    return {
        "false_accept": false_accept,
        "false_reject": false_reject,
        "false_accept_rate": false_accept / ng_count if ng_count else 0.0,
        "false_reject_rate": false_reject / ok_count if ok_count else 0.0,
        "weighted_cost": (
            false_accept * policy.false_accept_cost
            + false_reject * policy.false_reject_cost
        ),
    }


def _parse_feedback_row(row: dict[str, str]) -> ColorFeedbackSample:
    required = (
        "product",
        "area",
        "sample_id",
        "item_index",
        "diff",
        "threshold",
        "actual_is_ok",
    )
    missing = [field for field in required if not str(row.get(field) or "").strip()]
    if missing:
        raise ValueError(f"missing required field(s): {', '.join(missing)}")
    actual = str(row["actual_is_ok"]).strip().lower()
    if actual not in {"0", "1", "false", "true"}:
        raise ValueError("actual_is_ok must be 0/1 or false/true")
    failure_kind = str(row.get("failure_kind") or "threshold").strip().lower()
    if failure_kind not in {"threshold", "rule"}:
        raise ValueError("failure_kind must be threshold or rule")
    return ColorFeedbackSample(
        product=str(row["product"]).strip(),
        area=str(row["area"]).strip(),
        model_type=str(row.get("model_type") or "yolo").strip().lower(),
        checker_type=str(row.get("checker_type") or "enhanced").strip().lower(),
        threshold_key=str(row.get("threshold_key") or "global").strip().lower(),
        failure_kind=failure_kind,
        sample_id=str(row["sample_id"]).strip(),
        item_index=str(row["item_index"]).strip(),
        diff=_non_negative_float(row["diff"], "diff"),
        runtime_threshold=_non_negative_float(row["threshold"], "threshold"),
        actual_is_ok=actual in {"1", "true"},
    )


def _feedback_paths(paths: Iterable[str | Path]) -> Iterator[Path]:
    found: set[Path] = set()
    for raw_path in paths:
        path = Path(raw_path).expanduser().resolve()
        candidates = path.rglob("feedback.csv") if path.is_dir() else (path,)
        for candidate in candidates:
            if candidate.is_file() and candidate not in found:
                found.add(candidate)
                yield candidate


def _non_negative_float(value: Any, field: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be numeric") from exc
    if not (0.0 <= parsed < float("inf")):
        raise ValueError(f"{field} must be finite and non-negative")
    return parsed


def _mode_float(values: Iterable[float]) -> float:
    counts: dict[float, int] = {}
    for value in values:
        normalized = round(value, 12)
        counts[normalized] = counts.get(normalized, 0) + 1
    if not counts:
        raise ValueError("Cannot select threshold from an empty sample group")
    return min(counts, key=lambda value: (-counts[value], value))


def _public_to_config(checker_type: str, threshold: float) -> float:
    value = 1.0 - threshold if checker_type == "stats" else threshold
    if checker_type == "stats" and not 0.0 <= value <= 1.0:
        raise ValueError("Stats checker threshold must remain between 0 and 1")
    return round(value, 12)


def _target_config_path(root: Path, recommendation: dict[str, Any]) -> Path:
    segments = [
        str(recommendation.get("product") or ""),
        str(recommendation.get("area") or ""),
        str(recommendation.get("model_type") or "yolo"),
    ]
    if any(
        not segment
        or segment in {".", ".."}
        or "/" in segment
        or "\\" in segment
        for segment in segments
    ):
        raise ValueError("Unsafe product/area/model_type in calibration report")
    config_path = (root.joinpath(*segments) / "config.yaml").resolve()
    try:
        config_path.relative_to(root)
    except ValueError as exc:
        raise ValueError("Calibration target escapes models root") from exc
    if not config_path.is_file():
        raise FileNotFoundError(config_path)
    return config_path


def _validate_config_has_not_drifted(
    config: dict[str, Any], recommendation: dict[str, Any]
) -> None:
    expected_checker = str(recommendation.get("checker_type") or "").lower()
    observed_checker = str(
        config.get("color_checker_type") or "enhanced"
    ).lower()
    if expected_checker != observed_checker:
        raise ValueError(
            "Color checker type changed after the report was generated: "
            f"expected {expected_checker}, found {observed_checker}. "
            "Generate a new report."
        )
    expected = float(recommendation["current_config_value"])
    observed = _effective_config_threshold(config, recommendation)
    if abs(expected - observed) > 1e-9:
        raise ValueError(
            "Color threshold changed after the report was generated: "
            f"expected {expected}, found {observed}. Generate a new report."
        )


def _effective_config_threshold(
    config: dict[str, Any], recommendation: dict[str, Any]
) -> float:
    key = str(recommendation.get("threshold_key") or "global").lower()
    overrides = config.get("color_threshold_overrides")
    if key != "global" and isinstance(overrides, dict):
        for raw_key, raw_value in overrides.items():
            if str(raw_key).lower() == key:
                return float(raw_value)
    return float(config.get("color_score_threshold"))


def _set_config_threshold(
    config: dict[str, Any], recommendation: dict[str, Any]
) -> None:
    key = str(recommendation.get("threshold_key") or "global").lower()
    value = float(recommendation["suggested_config_value"])
    if key == "global":
        config["color_score_threshold"] = value
        return
    overrides = config.setdefault("color_threshold_overrides", {})
    if not isinstance(overrides, dict):
        raise ValueError("color_threshold_overrides must be a mapping")
    existing_key = next(
        (raw_key for raw_key in overrides if str(raw_key).lower() == key),
        key,
    )
    overrides[existing_key] = value


@contextmanager
def _config_lock(config_path: Path) -> Iterator[None]:
    lock_path = config_path.with_name(".color_threshold_calibration.lock")
    try:
        descriptor = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as exc:
        try:
            age = time.time() - lock_path.stat().st_mtime
        except OSError:
            age = 0.0
        if age <= 3600:
            raise RuntimeError(f"Color threshold config is locked: {config_path}") from exc
        lock_path.unlink(missing_ok=True)
        descriptor = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    try:
        os.write(descriptor, str(os.getpid()).encode())
        yield
    finally:
        os.close(descriptor)
        lock_path.unlink(missing_ok=True)


def _load_yaml_mapping(path: Path) -> dict[str, Any]:
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:
        raise ValueError(f"Invalid YAML config: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"Config must be a YAML mapping: {path}")
    return payload


def _load_json_mapping(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid calibration report: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError("Calibration report must be a JSON object")
    return payload


def _report_id(payload: dict[str, Any]) -> str:
    serialized = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(serialized).hexdigest()


def _write_yaml_atomic(path: Path, payload: dict[str, Any]) -> None:
    data = yaml.safe_dump(payload, allow_unicode=True, sort_keys=False).encode()
    _write_bytes_atomic(path, data)


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    data = json.dumps(payload, ensure_ascii=False, indent=2).encode()
    _write_bytes_atomic(path, data)


def _write_bytes_atomic(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_bytes(data)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _append_history_atomic(path: Path, receipt: dict[str, Any]) -> None:
    history: list[dict[str, Any]] = []
    if path.is_file():
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            raise ValueError(f"Color threshold history must be a JSON array: {path}")
        history = [item for item in payload if isinstance(item, dict)]
    history.append(receipt)
    _write_json_atomic(path, history)


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line contract for shadow report and approval."""
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    recommend = subparsers.add_parser("recommend", help="Build a shadow report")
    recommend.add_argument("feedback", nargs="+", type=Path)
    recommend.add_argument("--output", type=Path, required=True)
    recommend.add_argument("--minimum-total", type=int, default=30)
    recommend.add_argument("--minimum-ok", type=int, default=5)
    recommend.add_argument("--minimum-ng", type=int, default=5)
    recommend.add_argument("--maximum-false-accept-rate", type=float, default=0.0)
    apply_parser = subparsers.add_parser(
        "apply", help="Apply ready recommendations after explicit approval"
    )
    apply_parser.add_argument("report", type=Path)
    apply_parser.add_argument("--models-root", type=Path, required=True)
    apply_parser.add_argument("--approver", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run shadow recommendation or explicit approval from the CLI."""
    args = build_parser().parse_args(argv)
    if args.command == "recommend":
        policy = CalibrationPolicy(
            minimum_total=args.minimum_total,
            minimum_ok=args.minimum_ok,
            minimum_ng=args.minimum_ng,
            maximum_false_accept_rate=args.maximum_false_accept_rate,
        )
        report = build_recommendation(
            args.feedback, policy=policy, output_path=args.output
        )
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0
    receipt = apply_recommendations(
        args.report,
        models_root=args.models_root,
        approver=args.approver,
    )
    print(json.dumps(receipt, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

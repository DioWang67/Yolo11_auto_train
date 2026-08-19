import json
import logging
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Optional

import yaml

from picture_tool.utils.experiment import write_experiment
from picture_tool.utils.experiment import _load_metrics_csv  # type: ignore
from picture_tool.pipeline.utils import detect_existing_weights

import os

try:
    if os.environ.get("PYTEST_IS_RUNNING") == "1":
        raise ImportError("Bypass ultralytics during pytest to avoid Windows PyTorch DLL crashes")
    from ultralytics import YOLO  # type: ignore[import-untyped]
except Exception:  # pragma: no cover
    YOLO = None  # type: ignore


class EvaluationGateError(RuntimeError):
    """Raised when a trained model does not satisfy deployment criteria."""


@dataclass(frozen=True)
class EvaluationGateReport:
    """Normalized YOLO metrics and deployment-gate decision."""

    passed: bool
    metrics: dict[str, float]
    baseline_metrics: dict[str, float]
    failures: tuple[str, ...]
    report_path: Path


METRIC_ALIASES = {
    "precision": ("metrics/precision(b)", "metrics/precision", "precision"),
    "recall": ("metrics/recall(b)", "metrics/recall", "recall"),
    "map50": ("metrics/map50(b)", "metrics/map50", "map50"),
    "map50_95": (
        "metrics/map50-95(b)",
        "metrics/map50-95",
        "map50-95",
        "map50_95",
    ),
}


def _resolve_weights(config: dict) -> Path:
    # Prefer explicit weights via standard detection util instead of DRY violation
    weights_path, _ = detect_existing_weights(config, prefer=None)
    if not weights_path:
        raise FileNotFoundError("Could not detect any existing model weights to evaluate.")
    return Path(str(weights_path)).resolve()


def evaluate_yolo(
    config: dict, logger: Optional[logging.Logger] = None
) -> EvaluationGateReport:
    """Evaluate a trained YOLO model and enforce optional deployment gates."""
    logger = logger or logging.getLogger(__name__)
    if YOLO is None and os.environ.get("PYTEST_IS_RUNNING") != "1":
        raise RuntimeError("ultralytics is not available. Please install ultralytics.")

    ycfg = config.get("yolo_training", {})
    ecfg = config.get("yolo_evaluation", {})
    imgsz = int(ecfg.get("imgsz", ycfg.get("imgsz", 640)))
    device = str(ecfg.get("device", ycfg.get("device", "cpu")))
    # OOM Prevention: Limit default workers and batch size
    workers = int(ecfg.get("workers", ycfg.get("workers", 1)))
    batch = int(ecfg.get("batch", 4))
    split = str(ecfg.get("split", "val")).strip().lower()
    confidence = float(ecfg.get("conf", 0.001))
    if split not in {"val", "test"}:
        raise ValueError("yolo_evaluation.split must be 'val' or 'test'")
    if not 0.0 <= confidence <= 1.0:
        raise ValueError("yolo_evaluation.conf must be between 0 and 1")

    dataset_dir = Path(str(ycfg.get("dataset_dir", "./datasets/split_dataset")))
    data_yaml = (dataset_dir / "data.yaml").resolve()
    weights_path = _resolve_weights(config)
    if not weights_path.exists():
        raise FileNotFoundError(f"Weights not found: {weights_path}")
    if not data_yaml.exists():
        raise FileNotFoundError(f"data.yaml not found: {data_yaml}")

    logger.info(
        "Evaluating challenger: %s | data=%s split=%s conf=%.4f "
        "imgsz=%s device=%s workers=%s batch=%s",
        weights_path,
        data_yaml,
        split,
        confidence,
        imgsz,
        device,
        workers,
        batch,
    )
    validation_kwargs = {
        "data": str(data_yaml),
        "split": split,
        "conf": confidence,
        "imgsz": imgsz,
        "device": device,
        "workers": workers,
        "batch": batch,
    }
    model = YOLO(str(weights_path))
    results = model.val(**validation_kwargs)
    logger.info("Evaluation completed.")
    run_dir = weights_path.parent.parent
    artifacts = {
        "weights": weights_path,
        "data_yaml": data_yaml,
    }
    metrics: dict[str, float] = _extract_metrics(results)
    if hasattr(results, "results_file"):
        metrics_path = Path(str(results.results_file))
        metrics.update(_load_metrics_csv(metrics_path))
    gate = (ecfg.get("gate", {}) or {}) if isinstance(ecfg, dict) else {}
    baseline_metrics: dict[str, float] | None = None
    baseline_weights_value = str(gate.get("baseline_weights") or "").strip()
    if gate.get("enabled", False) and gate.get("compare_on_same_dataset", False):
        if not baseline_weights_value:
            baseline_metrics = {}
        else:
            baseline_weights = Path(baseline_weights_value).expanduser().resolve()
            if not baseline_weights.is_file():
                baseline_metrics = {}
            else:
                logger.info(
                    "Evaluating incumbent on the same split/confidence: %s",
                    baseline_weights,
                )
                baseline_results = YOLO(str(baseline_weights)).val(**validation_kwargs)
                baseline_metrics = _extract_metrics(baseline_results)
                if hasattr(baseline_results, "results_file"):
                    baseline_metrics.update(
                        _load_metrics_csv(Path(str(baseline_results.results_file)))
                    )
                artifacts["baseline_weights"] = baseline_weights
    report = _evaluate_gate(
        config,
        metrics,
        run_dir,
        baseline_metrics=baseline_metrics,
    )
    write_experiment(
        run_type="eval",
        config=config,
        run_dir=run_dir,
        metrics=metrics,
        artifacts=artifacts,
        extra={
            "imgsz": imgsz,
            "device": device,
            "split": split,
            "conf": confidence,
            "baseline_metrics": report.baseline_metrics,
        },
        results_csv=metrics_path if "metrics_path" in locals() else None,
    )
    if not report.passed:
        raise EvaluationGateError(
            "Deployment quality gate failed: " + "; ".join(report.failures)
        )
    return report


def _extract_metrics(results: Any) -> dict[str, float]:
    """Normalize common Ultralytics metric keys across supported versions."""
    raw = getattr(results, "results_dict", {})
    normalized_raw = {
        str(key).lower().replace(" ", ""): value
        for key, value in raw.items()
    } if isinstance(raw, dict) else {}
    metrics: dict[str, float] = {}
    for canonical, aliases in METRIC_ALIASES.items():
        for alias in aliases:
            if alias in normalized_raw:
                try:
                    metrics[canonical] = float(normalized_raw[alias])
                except (TypeError, ValueError):
                    pass
                break

    box = getattr(results, "box", None)
    for canonical, attribute in (
        ("precision", "mp"),
        ("recall", "mr"),
        ("map50", "map50"),
        ("map50_95", "map"),
    ):
        if canonical in metrics or box is None:
            continue
        try:
            metrics[canonical] = float(getattr(box, attribute))
        except (AttributeError, TypeError, ValueError):
            pass
    return metrics


def _evaluate_gate(
    config: dict[str, Any],
    metrics: dict[str, float],
    run_dir: Path,
    *,
    baseline_metrics: dict[str, float] | None = None,
) -> EvaluationGateReport:
    gate = ((config.get("yolo_evaluation", {}) or {}).get("gate", {}) or {})
    enabled = bool(gate.get("enabled", False))
    failures: list[str] = []
    baseline = (
        dict(baseline_metrics)
        if baseline_metrics is not None
        else _load_baseline_metrics(gate.get("baseline_manifest"))
    )
    if enabled and gate.get("require_metrics", True):
        missing = [key for key in METRIC_ALIASES if key not in metrics]
        if missing:
            failures.append("missing metrics: " + ", ".join(missing))

    thresholds = {
        "precision": gate.get("min_precision"),
        "recall": gate.get("min_recall"),
        "map50": gate.get("min_map50"),
        "map50_95": gate.get("min_map50_95"),
    }
    if enabled:
        if gate.get("require_baseline", False):
            missing_baseline = [key for key in METRIC_ALIASES if key not in baseline]
            if missing_baseline:
                failures.append(
                    "missing incumbent baseline metrics: "
                    + ", ".join(missing_baseline)
                )
        for name, raw_threshold in thresholds.items():
            if raw_threshold is None or name not in metrics:
                continue
            threshold = float(raw_threshold)
            if metrics[name] < threshold:
                failures.append(
                    f"{name}={metrics[name]:.4f} is below {threshold:.4f}"
                )
        max_regression = float(gate.get("max_regression", 0.05))
        for name, baseline_value in baseline.items():
            if name in metrics and metrics[name] < baseline_value - max_regression:
                failures.append(
                    f"{name} regressed from {baseline_value:.4f} "
                    f"to {metrics[name]:.4f}"
                )

    report_path = run_dir / "evaluation_gate.json"
    payload = {
        "passed": not failures,
        "enabled": enabled,
        "metrics": metrics,
        "baseline_metrics": baseline,
        "comparison_mode": (
            "same_dataset" if baseline_metrics is not None else "historical_manifest"
        ),
        "failures": failures,
    }
    _write_json_atomic(report_path, payload)
    return EvaluationGateReport(
        passed=not failures,
        metrics=metrics,
        baseline_metrics=baseline,
        failures=tuple(failures),
        report_path=report_path,
    )


def _load_baseline_metrics(path_value: Any) -> dict[str, float]:
    if not path_value:
        return {}
    path = Path(str(path_value))
    if not path.is_file():
        return {}
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except (OSError, UnicodeDecodeError, yaml.YAMLError):
        return {}
    raw = payload.get("evaluation_metrics") if isinstance(payload, dict) else None
    if not isinstance(raw, dict):
        return {}
    baseline: dict[str, float] = {}
    for key in METRIC_ALIASES:
        try:
            baseline[key] = float(raw[key])
        except (KeyError, TypeError, ValueError):
            continue
    return baseline


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        temporary.replace(path)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass

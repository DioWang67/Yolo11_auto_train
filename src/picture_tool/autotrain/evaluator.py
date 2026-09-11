"""Champion versus challenger evaluation.

Both models are measured on the *same* split, at the *same* confidence, in
the same process --- the comparison is only meaningful if nothing but the
weights differ. This mirrors what the existing deployment gate already does
via ``yolo_evaluation.gate.compare_on_same_dataset``; the difference is that
this path extracts per-class detail and returns a report instead of raising,
because a challenger being worse is a result, not an error.

The champion's weight file is hashed before and after. Production can
activate a new model at any moment through the existing release flow, and a
comparison whose baseline changed halfway through is not a comparison ---
so that outcome is reported as ``INVALIDATED`` rather than published.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping

from picture_tool.autotrain import AutoTrainError, golden as golden_module
from picture_tool.autotrain.metrics import (
    MetricComparison,
    ModelMetrics,
    compare,
    extract_metrics,
)

LOGGER = logging.getLogger(__name__)

#: The comparison completed and is trustworthy.
COMPLETED = "COMPLETED"
#: The champion's weights changed while the comparison was running.
INVALIDATED = "INVALIDATED"

DATA_YAML_NAME = "data.yaml"


class EvaluationError(AutoTrainError):
    """Raised when an evaluation could not be run at all."""


@dataclass(frozen=True)
class EvaluationReport:
    """Everything measured about one champion/challenger pair."""

    status: str
    comparison: MetricComparison
    golden: golden_module.GoldenStatus
    champion_weights: str
    challenger_weights: str
    champion_sha256: str
    data_yaml: str
    split: str
    confidence: float
    imgsz: int
    golden_comparison: MetricComparison | None = None
    detail: str = ""
    extra: Mapping[str, Any] = field(default_factory=dict)

    @property
    def is_valid(self) -> bool:
        return self.status == COMPLETED

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "detail": self.detail,
            "comparison": self.comparison.to_dict(),
            "golden": self.golden.to_dict(),
            "golden_comparison": (
                self.golden_comparison.to_dict() if self.golden_comparison else None
            ),
            "champion_weights": self.champion_weights,
            "challenger_weights": self.challenger_weights,
            "champion_sha256": self.champion_sha256,
            "data_yaml": self.data_yaml,
            "split": self.split,
            "confidence": self.confidence,
            "imgsz": self.imgsz,
            "extra": dict(self.extra),
        }


Validator = Callable[..., Any]


def evaluate_candidate(
    *,
    champion_weights: str | Path,
    challenger_weights: str | Path,
    data_yaml: str | Path,
    golden_status: golden_module.GoldenStatus,
    split: str = "test",
    confidence: float = 0.4,
    imgsz: int = 640,
    device: str = "cpu",
    batch: int = 4,
    workers: int = 0,
    validator: Validator | None = None,
) -> EvaluationReport:
    """Measure both models and return the comparison.

    ``validator`` is the injection point: it receives ``(weights, **kwargs)``
    and returns an ultralytics-shaped result, so tests never load a model.
    """
    champion = Path(champion_weights).expanduser()
    challenger = Path(challenger_weights).expanduser()
    dataset = Path(data_yaml).expanduser()

    if not challenger.is_file():
        raise EvaluationError(f"Challenger weights not found: {challenger}")
    if not dataset.is_file():
        raise EvaluationError(f"Evaluation dataset descriptor not found: {dataset}")
    if not champion.is_file():
        raise EvaluationError(
            f"Champion weights not found: {champion}. A challenger cannot be "
            "recommended without a baseline to beat."
        )

    run = validator or _default_validator()
    champion_sha_before = _sha256_file(champion)

    kwargs = {
        "data": str(dataset),
        "split": split,
        "conf": confidence,
        "imgsz": imgsz,
        "device": device,
        "batch": batch,
        "workers": workers,
    }

    LOGGER.info("Evaluating champion %s on %s", champion.name, dataset)
    champion_metrics = extract_metrics(run(str(champion), **kwargs), source="champion")
    LOGGER.info("Evaluating challenger %s on the same split", challenger.name)
    challenger_metrics = extract_metrics(
        run(str(challenger), **kwargs), source="challenger"
    )

    champion_sha_after = _sha256_file(champion)
    if champion_sha_before != champion_sha_after:
        return EvaluationReport(
            status=INVALIDATED,
            comparison=compare(champion_metrics, challenger_metrics),
            golden=golden_status,
            champion_weights=str(champion),
            challenger_weights=str(challenger),
            champion_sha256=champion_sha_before,
            data_yaml=str(dataset),
            split=split,
            confidence=confidence,
            imgsz=imgsz,
            detail=(
                "The champion's weights changed while the comparison was "
                "running, so the baseline is not the model that was measured. "
                "Re-run the evaluation against the current champion."
            ),
        )

    golden_comparison = _evaluate_golden(
        golden_status,
        run=run,
        champion=champion,
        challenger=challenger,
        kwargs=kwargs,
    )

    return EvaluationReport(
        status=COMPLETED,
        comparison=compare(champion_metrics, challenger_metrics),
        golden=golden_status,
        champion_weights=str(champion),
        challenger_weights=str(challenger),
        champion_sha256=champion_sha_after,
        data_yaml=str(dataset),
        split=split,
        confidence=confidence,
        imgsz=imgsz,
        golden_comparison=golden_comparison,
    )


def golden_data_yaml(status: golden_module.GoldenStatus) -> Path | None:
    """Locate the golden set's dataset descriptor, if it has one.

    A registered golden directory is not automatically an evaluable YOLO
    dataset --- it needs labels and a ``data.yaml``. Absence is reported by
    the caller rather than invented here.
    """
    if status.dataset is None:
        return None
    candidate = status.dataset.root / DATA_YAML_NAME
    return candidate if candidate.is_file() else None


def _evaluate_golden(
    status: golden_module.GoldenStatus,
    *,
    run: Validator,
    champion: Path,
    challenger: Path,
    kwargs: Mapping[str, Any],
) -> MetricComparison | None:
    """Measure both models on the golden set when one is usable."""
    if not status.is_passing:
        return None
    descriptor = golden_data_yaml(status)
    if descriptor is None:
        LOGGER.warning(
            "Golden dataset has no %s; skipping the golden comparison. Register "
            "a labelled YOLO dataset to enable it.",
            DATA_YAML_NAME,
        )
        return None

    golden_kwargs = dict(kwargs)
    golden_kwargs["data"] = str(descriptor)
    golden_kwargs["split"] = "val"
    try:
        champion_metrics = extract_metrics(
            run(str(champion), **golden_kwargs), source="champion-golden"
        )
        challenger_metrics = extract_metrics(
            run(str(challenger), **golden_kwargs), source="challenger-golden"
        )
    except Exception as exc:  # noqa: BLE001 - ultralytics raises many types
        LOGGER.warning("Golden evaluation failed: %s", exc)
        return None
    return compare(champion_metrics, challenger_metrics)


def _default_validator() -> Validator:
    """Import ultralytics lazily, and never during the test suite.

    ``PYTEST_IS_RUNNING`` is the repository's existing guard against loading
    torch DLLs under pytest on Windows; honouring it keeps this module
    importable in tests.
    """
    import os

    if os.environ.get("PYTEST_IS_RUNNING") == "1":
        raise EvaluationError(
            "Refusing to load ultralytics under pytest; inject a validator."
        )
    from ultralytics import YOLO

    def run(weights: str, **kwargs: Any) -> Any:
        return YOLO(weights).val(**kwargs)

    return run


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def metrics_from_mapping(payload: Mapping[str, Any]) -> ModelMetrics:
    """Rehydrate stored metrics, for reports built from a saved cycle."""
    return ModelMetrics.from_dict(payload)

"""Detection metrics, including the per-class detail the request asked for.

The existing :mod:`picture_tool.eval.yolo_evaluator` normalises four aggregate
numbers (precision, recall, mAP50, mAP50-95) and that is the right shape for
its deployment gate. A champion/challenger report needs more: a model can
improve on average while losing recall on the one class that matters, and an
overall number hides exactly that.

So this module reuses the same canonical names and alias handling, and adds
per-class precision/recall/mAP, the confusion matrix, and the false
positive/negative counts derived from it.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from picture_tool.eval.yolo_evaluator import METRIC_ALIASES

LOGGER = logging.getLogger(__name__)

#: Canonical aggregate metric names, in report order.
OVERALL_METRICS = ("precision", "recall", "map50", "map50_95")


@dataclass(frozen=True)
class ClassMetrics:
    """Per-class detection metrics."""

    name: str
    precision: float | None = None
    recall: float | None = None
    map50: float | None = None
    map50_95: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "precision": self.precision,
            "recall": self.recall,
            "map50": self.map50,
            "map50_95": self.map50_95,
        }


@dataclass(frozen=True)
class ModelMetrics:
    """Everything measured about one model on one evaluation split."""

    overall: Mapping[str, float] = field(default_factory=dict)
    per_class: Mapping[str, ClassMetrics] = field(default_factory=dict)
    confusion_matrix: tuple[tuple[float, ...], ...] = ()
    class_names: tuple[str, ...] = ()
    false_positives: int = 0
    false_negatives: int = 0
    source: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "overall": dict(self.overall),
            "per_class": {
                name: metrics.to_dict() for name, metrics in self.per_class.items()
            },
            "confusion_matrix": [list(row) for row in self.confusion_matrix],
            "class_names": list(self.class_names),
            "false_positives": self.false_positives,
            "false_negatives": self.false_negatives,
            "source": self.source,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ModelMetrics":
        per_class_raw = payload.get("per_class") or {}
        per_class = {
            str(name): ClassMetrics(
                name=str(name),
                precision=_optional_float(values.get("precision")),
                recall=_optional_float(values.get("recall")),
                map50=_optional_float(values.get("map50")),
                map50_95=_optional_float(values.get("map50_95")),
            )
            for name, values in per_class_raw.items()
            if isinstance(values, dict)
        }
        return cls(
            overall={
                str(k): float(v)
                for k, v in (payload.get("overall") or {}).items()
                if _optional_float(v) is not None
            },
            per_class=per_class,
            confusion_matrix=tuple(
                tuple(float(value) for value in row)
                for row in payload.get("confusion_matrix") or ()
            ),
            class_names=tuple(str(n) for n in payload.get("class_names") or ()),
            false_positives=int(payload.get("false_positives", 0)),
            false_negatives=int(payload.get("false_negatives", 0)),
            source=str(payload.get("source", "")),
        )


@dataclass(frozen=True)
class MetricDelta:
    """One metric's champion value, challenger value and difference."""

    name: str
    champion: float | None
    challenger: float | None

    @property
    def delta(self) -> float | None:
        if self.champion is None or self.challenger is None:
            return None
        return self.challenger - self.champion

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "champion": self.champion,
            "challenger": self.challenger,
            "delta": self.delta,
        }


@dataclass(frozen=True)
class MetricComparison:
    """Champion versus challenger across overall and per-class metrics."""

    overall: tuple[MetricDelta, ...]
    per_class_recall: tuple[MetricDelta, ...]
    per_class_precision: tuple[MetricDelta, ...]
    champion: ModelMetrics
    challenger: ModelMetrics

    def overall_delta(self, name: str) -> float | None:
        for item in self.overall:
            if item.name == name:
                return item.delta
        return None

    def recall_delta(self, class_name: str) -> float | None:
        for item in self.per_class_recall:
            if item.name == class_name:
                return item.delta
        return None

    def to_dict(self) -> dict[str, Any]:
        return {
            "overall": [item.to_dict() for item in self.overall],
            "per_class_recall": [item.to_dict() for item in self.per_class_recall],
            "per_class_precision": [
                item.to_dict() for item in self.per_class_precision
            ],
            "champion": self.champion.to_dict(),
            "challenger": self.challenger.to_dict(),
        }


def compare(champion: ModelMetrics, challenger: ModelMetrics) -> MetricComparison:
    """Build the champion/challenger comparison used by the report and policy.

    Classes are the union of both sides, so a class the challenger dropped
    shows up as a missing value rather than quietly disappearing from the
    comparison.
    """
    class_names = sorted(set(champion.per_class) | set(challenger.per_class))
    return MetricComparison(
        overall=tuple(
            MetricDelta(
                name=name,
                champion=_optional_float(champion.overall.get(name)),
                challenger=_optional_float(challenger.overall.get(name)),
            )
            for name in OVERALL_METRICS
        ),
        per_class_recall=tuple(
            MetricDelta(
                name=name,
                champion=_class_value(champion, name, "recall"),
                challenger=_class_value(challenger, name, "recall"),
            )
            for name in class_names
        ),
        per_class_precision=tuple(
            MetricDelta(
                name=name,
                champion=_class_value(champion, name, "precision"),
                challenger=_class_value(challenger, name, "precision"),
            )
            for name in class_names
        ),
        champion=champion,
        challenger=challenger,
    )


# ---------------------------------------------------------------------------
# Extraction from ultralytics results


def extract_metrics(results: Any, *, source: str = "") -> ModelMetrics:
    """Normalise one ultralytics validation result.

    Tolerant by design: ultralytics has moved these attributes between
    versions, and a missing per-class array should cost the per-class table,
    not the whole evaluation.
    """
    overall = _extract_overall(results)
    class_names = _extract_class_names(results)
    per_class = _extract_per_class(results, class_names)
    matrix = _extract_confusion_matrix(results)
    false_positives, false_negatives = _counts_from_matrix(matrix)
    return ModelMetrics(
        overall=overall,
        per_class=per_class,
        confusion_matrix=matrix,
        class_names=class_names,
        false_positives=false_positives,
        false_negatives=false_negatives,
        source=source,
    )


def _extract_overall(results: Any) -> dict[str, float]:
    raw = getattr(results, "results_dict", {})
    normalized = (
        {str(key).lower().replace(" ", ""): value for key, value in raw.items()}
        if isinstance(raw, dict)
        else {}
    )
    metrics: dict[str, float] = {}
    for canonical, aliases in METRIC_ALIASES.items():
        for alias in aliases:
            if alias in normalized:
                value = _optional_float(normalized[alias])
                if value is not None:
                    metrics[canonical] = value
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
        value = _optional_float(getattr(box, attribute, None))
        if value is not None:
            metrics[canonical] = value
    return metrics


def _extract_class_names(results: Any) -> tuple[str, ...]:
    names = getattr(results, "names", None)
    if isinstance(names, dict):
        return tuple(str(names[key]) for key in sorted(names))
    if isinstance(names, (list, tuple)):
        return tuple(str(name) for name in names)
    return ()


def _extract_per_class(
    results: Any, class_names: Sequence[str]
) -> dict[str, ClassMetrics]:
    box = getattr(results, "box", None)
    if box is None or not class_names:
        return {}

    indices = _as_list(getattr(box, "ap_class_index", None))
    if not indices:
        return {}

    precision = _as_list(getattr(box, "p", None))
    recall = _as_list(getattr(box, "r", None))
    ap50 = _as_list(getattr(box, "ap50", None))
    ap = _as_list(getattr(box, "ap", None))

    per_class: dict[str, ClassMetrics] = {}
    for position, class_index in enumerate(indices):
        index = _optional_int(class_index)
        if index is None or index < 0 or index >= len(class_names):
            continue
        name = class_names[index]
        per_class[name] = ClassMetrics(
            name=name,
            precision=_at(precision, position),
            recall=_at(recall, position),
            map50=_at(ap50, position),
            map50_95=_mean(_at_raw(ap, position)),
        )
    return per_class


def _extract_confusion_matrix(results: Any) -> tuple[tuple[float, ...], ...]:
    container = getattr(results, "confusion_matrix", None)
    matrix = getattr(container, "matrix", None) if container is not None else None
    if matrix is None:
        return ()
    try:
        return tuple(tuple(float(value) for value in row) for row in matrix)
    except (TypeError, ValueError):
        LOGGER.debug("Confusion matrix could not be normalised")
        return ()


def _counts_from_matrix(
    matrix: tuple[tuple[float, ...], ...]
) -> tuple[int, int]:
    """Derive false positive and false negative counts.

    Ultralytics' matrix has a trailing background row and column: the last
    column counts predictions with no matching ground truth (false
    positives), and the last row counts ground truth with no matching
    prediction (false negatives). A matrix without that extra band cannot
    express either, so both are reported as zero rather than guessed at.
    """
    if not matrix or len(matrix) < 2:
        return (0, 0)
    size = len(matrix)
    if any(len(row) != size for row in matrix):
        return (0, 0)
    background = size - 1
    false_positives = sum(
        matrix[row][background] for row in range(size) if row != background
    )
    false_negatives = sum(
        matrix[background][column] for column in range(size) if column != background
    )
    return (int(round(false_positives)), int(round(false_negatives)))


# ---------------------------------------------------------------------------


def _class_value(metrics: ModelMetrics, name: str, attribute: str) -> float | None:
    entry = metrics.per_class.get(name)
    return getattr(entry, attribute) if entry is not None else None


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, (list, tuple)):
        return list(value)
    return []


def _at(values: Sequence[Any], index: int) -> float | None:
    raw = _at_raw(values, index)
    return _optional_float(raw)


def _at_raw(values: Sequence[Any], index: int) -> Any:
    if index < 0 or index >= len(values):
        return None
    return values[index]


def _mean(value: Any) -> float | None:
    if isinstance(value, (list, tuple)):
        numbers = [n for n in (_optional_float(v) for v in value) if n is not None]
        return sum(numbers) / len(numbers) if numbers else None
    return _optional_float(value)


def _optional_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    if hasattr(value, "item"):
        try:
            value = value.item()
        except (AttributeError, TypeError, ValueError):
            return None
    if not isinstance(value, (int, float, str)):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _optional_int(value: Any) -> int | None:
    number = _optional_float(value)
    return int(number) if number is not None else None

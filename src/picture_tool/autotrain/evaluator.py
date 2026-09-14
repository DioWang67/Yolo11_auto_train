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
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import yaml

from picture_tool.autotrain import AutoTrainError, golden as golden_module
from picture_tool.autotrain.config import DEFAULT_MIN_GROUP_SAMPLES
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

#: The group was measured; its comparison is populated.
GROUP_MEASURED = "MEASURED"
#: Too few samples for a number that means anything.
GROUP_INSUFFICIENT = "INSUFFICIENT"
#: No ground truth could be located for this group's images.
GROUP_NO_LABELS = "NO_LABELS"
#: Measuring the group failed outright.
GROUP_FAILED = "FAILED"


class EvaluationError(AutoTrainError):
    """Raised when an evaluation could not be run at all."""


@dataclass(frozen=True)
class GroupEvaluation:
    """Champion versus challenger restricted to one golden group.

    A group whose numbers could not be produced carries the reason instead of
    a comparison. The two are never conflated: an unmeasured ``hard_case``
    group and one that held steady look identical if absence is rendered as
    "no regression".
    """

    group: str
    status: str
    sample_count: int
    comparison: MetricComparison | None = None
    detail: str = ""

    @property
    def is_measured(self) -> bool:
        return self.status == GROUP_MEASURED and self.comparison is not None

    def to_dict(self) -> dict[str, Any]:
        return {
            "group": self.group,
            "status": self.status,
            "sample_count": self.sample_count,
            "comparison": self.comparison.to_dict() if self.comparison else None,
            "detail": self.detail,
        }


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
    #: Per-group golden results, one entry per group the set declares.
    golden_groups: tuple[GroupEvaluation, ...] = ()
    detail: str = ""
    extra: Mapping[str, Any] = field(default_factory=dict)

    @property
    def is_valid(self) -> bool:
        return self.status == COMPLETED

    def group(self, name: str) -> GroupEvaluation | None:
        for item in self.golden_groups:
            if item.group == name:
                return item
        return None

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "detail": self.detail,
            "comparison": self.comparison.to_dict(),
            "golden": self.golden.to_dict(),
            "golden_comparison": (
                self.golden_comparison.to_dict() if self.golden_comparison else None
            ),
            "golden_groups": [item.to_dict() for item in self.golden_groups],
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
    min_group_samples: int = DEFAULT_MIN_GROUP_SAMPLES,
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
    golden_groups = _evaluate_golden_groups(
        golden_status,
        run=run,
        champion=champion,
        challenger=challenger,
        kwargs=kwargs,
        min_samples=min_group_samples,
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
        golden_groups=golden_groups,
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


def _evaluate_golden_groups(
    status: golden_module.GoldenStatus,
    *,
    run: Validator,
    champion: Path,
    challenger: Path,
    kwargs: Mapping[str, Any],
    min_samples: int,
) -> tuple[GroupEvaluation, ...]:
    """Measure each declared golden group separately.

    A single aggregate score cannot answer the question the split exists to
    ask: a challenger that improves on routine images while losing the hard
    ones is a worse model whose overall number went up. ultralytics reports
    only aggregates, so the only way to get the two numbers is to validate
    twice per group over a restricted image list.

    That costs two extra runs per group. The alternative --- deriving group
    scores from the combined run --- would be arithmetic, not measurement.
    """
    if not status.is_passing or status.dataset is None:
        return ()
    dataset = status.dataset
    counts = dataset.group_counts()
    if not counts:
        return ()

    descriptor = golden_data_yaml(status)
    if descriptor is None:
        return tuple(
            GroupEvaluation(
                group=group,
                status=GROUP_FAILED,
                sample_count=count,
                detail=(
                    f"The golden set has no {DATA_YAML_NAME}, so no group can "
                    "be validated."
                ),
            )
            for group, count in sorted(counts.items())
        )

    base = _load_yaml(descriptor)
    if base is None:
        return tuple(
            GroupEvaluation(
                group=group,
                status=GROUP_FAILED,
                sample_count=count,
                detail=f"{descriptor} could not be parsed as a dataset descriptor.",
            )
            for group, count in sorted(counts.items())
        )

    results: list[GroupEvaluation] = []
    with tempfile.TemporaryDirectory(prefix="autotrain-golden-group-") as workdir:
        for group in sorted(counts):
            results.append(
                _evaluate_one_group(
                    group,
                    dataset=dataset,
                    base=base,
                    workdir=Path(workdir),
                    run=run,
                    champion=champion,
                    challenger=challenger,
                    kwargs=kwargs,
                    min_samples=min_samples,
                )
            )
    return tuple(results)


def _evaluate_one_group(
    group: str,
    *,
    dataset: golden_module.GoldenDataset,
    base: Mapping[str, Any],
    workdir: Path,
    run: Validator,
    champion: Path,
    challenger: Path,
    kwargs: Mapping[str, Any],
    min_samples: int,
) -> GroupEvaluation:
    sample_ids = dataset.sample_ids_in_group(group)
    count = len(sample_ids)
    if count < min_samples:
        return GroupEvaluation(
            group=group,
            status=GROUP_INSUFFICIENT,
            sample_count=count,
            detail=(
                f"{count} sample(s) is below the {min_samples} required for a "
                "meaningful group score. Add samples or lower "
                "golden.min_group_samples deliberately."
            ),
        )

    images = [
        dataset.root / dataset.images[sample_id]
        for sample_id in sample_ids
        if sample_id in dataset.images
    ]
    present = [path for path in images if path.is_file()]
    if not present:
        return GroupEvaluation(
            group=group,
            status=GROUP_FAILED,
            sample_count=count,
            detail="None of this group's registered images are on disk.",
        )

    # One resolvable label is enough to prove the set follows the
    # images/labels layout ultralytics requires. Demanding one per image
    # would reject legitimate background images, which carry no label file by
    # convention; demanding none at all would let a wrongly laid out set
    # report every object as missed and call that a score.
    if not any(_has_label(path) for path in present):
        return GroupEvaluation(
            group=group,
            status=GROUP_NO_LABELS,
            sample_count=count,
            detail=(
                "No label file could be located for any image in this group. "
                "Ultralytics finds labels by replacing the 'images' path "
                "segment with 'labels'; a set not laid out that way would "
                "score every object as a miss."
            ),
        )

    descriptor = _write_group_descriptor(
        group, images=present, dataset_root=dataset.root, base=base, workdir=workdir
    )
    group_kwargs = dict(kwargs)
    group_kwargs["data"] = str(descriptor)
    group_kwargs["split"] = "val"
    try:
        champion_metrics = extract_metrics(
            run(str(champion), **group_kwargs), source=f"champion-golden-{group}"
        )
        challenger_metrics = extract_metrics(
            run(str(challenger), **group_kwargs), source=f"challenger-golden-{group}"
        )
    except Exception as exc:  # noqa: BLE001 - ultralytics raises many types
        LOGGER.warning("Golden group %s could not be evaluated: %s", group, exc)
        return GroupEvaluation(
            group=group,
            status=GROUP_FAILED,
            sample_count=count,
            detail=f"Validation failed for this group: {exc}",
        )

    return GroupEvaluation(
        group=group,
        status=GROUP_MEASURED,
        sample_count=count,
        comparison=compare(champion_metrics, challenger_metrics),
    )


def _write_group_descriptor(
    group: str,
    *,
    images: Sequence[Path],
    dataset_root: Path,
    base: Mapping[str, Any],
    workdir: Path,
) -> Path:
    """Write a dataset descriptor restricted to one group's images.

    The image list is a file of absolute paths rather than a directory, so
    the golden set itself is never rearranged, copied or written to --- the
    subset exists only as a list in a temporary directory.

    Everything else is inherited from the golden descriptor verbatim, which
    is what keeps the class ids identical to the combined run. Re-deriving
    ``names`` here would create a second place for the class contract to be
    wrong.
    """
    safe = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in group)
    list_path = workdir / f"{safe}.txt"
    list_path.write_text(
        "\n".join(str(path.resolve()) for path in images) + "\n", encoding="utf-8"
    )

    payload = dict(base)
    # An absolute ``val`` wins over ``path``, so the subset resolves the same
    # way regardless of where this temporary descriptor lives.
    payload["path"] = str(dataset_root)
    payload["val"] = str(list_path.resolve())
    # ``train`` is pointed at the same subset rather than removed. Ultralytics'
    # check_det_dataset requires both keys and raises SyntaxError without
    # them, so dropping it failed every group against the real library while
    # a fake validator --- which never parses this file --- saw nothing wrong.
    # Pointing it at the subset keeps the property that motivated removing it:
    # this descriptor cannot name the real training images.
    payload["train"] = str(list_path.resolve())
    payload.pop("test", None)

    descriptor = workdir / f"{safe}.yaml"
    descriptor.write_text(
        yaml.safe_dump(payload, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )
    return descriptor


def _has_label(image: Path) -> bool:
    label = _label_path(image)
    return label is not None and label.is_file()


def _label_path(image: Path) -> Path | None:
    """Mirror ultralytics' image-to-label path rule.

    Deliberately a copy of the rule rather than an import: this runs to decide
    whether a group *can* be measured, before any ultralytics import is
    allowed under pytest.
    """
    images_segment = f"{os.sep}images{os.sep}"
    labels_segment = f"{os.sep}labels{os.sep}"
    text = str(image)
    if images_segment not in text:
        return None
    head, _, tail = text.rpartition(images_segment)
    return Path(head + labels_segment + tail).with_suffix(".txt")


def _load_yaml(path: Path) -> dict[str, Any] | None:
    try:
        loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, yaml.YAMLError) as exc:
        LOGGER.warning("Golden descriptor %s could not be read: %s", path, exc)
        return None
    return loaded if isinstance(loaded, dict) else None


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

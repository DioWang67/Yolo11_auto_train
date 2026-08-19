"""Training-run artifact discovery helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional


@dataclass(frozen=True)
class TrainingRunArtifact:
    """Resolved training run and its primary runtime artifact.

    Args:
        kind: Model family, e.g. ``"yolo"`` or ``"anomalib"``.
        run_dir: Directory for the training run.
        primary_artifact: Main weight/checkpoint file.
    """

    kind: str
    run_dir: Path
    primary_artifact: Path


def find_yolo_run_artifact(
    config: dict,
    *,
    prefer: str | None = None,
) -> TrainingRunArtifact | None:
    """Find the newest usable YOLO run artifact from config.

    Args:
        config: Pipeline configuration.
        prefer: ``"position"`` to prefer position validation weights.

    Returns:
        A resolved training artifact, or ``None`` when no weights exist.
    """

    ycfg = config.get("yolo_training", {}) or {}
    position_cfg = ycfg.get("position_validation", {}) or {}
    eval_cfg = config.get("yolo_evaluation", {}) or {}
    project = Path(str(ycfg.get("project", "./runs/detect")))
    name_prefix = str(ycfg.get("name", "train"))

    explicit_weights = (
        [position_cfg.get("weights"), eval_cfg.get("weights")]
        if prefer == "position"
        else [eval_cfg.get("weights"), position_cfg.get("weights")]
    )
    for candidate in explicit_weights:
        artifact = _artifact_from_yolo_weight(candidate)
        if artifact:
            return artifact

    if not project.exists():
        return None

    candidates: list[tuple[float, TrainingRunArtifact]] = []
    for run_dir in project.iterdir():
        if not run_dir.is_dir() or not run_dir.name.startswith(name_prefix):
            continue
        for filename in ("best.pt", "last.pt", "best.onnx"):
            weight_path = run_dir / "weights" / filename
            if weight_path.exists():
                candidates.append(
                    (
                        weight_path.stat().st_mtime,
                        TrainingRunArtifact(
                            kind="yolo",
                            run_dir=run_dir.resolve(),
                            primary_artifact=weight_path.resolve(),
                        ),
                    )
                )
    if not candidates:
        return None
    return max(candidates, key=lambda item: item[0])[1]


def find_anomalib_run_artifact(
    search_roots: Iterable[Path],
) -> TrainingRunArtifact | None:
    """Find the newest Anomalib checkpoint below any search root.

    Args:
        search_roots: Candidate directories to scan.

    Returns:
        A resolved Anomalib artifact, or ``None`` when no checkpoint exists.
    """

    candidates: list[tuple[float, TrainingRunArtifact]] = []
    for root in search_roots:
        if not root.exists():
            continue
        for checkpoint in root.rglob("weights/lightning/*.ckpt"):
            run_dir = _infer_anomalib_run_dir(checkpoint)
            if run_dir is None:
                continue
            candidates.append(
                (
                    checkpoint.stat().st_mtime,
                    TrainingRunArtifact(
                        kind="anomalib",
                        run_dir=run_dir.resolve(),
                        primary_artifact=checkpoint.resolve(),
                    ),
                )
            )
    if not candidates:
        return None
    return max(candidates, key=lambda item: item[0])[1]


def _artifact_from_yolo_weight(
    path_value: Optional[str | Path],
) -> TrainingRunArtifact | None:
    if not path_value:
        return None
    weight_path = Path(str(path_value)).expanduser().resolve()
    if not weight_path.exists():
        return None
    run_dir = weight_path.parent.parent if weight_path.parent.name == "weights" else weight_path.parent
    return TrainingRunArtifact(
        kind="yolo",
        run_dir=run_dir.resolve(),
        primary_artifact=weight_path,
    )


def _infer_anomalib_run_dir(checkpoint: Path) -> Path | None:
    parts = checkpoint.parts
    if len(parts) < 4:
        return None
    try:
        weights_index = parts.index("weights")
    except ValueError:
        return None
    if checkpoint.parent.name != "lightning" or weights_index < 1:
        return None
    return Path(*parts[:weights_index])

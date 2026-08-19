from pathlib import Path
from typing import Iterable, Optional, Tuple

from picture_tool.pipeline.artifacts import find_yolo_run_artifact


def detect_existing_weights(
    config: dict, prefer: str | None = None
) -> Tuple[Optional[Path], Optional[Path]]:
    """
    Locate an existing trained weight file and its run directory (if available).

    Order of precedence:
    1. Explicit override on preferred section (position_validation or evaluation).
    2. Explicit override on the other section.
    3. Latest run under project/name* containing best/last.pt.
    """
    artifact = find_yolo_run_artifact(config, prefer=prefer)
    if artifact is None:
        return None, None
    return artifact.primary_artifact, artifact.run_dir


def mtime_latest(paths: Iterable[Path]) -> float:
    mts = []
    for p in paths:
        if p.is_file():
            mts.append(p.stat().st_mtime)
        elif p.is_dir():
            for sub in p.rglob("*"):
                if sub.is_file():
                    mts.append(sub.stat().st_mtime)
    return max(mts) if mts else 0.0


def exists_and_nonempty(p: Path) -> bool:
    return p.exists() and any(p.iterdir())

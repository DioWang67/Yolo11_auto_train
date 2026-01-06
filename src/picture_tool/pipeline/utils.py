from pathlib import Path
from typing import Iterable, Optional, Tuple


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
    ycfg = config.get("yolo_training", {}) or {}
    pv_cfg = ycfg.get("position_validation", {}) or {}
    ecfg = config.get("yolo_evaluation", {}) or {}
    project = Path(str(ycfg.get("project", "./runs/detect")))
    name_prefix = str(ycfg.get("name", "train"))

    def _candidate_path(path_val: Optional[str | Path]) -> Optional[Tuple[Path, Path]]:
        if not path_val:
            return None
        p = Path(str(path_val)).expanduser().resolve()
        if not p.exists():
            return None
        run_dir = p.parent.parent if p.parent.name == "weights" else p.parent
        return p, run_dir.resolve()

    preferred_first = (
        [pv_cfg.get("weights"), ecfg.get("weights")]
        if prefer == "position"
        else [ecfg.get("weights"), pv_cfg.get("weights")]
    )
    for candidate in preferred_first:
        resolved = _candidate_path(candidate)
        if resolved:
            return resolved

    if project.exists():
        runs = [
            p
            for p in project.iterdir()
            if p.is_dir() and p.name.startswith(name_prefix)
        ]
        candidates: list[Tuple[float, Path, Path]] = []
        for run in runs:
            for fname in ("best.pt", "last.pt"):
                w = run / "weights" / fname
                if w.exists():
                    candidates.append((w.stat().st_mtime, w.resolve(), run.resolve()))
        if candidates:
            _, w_path, run_dir = max(candidates, key=lambda entry: entry[0])
            return w_path, run_dir

    return None, None


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

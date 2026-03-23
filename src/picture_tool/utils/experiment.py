from __future__ import annotations

import json
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


def _git_rev() -> Optional[str]:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip() or None
    except (FileNotFoundError, PermissionError, OSError, subprocess.SubprocessError):
        return None


def _env_info() -> Dict[str, Any]:
    import platform
    import sys
    import os

    info: Dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
    }
    
    if os.environ.get("PYTEST_IS_RUNNING") == "1":
        info["torch_version"] = "mocked_for_test"
        info["cuda_available"] = False
        info["ultralytics_version"] = "mocked_for_test"
        return info

    try:
        import torch  # type: ignore

        tv = getattr(torch, "__version__", None)
        info["torch_version"] = str(tv) if tv is not None else None
        info["cuda_available"] = torch.cuda.is_available()
    except (ImportError, AttributeError):
        info["torch_version"] = None
        info["cuda_available"] = None
    try:
        import ultralytics  # type: ignore

        uv = getattr(ultralytics, "__version__", None)
        info["ultralytics_version"] = str(uv) if uv is not None else None
    except (ImportError, AttributeError):
        info["ultralytics_version"] = None
    return info


def _load_metrics_csv(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        import csv

        rows = list(csv.DictReader(path.open("r", encoding="utf-8")))
        return rows[-1] if rows else {}
    except (FileNotFoundError, ValueError, KeyError, OSError, TypeError):
        return {}


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, str):
        # TorchVersion and other str subclasses need explicit cast for YAML
        return str(obj)
    if isinstance(obj, (int, float, bool)) or obj is None:
        return obj
    if isinstance(obj, (list, tuple, set)):
        return [_jsonable(x) for x in obj]
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    try:
        return str(obj)
    except (TypeError, ValueError, AttributeError):
        return repr(obj)


def write_experiment(
    run_type: str,
    config: Dict[str, Any],
    run_dir: Path,
    metrics: Optional[Dict[str, Any]] = None,
    artifacts: Optional[Dict[str, Any]] = None,
    extra: Optional[Dict[str, Any]] = None,
    output_dir: Path | str = Path("reports/experiments"),
    results_csv: Optional[Path] = None,
) -> Path:
    """Persist a reproducible experiment record."""
    run_dir = Path(run_dir).resolve()
    out_dir = Path(output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_id = f"{run_type}-{ts}"
    payload: Dict[str, Any] = {
        "id": run_id,
        "type": run_type,
        "timestamp": ts,
        "git_commit": _git_rev(),
        "run_dir": str(run_dir),
        "env": _env_info(),
        "config": _jsonable(config),
        "artifacts": _jsonable(artifacts or {}),
        "metrics": _jsonable(metrics or {}),
    }
    if results_csv:
        payload["metrics_csv"] = _load_metrics_csv(Path(results_csv))
    if extra:
        payload["extra"] = _jsonable(extra)

    yaml_path = out_dir / f"{run_id}.yaml"
    with yaml_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(
            payload,
            f,
            sort_keys=False,
            allow_unicode=True,
        )
    json_path = out_dir / f"{run_id}.json"
    json_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return yaml_path

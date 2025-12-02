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
    except Exception:
        return None


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (list, tuple)):
        return [_jsonable(x) for x in obj]
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    return obj


def write_experiment(
    run_type: str,
    config: Dict[str, Any],
    run_dir: Path,
    metrics: Optional[Dict[str, Any]] = None,
    artifacts: Optional[Dict[str, Any]] = None,
    extra: Optional[Dict[str, Any]] = None,
    output_dir: Path | str = Path("reports/experiments"),
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
        "config": _jsonable(config),
        "artifacts": _jsonable(artifacts or {}),
        "metrics": _jsonable(metrics or {}),
    }
    if extra:
        payload["extra"] = _jsonable(extra)

    yaml_path = out_dir / f"{run_id}.yaml"
    yaml.safe_dump(payload, yaml_path.open("w", encoding="utf-8"), sort_keys=False, allow_unicode=True)
    json_path = out_dir / f"{run_id}.json"
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return yaml_path

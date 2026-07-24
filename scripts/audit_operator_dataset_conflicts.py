#!/usr/bin/env python3
"""Dry-run audit of historical ready datasets and operator job snapshots."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from picture_tool.quality.operator_dataset_conflicts import (  # noqa: E402
    analysis_payload,
    analyze_operator_dataset,
    write_json_atomic,
)


def audit_operator_data(
    data_root: str | Path,
    *,
    failed_jobs_only: bool = True,
) -> dict[str, Any]:
    """Return a no-mutation report for ready manifests and job snapshots."""
    root = Path(data_root).expanduser().resolve()
    reports: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []

    for manifest in sorted(root.glob("*/*/metadata/review_dataset_manifest.csv")):
        dataset_root = manifest.parent.parent
        reports.append(
            _analyze_scope(
                dataset_root,
                manifest,
                scope=f"ready:{dataset_root.relative_to(root).as_posix()}",
                errors=errors,
            )
        )

    jobs_root = root / ".operator_handoff" / "jobs"
    if jobs_root.is_dir():
        for job_dir in sorted(jobs_root.iterdir()):
            if not job_dir.is_dir():
                continue
            state = _job_state(job_dir / "status.json")
            if failed_jobs_only and state != "failed":
                continue
            dataset_container = job_dir / "dataset"
            for manifest in sorted(
                dataset_container.glob("*/*/metadata/review_dataset_manifest.csv")
            ):
                dataset_root = manifest.parent.parent
                reports.append(
                    _analyze_scope(
                        dataset_root,
                        manifest,
                        scope=f"job:{job_dir.name}:{state or 'unknown'}",
                        errors=errors,
                    )
                )

    conflict_count = sum(len(item.get("conflicts", [])) for item in reports)
    selection_count = sum(
        len(item.get("canonical_selections", [])) for item in reports
    )
    return {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "mode": "dry_run_only",
        "data_root": str(root),
        "failed_jobs_only": failed_jobs_only,
        "summary": {
            "scopes_checked": len(reports),
            "canonical_selection_count": selection_count,
            "blocking_conflict_count": conflict_count,
            "error_count": len(errors),
        },
        "reports": reports,
        "errors": errors,
        "mutation_performed": False,
    }


def _analyze_scope(
    dataset_root: Path,
    manifest: Path,
    *,
    scope: str,
    errors: list[dict[str, str]],
) -> dict[str, Any]:
    try:
        relative_parts = dataset_root.parts
        product = relative_parts[-2] if len(relative_parts) >= 2 else ""
        area = relative_parts[-1] if relative_parts else ""
        analysis = analyze_operator_dataset(
            dataset_root / "raw" / "images",
            dataset_root / "raw" / "labels",
            manifest,
            product=product,
            area=area,
        )
        return analysis_payload(analysis, scope=scope)
    except (OSError, UnicodeDecodeError, ValueError) as exc:
        errors.append({"scope": scope, "error": str(exc)})
        return {
            "scope": scope,
            "safe": False,
            "canonical_selections": [],
            "conflicts": [],
            "analysis_error": str(exc),
        }


def _job_state(path: Path) -> str:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return ""
    return str(payload.get("state") or "").strip().lower() if isinstance(payload, dict) else ""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=PROJECT_ROOT / "data",
        help="Yolo11_auto_train data directory",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional JSON report path; stdout is always printed",
    )
    parser.add_argument(
        "--all-jobs",
        action="store_true",
        help="Audit all job snapshots instead of failed jobs only",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = audit_operator_data(
        args.data_root,
        failed_jobs_only=not args.all_jobs,
    )
    if args.output is not None:
        write_json_atomic(args.output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    summary = report["summary"]
    if summary["error_count"]:
        return 1
    return 2 if summary["blocking_conflict_count"] else 0


if __name__ == "__main__":
    raise SystemExit(main())

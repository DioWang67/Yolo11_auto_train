import csv
import logging
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, List


def _read_last_metrics(results_csv: Path) -> Dict[str, str]:
    metrics: Dict[str, str] = {}
    if not results_csv.exists():
        return metrics
    with open(results_csv, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        if not rows:
            return metrics
        last = rows[-1]
        # pick common keys if available
        for k in [
            "metrics/mAP50-95",
            "metrics/mAP50",
            "metrics/precision(B)",
            "metrics/recall(B)",
            "val/box_loss",
            "val/cls_loss",
            "val/dfl_loss",
            "lr/pg0",
        ]:
            if k in last:
                metrics[k] = last[k]
    return metrics


def _candidate_runs(project: Path, name: str) -> List[Path]:
    if not project.exists():
        return []
    runs = [p for p in project.iterdir() if p.is_dir() and p.name.startswith(name)]
    # Prefer those with results.csv
    runs = [p for p in runs if (p / "results.csv").exists()]
    runs.sort(
        key=lambda p: (
            (p / "results.csv").stat().st_mtime
            if (p / "results.csv").exists()
            else p.stat().st_mtime
        ),
        reverse=True,
    )
    return runs


def generate_report(config: dict, logger: Optional[logging.Logger] = None) -> Path:
    """Generate a simple Markdown report from Ultralytics training artifacts."""
    logger = logger or logging.getLogger(__name__)

    ycfg = config.get("yolo_training", {})
    rcfg = config.get("report", {})
    dataset_dir = Path(
        str(ycfg.get("dataset_dir", "./datasets/split_dataset"))
    ).resolve()
    project = Path(str(ycfg.get("project", "./runs/detect"))).resolve()
    name = str(ycfg.get("name", "train"))
    # Select latest matching run (train, train2, ...)
    candidates = _candidate_runs(project, name)
    run_dir = (candidates[0] if candidates else (project / name)).resolve()
    output_root = Path(str(rcfg.get("output_dir", "./reports"))).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_dir = output_root / f"report_{name}_{stamp}"
    report_dir.mkdir(parents=True, exist_ok=True)

    # Copy artifacts
    artifacts = [
        "results.png",
        "confusion_matrix.png",
        "PR_curve.png",
        "F1_curve.png",
        "P_curve.png",
        "R_curve.png",
        "labels.jpg",
    ]
    for fname in artifacts:
        src = run_dir / fname
        if src.exists() and rcfg.get("include_artifacts", True):
            shutil.copy(src, report_dir / fname)

    # Dataset counts
    def _count_images(p: Path) -> int:
        if not p.exists():
            return 0
        return sum(1 for _ in p.glob("*.*"))

    n_train = _count_images(dataset_dir / "train" / "images")
    n_val = _count_images(dataset_dir / "val" / "images")
    n_test = _count_images(dataset_dir / "test" / "images")

    # Metrics from results.csv
    metrics = _read_last_metrics(run_dir / "results.csv")

    # Write Markdown report
    md = [
        f"# YOLO Report: {name}",
        "",
        f"- Run dir: {run_dir}",
        f"- Dataset: {dataset_dir}",
        f"- Images: train={n_train}, val={n_val}, test={n_test}",
        "",
        "## Metrics",
    ]
    if metrics:
        for k, v in metrics.items():
            md.append(f"- {k}: {v}")
    else:
        md.append("- (No metrics found in results.csv)")

    md.extend(
        [
            "",
            "## Artifacts",
        ]
    )
    for fname in artifacts:
        if (report_dir / fname).exists():
            md.append(f"![{fname}]({(report_dir / fname).name})")

    report_path = report_dir / "report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md))

    logger.info(f"Report generated: {report_path}")
    return report_path

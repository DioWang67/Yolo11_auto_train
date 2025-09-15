import csv
import logging
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import matplotlib.pyplot as plt
from matplotlib import patches

from picture_tool.utils.io_utils import DEFAULT_IMAGE_EXTS


@dataclass
class LintConfig:
    image_dir: Path
    label_dir: Path
    output_dir: Path
    num_preview: int = 20
    preview_cols: int = 5
    seed: int = 42
    class_names: Optional[List[str]] = None


def _read_labels(label_path: Path) -> List[List[float]]:
    if not label_path.exists():
        return []
    lines = [ln.strip() for ln in label_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    labels: List[List[float]] = []
    for ln in lines:
        parts = ln.split()
        if len(parts) >= 5:
            try:
                cls = int(float(parts[0]))
                x, y, w, h = map(float, parts[1:5])
                labels.append([cls, x, y, w, h])
            except Exception:
                continue
    return labels


def _list_files(directory: Path, exts: Tuple[str, ...]) -> Dict[str, Path]:
    mapping: Dict[str, Path] = {}
    if not directory.exists():
        return mapping
    for p in directory.iterdir():
        if p.is_file() and (p.suffix.lower() in exts):
            mapping[p.stem] = p
    return mapping


def _validate_labels(labels: List[List[float]], num_classes: Optional[int]) -> List[str]:
    issues: List[str] = []
    for row in labels:
        cls, x, y, w, h = row
        if num_classes is not None and (cls < 0 or cls >= num_classes):
            issues.append(f"class_out_of_range:{cls}")
        # Bounds check
        for name, v in (("x", x), ("y", y), ("w", w), ("h", h)):
            if not (0.0 <= v <= 1.0):
                issues.append(f"{name}_out_of_bounds:{v:.4f}")
        # Positive area
        if w <= 0 or h <= 0:
            issues.append("non_positive_area")
        area = w * h
        if area < 1e-5:
            issues.append("tiny_box")
        if area > 0.8:
            issues.append("huge_box")
    if not labels:
        issues.append("empty_labels")
    return issues


def lint_dataset(config: dict, logger: Optional[logging.Logger] = None) -> Path:
    logger = logger or logging.getLogger(__name__)
    lcfg = config.get("dataset_lint", {})
    image_dir = Path(str(lcfg.get("image_dir", "./data/augmented/images")))
    label_dir = Path(str(lcfg.get("label_dir", "./data/augmented/labels")))
    output_dir = Path(str(lcfg.get("output_dir", "./reports/lint")))
    output_dir.mkdir(parents=True, exist_ok=True)
    class_names = config.get("yolo_training", {}).get("class_names")
    num_classes = len(class_names) if isinstance(class_names, list) and class_names else None

    img_map = _list_files(image_dir, DEFAULT_IMAGE_EXTS)
    lbl_map = _list_files(label_dir, (".txt",))
    stems = sorted(set(img_map.keys()) | set(lbl_map.keys()))

    issues_path = output_dir / "lint.csv"
    counts: Dict[str, int] = {}
    cls_hist: Dict[int, int] = {}
    with open(issues_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["stem", "issue", "detail"]) 
        for stem in stems:
            if stem not in img_map:
                writer.writerow([stem, "missing_image", str(label_dir / f"{stem}.txt")])
                counts["missing_image"] = counts.get("missing_image", 0) + 1
                continue
            if stem not in lbl_map:
                writer.writerow([stem, "missing_label", str(image_dir / f"{stem}.jpg|png|...")])
                counts["missing_label"] = counts.get("missing_label", 0) + 1
                continue
            labels = _read_labels(lbl_map[stem])
            for row in labels:
                cls = int(row[0])
                cls_hist[cls] = cls_hist.get(cls, 0) + 1
            errors = _validate_labels(labels, num_classes)
            for err in errors:
                writer.writerow([stem, err, ""])
                counts[err] = counts.get(err, 0) + 1

    # Write summary
    summary_md = output_dir / "summary.md"
    total = len(stems)
    with open(summary_md, "w", encoding="utf-8") as f:
        f.write(f"# Dataset Lint Summary\n\n")
        f.write(f"- Images scanned: {total}\n")
        for k in sorted(counts):
            f.write(f"- {k}: {counts[k]}\n")
        if cls_hist:
            f.write("\n## Class Histogram\n")
            for cid in sorted(cls_hist):
                cname = class_names[cid] if (isinstance(class_names, list) and cid < len(class_names)) else str(cid)
                f.write(f"- {cname} ({cid}): {cls_hist[cid]}\n")

    logger.info(f"Dataset lint written to: {issues_path} and {summary_md}")
    return output_dir


def _draw_boxes(ax, img, labels: List[List[float]], title: str = ""):
    ax.imshow(img)
    ax.set_axis_off()
    if title:
        ax.set_title(title, fontsize=8)
    h, w = img.shape[:2]
    for row in labels:
        _, x, y, bw, bh = row
        x1 = (x - bw / 2) * w
        y1 = (y - bh / 2) * h
        ww = bw * w
        hh = bh * h
        ax.add_patch(patches.Rectangle((x1, y1), ww, hh, fill=False, edgecolor='lime', linewidth=1))


def preview_dataset(config: dict, logger: Optional[logging.Logger] = None) -> Path:
    logger = logger or logging.getLogger(__name__)
    pcfg = config.get("aug_preview", {})
    image_dir = Path(str(pcfg.get("image_dir", "./data/augmented/images")))
    label_dir = Path(str(pcfg.get("label_dir", "./data/augmented/labels")))
    output_dir = Path(str(pcfg.get("output_dir", "./reports/preview")))
    num_samples = int(pcfg.get("num_samples", 16))
    cols = int(pcfg.get("cols", 4))
    seed = int(pcfg.get("seed", 42))
    output_dir.mkdir(parents=True, exist_ok=True)

    img_map = _list_files(image_dir, DEFAULT_IMAGE_EXTS)
    lbl_map = _list_files(label_dir, (".txt",))
    stems = sorted(set(img_map.keys()) & set(lbl_map.keys()))
    if not stems:
        raise FileNotFoundError("No paired images/labels found for preview")
    random.Random(seed).shuffle(stems)
    sel = stems[:num_samples]

    rows = math.ceil(len(sel) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    # Normalize axes to a flat list of Axes
    if isinstance(axes, (list, tuple)):
        axes = list(axes)
    elif hasattr(axes, 'ravel'):
        try:
            axes = list(axes.ravel())
        except Exception:
            axes = [axes]
    else:
        axes = [axes]

    for i, stem in enumerate(sel):
        ax = axes[i]
        img_bgr = cv2.imread(str(img_map[stem]))
        if img_bgr is None:
            ax.set_axis_off()
            ax.set_title(f"Missing image: {stem}", fontsize=8)
            continue
        img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        labels = _read_labels(lbl_map[stem])
        _draw_boxes(ax, img, labels, stem)

    # Hide remaining axes
    for j in range(i + 1, rows * cols):
        axes[j].set_axis_off()

    out_path = output_dir / "preview.png"
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info(f"Preview saved to: {out_path}")
    return out_path

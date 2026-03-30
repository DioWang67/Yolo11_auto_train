"""Auto-generate position validation config from training results.

Uses statistical aggregation (mean ± std) instead of min/max envelope,
supports multi-instance same-class objects via ``ClassName#N`` indexed keys,
and auto-computes a sensible tolerance from calibration σ when not explicitly set.
"""

import logging
import math
import os
import yaml
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, MutableMapping, Tuple

try:
    if os.environ.get("PYTEST_IS_RUNNING") == "1":
        raise ImportError("Bypass ultralytics during pytest")
    from ultralytics import YOLO  # type: ignore
except ImportError:
    YOLO = None  # type: ignore

from picture_tool.utils.normalization import normalize_imgsz
from picture_tool.position.yolo_position_validator import _resolve_sample_images


# ---------------------------------------------------------------------------
# Statistics helpers
# ---------------------------------------------------------------------------

def _mean(values: List[float]) -> float:
    return sum(values) / len(values)


def _stdev(values: List[float]) -> float:
    if len(values) < 2:
        return 0.0
    m = _mean(values)
    return math.sqrt(sum((v - m) ** 2 for v in values) / (len(values) - 1))


def _statistical_aggregate(boxes: List[List[int]]) -> Dict[str, Any]:
    """Compute mean-based expected box with statistical metadata.

    Instead of min/max envelope (which inflates with outliers), this computes:
    - Mean center (cx, cy) and mean bbox size (w, h)
    - Expected box = mean center ± mean half-size
    - Standard deviation of centers for tolerance estimation
    """
    centers_x = [(b[0] + b[2]) / 2.0 for b in boxes]
    centers_y = [(b[1] + b[3]) / 2.0 for b in boxes]
    widths = [float(b[2] - b[0]) for b in boxes]
    heights = [float(b[3] - b[1]) for b in boxes]

    mean_cx = _mean(centers_x)
    mean_cy = _mean(centers_y)
    mean_w = _mean(widths)
    mean_h = _mean(heights)
    sigma_cx = _stdev(centers_x)
    sigma_cy = _stdev(centers_y)

    x1 = int(round(mean_cx - mean_w / 2))
    y1 = int(round(mean_cy - mean_h / 2))
    x2 = int(round(mean_cx + mean_w / 2))
    y2 = int(round(mean_cy + mean_h / 2))

    return {
        "x1": x1, "y1": y1, "x2": x2, "y2": y2,
        "cx": round(mean_cx, 2),
        "cy": round(mean_cy, 2),
        "sigma_cx": round(sigma_cx, 2),
        "sigma_cy": round(sigma_cy, 2),
        "count": len(boxes),
    }


# ---------------------------------------------------------------------------
# Multi-instance clustering
# ---------------------------------------------------------------------------

def _mode_count(values: List[int]) -> int:
    """Return the most common value (mode) from a list of counts."""
    if not values:
        return 1
    counter = Counter(values)
    return counter.most_common(1)[0][0]


def _simple_kmeans(
    points: List[Tuple[float, float]], k: int, max_iter: int = 50
) -> List[List[int]]:
    """Minimal K-means on 2D points. Returns list of K groups (indices).

    No external dependency required. Sufficient for small N (typically < 100).
    """
    n = len(points)
    if k <= 0 or k > n:
        return [list(range(n))]

    # Initialize centroids by evenly spaced selection from sorted points
    sorted_indices = sorted(range(n), key=lambda i: (points[i][0], points[i][1]))
    step = max(1, n // k)
    centroids = [points[sorted_indices[min(i * step, n - 1)]] for i in range(k)]

    assignments = [0] * n
    for _ in range(max_iter):
        changed = False
        # Assign each point to nearest centroid
        for i, (px, py) in enumerate(points):
            best_c = 0
            best_dist = float("inf")
            for c, (ccx, ccy) in enumerate(centroids):
                d = (px - ccx) ** 2 + (py - ccy) ** 2
                if d < best_dist:
                    best_dist = d
                    best_c = c
            if assignments[i] != best_c:
                assignments[i] = best_c
                changed = True

        if not changed:
            break

        # Recompute centroids
        for c in range(k):
            members = [(points[i][0], points[i][1]) for i in range(n) if assignments[i] == c]
            if members:
                centroids[c] = (_mean([m[0] for m in members]), _mean([m[1] for m in members]))

    groups: List[List[int]] = [[] for _ in range(k)]
    for i, c in enumerate(assignments):
        groups[c].append(i)
    return groups


def _cluster_multi_instance(
    boxes: List[List[int]],
    per_image_counts: List[int],
    logger: logging.Logger,
    class_name: str,
) -> Dict[str, Dict[str, Any]]:
    """Cluster detections for a class with multiple instances per image.

    Returns a dict mapping indexed keys (e.g., ``Black#0``, ``Black#1``)
    to their statistical aggregate.
    """
    k = _mode_count(per_image_counts)
    if k <= 1:
        return {class_name: _statistical_aggregate(boxes)}

    centers = [((b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0) for b in boxes]
    groups = _simple_kmeans(centers, k)

    # Sort clusters by x then y for stable ordering
    cluster_data: List[Tuple[float, float, List[List[int]]]] = []
    for group_indices in groups:
        if not group_indices:
            continue
        group_boxes = [boxes[i] for i in group_indices]
        cx = _mean([(b[0] + b[2]) / 2.0 for b in group_boxes])
        cy = _mean([(b[1] + b[3]) / 2.0 for b in group_boxes])
        cluster_data.append((cx, cy, group_boxes))

    cluster_data.sort(key=lambda t: (t[0], t[1]))

    if len(per_image_counts) > 2:
        count_variance = _stdev([float(c) for c in per_image_counts])
        if count_variance > 1.0:
            logger.warning(
                "Class '%s': per-image count varies significantly (σ=%.1f). "
                "Detected mode K=%d but counts range %d–%d. "
                "Consider manual review of position config.",
                class_name, count_variance, k,
                min(per_image_counts), max(per_image_counts),
            )

    result: Dict[str, Dict[str, Any]] = {}
    for idx, (_, _, group_boxes) in enumerate(cluster_data):
        key = f"{class_name}#{idx}"
        result[key] = _statistical_aggregate(group_boxes)

    logger.info(
        "Class '%s': detected %d instances per image, "
        "emitting indexed keys %s",
        class_name, len(cluster_data),
        list(result.keys()),
    )
    return result


# ---------------------------------------------------------------------------
# Main generator
# ---------------------------------------------------------------------------

class PositionConfigGenerator:
    @staticmethod
    def generate(
        config: MutableMapping[str, Any], run_dir: Path, logger: logging.Logger
    ) -> Optional[Path]:
        """Automatically derive a position config from latest training results.

        Improvements over the legacy min/max approach:
        - **Statistical aggregation**: expected boxes use mean center ± mean size
        - **Multi-instance support**: same-class objects are clustered and emitted
          as ``ClassName#0``, ``ClassName#1``, etc.
        - **Auto-tolerance**: when tolerance is not explicitly set, it is computed
          from the calibration σ (3σ rule → 99.7% of good samples pass).
        """
        if YOLO is None:
            logger.info(
                "Skipping auto position config generation: ultralytics not available."
            )
            return None

        ycfg = config.get("yolo_training")
        if not isinstance(ycfg, MutableMapping):
            return None
        pos_cfg = ycfg.get("position_validation")
        if not isinstance(pos_cfg, MutableMapping):
            return None
        if not pos_cfg.get("enabled"):
            return None
        auto_generate = pos_cfg.get("auto_generate", True)
        if not auto_generate:
            logger.info("Auto position config generation disabled via config.")
            return None

        product = pos_cfg.get("product")
        area = pos_cfg.get("area")
        if not product or not area:
            logger.warning(
                "Auto position config generation skipped: "
                "product/area must be specified when position validation is enabled."
            )
            return None

        imgsz_value = pos_cfg.get("imgsz") or ycfg.get("imgsz") or 640
        imgsz_norm = normalize_imgsz(imgsz_value) or [640, 640]
        imgsz_int = imgsz_norm[0]

        dataset_dir_val = ycfg.get("dataset_dir")
        if dataset_dir_val:
            dataset_dir = Path(str(dataset_dir_val))
        else:
            dataset_dir = Path("data/split")

        sample_dir_value = pos_cfg.get("sample_dir") or (dataset_dir / "val" / "images")
        sample_dir = Path(str(sample_dir_value)).resolve()

        ImageSuffixes = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}

        try:
            images = _resolve_sample_images(sample_dir, ImageSuffixes)
        except (FileNotFoundError, OSError) as exc:
            logger.warning("Auto position config generation skipped: %s", exc)
            return None

        weights_path = run_dir / "weights" / "best.pt"
        if not weights_path.exists():
            weights_path = run_dir / "weights" / "last.pt"
        if not weights_path.exists():
            logger.warning(
                "Auto position config generation skipped: "
                "unable to locate weights under %s",
                run_dir,
            )
            return None

        try:
            from picture_tool.position.yolo_position_validator import (
                convert_results_to_detections,
            )
        except (ImportError, AttributeError) as exc:  # pragma: no cover
            logger.warning("Auto position config generation skipped: %s", exc)
            return None

        device_value = pos_cfg.get("device") or ycfg.get("device") or "cpu"
        conf_value = pos_cfg.get("conf")
        if conf_value is None:
            conf_value = 0.25
        try:
            model = YOLO(str(weights_path))
        except (FileNotFoundError, RuntimeError, OSError) as exc:  # pragma: no cover
            logger.warning(
                "Auto position config generation skipped: "
                "failed to load weights (%s)",
                exc,
            )
            return None

        # ------------------------------------------------------------------
        # Collect detections — track per-image counts for multi-instance
        # ------------------------------------------------------------------
        boxes_by_class: Dict[str, List[List[int]]] = {}
        per_image_class_counts: Dict[str, List[int]] = {}

        for img_path in images:
            try:
                results = model(
                    str(img_path),
                    imgsz=imgsz_norm[0],
                    device=str(device_value),
                    conf=float(conf_value),
                    verbose=False,
                )
            except (RuntimeError, OSError, AttributeError) as exc:  # pragma: no cover
                logger.warning(
                    "Auto position config: inference failed for %s (%s)",
                    img_path.name, exc,
                )
                continue

            # Count detections per class in this image
            image_class_counter: Dict[str, int] = {}

            for res in results:
                detections = convert_results_to_detections(res, imgsz_int)
                for det in detections:
                    cls = det.get("class")
                    bbox = det.get("bbox")
                    if not isinstance(cls, str):
                        cls = str(cls)
                    if not isinstance(bbox, list) or len(bbox) != 4:
                        continue
                    try:
                        box_vals = [int(round(float(v))) for v in bbox]
                    except (TypeError, ValueError):
                        continue
                    boxes_by_class.setdefault(cls, []).append(box_vals)
                    image_class_counter[cls] = image_class_counter.get(cls, 0) + 1

            for cls, count in image_class_counter.items():
                per_image_class_counts.setdefault(cls, []).append(count)

        if not boxes_by_class:
            logger.warning(
                "Auto position config generation skipped: "
                "no detections collected from sample images."
            )
            return None

        # ------------------------------------------------------------------
        # Build expected_boxes with statistical aggregation + clustering
        # ------------------------------------------------------------------
        expected_boxes: Dict[str, Dict[str, Any]] = {}
        all_sigmas: List[float] = []

        for cls, bxs in boxes_by_class.items():
            counts = per_image_class_counts.get(cls, [1])
            k = _mode_count(counts)

            if k > 1 and len(bxs) >= k:
                clustered = _cluster_multi_instance(bxs, counts, logger, cls)
                for key, stats in clustered.items():
                    expected_boxes[key] = stats
                    all_sigmas.append(stats["sigma_cx"])
                    all_sigmas.append(stats["sigma_cy"])
            else:
                stats = _statistical_aggregate(bxs)
                expected_boxes[cls] = stats
                all_sigmas.append(stats["sigma_cx"])
                all_sigmas.append(stats["sigma_cy"])

        if not expected_boxes:
            logger.warning(
                "Auto position config generation skipped: "
                "expected boxes could not be computed."
            )
            return None

        # ------------------------------------------------------------------
        # Resolve tolerance: auto-compute from σ if not explicitly set
        # ------------------------------------------------------------------
        explicit_tolerance = pos_cfg.get("tolerance_override")
        if explicit_tolerance is not None:
            tolerance_value = float(explicit_tolerance)
        elif "tolerance" in pos_cfg and float(pos_cfg.get("tolerance", 0.0)) > 0:
            tolerance_value = float(pos_cfg["tolerance"])
        else:
            # Auto-compute: 3σ rule with minimum floor of 5px
            max_sigma = max(all_sigmas) if all_sigmas else 0.0
            auto_tolerance_px = max(max_sigma * 3.0, 5.0)
            tolerance_value = round((auto_tolerance_px / imgsz_int) * 100.0, 2)
            logger.info(
                "Auto-computed tolerance: %.2f%% (%.1fpx) from max σ=%.2fpx "
                "(3σ rule, min 5px floor, imgsz=%d)",
                tolerance_value, auto_tolerance_px, max_sigma, imgsz_int,
            )

        area_block: Dict[str, Any] = {
            "enabled": True,
            "mode": str(pos_cfg.get("mode", "center")),
            "tolerance": float(tolerance_value),
            "expected_boxes": expected_boxes,
            "imgsz": imgsz_norm[0],
        }

        position_config = {str(product): {str(area): area_block}}
        out_path = (run_dir / "auto_position_config.yaml").resolve()
        try:
            with open(out_path, "w", encoding="utf-8") as fh:
                yaml.safe_dump(
                    position_config, fh, allow_unicode=True, sort_keys=False
                )
        except (FileNotFoundError, OSError, yaml.YAMLError) as exc:  # pragma: no cover
            logger.warning("Auto position config generation failed: %s", exc)
            return None

        previous_path = pos_cfg.get("config_path")
        if previous_path and previous_path != str(out_path):
            pos_cfg["previous_config_path"] = previous_path
        pos_cfg["config_path"] = str(out_path)
        if pos_cfg.get("config"):
            pos_cfg.pop("config", None)
        if not pos_cfg.get("sample_dir"):
            pos_cfg["sample_dir"] = str(sample_dir)
        if not pos_cfg.get("weights"):
            pos_cfg["weights"] = str(weights_path)
        logger.info("Auto-generated position config at %s", out_path)
        return out_path

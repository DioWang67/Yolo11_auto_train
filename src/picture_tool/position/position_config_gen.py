"""Auto-generate position validation config from training results.

Uses statistical aggregation (mean ± std) instead of min/max envelope,
supports multi-instance same-class objects via ``ClassName#N`` indexed keys,
and auto-computes a sensible tolerance from calibration σ when not explicitly set.
"""

import logging
import math
import os
import hashlib
import yaml
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, MutableMapping, Tuple

try:
    if os.environ.get("PYTEST_IS_RUNNING") == "1":
        raise ImportError("Bypass ultralytics during pytest")
    from ultralytics import YOLO  # type: ignore
except ImportError:
    YOLO = None  # type: ignore

from picture_tool.utils.normalization import normalize_imgsz
from picture_tool.position.yolo_position_validator import _resolve_sample_images
from picture_tool.position.position_calibration import (
    PositionCalibrationError,
    collect_yolo_calibration_dataset,
    write_calibration_manifest,
)


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


def _median(values: List[float]) -> float:
    if not values:
        raise ValueError("Cannot compute a median from an empty list.")
    ordered = sorted(float(value) for value in values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2.0


def _quantile(values: List[float], quantile: float) -> float:
    if not values:
        raise ValueError("Cannot compute a quantile from an empty list.")
    if not 0.0 <= quantile <= 1.0:
        raise ValueError("quantile must be between 0 and 1.")
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * quantile
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


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


def _robust_statistical_aggregate(boxes: List[List[int]]) -> Dict[str, Any]:
    """Use medians and P99 center distance to resist calibration outliers."""

    centers_x = [(box[0] + box[2]) / 2.0 for box in boxes]
    centers_y = [(box[1] + box[3]) / 2.0 for box in boxes]
    widths = [float(box[2] - box[0]) for box in boxes]
    heights = [float(box[3] - box[1]) for box in boxes]
    median_cx = _median(centers_x)
    median_cy = _median(centers_y)
    median_width = _median(widths)
    median_height = _median(heights)
    distances = [
        math.hypot(cx - median_cx, cy - median_cy)
        for cx, cy in zip(centers_x, centers_y)
    ]
    mad_cx = _median([abs(value - median_cx) for value in centers_x])
    mad_cy = _median([abs(value - median_cy) for value in centers_y])
    return {
        "x1": int(round(median_cx - median_width / 2.0)),
        "y1": int(round(median_cy - median_height / 2.0)),
        "x2": int(round(median_cx + median_width / 2.0)),
        "y2": int(round(median_cy + median_height / 2.0)),
        "cx": round(median_cx, 2),
        "cy": round(median_cy, 2),
        "sigma_cx": round(_stdev(centers_x), 2),
        "sigma_cy": round(_stdev(centers_y), 2),
        "mad_cx": round(mad_cx, 2),
        "mad_cy": round(mad_cy, 2),
        "p99_center_distance": round(_quantile(distances, 0.99), 2),
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
    aggregate: Callable[[List[List[int]]], Dict[str, Any]] = _statistical_aggregate,
) -> Dict[str, Dict[str, Any]]:
    """Cluster detections for a class with multiple instances per image.

    Returns a dict mapping indexed keys (e.g., ``Black#0``, ``Black#1``)
    to their statistical aggregate.
    """
    k = _mode_count(per_image_counts)
    if k <= 1:
        return {class_name: aggregate(boxes)}

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
        result[key] = aggregate(group_boxes)

    logger.info(
        "Class '%s': detected %d instances per image, "
        "emitting indexed keys %s",
        class_name, len(cluster_data),
        list(result.keys()),
    )
    return result


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _collect_prediction_calibration(
    *,
    pos_cfg: MutableMapping[str, Any],
    ycfg: MutableMapping[str, Any],
    dataset_dir: Path,
    run_dir: Path,
    imgsz: int,
    logger: logging.Logger,
) -> tuple[
    Dict[str, List[List[int]]],
    Dict[str, List[int]],
    Dict[str, Any],
]:
    """Legacy adapter retained for explicitly configured prediction baselines."""

    if YOLO is None:
        raise RuntimeError(
            "Prediction-based position calibration requires ultralytics."
        )
    sample_dir = Path(
        str(pos_cfg.get("sample_dir") or dataset_dir / "val" / "images")
    ).resolve()
    images = _resolve_sample_images(sample_dir)
    weights_path = run_dir / "weights" / "best.pt"
    if not weights_path.exists():
        weights_path = run_dir / "weights" / "last.pt"
    if not weights_path.exists():
        raise FileNotFoundError(
            f"Unable to locate prediction calibration weights under {run_dir}"
        )
    from picture_tool.position.yolo_position_validator import (
        convert_results_to_detections,
    )

    device = str(pos_cfg.get("device") or ycfg.get("device") or "cpu")
    conf = float(pos_cfg.get("conf") or 0.25)
    model = YOLO(str(weights_path))
    boxes_by_class: Dict[str, List[List[int]]] = {}
    per_image_class_counts: Dict[str, List[int]] = {}
    for image_path in images:
        results = model(
            str(image_path),
            imgsz=imgsz,
            device=device,
            conf=conf,
            verbose=False,
        )
        image_counts: Dict[str, int] = {}
        for result in results:
            for detection in convert_results_to_detections(result, imgsz):
                class_name = str(detection.get("class"))
                bbox = detection.get("bbox")
                if not isinstance(bbox, list) or len(bbox) != 4:
                    continue
                values = [int(round(float(value))) for value in bbox]
                boxes_by_class.setdefault(class_name, []).append(values)
                image_counts[class_name] = image_counts.get(class_name, 0) + 1
        for class_name, count in image_counts.items():
            per_image_class_counts.setdefault(class_name, []).append(count)
    if not boxes_by_class:
        raise RuntimeError(
            "Prediction-based position calibration produced no detections."
        )
    logger.warning(
        "Using legacy prediction-based position calibration for %s images. "
        "Human-verified labels are required for production promotion.",
        len(images),
    )
    return (
        boxes_by_class,
        per_image_class_counts,
        {
            "source": "challenger_predictions_legacy",
            "sample_count": len(images),
            "weights_sha256": _sha256_file(weights_path),
        },
    )


# ---------------------------------------------------------------------------
# Main generator
# ---------------------------------------------------------------------------

class PositionConfigGenerator:
    @staticmethod
    def generate(
        config: MutableMapping[str, Any], run_dir: Path, logger: logging.Logger
    ) -> Optional[Path]:
        """Derive a position config from human labels or legacy predictions.

        Human-verified YOLO labels are the default source. The prediction source
        remains available only for explicit legacy configurations.
        """
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
        dataset_dir = Path(str(ycfg.get("dataset_dir") or "data/split"))
        calibration_source = str(
            pos_cfg.get("calibration_source") or "labels"
        ).strip().lower()
        calibration_manifest_path: Path | None = None
        calibration_summary: Dict[str, Any]
        aggregate = _statistical_aggregate

        if calibration_source == "labels":
            calibration_image_dir = Path(
                str(
                    pos_cfg.get("calibration_image_dir")
                    or dataset_dir / "train" / "images"
                )
            ).resolve()
            calibration_label_dir = Path(
                str(
                    pos_cfg.get("calibration_label_dir")
                    or dataset_dir / "train" / "labels"
                )
            ).resolve()
            class_names = ycfg.get("class_names")
            if not isinstance(class_names, list) or not class_names:
                raise PositionCalibrationError(
                    "Human position calibration requires yolo_training.class_names."
                )
            calibration = collect_yolo_calibration_dataset(
                image_dir=calibration_image_dir,
                label_dir=calibration_label_dir,
                class_names=class_names,
                imgsz=imgsz_int,
                require_all_classes=bool(
                    pos_cfg.get("calibration_require_all_classes", True)
                ),
                exclude_augmented=bool(
                    pos_cfg.get("calibration_exclude_augmented", True)
                ),
            )
            minimum_samples = int(pos_cfg.get("calibration_min_samples", 3))
            if len(calibration.samples) < minimum_samples:
                raise PositionCalibrationError(
                    "Insufficient complete position calibration samples: "
                    f"required={minimum_samples}, actual={len(calibration.samples)}"
                )
            boxes_by_class = {
                name: [list(box) for box in boxes]
                for name, boxes in calibration.boxes_by_class.items()
            }
            per_image_class_counts = {
                name: list(counts)
                for name, counts in calibration.per_image_class_counts.items()
            }
            calibration_manifest_path = (
                run_dir / "position_calibration_manifest.json"
            ).resolve()
            manifest_payload = calibration.manifest_payload(
                product=str(product),
                area=str(area),
                imgsz=imgsz_int,
            )
            write_calibration_manifest(
                calibration_manifest_path,
                manifest_payload,
            )
            calibration_summary = {
                "source": "human_verified_yolo_labels",
                "sample_count": len(calibration.samples),
                "dataset_sha256": calibration.dataset_sha256,
                "manifest_sha256": _sha256_file(calibration_manifest_path),
            }
            aggregate = _robust_statistical_aggregate
        elif calibration_source == "predictions":
            (
                boxes_by_class,
                per_image_class_counts,
                calibration_summary,
            ) = _collect_prediction_calibration(
                pos_cfg=pos_cfg,
                ycfg=ycfg,
                dataset_dir=dataset_dir,
                run_dir=run_dir,
                imgsz=imgsz_int,
                logger=logger,
            )
        else:
            raise ValueError(
                "position_validation.calibration_source must be "
                "'labels' or 'predictions'."
            )

        # ------------------------------------------------------------------
        # Build expected_boxes with statistical aggregation + clustering
        # ------------------------------------------------------------------
        expected_boxes: Dict[str, Dict[str, Any]] = {}
        all_sigmas: List[float] = []
        all_p99_distances: List[float] = []

        for cls, bxs in boxes_by_class.items():
            counts = per_image_class_counts.get(cls, [1])
            k = _mode_count(counts)

            if k > 1 and len(bxs) >= k:
                clustered = _cluster_multi_instance(
                    bxs,
                    counts,
                    logger,
                    cls,
                    aggregate=aggregate,
                )
                for key, stats in clustered.items():
                    expected_boxes[key] = stats
                    all_sigmas.append(stats["sigma_cx"])
                    all_sigmas.append(stats["sigma_cy"])
                    if "p99_center_distance" in stats:
                        all_p99_distances.append(
                            float(stats["p99_center_distance"])
                        )
            else:
                stats = aggregate(bxs)
                expected_boxes[cls] = stats
                all_sigmas.append(stats["sigma_cx"])
                all_sigmas.append(stats["sigma_cy"])
                if "p99_center_distance" in stats:
                    all_p99_distances.append(float(stats["p99_center_distance"]))

        if not expected_boxes:
            logger.warning(
                "Auto position config generation skipped: "
                "expected boxes could not be computed."
            )
            return None

        # ------------------------------------------------------------------
        # Resolve tolerance: auto-compute from σ if not explicitly set
        # ------------------------------------------------------------------
        mode = str(pos_cfg.get("mode") or "center").strip().lower()
        explicit_tolerance = pos_cfg.get("tolerance_override")
        configured_unit = str(
            pos_cfg.get("tolerance_unit") or "percent"
        ).lower()
        if explicit_tolerance is not None:
            tolerance_value = float(explicit_tolerance)
            tolerance_unit = configured_unit
        elif "tolerance" in pos_cfg and float(pos_cfg.get("tolerance", 0.0)) > 0:
            tolerance_value = float(pos_cfg["tolerance"])
            tolerance_unit = configured_unit
        elif mode == "iou":
            raise ValueError(
                "IoU position mode requires an explicit positive tolerance "
                "(minimum IoU as 0-1 or percent)."
            )
        else:
            max_sigma = max(all_sigmas) if all_sigmas else 0.0
            robust_distance = (
                max(all_p99_distances) if all_p99_distances else max_sigma * 3.0
            )
            auto_tolerance_px = max(robust_distance, 5.0)
            tolerance_value = round(auto_tolerance_px, 2)
            tolerance_unit = "pixel"
            logger.info(
                "Auto-computed position tolerance: %.2fpx "
                "(P99/3σ, min 5px floor, imgsz=%d)",
                tolerance_value,
                imgsz_int,
            )
        if not math.isfinite(tolerance_value) or tolerance_value <= 0.0:
            raise ValueError("Position tolerance must be a finite positive number.")
        if mode == "iou":
            normalized_iou = (
                tolerance_value / 100.0
                if tolerance_value > 1.0
                else tolerance_value
            )
            if normalized_iou > 1.0:
                raise ValueError(
                    "IoU position tolerance must be between 0-1 or 0-100%."
                )

        area_block: Dict[str, Any] = {
            "enabled": True,
            "mode": mode,
            "tolerance": float(tolerance_value),
            "tolerance_unit": tolerance_unit,
            "expected_boxes": expected_boxes,
            "imgsz": imgsz_norm[0],
            "calibration": calibration_summary,
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
        if calibration_manifest_path is not None:
            pos_cfg["calibration_manifest_path"] = str(
                calibration_manifest_path
            )
        if pos_cfg.get("config"):
            pos_cfg.pop("config", None)
        logger.info("Auto-generated position config at %s", out_path)
        return out_path

import logging
import shutil
from pathlib import Path
from typing import Any, Dict, List, Mapping, MutableMapping, Optional, Sequence

import yaml

try:
    from ultralytics import YOLO  # type: ignore[import-untyped]
except Exception:  # pragma: no cover
    YOLO = None  # type: ignore

ImageSuffixes = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def _ensure_data_yaml(
    dataset_dir: Path, names: List[str], out_path: Optional[Path] = None
) -> Path:
    dataset_dir = dataset_dir.resolve()
    data = {
        "path": str(dataset_dir),
        "train": "train/images",
        "val": "val/images",
        "test": "test/images",
        "names": names,
    }
    out_path = out_path or (dataset_dir / "data.yaml")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False)
    return out_path


def _normalize_imgsz(value: Any) -> Optional[List[int]]:
    """Normalize imgsz representations to [width, height]."""
    if value in (None, "", []):
        return None
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        ints: List[int] = []
        for item in value:
            if item in (None, ""):
                continue
            try:
                ints.append(int(float(item)))
            except (TypeError, ValueError):
                continue
        if not ints:
            return None
        if len(ints) == 1:
            ints *= 2
        elif len(ints) >= 2:
            ints = ints[:2]
            if len(ints) == 1:
                ints.append(ints[0])
        if len(ints) < 2:
            ints.append(ints[0])
        return [ints[0], ints[1]]
    try:
        val = int(float(value))
    except (TypeError, ValueError):
        return None
    return [val, val]


def _normalize_name_sequence(value: Any) -> List[str]:
    """Convert various name representations (list or dict) to a string list."""
    if isinstance(value, Mapping):
        items = []
        for key, val in value.items():
            try:
                sort_key = int(key)
            except (TypeError, ValueError):
                sort_key = key
            items.append((sort_key, str(val)))
        try:
            items.sort(key=lambda item: item[0])
        except TypeError:
            items.sort(key=lambda item: str(item[0]))
        return [name for _, name in items if name]
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [str(item) for item in value if item not in (None, "")]
    if value not in (None, ""):
        return [str(value)]
    return []


def _load_class_names_from_run(
    run_dir: Path,
    logger: logging.Logger,
    fallback: Any,
) -> List[str]:
    """Load class names from the Ultralytics run directory or fallback."""
    args_path = run_dir / "args.yaml"
    if args_path.exists():
        try:
            args_data = yaml.safe_load(args_path.read_text(encoding="utf-8")) or {}
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.warning(
                "Detection config export: failed to read %s (%s)", args_path, exc
            )
            args_data = {}
        if isinstance(args_data, Mapping):
            names = _normalize_name_sequence(args_data.get("names"))
            if names:
                return names
            data_entry = args_data.get("data")
            if data_entry:
                data_path = Path(str(data_entry))
                if not data_path.is_absolute():
                    data_path = (run_dir / data_path).resolve()
                if data_path.exists():
                    try:
                        data_yaml = (
                            yaml.safe_load(data_path.read_text(encoding="utf-8")) or {}
                        )
                    except Exception as exc:  # pragma: no cover
                        logger.warning(
                            "Detection config export: failed to read dataset yaml %s (%s)",
                            data_path,
                            exc,
                        )
                        data_yaml = {}
                    if isinstance(data_yaml, Mapping):
                        names = _normalize_name_sequence(data_yaml.get("names"))
                        if names:
                            return names
                else:
                    logger.warning(
                        "Detection config export: dataset yaml %s not found (from args)",
                        data_path,
                    )
    return _normalize_name_sequence(fallback)


def _load_mapping_from_source(source: Any, logger: logging.Logger) -> Dict[str, Any]:
    """Load YAML or mapping based position config data."""
    if source in (None, "", {}):
        return {}
    if isinstance(source, Mapping):
        return {str(k): v for k, v in source.items()}
    try:
        path = Path(str(source))
    except Exception:
        return {}
    if not path.exists():
        logger.warning(
            "Detection config export skipped: config file not found at %s", path
        )
        return {}
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception as exc:  # pragma: no cover - defensive logging around YAML errors
        logger.error(
            "Detection config export skipped: failed to read %s (%s)", path, exc
        )
        return {}
    if isinstance(data, Mapping):
        return {str(k): v for k, v in data.items()}
    logger.warning(
        "Detection config export skipped: %s did not contain a mapping", path
    )
    return {}


def _apply_area_overrides(
    area_cfg: Mapping[str, Any],
    export_cfg: Mapping[str, Any],
    imgsz: Optional[List[int]],
    logger: logging.Logger,
    warned: Dict[str, bool],
) -> Dict[str, Any]:
    """Apply tolerance/imgsz overrides to a single area block."""
    area_dict = dict(area_cfg)
    if export_cfg.get("tolerance") is not None:
        try:
            area_dict["tolerance"] = float(export_cfg["tolerance"])
        except (TypeError, ValueError):
            if not warned.get("tolerance"):
                logger.warning(
                    "Detection config export: tolerance override %r is not numeric",
                    export_cfg.get("tolerance"),
                )
                warned["tolerance"] = True
    tol_unit = export_cfg.get("tolerance_unit")
    if tol_unit:
        area_dict["tolerance_unit"] = str(tol_unit)
    if imgsz:
        area_dict["imgsz"] = (
            imgsz[0] if len(imgsz) == 2 and imgsz[0] == imgsz[1] else imgsz
        )
    return area_dict


def _prepare_position_config(
    raw_config: Mapping[str, Any],
    export_cfg: Mapping[str, Any],
    product: Optional[Any],
    area: Optional[Any],
    imgsz: Optional[List[int]],
    logger: logging.Logger,
) -> Dict[str, Any]:
    """Filter and adjust the position config according to export settings."""
    warned: Dict[str, bool] = {}
    include_all = bool(export_cfg.get("include_all_products"))
    if not include_all and product and area:
        product_key = str(product)
        area_key = str(area)
        product_block = raw_config.get(product_key)
        if isinstance(product_block, Mapping):
            area_block = product_block.get(area_key)
            if isinstance(area_block, Mapping):
                return {
                    product_key: {
                        area_key: _apply_area_overrides(
                            area_block, export_cfg, imgsz, logger, warned
                        )
                    }
                }
            logger.warning(
                "Detection config export: area '%s' not present in position config for product '%s'",
                area_key,
                product_key,
            )
            return {}
        logger.warning(
            "Detection config export: product '%s' not present in position config",
            product_key,
        )
        return {}

    filtered: Dict[str, Any] = {}
    for prod_key, areas in raw_config.items():
        if not isinstance(areas, Mapping):
            continue
        area_map: Dict[str, Any] = {}
        for area_key, area_cfg in areas.items():
            if not isinstance(area_cfg, Mapping):
                continue
            area_map[str(area_key)] = _apply_area_overrides(
                area_cfg, export_cfg, imgsz, logger, warned
            )
        if area_map:
            filtered[str(prod_key)] = area_map
    return filtered


def _normalize_expected_items(
    value: Any,
    product: Optional[Any] = None,
    area: Optional[Any] = None,
) -> Dict[str, Dict[str, List[str]]]:
    """Ensure expected item mapping contains string keys and value lists."""
    if not isinstance(value, Mapping):
        if (
            isinstance(value, Sequence)
            and not isinstance(value, (str, bytes, bytearray, Mapping))
            and product
            and area
        ):
            return {
                str(product): {
                    str(area): [str(item) for item in value if item is not None]
                }
            }
        return {}
    normalized: Dict[str, Dict[str, List[str]]] = {}
    for prod_key, areas in value.items():
        if not isinstance(areas, Mapping):
            continue
        area_map: Dict[str, List[str]] = {}
        for area_key, items in areas.items():
            if isinstance(items, Sequence) and not isinstance(
                items, (str, bytes, bytearray)
            ):
                area_map[str(area_key)] = [str(item) for item in items]
        if area_map:
            normalized[str(prod_key)] = area_map
    return normalized


def _resolve_sample_images(directory: Path) -> List[Path]:
    if not directory.exists():
        raise FileNotFoundError(
            f"Position validation sample_dir not found: {directory}"
        )
    images = [
        p
        for p in sorted(directory.iterdir())
        if p.is_file() and p.suffix.lower() in ImageSuffixes
    ]
    if not images:
        raise FileNotFoundError(f"No images available under {directory}")
    return images


def _auto_generate_position_config(
    config: MutableMapping[str, Any],
    run_dir: Path,
    logger: logging.Logger,
) -> Optional[Path]:
    """Automatically derive a position config from latest training results."""
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
            "Auto position config generation skipped: product/area must be specified when position validation is enabled."
        )
        return None

    imgsz_value = pos_cfg.get("imgsz") or ycfg.get("imgsz") or 640
    imgsz_norm = _normalize_imgsz(imgsz_value) or [640, 640]
    imgsz_int = (
        imgsz_norm[0]
        if len(imgsz_norm) == 2 and imgsz_norm[0] == imgsz_norm[1]
        else imgsz_norm[0]
    )

    dataset_dir = Path(str(ycfg.get("dataset_dir", "./data/split")))
    sample_dir_value = pos_cfg.get("sample_dir") or (dataset_dir / "val" / "images")
    sample_dir = Path(str(sample_dir_value)).resolve()
    try:
        images = _resolve_sample_images(sample_dir)
    except Exception as exc:
        logger.warning("Auto position config generation skipped: %s", exc)
        return None

    weights_path = run_dir / "weights" / "best.pt"
    if not weights_path.exists():
        weights_path = run_dir / "weights" / "last.pt"
    if not weights_path.exists():
        logger.warning(
            "Auto position config generation skipped: unable to locate weights under %s",
            run_dir,
        )
        return None

    try:
        from picture_tool.position.yolo_position_validator import (
            convert_results_to_detections,
        )
    except Exception as exc:  # pragma: no cover
        logger.warning("Auto position config generation skipped: %s", exc)
        return None

    device_value = pos_cfg.get("device") or ycfg.get("device") or "cpu"
    conf_value = pos_cfg.get("conf")
    if conf_value is None:
        conf_value = 0.25
    try:
        model = YOLO(str(weights_path))
    except Exception as exc:  # pragma: no cover
        logger.warning(
            "Auto position config generation skipped: failed to load weights (%s)", exc
        )
        return None

    boxes_by_class: Dict[str, List[List[int]]] = {}
    for img_path in images:
        try:
            results = model(
                str(img_path),
                imgsz=imgsz_norm if len(imgsz_norm) > 1 else imgsz_norm[0],
                device=str(device_value),
                conf=float(conf_value),
            )
        except Exception as exc:  # pragma: no cover
            logger.warning(
                "Auto position config: inference failed for %s (%s)", img_path.name, exc
            )
            continue
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

    if not boxes_by_class:
        logger.warning(
            "Auto position config generation skipped: no detections collected from sample images."
        )
        return None

    def aggregate(boxes: List[List[int]]) -> Dict[str, int]:
        x1 = min(b[0] for b in boxes)
        y1 = min(b[1] for b in boxes)
        x2 = max(b[2] for b in boxes)
        y2 = max(b[3] for b in boxes)
        return {"x1": int(x1), "y1": int(y1), "x2": int(x2), "y2": int(y2)}

    expected_boxes = {cls: aggregate(bxs) for cls, bxs in boxes_by_class.items()}
    if not expected_boxes:
        logger.warning(
            "Auto position config generation skipped: expected boxes could not be computed."
        )
        return None

    tolerance_value = pos_cfg.get("tolerance_override")
    if tolerance_value is None:
        tolerance_value = (
            float(pos_cfg.get("tolerance", 0.0)) if "tolerance" in pos_cfg else 0.0
        )

    area_block: Dict[str, Any] = {
        "enabled": True,
        "mode": str(pos_cfg.get("mode", "bbox")),
        "tolerance": float(tolerance_value),
        "expected_boxes": expected_boxes,
    }
    area_block["imgsz"] = (
        imgsz_norm[0]
        if len(imgsz_norm) == 2 and imgsz_norm[0] == imgsz_norm[1]
        else imgsz_norm
    )

    position_config = {str(product): {str(area): area_block}}
    out_path = (run_dir / "auto_position_config.yaml").resolve()
    try:
        with open(out_path, "w", encoding="utf-8") as fh:
            yaml.safe_dump(position_config, fh, allow_unicode=True, sort_keys=False)
    except Exception as exc:  # pragma: no cover
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


def _maybe_export_detection_config(
    config: MutableMapping[str, Any],
    run_dir: Path,
    logger: logging.Logger,
) -> Optional[Path]:
    """Create the post-training detection config file when enabled."""
    ycfg = config.get("yolo_training")
    if not isinstance(ycfg, Mapping):
        return None
    export_cfg = ycfg.get("export_detection_config")
    if not isinstance(export_cfg, Mapping) or not export_cfg.get("enabled"):
        return None

    weights_name = str(export_cfg.get("weights_name") or "best.pt")
    weights_path = (run_dir / "weights" / weights_name).resolve()
    if not weights_path.exists():
        logger.warning(
            "Detection config export skipped: unable to find weights at %s",
            weights_path,
        )
        return None

    output_path_value = export_cfg.get("output_path")
    if output_path_value:
        output_path = Path(str(output_path_value))
    else:
        output_path = run_dir / "detection_config.yaml"
    output_path = output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    device = str(export_cfg.get("device") or ycfg.get("device") or "cpu")
    conf_thres = float(export_cfg.get("conf_thres", 0.25))
    iou_thres = float(export_cfg.get("iou_thres", 0.45))
    timeout_val = int(export_cfg.get("timeout", 1))
    enable_yolo = bool(export_cfg.get("enable_yolo", True))
    output_dir = str(export_cfg.get("output_dir", "Result"))

    imgsz = _normalize_imgsz(export_cfg.get("imgsz"))
    if imgsz is None:
        imgsz = _normalize_imgsz(ycfg.get("imgsz"))
    if imgsz is None:
        imgsz = [640, 640]

    pos_cfg = ycfg.get("position_validation")
    if not isinstance(pos_cfg, Mapping):
        pos_cfg = {}
    product = export_cfg.get("current_product") or pos_cfg.get("product")
    area = export_cfg.get("area") or pos_cfg.get("area")
    class_names = _load_class_names_from_run(
        run_dir,
        logger,
        ycfg.get("class_names"),
    )

    expected_items = _normalize_expected_items(
        export_cfg.get("expected_items"), product, area
    )

    include_position_config = bool(
        export_cfg.get("position_config") or export_cfg.get("position_config_path")
    )
    if not include_position_config and pos_cfg.get("enabled"):
        include_position_config = True

    position_config: Dict[str, Any] = {}
    if include_position_config:
        position_sources = [
            export_cfg.get("position_config"),
            export_cfg.get("position_config_path"),
            pos_cfg.get("config"),
            pos_cfg.get("config_path"),
        ]
        raw_position_config: Dict[str, Any] = {}
        for source in position_sources:
            data = _load_mapping_from_source(source, logger)
            if data:
                raw_position_config = data
                break
        if raw_position_config:
            position_config = _prepare_position_config(
                raw_position_config, export_cfg, product, area, imgsz, logger
            )
            if not position_config:
                position_config = raw_position_config
        if not expected_items and position_config:
            if product and area:
                product_key = str(product)
                area_key = str(area)
                product_block = position_config.get(product_key)
                if isinstance(product_block, Mapping):
                    area_block = product_block.get(area_key)
                    if isinstance(area_block, Mapping):
                        boxes = area_block.get("expected_boxes")
                        if isinstance(boxes, Mapping) and boxes:
                            expected_items = {
                                product_key: {
                                    area_key: [str(name) for name in boxes.keys()]
                                }
                            }
            else:
                collected: Dict[str, Dict[str, List[str]]] = {}
                for prod_key, areas in position_config.items():
                    if not isinstance(areas, Mapping):
                        continue
                    area_map: Dict[str, List[str]] = {}
                    for area_key, area_cfg in areas.items():
                        if not isinstance(area_cfg, Mapping):
                            continue
                        boxes = area_cfg.get("expected_boxes")
                        if isinstance(boxes, Mapping) and boxes:
                            area_map[str(area_key)] = [
                                str(name) for name in boxes.keys()
                            ]
                    if area_map:
                        collected[str(prod_key)] = area_map
                if collected:
                    expected_items = collected

    if (
        not expected_items
        and product
        and area
        and isinstance(class_names, Sequence)
        and not isinstance(class_names, (str, bytes, bytearray))
    ):
        expected_items = {
            str(product): {
                str(area): [str(name) for name in class_names if name is not None]
            }
        }

    payload: Dict[str, Any] = {
        "weights": str(weights_path),
        "device": device,
        "conf_thres": conf_thres,
        "iou_thres": iou_thres,
        "imgsz": imgsz,
        "timeout": timeout_val,
        "enable_yolo": enable_yolo,
        "output_dir": output_dir,
    }
    if product:
        payload["current_product"] = str(product)
    if area:
        payload["current_area"] = str(area)
    if expected_items:
        payload["expected_items"] = expected_items
    if position_config:
        payload["position_config"] = position_config

    with open(output_path, "w", encoding="utf-8") as fh:
        yaml.safe_dump(payload, fh, allow_unicode=True, sort_keys=False)
    logger.info("Detection config exported to %s", output_path)
    return output_path


def _bundle_run_artifacts(
    config: MutableMapping[str, Any],
    run_dir: Path,
    detection_config_path: Optional[Path],
    generated_position_path: Optional[Path],
    logger: logging.Logger,
) -> Optional[Path]:
    """Collect selected training artifacts into a single directory."""
    ycfg = config.get("yolo_training")
    if not isinstance(ycfg, Mapping):
        return None
    bundle_cfg = ycfg.get("artifact_bundle")
    if not isinstance(bundle_cfg, Mapping) or not bundle_cfg.get("enabled"):
        return None

    dir_name = str(bundle_cfg.get("dir_name") or "bundle")
    base_dir_value = bundle_cfg.get("base_dir")
    if base_dir_value:
        bundle_dir = Path(str(base_dir_value))
        if not bundle_dir.is_absolute():
            bundle_dir = (run_dir / bundle_dir).resolve()
    else:
        bundle_dir = (run_dir / dir_name).resolve()
    if not bundle_dir.is_absolute():
        bundle_dir = (run_dir / bundle_dir).resolve()
    bundle_dir.mkdir(parents=True, exist_ok=True)

    def copy_into_bundle(path: Path) -> None:
        path = path.resolve()
        if not path.exists():
            return
        destination = (bundle_dir / path.name).resolve()
        if destination == path:
            return
        try:
            shutil.copy2(path, destination)
            logger.info("Bundled artifact: %s -> %s", path, destination)
        except Exception as exc:  # pragma: no cover
            logger.warning("Failed to bundle %s: %s", path, exc)

    if bundle_cfg.get("include_detection_config", True):
        if detection_config_path:
            copy_into_bundle(Path(detection_config_path))
        else:
            default_detection = run_dir / "detection_config.yaml"
            if default_detection.exists():
                copy_into_bundle(default_detection)

    if bundle_cfg.get("include_position_config", True):
        candidate: Optional[Path] = None
        if generated_position_path:
            candidate = Path(generated_position_path)
        else:
            pos_cfg = ycfg.get("position_validation")
            if isinstance(pos_cfg, Mapping):
                pc_path = pos_cfg.get("config_path")
                if pc_path:
                    candidate = Path(str(pc_path))
        if candidate and candidate.exists():
            copy_into_bundle(candidate)

    if bundle_cfg.get("include_weights", True):
        weights_dir = run_dir / "weights"
        for name in ("best.pt", "last.pt"):
            candidate = weights_dir / name
            if candidate.exists():
                copy_into_bundle(candidate)

    if bundle_cfg.get("include_results_csv", True):
        results_csv = run_dir / "results.csv"
        if results_csv.exists():
            copy_into_bundle(results_csv)

    if bundle_cfg.get("include_args_yaml", True):
        args_yaml = run_dir / "args.yaml"
        if args_yaml.exists():
            copy_into_bundle(args_yaml)

    return bundle_dir


def train_yolo(config: dict, logger: Optional[logging.Logger] = None) -> Path:
    """Train YOLO using Ultralytics and return run directory.

    Expects config['yolo_training'] block with keys:
      - dataset_dir, class_names, model, epochs, imgsz, batch, device, project, name
    """
    logger = logger or logging.getLogger(__name__)
    ycfg = config.get("yolo_training", {})

    dataset_dir = Path(ycfg.get("dataset_dir", "./datasets/split_dataset")).resolve()
    names = ycfg.get("class_names") or []
    if not isinstance(names, list) or not names:
        raise ValueError("yolo_training.class_names must be a non-empty list")

    # Prepare data.yaml
    data_yaml = _ensure_data_yaml(dataset_dir, names)
    logger.info(f"Prepared data.yaml at: {data_yaml}")

    # Resolve model weights
    model_cfg = ycfg.get("model", "yolo11n.pt")
    model_path = Path(str(model_cfg))
    if model_path.exists():
        model_arg = str(model_path)
    else:
        # allow using model name directly (e.g., 'yolo11n.pt')
        model_arg = str(model_cfg)

    epochs = int(ycfg.get("epochs", 100))
    imgsz = int(ycfg.get("imgsz", 640))
    batch = int(ycfg.get("batch", 16))
    device = str(ycfg.get("device", "cpu"))
    project = str(ycfg.get("project", "./runs/detect"))
    name = str(ycfg.get("name", "train"))

    if YOLO is None:
        raise RuntimeError("ultralytics is not available. Please install ultralytics.")

    logger.info(
        f"Starting YOLO training | model={model_arg} epochs={epochs} imgsz={imgsz} batch={batch} device={device}"
    )
    model = YOLO(model_arg)
    train_results = model.train(
        data=str(data_yaml),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        project=project,
        name=name,
    )
    run_dir: Optional[Path] = None
    if hasattr(train_results, "save_dir"):
        candidate = getattr(train_results, "save_dir")
        if candidate:
            run_dir = Path(str(candidate)).resolve()
    trainer = getattr(model, "trainer", None)
    if run_dir is None and trainer is not None and getattr(trainer, "save_dir", None):
        run_dir = Path(str(trainer.save_dir)).resolve()
    if run_dir is None:
        run_dir = (Path(project) / name).resolve()
    logger.info(f"Training completed. Run directory: {run_dir}")

    generated_cfg = _auto_generate_position_config(config, run_dir, logger)
    if generated_cfg:
        logger.info("Position config generated automatically: %s", generated_cfg)

    position_cfg = ycfg.get("position_validation", {})
    if isinstance(position_cfg, Mapping) and position_cfg.get("enabled"):
        try:
            from picture_tool.position.yolo_position_validator import (
                run_position_validation,
            )

            run_position_validation(config, run_dir, logger=logger)
        except Exception as exc:
            logger.error(f"Position validation failed: {exc}")

    detection_cfg_path = _maybe_export_detection_config(config, run_dir, logger=logger)
    _bundle_run_artifacts(
        config,
        run_dir,
        detection_cfg_path,
        generated_cfg,
        logger,
    )
    return run_dir

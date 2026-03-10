import logging
import yaml
from pathlib import Path
from typing import Any, Dict, Optional, MutableMapping, Mapping, List, Sequence
from picture_tool.utils.normalization import normalize_imgsz, normalize_name_sequence


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
        except (FileNotFoundError, yaml.YAMLError, UnicodeDecodeError, OSError) as exc:
            logger.warning(
                "Detection config export: failed to read %s (%s)", args_path, exc
            )
            args_data = {}
        if isinstance(args_data, Mapping):
            names = normalize_name_sequence(args_data.get("names"))
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
                    except (FileNotFoundError, yaml.YAMLError, UnicodeDecodeError, OSError) as exc:
                        logger.warning(
                            "Detection config export: failed to read dataset yaml %s (%s)",
                            data_path,
                            exc,
                        )
                        data_yaml = {}
                    if isinstance(data_yaml, Mapping):
                        names = normalize_name_sequence(data_yaml.get("names"))
                        if names:
                            return names
                else:
                    logger.warning(
                        "Detection config export: dataset yaml %s not found (from args)",
                        data_path,
                    )
    return normalize_name_sequence(fallback)


def _load_mapping_from_source(source: Any, logger: logging.Logger) -> Dict[str, Any]:
    if source in (None, "", {}):
        return {}
    if isinstance(source, Mapping):
        return {str(k): v for k, v in source.items()}
    try:
        path = Path(str(source))
    except (TypeError, ValueError, OSError):
        return {}
    if not path.exists():
        logger.warning(
            "Detection config export skipped: config file not found at %s", path
        )
        return {}
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except (FileNotFoundError, yaml.YAMLError, UnicodeDecodeError, OSError) as exc:
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


class DetectionConfigExporter:
    @staticmethod
    def export(
        config: MutableMapping[str, Any],
        run_dir: Path,
        logger: logging.Logger,
        include_position: bool,
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

        imgsz = normalize_imgsz(export_cfg.get("imgsz"))
        if imgsz is None:
            imgsz = normalize_imgsz(ycfg.get("imgsz"))
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

        explicit_position = bool(
            export_cfg.get("position_config") or export_cfg.get("position_config_path")
        )
        include_position_config = explicit_position or (
            include_position and pos_cfg.get("enabled")
        )

        position_config: Dict[str, Any] = {}
        # 1. First, try to get expected_items directly from export_cfg
        expected_items: Optional[Dict[str, Dict[str, List[str]]]] = export_cfg.get("expected_items")

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

            # Extract expected items from position config if available
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
            "weights": weights_name,
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

        # Inject extra custom metadata/configurations defined in export_detection_config
        # (e.g., pipeline, steps, enable_color_check, color_model_path, model_version)
        reserved_keys = {
            "enabled", "output_path", "weights_name", "device", "conf_thres",
            "iou_thres", "timeout", "enable_yolo", "output_dir", "imgsz",
            "current_product", "area", "position_config", "position_config_path",
            "include_all_products", "tolerance", "tolerance_unit"
        }
        for k, v in export_cfg.items():
            if k not in reserved_keys and k not in payload:
                payload[k] = v

        # 始終包含 position_config
        if include_position_config and position_config:
            payload["position_config"] = position_config
        elif include_position_config and include_position:
            # 提供預設範本
            payload["position_config"] = {
                "# NOTE": "Position validation disabled. Default template for reference.",
                "enabled": False,
                "ProductA": {
                    "Area1": {
                        "tolerance": 10,
                        "expected_boxes": {
                            "ClassName1": {"x1": 100, "y1": 100, "x2": 200, "y2": 200}
                        },
                    }
                },
            }
            logger.info("Position validation not configured, added default template")

        with open(output_path, "w", encoding="utf-8") as fh:
            yaml.safe_dump(payload, fh, allow_unicode=True, sort_keys=False)
        logger.info("Detection config exported to %s", output_path)
        return output_path

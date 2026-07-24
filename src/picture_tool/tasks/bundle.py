import logging
import zipfile
from pathlib import Path
from typing import List, Any, Mapping
from picture_tool.pipeline.core import Task
from picture_tool.pipeline.utils import detect_existing_weights
from picture_tool.tasks.deployment_target import resolve_yolo_deployment_target


PLACEHOLDER_TARGET_NAMES = {"", "project", "train", "default"}


def validate_deployment_target(product: str, area: str) -> None:
    """Validate product/area names before writing inference model artifacts.

    Args:
        product: Product directory name under ``yolo11_inference/models``.
        area: Area directory name under the product.

    Raises:
        ValueError: If either value is empty or still uses a template default.
    """
    if product.strip().lower() in PLACEHOLDER_TARGET_NAMES:
        raise ValueError(
            "Deployment product is not specific enough. Set "
            "yolo_training.deploy.product, artifact_bundle.product, or pass "
            "--product like PCBA1:A."
        )
    if area.strip().lower() in PLACEHOLDER_TARGET_NAMES:
        raise ValueError(
            "Deployment area is not specific enough. Set yolo_training.deploy.area, "
            "artifact_bundle.area, or pass --product like PCBA1:A."
        )


def _rewrite_nested_product_area_keys(
    value: Any,
    product: str,
    area: str | None,
) -> Any:
    """Return a copy of a product/area mapping keyed by the deploy target."""
    if not isinstance(value, Mapping):
        return value

    result = {str(k): v for k, v in value.items()}
    if product not in result and result:
        first_product = next(
            (key for key, item in result.items() if isinstance(item, Mapping)),
            None,
        )
        if first_product is None:
            return result
        result[product] = result.pop(first_product)

    if area and isinstance(result.get(product), Mapping):
        area_map = {str(k): v for k, v in result[product].items()}
        if area not in area_map and area_map:
            first_area = next(iter(area_map))
            area_map[area] = area_map.pop(first_area)
        if area in area_map:
            area_map = {area: area_map[area]}
        result[product] = area_map

    return result


def rewrite_detection_config(
    det_cfg_data: dict,
    product: str,
    area: str | None = None,
) -> dict:
    """Return a copy of *det_cfg_data* with paths and keys rewritten for deployment.

    Shared by both :func:`run_artifact_bundle` and the deploy task so the
    two output formats stay in sync automatically.
    """
    data = dict(det_cfg_data)

    # 1. Fix weights path: ensure it lives under 'weights/' sub-folder
    old_weights = data.get("weights", "")
    if old_weights:
        data["weights"] = f"weights/{Path(old_weights).name}"

    # 2. Fix color_model_path: just the filename (sits next to config.yaml)
    color_model = data.get("color_model_path", "")
    if color_model:
        data["color_model_path"] = Path(color_model).name

    # 3. Replace generic product/area keys in expected_items with the real target.
    exp_items = data.get("expected_items", {})
    if exp_items:
        data["expected_items"] = _rewrite_nested_product_area_keys(
            exp_items, product, area
        )
    data["current_product"] = product
    if area:
        data["current_area"] = area

    # 4. Replace generic product/area keys in position_config.
    pos_cfg = data.get("position_config", {})
    if pos_cfg:
        data["position_config"] = _rewrite_nested_product_area_keys(
            pos_cfg, product, area
        )

    return data


def select_runtime_weight(run_dir: Path, configured_weight: str | None) -> Path:
    """Choose the runtime weight file for an inference-ready bundle.

    Args:
        run_dir: Ultralytics run directory containing ``weights``.
        configured_weight: Weight name referenced by ``detection_config.yaml``.

    Returns:
        Existing weight path, preferring the contracted runtime export, then
        the configured file and ONNX.

    Raises:
        FileNotFoundError: If no supported runtime weight exists.
    """
    weights_dir = run_dir / "weights"
    candidate_names: list[str] = []
    contracted_runtime = _contracted_runtime_name(run_dir)
    if contracted_runtime:
        candidate_names.append(contracted_runtime)
    if configured_weight:
        candidate_names.append(Path(configured_weight).name)
    candidate_names.extend(["best.onnx", "best.pt", "last.pt"])

    seen: set[str] = set()
    for name in candidate_names:
        if not name or name in seen:
            continue
        seen.add(name)
        candidate = weights_dir / name
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        f"No supported runtime weight found in {weights_dir} "
        "(expected configured weight, best.onnx, best.pt, or last.pt)."
    )


def _contracted_runtime_name(run_dir: Path) -> str | None:
    """Return a safe runtime filename from the export-lineage contract."""
    contract_path = run_dir / "runtime_export_manifest.json"
    if not contract_path.is_file():
        return None
    try:
        import json

        payload = json.loads(contract_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        return None
    runtime_value = str(payload.get("runtime_file") or "").strip()
    if not runtime_value:
        return None
    runtime_path = (run_dir / runtime_value).resolve()
    weights_dir = (run_dir / "weights").resolve()
    if runtime_path.parent != weights_dir or not runtime_path.is_file():
        return None
    return runtime_path.name


def find_color_model_source(run_dir: Path, color_model_name: str) -> Path | None:
    """Find the color model/stat file that should be deployed.

    Args:
        run_dir: Ultralytics run directory.
        color_model_name: Filename referenced by ``color_model_path``.

    Returns:
        Existing color model path, or ``None`` when no candidate exists.
    """
    if not color_model_name:
        return None

    requested = Path(color_model_name).name
    exact_candidates = [
        run_dir.parent / "quality" / "color" / requested,
        run_dir / requested,
    ]
    for candidate in exact_candidates:
        if candidate.exists():
            return candidate

    if requested not in {"color_stats.json", "stats.json"}:
        return None

    candidates = [
        run_dir.parent / "quality" / "color" / "stats.json",
        run_dir / "color_stats.json",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def run_artifact_bundle(config, args):
    """Bundle training artifacts into a zip that can be extracted directly
    into ``yolo11_inference/models/`` and be discovered by ModelManager.

    Zip layout mirrors the inference directory convention::

        {product}/{area}/yolo/
        ├── config.yaml
        ├── weights/best.pt
        ├── weights/last.pt       (optional)
        ├── weights/best.onnx     (optional)
        ├── stats.json            (optional)
        ├── results.csv           (optional)
        └── args.yaml             (optional)
    """
    import yaml

    logger = logging.getLogger(__name__)
    ycfg = config.get("yolo_training", {})
    bcfg = ycfg.get("artifact_bundle", {})
    if not bcfg.get("enabled", False):
        logger.info("Artifact bundle disabled.")
        return

    # Determine source run directory
    _, run_dir = detect_existing_weights(config)
    if not run_dir or not run_dir.exists():
        raise FileNotFoundError(
            "No YOLO training run found for artifact_bundle. Run yolo_train first "
            "or configure yolo_evaluation.weights / yolo_training.position_validation.weights."
        )

    target = resolve_yolo_deployment_target(
        config, args, artifact_config_key="artifact_bundle"
    )
    product = target.product
    area = target.area
    validate_deployment_target(product, area)

    # Prefix for all paths inside the zip — mirrors inference models/ layout
    zip_prefix = f"{product}/{area}/yolo"

    out_dir = Path(bcfg.get("base_dir") or run_dir)
    dir_name = bcfg.get("dir_name", "bundle")
    zip_name = f"{product}_{dir_name}.zip"
    zip_path = out_dir / zip_name

    logger.info(f"Bundling deployment-ready artifacts from {run_dir} into {zip_path}...")

    # Load detection config to rewrite it — abort if missing, as the resulting
    # ZIP would be unusable (no config.yaml = ModelCatalog cannot load it).
    det_cfg_path = run_dir / "detection_config.yaml"
    if not det_cfg_path.exists():
        raise FileNotFoundError(
            "detection_config.yaml is required to build an inference package. "
            f"Expected: {det_cfg_path}. Run yolo_train or export detection config first."
        )

    try:
        with open(det_cfg_path, "r", encoding="utf-8") as f:
            det_cfg_data = yaml.safe_load(f) or {}
    except (OSError, yaml.YAMLError) as exc:
        raise RuntimeError(f"Failed to read detection_config.yaml: {exc}") from exc

    det_cfg_data = rewrite_detection_config(det_cfg_data, str(product), str(area))
    selected_weight = select_runtime_weight(
        run_dir, str(det_cfg_data.get("weights") or "")
    )
    det_cfg_data["weights"] = selected_weight.name

    # Override weights / color_model_path to use full inference-relative paths
    # so ModelManager can resolve them from the project root.
    old_weights = det_cfg_data.get("weights", "")
    if old_weights:
        weights_filename = Path(old_weights).name
        det_cfg_data["weights"] = (
            f"models/{product}/{area}/yolo/weights/{weights_filename}"
        )

    color_model = det_cfg_data.get("color_model_path", "")
    color_source: Path | None = None
    if color_model:
        color_filename = Path(color_model).name
        det_cfg_data["color_model_path"] = (
            f"models/{product}/{area}/yolo/{color_filename}"
        )
        color_source = find_color_model_source(run_dir, color_filename)
        if det_cfg_data.get("enable_color_check") and color_source is None:
            raise FileNotFoundError(
                "enable_color_check is true but no color model/stat file was found "
                f"for {color_filename}."
            )

    # Prepare files to copy verbatim
    files_to_zip: list[tuple[Path, str]] = []

    # Optional explicitly included files
    inclusion_map = {
        "include_results_csv": [run_dir / "results.csv"],
        "include_args_yaml": [run_dir / "args.yaml"],
    }

    for key, candidates in inclusion_map.items():
        if bcfg.get(key, False):
            for cand in candidates:
                if cand.exists():
                    files_to_zip.append((cand, f"{zip_prefix}/{cand.name}"))

    # Build ZIP
    try:
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            # Write rewritten config.yaml
            config_yaml_str = yaml.safe_dump(det_cfg_data, allow_unicode=True, sort_keys=False)
            zf.writestr(f"{zip_prefix}/config.yaml", config_yaml_str)

            # Write weights
            if bcfg.get("include_weights", True):
                weight_cands = [
                    selected_weight.name,
                    "best.onnx",
                    "best.pt",
                    "last.pt",
                ]
                written_weights: set[str] = set()
                for wc in weight_cands:
                    if wc in written_weights:
                        continue
                    wp = run_dir / "weights" / wc
                    if wp.exists():
                        zf.write(wp, arcname=f"{zip_prefix}/weights/{wc}")
                        written_weights.add(wc)

            # Write color stats — use the filename referenced in config so
            # ModelManager can find it.  Search same locations as deploy task.
            color_cfg_name = Path(det_cfg_data.get("color_model_path", "")).name
            if color_source:
                arcname = (
                    f"{zip_prefix}/{color_cfg_name}"
                    if color_cfg_name
                    else f"{zip_prefix}/{color_source.name}"
                )
                zf.write(color_source, arcname=arcname)

            # Write verbatim files
            for src, arcname in files_to_zip:
                zf.write(src, arcname=arcname)

        logger.info(
            "Deployment Bundle created: %s\n"
            "  → 解壓到 yolo11_inference/models/ 即可直接使用:\n"
            "    unzip %s -d /path/to/yolo11_inference/models/",
            zip_path, zip_path.name,
        )
    except (FileNotFoundError, PermissionError, OSError) as e:
        raise RuntimeError(f"Failed to create artifact bundle: {e}") from e


TASKS: List[Any] = [
    Task(
        name="artifact_bundle",
        run=run_artifact_bundle,
        description="Bundle training artifacts (Zip).",
        dependencies=[],
    ),
]

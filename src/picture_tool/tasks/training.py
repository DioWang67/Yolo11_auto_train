import logging
import json
import re
from pathlib import Path
from typing import Any
from picture_tool.train.anomalib_trainer import (
    find_existing_anomalib_run,
    package_anomalib_run,
    train_anomalib,
)
from picture_tool.train.yolo_trainer import train_yolo
from picture_tool.eval.yolo_evaluator import evaluate_yolo
from picture_tool.position import load_position_config, run_position_validation
from picture_tool.position.position_gate import (
    PositionGatePolicy,
    evaluate_position_gate,
    load_json_mapping,
    write_position_gate_report,
)
from picture_tool.path_resolver import parse_project_area_override
from picture_tool.pipeline.artifacts import find_anomalib_run_artifact
from picture_tool.pipeline.utils import detect_existing_weights, mtime_latest
from picture_tool.pipeline.core import Task
from picture_tool.utils.onnx_exporter import OnnxExporter
from picture_tool.position.position_config_gen import PositionConfigGenerator
from picture_tool.utils.detection_config import DetectionConfigExporter
from picture_tool.utils.hashing import compute_dir_hash, compute_config_hash
from picture_tool.quality.dataset_readiness import validate_training_dataset
from picture_tool.constants import DEFAULT_RUNS_DIR, DEFAULT_SPLITS_DIR


def run_yolo_train(config, args):
    logger = logging.getLogger(__name__)
    validate_training_dataset(config, logger=logger)
    run_dir = train_yolo(config, args=args, logger=logger)

    # Post-training steps
    # 1. Position Config Generation
    try:
        generated_position_config = PositionConfigGenerator.generate(
            config,
            run_dir,
            logger,
        )
        position_cfg = (
            (config.get("yolo_training", {}) or {}).get(
                "position_validation",
                {},
            )
            or {}
        )
        selected_tasks = {
            str(task) for task in (getattr(args, "tasks", []) or [])
        }
        position_required = bool(
            position_cfg.get("enabled")
            and position_cfg.get("auto_generate", True)
            and (not selected_tasks or "position_validation" in selected_tasks)
        )
        if position_required and generated_position_config is None:
            raise RuntimeError(
                "Position validation is enabled, but no position configuration "
                "was generated."
            )
    except (FileNotFoundError, ValueError, RuntimeError, OSError) as exc:
        logger.error("Position config generation failed: %s", exc)
        raise RuntimeError(
            f"Pipeline aborted: Position config generation failed - {exc}"
        ) from exc

    # 2. ONNX Export
    try:
        exported_runtime = OnnxExporter.ensure(config, run_dir, logger)
        yolo_cfg = config.get("yolo_training", {}) or {}
        runtime_cfg = yolo_cfg.get("export_runtime", {}) or {}
        onnx_cfg = yolo_cfg.get("export_onnx", {}) or {}
        export_required = bool(
            runtime_cfg.get("enabled", False)
            or onnx_cfg.get("enabled", False)
            or OnnxExporter.is_enabled(config)
        )
        if export_required and exported_runtime is None:
            raise RuntimeError(
                "Configured YOLO runtime export did not produce a validated artifact."
            )
    except (ImportError, FileNotFoundError, RuntimeError, OSError) as exc:
        logger.error("ONNX export failed: %s", exc)
        raise RuntimeError(
            f"Pipeline aborted: ONNX export failed - {exc}"
        ) from exc

    # 3. Detection Config Export
    ycfg = config.get("yolo_training", {})
    pos_cfg = ycfg.get("position_validation", {})
    position_requested = (
        not selected_tasks or "position_validation" in selected_tasks
    )
    position_validation_active = bool(
        isinstance(pos_cfg, dict) and pos_cfg.get("enabled") and position_requested
    )

    try:
        DetectionConfigExporter.export(
            config, run_dir, logger, include_position=position_validation_active
        )
    except (FileNotFoundError, ValueError, OSError) as exc:
        logger.error("Detection config export failed: %s", exc)
        raise RuntimeError(
            f"Pipeline aborted: Detection config export failed - {exc}"
        ) from exc


def run_anomalib_train(config, args):
    """Run Anomalib training from ``anomalib_training`` config."""
    train_anomalib(config, logger=logging.getLogger(__name__))


def run_anomalib_package(config, args):
    """Package a trained Anomalib run into an inference-ready zip."""
    logger = logging.getLogger(__name__)
    package_cfg = config.get("anomalib_package", {}) or {}
    product, area = _resolve_anomalib_package_product_area(config, args)
    run_dir = _resolve_anomalib_package_run_dir(
        config, args, product=product, area=area
    )

    result = package_anomalib_run(
        run_dir,
        output_dir=Path(str(package_cfg.get("output_dir", "runs/anomalib_packages"))),
        product=product,
        area=area,
        threshold=float(package_cfg.get("threshold", 0.5)),
        force=bool(package_cfg.get("force", False) or getattr(args, "force", False)),
    )
    logger.info("Anomalib inference package written: %s", result.zip_path)
    if result.baseline_only:
        logger.warning(
            "Anomalib package is baseline-only; threshold is not deployment-grade."
        )


def _resolve_anomalib_package_product_area(
    config: dict[str, Any], args
) -> tuple[str, str]:
    package_cfg = config.get("anomalib_package", {}) or {}
    train_cfg = config.get("anomalib_training", {}) or {}
    override = getattr(args, "product", None)
    if override:
        parsed = parse_project_area_override(str(override))
        product = parsed.project
        area = parsed.area or str(
            package_cfg.get("area") or train_cfg.get("area") or ""
        )
    else:
        product = str(
            package_cfg.get("product")
            or train_cfg.get("product")
            or _product_from_run_name(str(train_cfg.get("name", "")))
        )
        area = str(
            package_cfg.get("area")
            or train_cfg.get("area")
            or _area_from_run_name(str(train_cfg.get("name", "")))
        )

    if not product or product.lower() in {"none", "project"}:
        raise ValueError(
            "Set GUI Product to '<product>,<area>' or configure anomalib_package.product."
        )
    if not area or area.lower() in {"none", "area"}:
        raise ValueError(
            "Set GUI Product to '<product>,<area>' or configure anomalib_package.area."
        )
    return product, area


def _resolve_anomalib_package_run_dir(
    config: dict[str, Any],
    args,
    *,
    product: str,
    area: str,
) -> Path:
    package_cfg = config.get("anomalib_package", {}) or {}
    override = getattr(args, "product", None)
    run_dir_value = package_cfg.get("run_dir")
    if run_dir_value and not override:
        return Path(str(run_dir_value))

    search_roots = _anomalib_package_search_roots(config, package_cfg, product, area)
    run_dir = _find_latest_anomalib_run_for_target(search_roots)
    if run_dir is not None:
        return run_dir

    if run_dir_value:
        return Path(str(run_dir_value))

    existing = find_existing_anomalib_run(config)
    if existing is not None:
        return existing

    searched = ", ".join(str(path) for path in search_roots)
    raise FileNotFoundError(
        "No Anomalib checkpoint found for packaging. Checked: "
        f"{searched}. Run anomalib_train first or set anomalib_package.run_dir."
    )


def _anomalib_package_search_roots(
    config: dict[str, Any],
    package_cfg: dict[str, Any],
    product: str,
    area: str,
) -> list[Path]:
    train_cfg = config.get("anomalib_training", {}) or {}
    roots: list[Path] = []
    explicit_search_root = package_cfg.get("search_root")
    if explicit_search_root:
        roots.append(Path(str(explicit_search_root)) / product / area)

    output_project = train_cfg.get("project")
    if output_project:
        project_path = Path(str(output_project))
        roots.extend([project_path / product / area, project_path])

    roots.append(Path("runs") / "anomalib" / product / area)

    unique: list[Path] = []
    seen: set[Path] = set()
    for root in roots:
        normalized = root
        if normalized not in seen:
            seen.add(normalized)
            unique.append(normalized)
    return unique


def _find_latest_anomalib_run_for_target(search_roots: list[Path]) -> Path | None:
    artifact = find_anomalib_run_artifact(search_roots)
    return artifact.run_dir if artifact else None


def skip_anomalib_train(config, args):
    """Skip Anomalib training when a checkpoint already exists."""
    if getattr(args, "force", False):
        return None
    try:
        run_dir = find_existing_anomalib_run(config)
    except (KeyError, TypeError, ValueError):
        return None
    if run_dir is not None:
        return f"Found existing Anomalib checkpoint in {run_dir}; skipping training."
    return None


def _product_from_run_name(name: str) -> str:
    parts = [part for part in name.split("_") if part]
    return parts[0] if parts else ""


def _area_from_run_name(name: str) -> str:
    parts = [part for part in name.split("_") if part]
    return parts[1] if len(parts) > 1 else ""


def run_yolo_evaluation(config, args):
    weights_path, run_dir = detect_existing_weights(config, prefer=None)
    if weights_path:
        ecfg = config.setdefault("yolo_evaluation", {})
        ecfg["weights"] = str(weights_path)
        logging.getLogger(__name__).info(
            "Using existing weights for evaluation: %s (run_dir=%s)",
            weights_path,
            run_dir,
        )
    else:
        logging.getLogger(__name__).warning(
            "No existing weights detected before evaluation; will rely on default resolution."
        )
    evaluate_yolo(config)


def run_position_validation_task(config, args):
    """Run offline position validation using trained weights and sample images."""
    ycfg = config.get("yolo_training", {}) if isinstance(config, dict) else {}
    configured_position = ycfg.get("position_validation", {}) or {}
    if not configured_position.get("enabled", False):
        logging.getLogger(__name__).info(
            "Position validation skipped; the existing disabled station "
            "position contract will be preserved."
        )
        return None
    run_root = Path(str(ycfg.get("project", DEFAULT_RUNS_DIR / "project")))
    run_name = str(ycfg.get("name", "train"))
    default_run_dir = run_root / run_name
    weights_path, detected_run_dir = detect_existing_weights(config, prefer="position")
    run_dir = detected_run_dir or default_run_dir

    if weights_path:
        pv_cfg = ycfg.get("position_validation", {}) or {}
        pv_cfg["weights"] = str(weights_path)
        ycfg["position_validation"] = pv_cfg
        config["yolo_training"] = ycfg
        logging.getLogger(__name__).info(
            "Using existing weights for position validation: %s (run_dir=%s)",
            weights_path,
            run_dir,
        )

    # Fallback: If no usable config is provided, check for auto-generated one in run_dir
    pv_cfg = ycfg.get("position_validation", {})

    logging.getLogger(__name__).debug(
        "Checking position config: keys=%s, run_dir=%s, exists=%s",
        list(pv_cfg.keys()),
        run_dir,
        run_dir.exists(),
    )

    auto_conf = run_dir / "auto_position_config.yaml"
    configured_source = pv_cfg.get("config") or pv_cfg.get("config_path")
    product = str(pv_cfg.get("product") or "")
    area = str(pv_cfg.get("area") or "")
    should_use_auto_config = False

    if not configured_source:
        should_use_auto_config = True
    elif auto_conf.exists() and product and area:
        should_use_auto_config = not _position_config_has_target(
            configured_source, product, area
        )

    if should_use_auto_config:
        if auto_conf.exists():
            pv_cfg["config"] = str(auto_conf)
            ycfg["position_validation"] = pv_cfg
            config["yolo_training"] = ycfg
            logging.getLogger(__name__).info(
                "Using auto-generated position config for %s/%s: %s",
                product or "<unknown>",
                area or "<unknown>",
                auto_conf,
            )
        else:
            raise FileNotFoundError(
                "position_validation is enabled but no usable position config "
                "was found for "
                f"{product or '<unknown>'}/{area or '<unknown>'} and no "
                f"auto_position_config.yaml exists in {run_dir}."
            )
    else:
        logging.getLogger(__name__).debug(
            "Position config source selected: config=%s, config_path=%s",
            pv_cfg.get("config"),
            pv_cfg.get("config_path"),
        )

    if not run_dir.exists():
        raise FileNotFoundError(
            "No trained run found for position_validation. "
            f"Checked {run_dir} (project={run_root}, name prefix={run_name}). "
            "Provide yolo_training.position_validation.weights or run yolo_train manually."
        )

    report_path = run_position_validation(
        config,
        run_dir,
        logger=logging.getLogger(__name__),
    )
    if report_path is None:
        raise RuntimeError(
            "Position validation is enabled but did not produce a report."
        )
    _run_position_gate(config, run_dir, report_path)
    return report_path


def _run_position_gate(
    config: dict[str, Any],
    run_dir: Path,
    report_path: Path,
) -> Path | None:
    ycfg = config.get("yolo_training", {}) or {}
    position_cfg = ycfg.get("position_validation", {}) or {}
    gate_cfg = position_cfg.get("gate", {}) or {}
    if not isinstance(gate_cfg, dict) or not gate_cfg.get("enabled", False):
        return None

    candidate_report = load_json_mapping(
        report_path,
        "position validation report",
    )
    baseline_path_value = gate_cfg.get("baseline_report_path")
    baseline_path = (
        Path(str(baseline_path_value)).resolve()
        if baseline_path_value
        else None
    )
    baseline_report = (
        load_json_mapping(baseline_path, "baseline position report")
        if baseline_path is not None
        else None
    )
    calibration_path_value = position_cfg.get("calibration_manifest_path")
    calibration_path = (
        Path(str(calibration_path_value)).resolve()
        if calibration_path_value
        else None
    )
    calibration_manifest = (
        load_json_mapping(calibration_path, "position calibration manifest")
        if calibration_path is not None
        else None
    )
    policy = PositionGatePolicy.from_mapping(gate_cfg)
    decision = evaluate_position_gate(
        candidate_report,
        policy=policy,
        baseline_report=baseline_report,
        calibration_manifest=calibration_manifest,
    )
    gate_path = (run_dir / "position_gate.json").resolve()
    write_position_gate_report(
        gate_path,
        decision,
        product=str(position_cfg.get("product") or ""),
        area=str(position_cfg.get("area") or ""),
        candidate_report_path=report_path,
        baseline_report_path=baseline_path,
        calibration_manifest_path=calibration_path,
    )
    if not decision.passed:
        raise RuntimeError(
            "Position deployment gate failed: "
            + "; ".join(decision.failures)
        )
    return gate_path


def _position_config_has_target(source, product: str, area: str) -> bool:
    """Return whether a position config source contains product/area."""
    try:
        position_config = load_position_config(source)
    except (FileNotFoundError, OSError, TypeError, ValueError) as exc:
        logging.getLogger(__name__).warning(
            "Unable to read configured position config %r: %s", source, exc
        )
        return False
    product_cfg = position_config.get(product)
    return bool(product_cfg and product_cfg.get(area))


def _find_latest_run_dir(project: Path, name: str) -> Path | None:
    """Return the most recently modified run directory matching the Ultralytics
    versioning pattern: ``name``, ``name2``, ``name3``, etc."""
    if not project.exists():
        return None
    pattern = re.compile(r"^" + re.escape(name) + r"\d*$")
    candidates = [p for p in project.iterdir() if p.is_dir() and pattern.match(p.name)]
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def skip_yolo_train(config, args):
    if getattr(args, "force", False):
        return None
    y = config.get("yolo_training", {})
    completed_job_checkpoint = Path(
        str(y.get("completed_job_checkpoint") or "")
    )
    if (
        str(y.get("completed_job_checkpoint") or "").strip()
        and completed_job_checkpoint.suffix.lower() == ".pt"
        and completed_job_checkpoint.is_file()
    ):
        return (
            "Operator job training already completed successfully; "
            f"reusing {completed_job_checkpoint}."
        )

    dataset_dir = Path(
        str(
            (config.get("yolo_training", {}) or {}).get(
                "dataset_dir", DEFAULT_SPLITS_DIR
            )
        )
    )
    split_is_populated = all(
        (dataset_dir / split / "images").exists() for split in ("train", "val")
    )
    if split_is_populated:
        validate_training_dataset(config, logger=logging.getLogger(__name__))

    project = Path(str(y.get("project", DEFAULT_RUNS_DIR / "detect")))
    name = str(y.get("name", "train"))

    run_dir = _find_latest_run_dir(project, name)
    if run_dir is None:
        return None

    metadata_path = run_dir / "last_run_metadata.json"
    if not metadata_path.exists():
        return None

    # Check hashes
    try:
        with open(metadata_path, "r") as f:
            stored_meta = json.load(f)

        dataset_dir = Path(str(y.get("dataset_dir", DEFAULT_SPLITS_DIR))).resolve()
        current_data_hash = compute_dir_hash(dataset_dir)
        current_cfg_hash = compute_config_hash(y)

        if (
            stored_meta.get("dataset_hash") == current_data_hash
            and stored_meta.get("config_hash") == current_cfg_hash
            and (run_dir / "weights" / "best.pt").exists()
        ):
            return f"Skipping training: dataset and config match last run ({run_dir.name})."

    except (FileNotFoundError, json.JSONDecodeError, KeyError, ValueError, OSError):
        # Fallback to mtime if hash check fails or file corrupt
        pass

    # Legacy mtime fallback
    weights = run_dir / "weights" / "best.pt"
    dataset_dir = Path(str(y.get("dataset_dir", DEFAULT_SPLITS_DIR)))
    auto_conf = run_dir / "auto_position_config.yaml"
    if weights.exists() and auto_conf.exists():
        if weights.stat().st_mtime >= mtime_latest([dataset_dir]):
            return f"Found latest best.pt in {run_dir.name} (mtime check); skipping training."

    return None


TASKS = [
    Task(
        name="yolo_train",
        run=run_yolo_train,
        skip_fn=skip_yolo_train,
        description="Train YOLO model.",
        dependencies=["dataset_splitter"],
    ),
    Task(
        name="anomalib_train",
        run=run_anomalib_train,
        skip_fn=skip_anomalib_train,
        description="Train an Anomalib model from normal/abnormal image folders.",
        dependencies=[],
    ),
    Task(
        name="anomalib_package",
        run=run_anomalib_package,
        description="Package a trained Anomalib run for yolo11_inference.",
        dependencies=[],
    ),
    Task(
        name="yolo_evaluation",
        run=run_yolo_evaluation,
        description="Evaluate YOLO model.",
        dependencies=["yolo_train"],
    ),
    Task(
        name="position_validation",
        run=run_position_validation_task,
        description="Offline position validation.",
        dependencies=[],  # 移除硬依賴,改為運行時檢查權重
    ),
]

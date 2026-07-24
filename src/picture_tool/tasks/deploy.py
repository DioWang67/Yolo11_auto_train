"""Deploy trained artifacts directly into a yolo11_inference models directory.

After ``yolo_train`` completes, this task copies ``config.yaml`` and weights
into ``{inference_models_dir}/{product}/{area}/yolo/`` so that inference can
discover the model without manual path edits.
"""

from __future__ import annotations

import datetime
import hashlib
import json
import logging
import re
import shutil
import time
from pathlib import Path
from typing import Any, List, Tuple

import yaml

from picture_tool.pipeline.core import Task
from picture_tool.pipeline.utils import detect_existing_weights
from picture_tool.tasks.bundle import (
    find_color_model_source,
    rewrite_detection_config,
    select_runtime_weight,
    validate_deployment_target,
)
from picture_tool.tasks.deployment_target import resolve_yolo_deployment_target


_VERSION_RE = re.compile(r"_v(\d+)\.(\d+)\.(\d+)_")

STATION_LOCAL_FIELDS = {
    "exposure_time",
    "gain",
    "light_brightness",
    "calibration",
    "output_dir",
    "save_original",
    "save_processed",
    "save_annotated",
    "save_crops",
    "save_fail_only",
    "jpeg_quality",
    "png_compression",
    "max_crops_per_frame",
    "buffer_limit",
    "flush_interval",
}


def _parse_version(filename: str) -> Tuple[int, int, int] | None:
    """Parse a semantic version tuple from a versioned weights filename."""
    match = _VERSION_RE.search(Path(filename).name)
    if not match:
        return None
    return (int(match.group(1)), int(match.group(2)), int(match.group(3)))


def _version_str(version: Tuple[int, int, int]) -> str:
    """Format a semantic version tuple."""
    return f"{version[0]}.{version[1]}.{version[2]}"


def _resolve_version(
    version_cfg: str,
    weights_dest: Path,
    product: str,
    area: str,
    extension: str = ".pt",
) -> Tuple[int, int, int]:
    """Return the deployment version tuple.

    Args:
        version_cfg: Explicit ``major.minor.patch`` value or ``auto``.
        weights_dest: Destination weights directory.
        product: Deployment product name.
        area: Deployment area name.
        extension: Runtime weight extension to scan.

    Returns:
        Version tuple for the new deployed weight.

    Raises:
        ValueError: If an explicit version string is invalid.
    """
    prefix = f"{product}_{area}_v"

    if version_cfg and version_cfg.lower() != "auto":
        parts = version_cfg.split(".")
        if len(parts) == 3:
            try:
                return (int(parts[0]), int(parts[1]), int(parts[2]))
            except ValueError:
                pass
        raise ValueError(
            f"deploy.version '{version_cfg}' is not a valid semantic version "
            "(expected 'major.minor.patch' or 'auto')."
        )

    existing: list[Tuple[int, int, int]] = []
    if weights_dest.exists():
        for path in weights_dest.glob(f"{prefix}*{extension}"):
            version = _parse_version(path.name)
            if version:
                existing.append(version)

    if not existing:
        return (1, 0, 0)

    latest = max(existing)
    return (latest[0], latest[1], latest[2] + 1)


def _acquire_deploy_lock(dest_dir: Path, timeout_seconds: float) -> Path:
    """Create a per-target lock directory around versioning and file writes.

    Args:
        dest_dir: Target ``models/<product>/<area>/yolo`` directory.
        timeout_seconds: Maximum seconds to wait for a previous deploy.

    Returns:
        Created lock directory path.

    Raises:
        TimeoutError: If another deploy keeps the lock too long.
    """
    lock_dir = dest_dir / ".deploy.lock"
    deadline = time.monotonic() + max(timeout_seconds, 0.0)
    while True:
        try:
            lock_dir.mkdir()
            return lock_dir
        except FileExistsError:
            if time.monotonic() >= deadline:
                raise TimeoutError(f"Timed out waiting for deploy lock at {lock_dir}.")
            time.sleep(0.2)


def _release_deploy_lock(lock_dir: Path) -> None:
    """Release a deploy lock created by this process."""
    try:
        lock_dir.rmdir()
    except FileNotFoundError:
        return


def _sha256_file(path: Path) -> str:
    """Return the SHA-256 identity of a deployment artifact."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_runtime_training_pair(run_dir: Path, runtime_path: Path) -> Path:
    """Validate and return the exact PT checkpoint used for a runtime export."""
    resolved_run_dir = run_dir.resolve()
    resolved_runtime = runtime_path.resolve()
    if resolved_runtime.suffix.lower() == ".pt":
        return resolved_runtime

    contract_path = resolved_run_dir / "runtime_export_manifest.json"
    if not contract_path.is_file():
        raise FileNotFoundError(
            "Runtime export lineage contract is missing; re-export the runtime "
            f"artifact before deployment: {contract_path}"
        )
    try:
        payload = json.loads(contract_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Runtime export lineage contract is invalid: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError("Runtime export lineage contract must contain an object.")

    runtime_value = str(payload.get("runtime_file") or "").strip()
    training_value = str(payload.get("training_weight_file") or "").strip()
    if not runtime_value or not training_value:
        raise ValueError("Runtime export lineage contract is incomplete.")
    contracted_runtime = (resolved_run_dir / runtime_value).resolve()
    training_weight = (resolved_run_dir / training_value).resolve()
    if not contracted_runtime.is_relative_to(resolved_run_dir):
        raise ValueError("Runtime export lineage contains an unsafe runtime path.")
    if not training_weight.is_relative_to(resolved_run_dir):
        raise ValueError("Runtime export lineage contains an unsafe training path.")
    if contracted_runtime != resolved_runtime:
        raise ValueError(
            "Selected runtime artifact does not match runtime_export_manifest.json."
        )
    if training_weight.suffix.lower() != ".pt" or not training_weight.is_file():
        raise FileNotFoundError(
            f"Paired runtime training checkpoint is unavailable: {training_weight}"
        )
    runtime_sha256 = str(payload.get("runtime_sha256") or "").strip().lower()
    training_sha256 = str(payload.get("training_weight_sha256") or "").strip().lower()
    if not runtime_sha256 or _sha256_file(resolved_runtime) != runtime_sha256:
        raise ValueError("Runtime export checksum does not match its lineage contract.")
    if not training_sha256 or _sha256_file(training_weight) != training_sha256:
        raise ValueError("Training checkpoint checksum does not match runtime lineage.")
    return training_weight


def _atomic_copy_verified(source: Path, destination: Path) -> str:
    """Copy a file through a temporary path, verify it, then publish atomically."""
    temporary = destination.with_name(f".{destination.name}.tmp")
    try:
        shutil.copy2(source, temporary)
        source_hash = _sha256_file(source)
        copied_hash = _sha256_file(temporary)
        if source_hash != copied_hash:
            raise OSError(
                f"Artifact checksum mismatch while copying {source} to {destination}"
            )
        temporary.replace(destination)
        return source_hash
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _write_yaml_atomic(path: Path, payload: dict[str, Any]) -> None:
    """Publish YAML with an atomic same-directory replacement."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        temporary.write_text(
            yaml.safe_dump(payload, allow_unicode=True, sort_keys=False),
            encoding="utf-8",
        )
        temporary.replace(path)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _preserve_station_fields(
    generated: dict[str, Any], existing_config_path: Path, deploy_cfg: dict[str, Any]
) -> dict[str, Any]:
    """Keep machine-local camera/output settings across model deployments."""
    if not deploy_cfg.get("preserve_station_settings", True):
        return generated
    if not existing_config_path.exists():
        return generated
    existing = yaml.safe_load(existing_config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(existing, dict):
        return generated
    configured_fields = deploy_cfg.get("station_fields")
    fields = (
        {str(field) for field in configured_fields}
        if isinstance(configured_fields, list)
        else STATION_LOCAL_FIELDS
    )
    merged = dict(generated)
    for field in fields:
        if field in existing:
            merged[field] = existing[field]
    return merged


def _load_training_metadata(run_dir: Path) -> dict[str, Any]:
    """Load optional dataset/config hashes recorded by the trainer."""
    path = run_dir / "last_run_metadata.json"
    if not path.exists():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _load_evaluation_gate(run_dir: Path, *, required: bool) -> dict[str, Any]:
    """Load the evaluation decision required before an operator deployment."""
    path = run_dir / "evaluation_gate.json"
    if not path.exists():
        if required:
            raise ValueError(f"Evaluation gate report not found: {path}")
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Unable to read evaluation gate report: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError("Evaluation gate report must be an object.")
    if required and not payload.get("passed", False):
        failures = payload.get("failures") or []
        raise ValueError(
            "Deployment blocked by evaluation gate: "
            + "; ".join(str(item) for item in failures)
        )
    return payload


def run_deploy(config: dict, args: Any) -> None:
    """Copy training artifacts to the inference models directory.

    Args:
        config: Pipeline configuration containing ``yolo_training.deploy``.
        args: Runtime args. Currently unused by this task.

    Raises:
        FileNotFoundError: If required config, weights, or color model files are missing.
        TimeoutError: If another deploy holds the target lock too long.
        ValueError: If product, area, or version config is invalid.
    """
    logger = logging.getLogger(__name__)
    ycfg = config.get("yolo_training", {})
    dcfg = ycfg.get("deploy", {})
    if not dcfg.get("enabled", False):
        logger.info("Deploy task disabled (yolo_training.deploy.enabled is false).")
        return

    target = resolve_yolo_deployment_target(config, args)
    product = target.product
    area = target.area
    validate_deployment_target(product, area)

    raw_dir = dcfg.get("inference_models_dir")
    if not raw_dir:
        raise ValueError(
            "deploy.inference_models_dir is not set. Configure the yolo11_inference "
            "models directory before running deploy."
        )

    inference_models_dir = Path(raw_dir).expanduser()
    if not inference_models_dir.is_absolute():
        inference_models_dir = Path.cwd() / inference_models_dir
    inference_models_dir = inference_models_dir.resolve()

    _, run_dir = detect_existing_weights(config)
    if not run_dir or not run_dir.exists():
        raise FileNotFoundError(
            "No YOLO training run found for deploy. Run yolo_train first "
            "or configure yolo_evaluation.weights / yolo_training.position_validation.weights."
        )

    gate_required = bool(
        ((config.get("yolo_evaluation", {}) or {}).get("gate", {}) or {}).get(
            "enabled", False
        )
    )
    evaluation_gate = _load_evaluation_gate(run_dir, required=gate_required)

    det_cfg_path = run_dir / "detection_config.yaml"
    if not det_cfg_path.exists():
        logger.error(
            "detection_config.yaml not found at %s; deploy aborted.", det_cfg_path
        )
        raise FileNotFoundError(det_cfg_path)

    try:
        det_cfg_data = yaml.safe_load(det_cfg_path.read_text(encoding="utf-8")) or {}
        det_cfg_data = rewrite_detection_config(det_cfg_data, product, area)
    except (OSError, UnicodeDecodeError, yaml.YAMLError, ValueError, TypeError, KeyError) as exc:
        logger.error("Failed to read detection config: %s", exc)
        raise

    selected_weights_path = select_runtime_weight(
        run_dir, str(det_cfg_data.get("weights") or "")
    )
    selected_weights_name = selected_weights_path.name
    selected_extension = selected_weights_path.suffix or ".pt"
    training_weight_source = _resolve_runtime_training_pair(
        run_dir,
        selected_weights_path,
    )
    if not training_weight_source.is_file():
        raise FileNotFoundError(
            "Deployment requires the PT checkpoint paired with the runtime "
            f"artifact: {training_weight_source}"
        )
    dest_dir = inference_models_dir / product / area / "yolo"
    weights_dest = dest_dir / "weights"
    dest_dir.mkdir(parents=True, exist_ok=True)
    weights_dest.mkdir(exist_ok=True)

    color_cfg_name = Path(det_cfg_data.get("color_model_path", "")).name
    if det_cfg_data.get("enable_color_check") and not color_cfg_name:
        raise FileNotFoundError(
            "enable_color_check is true but color_model_path is not configured."
        )
    color_source = find_color_model_source(run_dir, color_cfg_name)
    color_source_kind = "training_run"
    if (
        color_source is None
        and color_cfg_name
        and dcfg.get("preserve_station_settings", True)
    ):
        station_color = (dest_dir / color_cfg_name).resolve()
        if station_color.parent != dest_dir.resolve():
            raise ValueError("Resolved station color model path is unsafe.")
        if station_color.is_file():
            color_source = station_color
            color_source_kind = "existing_station"
    if det_cfg_data.get("enable_color_check") and color_source is None:
        raise FileNotFoundError(
            "enable_color_check is true but no color model/stat file was found "
            f"for {color_cfg_name}."
        )

    lock_dir = _acquire_deploy_lock(dest_dir, float(dcfg.get("lock_timeout", 30.0)))
    try:
        version = _resolve_version(
            dcfg.get("version", "auto"),
            weights_dest,
            product,
            area,
            selected_extension,
        )
        date_str = datetime.datetime.now().strftime("%Y%m%d")
        versioned_name = (
            f"{product}_{area}_v{_version_str(version)}_{date_str}{selected_extension}"
        )
        versioned_path = weights_dest / versioned_name
        paired_training_name = (
            versioned_name
            if selected_extension.lower() == ".pt"
            else f"{Path(versioned_name).stem}.training.pt"
        )
        paired_training_path = weights_dest / paired_training_name

        if versioned_path.exists() and not dcfg.get("force", False):
            raise FileExistsError(
                f"Deployment version already exists: {versioned_path}. "
                "Use version=auto or deploy.force=true intentionally."
            )

        logger.info(
            "Deploying %s/%s/yolo to %s (version %s)",
            product,
            area,
            dest_dir,
            _version_str(version),
        )

        weight_sha256 = _atomic_copy_verified(selected_weights_path, versioned_path)
        if paired_training_path == versioned_path:
            paired_training_sha256 = weight_sha256
        else:
            paired_training_sha256 = _atomic_copy_verified(
                training_weight_source,
                paired_training_path,
            )
        _atomic_copy_verified(
            selected_weights_path, weights_dest / selected_weights_name
        )
        logger.info(
            "Copied runtime weight %s as %s.", selected_weights_name, versioned_name
        )

        for filename in [
            "best.pt",
            "last.pt",
            "best.onnx",
            "best.engine",
            "best.torchscript",
        ]:
            src = run_dir / "weights" / filename
            if src.exists() and src.name != selected_weights_name:
                _atomic_copy_verified(src, weights_dest / filename)
                logger.info("Copied %s.", filename)

        for src_dir in _iter_export_dirs(run_dir / "weights"):
            dest_export_dir = weights_dest / src_dir.name
            if dest_export_dir.exists():
                shutil.rmtree(dest_export_dir)
            shutil.copytree(src_dir, dest_export_dir)
            logger.info("Copied exported runtime directory %s.", src_dir.name)

        color_model_sha256: str | None = None
        if color_source and color_cfg_name:
            color_destination = (dest_dir / color_cfg_name).resolve()
            if color_source.resolve() == color_destination:
                color_model_sha256 = _sha256_file(color_destination)
                logger.info("Preserved station colour model %s.", color_cfg_name)
            else:
                color_model_sha256 = _atomic_copy_verified(
                    color_source, color_destination
                )
                logger.info(
                    "Copied colour model from %s to %s.",
                    color_source,
                    color_cfg_name,
                )

        deploy_config = dict(det_cfg_data)
        deploy_config["weights"] = (
            f"models/{product}/{area}/yolo/weights/{versioned_name}"
        )
        if color_cfg_name:
            deploy_config["color_model_path"] = (
                f"models/{product}/{area}/yolo/{color_cfg_name}"
            )

        config_path = dest_dir / "config.yaml"
        deploy_config = _preserve_station_fields(deploy_config, config_path, dcfg)
        training_metadata = _load_training_metadata(run_dir)
        trained_at = training_metadata.get(
            "trained_at"
        ) or datetime.datetime.fromtimestamp(
            selected_weights_path.stat().st_mtime
        ).astimezone().isoformat(timespec="seconds")
        deployed_at = datetime.datetime.now().astimezone().isoformat(timespec="seconds")
        config_snapshot_relative = f"versions/{versioned_name}.config.yaml"
        manifest = {
            "schema_version": 2,
            "deployed_version": _version_str(version),
            "deployed_date": date_str,
            "deployed_at": deployed_at,
            "trained_at": trained_at,
            "deployed_file": versioned_name,
            "model_type": "yolo",
            "runtime_format": selected_extension.lstrip("."),
            "file_size": versioned_path.stat().st_size,
            "weight_sha256": weight_sha256,
            "training_weight_file": paired_training_name,
            "training_weight_sha256": paired_training_sha256,
            "runtime_source_training_weight": (
                selected_weights_name
                if selected_extension.lower() == ".pt"
                else training_weight_source.name
            ),
            "deployed_to": str(config_path),
            "product": product,
            "area": area,
            "dataset_hash": training_metadata.get("dataset_hash"),
            "training_config_hash": training_metadata.get("config_hash"),
            "evaluation_metrics": evaluation_gate.get("metrics", {}),
            "evaluation_gate_passed": evaluation_gate.get("passed"),
            "config_snapshot": config_snapshot_relative,
            "color_model_file": color_cfg_name or None,
            "color_model_sha256": color_model_sha256,
            "color_model_source": color_source_kind if color_model_sha256 else None,
        }
        # Each deployed weight keeps immutable metadata and the matching runtime
        # config. This makes later rollback deterministic instead of guessing
        # which thresholds/classes belonged to an old artifact.
        _write_yaml_atomic(dest_dir / config_snapshot_relative, deploy_config)
        _write_yaml_atomic(
            versioned_path.with_name(f"{versioned_path.name}.manifest.yaml"),
            manifest,
        )
        _write_yaml_atomic(dest_dir / "deployment_manifest.yaml", manifest)
        # Publish config last: inference never observes a pointer to an
        # unverified or partially copied runtime artifact.
        _write_yaml_atomic(config_path, deploy_config)
        logger.info("config.yaml written with weights %s.", versioned_name)
        try:
            (run_dir / "version_manifest.yaml").write_text(
                yaml.safe_dump(manifest, allow_unicode=True, sort_keys=False),
                encoding="utf-8",
            )
            logger.info("version_manifest.yaml written to %s.", run_dir)
        except OSError as exc:
            logger.warning("Could not write version_manifest.yaml: %s", exc)

        logger.info(
            "Deploy complete: %s/%s/yolo, version=%s, file=%s, trained_at=%s",
            product,
            area,
            _version_str(version),
            versioned_name,
            trained_at,
        )
    finally:
        _release_deploy_lock(lock_dir)


TASKS: List[Any] = [
    Task(
        name="deploy",
        run=run_deploy,
        description="Deploy artifacts directly to yolo11_inference models directory.",
        dependencies=["yolo_evaluation"],
    ),
]


def _iter_export_dirs(weights_dir: Path) -> list[Path]:
    """Return directory-based YOLO runtime artifacts created by Ultralytics."""
    if not weights_dir.exists():
        return []
    return sorted(
        path
        for path in weights_dir.iterdir()
        if path.is_dir() and path.name.endswith("_openvino_model")
    )

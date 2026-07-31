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
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, List, Tuple

import yaml

from picture_tool.pipeline.core import Task
from picture_tool.pipeline.utils import detect_existing_weights
from picture_tool.runtime_pair_deployment import PairVerification, verify_runtime_pair
from picture_tool.tasks.bundle import (
    find_color_model_source,
    rewrite_detection_config,
    select_runtime_weight,
    validate_deployment_target,
)
from picture_tool.tasks.deployment_target import resolve_yolo_deployment_target
from picture_tool.utils.onnx_exporter import OnnxExporter


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
DEPLOYMENT_OWNED_FIELDS = frozenset(
    {"weights", "position_config", "color_model_path"}
)


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
        # Version identity belongs to the station model, not to a particular
        # runtime extension. Switching a station from PT to ONNX must advance
        # from v1.0.4 to v1.0.5 instead of restarting at v1.0.0.
        for path in weights_dest.glob(f"{prefix}*"):
            if not path.is_file():
                continue
            if path.suffix.lower() not in {
                ".pt",
                ".onnx",
                ".engine",
                ".torchscript",
                extension.lower(),
            }:
                continue
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
    forbidden_fields = fields & DEPLOYMENT_OWNED_FIELDS
    if forbidden_fields:
        raise ValueError(
            "deploy.station_fields cannot preserve deployment-owned fields: "
            + ", ".join(sorted(forbidden_fields))
        )
    merged = dict(generated)
    for field in fields:
        if field in existing:
            merged[field] = existing[field]
    return merged


def _position_area_mapping(
    config: dict[str, Any],
    product: str,
    area: str,
) -> dict[str, Any] | None:
    position_config = config.get("position_config")
    if not isinstance(position_config, dict):
        return None
    product_config = position_config.get(product)
    if not isinstance(product_config, dict):
        return None
    area_config = product_config.get(area)
    return area_config if isinstance(area_config, dict) else None


def _apply_position_activation_policy(
    generated: dict[str, Any],
    existing_config_path: Path,
    deploy_cfg: dict[str, Any],
    *,
    product: str,
    area: str,
) -> dict[str, Any]:
    """Keep runtime activation explicit while still deploying calibrated rules."""

    candidate_area = _position_area_mapping(generated, product, area)
    if candidate_area is None:
        return generated
    policy = str(
        deploy_cfg.get("position_activation") or "preserve"
    ).strip().lower()
    if policy not in {"preserve", "enable", "disable"}:
        raise ValueError(
            "deploy.position_activation must be preserve, enable, or disable."
        )

    enabled = False
    if policy == "enable":
        enabled = True
    elif policy == "disable":
        enabled = False
    elif existing_config_path.is_file():
        try:
            existing = (
                yaml.safe_load(existing_config_path.read_text(encoding="utf-8"))
                or {}
            )
        except (OSError, UnicodeDecodeError, yaml.YAMLError) as exc:
            raise ValueError(
                f"Unable to preserve station position activation: {exc}"
            ) from exc
        if isinstance(existing, dict):
            existing_area = _position_area_mapping(existing, product, area)
            enabled = bool(
                existing_area.get("enabled", False)
                if existing_area is not None
                else False
            )

    merged = dict(generated)
    position_config = dict(merged.get("position_config") or {})
    product_config = dict(position_config.get(product) or {})
    updated_area = dict(candidate_area)
    updated_area["enabled"] = enabled
    product_config[area] = updated_area
    position_config[product] = product_config
    merged["position_config"] = position_config
    return merged


def _preserve_disabled_station_position_contract(
    generated: dict[str, Any],
    existing_config_path: Path,
    deploy_cfg: dict[str, Any],
    *,
    product: str,
    area: str,
) -> dict[str, Any]:
    """Preserve an inactive station contract when this job has no golden set."""
    policy = str(
        deploy_cfg.get("position_contract_policy") or "validate_candidate"
    ).strip().lower()
    if policy == "validate_candidate":
        return generated
    if policy != "preserve_disabled_station":
        raise ValueError(
            "deploy.position_contract_policy must be validate_candidate or "
            "preserve_disabled_station."
        )
    try:
        existing = (
            yaml.safe_load(existing_config_path.read_text(encoding="utf-8")) or {}
        )
    except (OSError, UnicodeDecodeError, yaml.YAMLError) as exc:
        raise ValueError(
            f"Unable to preserve station position contract: {exc}"
        ) from exc
    if not isinstance(existing, dict):
        raise ValueError("Existing station config must be a mapping.")
    existing_area = _position_area_mapping(existing, product, area)
    if existing_area is None or bool(existing_area.get("enabled", False)):
        raise ValueError(
            "An active or missing station position contract cannot bypass "
            "position validation."
        )
    existing_position = existing.get("position_config")
    if not isinstance(existing_position, dict):
        raise ValueError("Existing station position contract is invalid.")
    merged = dict(generated)
    merged["position_config"] = existing_position
    return merged


def _load_position_gate(
    run_dir: Path,
    *,
    required: bool,
) -> tuple[dict[str, Any], Path | None]:
    path = run_dir / "position_gate.json"
    if not path.is_file():
        if required:
            raise ValueError(f"Position gate report not found: {path}")
        return {}, None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Unable to read position gate report: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError("Position gate report must contain an object.")
    if required and payload.get("passed") is not True:
        failures = payload.get("failures") or []
        raise ValueError(
            "Deployment blocked by position gate: "
            + "; ".join(str(item) for item in failures)
        )
    return payload, path


def _sha256_json(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _verified_position_evidence(
    gate: dict[str, Any],
    *,
    path_field: str,
    hash_field: str,
    label: str,
    required: bool = False,
) -> tuple[Path | None, str | None]:
    raw_path = str(gate.get(path_field) or "").strip()
    if not raw_path:
        if required:
            raise ValueError(f"Position gate has no {label} path.")
        return None, None
    path = Path(raw_path).resolve()
    if not path.is_file():
        raise ValueError(f"Position {label} was not found: {path}")
    expected_hash = str(gate.get(hash_field) or "").strip().lower()
    if (
        len(expected_hash) != 64
        or any(character not in "0123456789abcdef" for character in expected_hash)
    ):
        raise ValueError(f"Position gate has no valid {label} checksum.")
    actual_hash = _sha256_file(path)
    if actual_hash != expected_hash:
        raise ValueError(
            f"Position {label} checksum changed after gate evaluation."
        )
    return path, actual_hash


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


def _load_model_acceptance_report(
    report_path: Path,
    *,
    required: bool,
) -> dict[str, Any]:
    """Load and validate the deployment-blocking acceptance decision."""
    if not report_path.is_file():
        if required:
            raise ValueError(
                f"Model acceptance gate report not found: {report_path}"
            )
        return {}
    try:
        payload = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(
            f"Unable to read model acceptance gate report: {exc}"
        ) from exc
    if not isinstance(payload, dict):
        raise ValueError("Model acceptance gate report must be an object.")
    if required and not payload.get("passed", False):
        failures = payload.get("failures") or []
        raise ValueError(
            "Deployment blocked by model acceptance gate: "
            + "; ".join(str(item) for item in failures)
        )
    return payload


def _run_model_acceptance_gate(
    *,
    config: dict[str, Any],
    run_dir: Path,
    inference_project_root: Path,
    inference_models_dir: Path,
    product: str,
    area: str,
    candidate_config: dict[str, Any],
    candidate_weight: Path,
    color_model: Path | None,
    logger: logging.Logger,
) -> tuple[dict[str, Any], Path | None]:
    """Run the inference-owned headless gate against a job-scoped bundle."""
    ycfg = config.get("yolo_training", {}) or {}
    deploy_cfg = ycfg.get("deploy", {}) or {}
    gate_cfg = deploy_cfg.get("acceptance_gate", {}) or {}
    if not isinstance(gate_cfg, dict) or not gate_cfg.get("enabled", False):
        logger.info("Model acceptance gate disabled.")
        return {}, None

    dataset_root = Path(str(gate_cfg.get("dataset_root") or "")).expanduser()
    snapshot_manifest = Path(
        str(gate_cfg.get("snapshot_manifest") or "")
    ).expanduser()
    if not dataset_root.is_absolute():
        dataset_root = (inference_project_root / dataset_root).resolve()
    else:
        dataset_root = dataset_root.resolve()
    if not snapshot_manifest.is_absolute():
        snapshot_manifest = (dataset_root / snapshot_manifest).resolve()
    else:
        snapshot_manifest = snapshot_manifest.resolve()
    if not dataset_root.is_dir():
        raise FileNotFoundError(
            f"Acceptance dataset root not found: {dataset_root}"
        )
    if not snapshot_manifest.is_file():
        raise FileNotFoundError(
            f"Acceptance snapshot manifest not found: {snapshot_manifest}"
        )

    runner_path = inference_project_root / "app" / "acceptance" / "headless.py"
    global_config = inference_project_root / "config.yaml"
    if not runner_path.is_file() or not global_config.is_file():
        raise FileNotFoundError(
            "Inference acceptance runtime is incomplete: "
            f"runner={runner_path}, config={global_config}"
        )

    candidate_models_root = run_dir / "acceptance_candidate" / "models"
    candidate_station_dir = (
        candidate_models_root / product / area / "yolo"
    )
    candidate_config_path = candidate_station_dir / "config.yaml"
    _write_yaml_atomic(candidate_config_path, candidate_config)
    report_path = run_dir / "model_acceptance_gate.json"

    command = [
        sys.executable,
        "-m",
        "app.acceptance.headless",
        "--project-root",
        str(inference_project_root),
        "--models-root",
        str(candidate_models_root),
        "--global-config",
        str(global_config),
        "--color-revisions-root",
        str(inference_models_dir.parent / ".color_revisions"),
        "--dataset-root",
        str(dataset_root),
        "--snapshot-manifest",
        str(snapshot_manifest),
        "--report",
        str(report_path),
        "--product",
        product,
        "--area",
        area,
        "--inference-type",
        "yolo",
        "--candidate-version",
        "candidate",
        "--candidate-weight",
        str(candidate_weight),
        "--candidate-config",
        str(candidate_config_path),
        "--min-confirmed",
        str(int(gate_cfg.get("min_confirmed", 1))),
        "--max-false-positives",
        str(int(gate_cfg.get("max_false_positives", 0))),
        "--max-false-negatives",
        str(int(gate_cfg.get("max_false_negatives", 0))),
        "--max-regressions",
        str(int(gate_cfg.get("max_regressions", 0))),
    ]
    if color_model is not None:
        command.extend(["--color-model", str(color_model)])
    if not bool(gate_cfg.get("require_all_confirmed", True)):
        command.append("--allow-pending")
    if not bool(gate_cfg.get("require_no_errors", True)):
        command.append("--allow-errors")

    timeout_seconds = max(float(gate_cfg.get("timeout_seconds", 1800.0)), 1.0)
    logger.info(
        "Running frozen model acceptance set for %s/%s (%s).",
        product,
        area,
        snapshot_manifest.parent.name,
    )
    try:
        completed = subprocess.run(
            command,
            cwd=inference_project_root,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout_seconds,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise TimeoutError(
            f"Model acceptance exceeded {timeout_seconds:.0f} seconds."
        ) from exc
    runner_log_path = run_dir / "model_acceptance_runner.log"
    runner_log_path.write_text(
        completed.stdout
        + ("\n[stderr]\n" + completed.stderr if completed.stderr else ""),
        encoding="utf-8",
    )
    for line in completed.stdout.splitlines():
        if line.startswith("[acceptance]"):
            logger.info("%s", line)
    if completed.returncode != 0:
        diagnostic_lines = (
            completed.stderr.splitlines() or completed.stdout.splitlines()
        )[-20:]
        for line in diagnostic_lines:
            logger.warning("%s", line)

    report = _load_model_acceptance_report(report_path, required=True)
    if completed.returncode != 0:
        raise ValueError(
            "Model acceptance runner failed with exit code "
            f"{completed.returncode}."
        )
    candidate = report.get("candidate") or {}
    if candidate.get("sha256") != _sha256_file(candidate_weight):
        raise ValueError(
            "Model acceptance report does not match the candidate weight."
        )
    if candidate.get("runtime_config_sha256") != _sha256_file(
        candidate_config_path
    ):
        raise ValueError(
            "Model acceptance report does not match the candidate config."
        )
    return report, report_path


def _verify_deployment_runtime_pair(
    config: dict[str, Any],
    runtime_path: Path,
    training_weight_path: Path,
) -> PairVerification | None:
    """Run the deployment-only ONNX/PT equivalence gate when configured."""
    if runtime_path.suffix.lower() != ".onnx":
        return None
    ycfg = config.get("yolo_training", {}) or {}
    deploy_cfg = ycfg.get("deploy", {}) or {}
    verification_cfg = deploy_cfg.get("runtime_pair_verification", {}) or {}
    if not isinstance(verification_cfg, dict) or not verification_cfg.get(
        "enabled", False
    ):
        return None

    configured_size = verification_cfg.get("input_size", ycfg.get("imgsz", 640))
    if isinstance(configured_size, (list, tuple)):
        input_size = max(int(value) for value in configured_size)
    else:
        input_size = int(configured_size or 640)
    return verify_runtime_pair(
        runtime_path,
        training_weight_path,
        input_size=input_size,
        rtol=float(verification_cfg.get("rtol", 1e-3)),
        atol=float(verification_cfg.get("atol", 1e-3)),
    )


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

    exported_runtime = OnnxExporter.ensure(config, run_dir, logger)
    if OnnxExporter.is_enabled(config) and exported_runtime is None:
        raise RuntimeError(
            "Deployment requires a validated runtime export, but export failed."
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

    candidate_position_area = _position_area_mapping(
        det_cfg_data,
        product,
        area,
    )
    # A disabled candidate can become enabled through the preserve/enable
    # activation policy. Therefore every deployed position contract requires
    # gate evidence, independent of its pre-policy enabled flag.
    position_contract_policy = str(
        dcfg.get("position_contract_policy") or "validate_candidate"
    ).strip().lower()
    if position_contract_policy not in {
        "validate_candidate",
        "preserve_disabled_station",
    }:
        raise ValueError(
            "deploy.position_contract_policy must be validate_candidate or "
            "preserve_disabled_station."
        )
    position_gate_required = (
        candidate_position_area is not None
        and position_contract_policy == "validate_candidate"
    )
    position_gate, position_gate_path = _load_position_gate(
        run_dir,
        required=position_gate_required,
    )
    if position_gate:
        if str(position_gate.get("product") or "") != product:
            raise ValueError("Position gate product does not match deploy target.")
        if str(position_gate.get("area") or "") != area:
            raise ValueError("Position gate area does not match deploy target.")
    position_report_path, position_report_sha256 = _verified_position_evidence(
        position_gate,
        path_field="candidate_report",
        hash_field="candidate_report_sha256",
        label="validation evidence",
        required=position_gate_required,
    )
    position_baseline_path, position_baseline_sha256 = (
        _verified_position_evidence(
            position_gate,
            path_field="baseline_report",
            hash_field="baseline_report_sha256",
            label="baseline evidence",
        )
    )
    position_calibration_path, position_calibration_sha256 = (
        _verified_position_evidence(
            position_gate,
            path_field="calibration_manifest",
            hash_field="calibration_manifest_sha256",
            label="calibration evidence",
        )
    )

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
    pair_verification = _verify_deployment_runtime_pair(
        config,
        selected_weights_path,
        training_weight_source,
    )
    if pair_verification is not None:
        logger.info(
            "ONNX/PT pair verified (max_abs_error=%.6g, mean_abs_error=%.6g).",
            pair_verification.comparison.max_abs_error,
            pair_verification.comparison.mean_abs_error,
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

    candidate_deploy_config = dict(det_cfg_data)
    candidate_deploy_config["weights"] = str(selected_weights_path.resolve())
    if color_source is not None and color_cfg_name:
        candidate_deploy_config["color_model_path"] = str(color_source.resolve())
    station_config_path = dest_dir / "config.yaml"
    candidate_deploy_config = _preserve_station_fields(
        candidate_deploy_config,
        station_config_path,
        dcfg,
    )
    candidate_deploy_config = _preserve_disabled_station_position_contract(
        candidate_deploy_config,
        station_config_path,
        dcfg,
        product=product,
        area=area,
    )
    candidate_deploy_config = _apply_position_activation_policy(
        candidate_deploy_config,
        station_config_path,
        dcfg,
        product=product,
        area=area,
    )
    acceptance_report, acceptance_report_path = _run_model_acceptance_gate(
        config=config,
        run_dir=run_dir,
        inference_project_root=inference_models_dir.parent,
        inference_models_dir=inference_models_dir,
        product=product,
        area=area,
        candidate_config=candidate_deploy_config,
        candidate_weight=selected_weights_path,
        color_model=color_source,
        logger=logger,
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
        accepted_weight_sha256 = (
            (acceptance_report.get("candidate") or {}).get("sha256")
            if acceptance_report
            else None
        )
        if (
            accepted_weight_sha256 is not None
            and accepted_weight_sha256 != weight_sha256
        ):
            raise ValueError(
                "Candidate weight changed after model acceptance completed."
            )
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

        position_gate_relative: str | None = None
        position_report_relative: str | None = None
        position_baseline_relative: str | None = None
        position_calibration_relative: str | None = None
        acceptance_report_relative: str | None = None
        acceptance_report_sha256: str | None = None
        evidence_dir = dest_dir / "versions"
        if acceptance_report_path is not None:
            evidence_dir.mkdir(parents=True, exist_ok=True)
            acceptance_destination = (
                evidence_dir / f"{versioned_name}.model_acceptance.json"
            )
            _atomic_copy_verified(
                acceptance_report_path,
                acceptance_destination,
            )
            acceptance_report_relative = (
                f"versions/{acceptance_destination.name}"
            )
            acceptance_report_sha256 = _sha256_file(
                acceptance_destination
            )
        if position_gate_path is not None:
            evidence_dir.mkdir(parents=True, exist_ok=True)
            gate_destination = evidence_dir / f"{versioned_name}.position_gate.json"
            _atomic_copy_verified(position_gate_path, gate_destination)
            position_gate_relative = f"versions/{gate_destination.name}"
        if position_report_path is not None:
            evidence_dir.mkdir(parents=True, exist_ok=True)
            report_destination = (
                evidence_dir / f"{versioned_name}.position_validation.json"
            )
            _atomic_copy_verified(position_report_path, report_destination)
            position_report_relative = f"versions/{report_destination.name}"
            position_report_sha256 = _sha256_file(report_destination)
        if position_baseline_path is not None:
            evidence_dir.mkdir(parents=True, exist_ok=True)
            baseline_destination = (
                evidence_dir / f"{versioned_name}.position_baseline.json"
            )
            _atomic_copy_verified(position_baseline_path, baseline_destination)
            position_baseline_relative = f"versions/{baseline_destination.name}"
            position_baseline_sha256 = _sha256_file(baseline_destination)
        if position_calibration_path is not None:
            evidence_dir.mkdir(parents=True, exist_ok=True)
            calibration_destination = (
                evidence_dir / f"{versioned_name}.position_calibration.json"
            )
            _atomic_copy_verified(
                position_calibration_path,
                calibration_destination,
            )
            position_calibration_relative = (
                f"versions/{calibration_destination.name}"
            )
            position_calibration_sha256 = _sha256_file(calibration_destination)

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
        deploy_config = _preserve_disabled_station_position_contract(
            deploy_config,
            config_path,
            dcfg,
            product=product,
            area=area,
        )
        deploy_config = _apply_position_activation_policy(
            deploy_config,
            config_path,
            dcfg,
            product=product,
            area=area,
        )
        deployed_position_area = _position_area_mapping(
            deploy_config,
            product,
            area,
        )
        position_config_sha256 = (
            _sha256_json(deploy_config.get("position_config"))
            if deployed_position_area is not None
            else None
        )
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
            "runtime_pair_verified": pair_verification is not None,
            "runtime_pair_verification": (
                {
                    "input_size": pair_verification.input_size,
                    "runtime_shape": list(
                        pair_verification.comparison.runtime_shape
                    ),
                    "training_shape": list(
                        pair_verification.comparison.training_shape
                    ),
                    "max_abs_error": pair_verification.comparison.max_abs_error,
                    "mean_abs_error": pair_verification.comparison.mean_abs_error,
                    "p99_abs_error": pair_verification.comparison.p99_abs_error,
                }
                if pair_verification is not None
                else None
            ),
            "deployed_to": str(config_path),
            "product": product,
            "area": area,
            "dataset_hash": training_metadata.get("dataset_hash"),
            "training_config_hash": training_metadata.get("config_hash"),
            "evaluation_metrics": evaluation_gate.get("metrics", {}),
            "evaluation_gate_passed": evaluation_gate.get("passed"),
            "model_acceptance_gate_passed": (
                acceptance_report.get("passed")
                if acceptance_report
                else None
            ),
            "model_acceptance_metrics": acceptance_report.get("metrics", {}),
            "model_acceptance_baseline_metrics": acceptance_report.get(
                "baseline_metrics", {}
            ),
            "model_acceptance_report": acceptance_report_relative,
            "model_acceptance_report_sha256": acceptance_report_sha256,
            "model_acceptance_snapshot_sha256": (
                (acceptance_report.get("dataset") or {}).get(
                    "snapshot_manifest_sha256"
                )
                if acceptance_report
                else None
            ),
            "config_snapshot": config_snapshot_relative,
            "color_model_file": color_cfg_name or None,
            "color_model_sha256": color_model_sha256,
            "color_model_source": color_source_kind if color_model_sha256 else None,
            "position_runtime_enabled": bool(
                deployed_position_area.get("enabled", False)
                if deployed_position_area is not None
                else False
            ),
            "position_config_sha256": position_config_sha256,
            "position_gate_required": position_gate_required,
            "position_gate_passed": (
                position_gate.get("passed")
                if position_gate
                else None
            ),
            "position_gate_sha256": (
                _sha256_file(position_gate_path)
                if position_gate_path is not None
                else None
            ),
            "position_gate_report": position_gate_relative,
            "position_validation_sha256": position_report_sha256,
            "position_validation_report": position_report_relative,
            "position_baseline_sha256": position_baseline_sha256,
            "position_baseline_report": position_baseline_relative,
            "position_calibration_sha256": position_calibration_sha256,
            "position_calibration_manifest": position_calibration_relative,
            "position_metrics": position_gate.get("metrics", {}),
            "position_baseline_metrics": position_gate.get(
                "baseline_metrics"
            ),
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

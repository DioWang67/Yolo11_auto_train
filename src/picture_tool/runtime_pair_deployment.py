"""Verify and deploy an ONNX runtime together with its exact PT checkpoint.

This module is the guarded import path for models trained outside the regular
pipeline.  A runtime is never paired by file name alone: both artifacts are
hashed and their raw model outputs must be numerically equivalent before a
deployment manifest is published.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import re
import shutil
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import yaml

from picture_tool.workspace_paths import WorkspaceConfigurationError, WorkspacePaths


_VERSION_RE = re.compile(r"_v(\d+)\.(\d+)\.(\d+)_")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_DEFAULT_INPUT_SIZE = 640
_DEFAULT_RTOL = 1e-3
_DEFAULT_ATOL = 1e-3


class RuntimePairError(RuntimeError):
    """Raised when a runtime/checkpoint pair cannot be trusted or deployed."""


@dataclass(frozen=True)
class NumericalComparison:
    """Result returned by a runtime-specific numerical comparison adapter."""

    runtime_shape: tuple[int, ...]
    training_shape: tuple[int, ...]
    max_abs_error: float
    mean_abs_error: float
    p99_abs_error: float
    passed: bool
    class_names: tuple[str, ...] = ()


@dataclass(frozen=True)
class PairVerification:
    """Immutable identity and equivalence proof for two model artifacts."""

    runtime_path: Path
    training_weight_path: Path
    runtime_sha256: str
    training_weight_sha256: str
    input_size: int
    comparison: NumericalComparison


@dataclass(frozen=True)
class PairDeployment:
    """Paths and version published by :func:`deploy_runtime_pair`."""

    version: str
    runtime_path: Path
    training_weight_path: Path
    manifest_path: Path
    config_path: Path


ComparisonRunner = Callable[[Path, Path, int, float, float], NumericalComparison]


def sha256_file(path: Path) -> str:
    """Return the SHA-256 identity of a regular file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_runtime_pair(
    runtime_path: Path,
    training_weight_path: Path,
    *,
    input_size: int = _DEFAULT_INPUT_SIZE,
    rtol: float = _DEFAULT_RTOL,
    atol: float = _DEFAULT_ATOL,
    comparison_runner: ComparisonRunner | None = None,
) -> PairVerification:
    """Prove that an ONNX runtime and PT checkpoint produce the same outputs.

    Hashes are calculated both before and after inference to close the usual
    time-of-check/time-of-use window.  The default adapter executes both models
    on the same deterministic tensor; tests may inject a pure comparison
    adapter without importing the ML runtime.
    """
    runtime = runtime_path.expanduser().resolve()
    training_weight = training_weight_path.expanduser().resolve()
    _validate_source_artifact(runtime, ".onnx", "runtime model")
    _validate_source_artifact(training_weight, ".pt", "training checkpoint")
    if input_size <= 0:
        raise RuntimePairError("Input size must be a positive integer.")
    if rtol < 0 or atol < 0:
        raise RuntimePairError("Numerical tolerances must not be negative.")

    runtime_sha256 = sha256_file(runtime)
    training_sha256 = sha256_file(training_weight)
    runner = comparison_runner or _compare_yolo_outputs
    try:
        comparison = runner(runtime, training_weight, input_size, rtol, atol)
    except RuntimePairError:
        raise
    except (ImportError, OSError, RuntimeError, ValueError, TypeError) as exc:
        raise RuntimePairError(f"Unable to compare ONNX and PT outputs: {exc}") from exc

    if sha256_file(runtime) != runtime_sha256:
        raise RuntimePairError("ONNX file changed while its pair was being verified.")
    if sha256_file(training_weight) != training_sha256:
        raise RuntimePairError("PT file changed while its pair was being verified.")
    if comparison.runtime_shape != comparison.training_shape:
        raise RuntimePairError(
            "ONNX/PT output shapes differ: "
            f"onnx={comparison.runtime_shape}, pt={comparison.training_shape}."
        )
    if not comparison.passed:
        raise RuntimePairError(
            "ONNX/PT numerical outputs do not match; deployment was blocked "
            f"(max_abs_error={comparison.max_abs_error:.6g}, "
            f"mean_abs_error={comparison.mean_abs_error:.6g})."
        )
    return PairVerification(
        runtime_path=runtime,
        training_weight_path=training_weight,
        runtime_sha256=runtime_sha256,
        training_weight_sha256=training_sha256,
        input_size=input_size,
        comparison=comparison,
    )


def validate_export_contract(
    contract_path: Path,
    verification: PairVerification,
) -> Mapping[str, Any]:
    """Validate a remote export contract against locally downloaded bytes."""
    path = contract_path.expanduser().resolve()
    if not path.is_file():
        raise RuntimePairError(f"Runtime export contract was not found: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimePairError(f"Runtime export contract is invalid: {exc}") from exc
    if not isinstance(payload, dict):
        raise RuntimePairError("Runtime export contract must contain a JSON object.")
    if payload.get("schema_version") != 1:
        raise RuntimePairError("Unsupported runtime export contract schema.")
    if str(payload.get("runtime_format") or "").lower() != "onnx":
        raise RuntimePairError("Runtime export contract does not describe ONNX.")

    contracted_runtime_hash = str(payload.get("runtime_sha256") or "").lower()
    contracted_training_hash = str(
        payload.get("training_weight_sha256") or ""
    ).lower()
    if not _SHA256_RE.fullmatch(contracted_runtime_hash):
        raise RuntimePairError("Runtime export contract has an invalid ONNX checksum.")
    if not _SHA256_RE.fullmatch(contracted_training_hash):
        raise RuntimePairError("Runtime export contract has an invalid PT checksum.")
    if contracted_runtime_hash != verification.runtime_sha256:
        raise RuntimePairError("Downloaded ONNX checksum differs from export contract.")
    if contracted_training_hash != verification.training_weight_sha256:
        raise RuntimePairError("Downloaded PT checksum differs from export contract.")
    return payload


def deploy_runtime_pair(
    verification: PairVerification,
    *,
    inference_models_dir: Path,
    product: str,
    area: str,
    export_contract: Mapping[str, Any] | None = None,
    lock_timeout: float = 30.0,
) -> PairDeployment:
    """Atomically publish a verified runtime/checkpoint pair to one station.

    Large model copies are staged and checksummed before taking the station
    lock.  The configuration and manifest are replaced only after both model
    files are present, and the previous configuration is restored if manifest
    publication fails.
    """
    _validate_segment(product, "product")
    _validate_segment(area, "area")
    models_dir = inference_models_dir.expanduser().resolve()
    if not models_dir.is_dir():
        raise RuntimePairError(f"Inference models directory was not found: {models_dir}")
    station_dir = (models_dir / product / area / "yolo").resolve()
    if not station_dir.is_relative_to(models_dir):
        raise RuntimePairError("Deployment target escapes the inference models directory.")
    config_path = station_dir / "config.yaml"
    if not config_path.is_file():
        raise RuntimePairError(f"Station config was not found: {config_path}")
    config = _load_yaml_mapping(config_path, "station config")
    _validate_verification_files(verification)

    weights_dir = station_dir / "weights"
    versions_dir = station_dir / "versions"
    weights_dir.mkdir(parents=True, exist_ok=True)
    versions_dir.mkdir(parents=True, exist_ok=True)
    token = uuid.uuid4().hex
    staged_runtime = weights_dir / f".runtime-pair-{token}.onnx.tmp"
    staged_training = weights_dir / f".runtime-pair-{token}.pt.tmp"
    lock_dir: Path | None = None
    published_runtime: Path | None = None
    published_training: Path | None = None
    published_snapshot: Path | None = None
    deployment_succeeded = False
    try:
        _stage_verified_copy(
            verification.runtime_path,
            staged_runtime,
            verification.runtime_sha256,
        )
        _stage_verified_copy(
            verification.training_weight_path,
            staged_training,
            verification.training_weight_sha256,
        )
        lock_dir = _acquire_lock(station_dir, lock_timeout)
        _validate_verification_files(verification)

        version_tuple = _next_version(weights_dir, product, area)
        version = ".".join(str(part) for part in version_tuple)
        date_text = dt.datetime.now().strftime("%Y%m%d")
        runtime_name = f"{product}_{area}_v{version}_{date_text}.onnx"
        training_name = f"{Path(runtime_name).stem}.training.pt"
        published_runtime = weights_dir / runtime_name
        published_training = weights_dir / training_name
        if published_runtime.exists() or published_training.exists():
            raise RuntimePairError(
                "Resolved deployment version already exists; retry after the other "
                "deployment completes."
            )
        staged_runtime.replace(published_runtime)
        staged_training.replace(published_training)

        deployed_config = dict(config)
        deployed_config["weights"] = (
            f"models/{product}/{area}/yolo/weights/{runtime_name}"
        )
        published_snapshot = versions_dir / f"{runtime_name}.config.yaml"
        _write_yaml_atomic(published_snapshot, deployed_config)

        now = dt.datetime.now().astimezone().isoformat(timespec="seconds")
        source_kind = (
            "export_contract_and_numerical_equivalence"
            if export_contract is not None
            else "numerically_verified_legacy_import"
        )
        manifest = {
            "schema_version": 2,
            "deployed_version": version,
            "deployed_date": date_text,
            "deployed_at": now,
            "trained_at": dt.datetime.fromtimestamp(
                verification.training_weight_path.stat().st_mtime
            )
            .astimezone()
            .isoformat(timespec="seconds"),
            "deployed_file": runtime_name,
            "model_type": "yolo",
            "runtime_format": "onnx",
            "file_size": published_runtime.stat().st_size,
            "weight_sha256": verification.runtime_sha256,
            "training_weight_file": training_name,
            "training_weight_sha256": verification.training_weight_sha256,
            "runtime_source_training_weight": verification.training_weight_path.name,
            "deployed_to": str(config_path),
            "product": product,
            "area": area,
            "config_snapshot": f"versions/{published_snapshot.name}",
            "pair_verification": {
                "method": source_kind,
                "input_size": verification.input_size,
                **asdict(verification.comparison),
            },
        }
        previous_config = config_path.read_bytes()
        manifest_path = station_dir / "deployment_manifest.yaml"
        previous_manifest = manifest_path.read_bytes() if manifest_path.is_file() else None
        try:
            _write_yaml_atomic(config_path, deployed_config)
            _write_yaml_atomic(manifest_path, manifest)
        except (OSError, UnicodeError, yaml.YAMLError) as exc:
            _write_bytes_atomic(config_path, previous_config)
            if previous_manifest is None:
                manifest_path.unlink(missing_ok=True)
            else:
                _write_bytes_atomic(manifest_path, previous_manifest)
            raise RuntimePairError(f"Unable to publish deployment metadata: {exc}") from exc

        deployment = PairDeployment(
            version=version,
            runtime_path=published_runtime,
            training_weight_path=published_training,
            manifest_path=manifest_path,
            config_path=config_path,
        )
        deployment_succeeded = True
        return deployment
    finally:
        if not deployment_succeeded:
            for candidate in (
                published_runtime,
                published_training,
                published_snapshot,
            ):
                if candidate is not None:
                    candidate.unlink(missing_ok=True)
        staged_runtime.unlink(missing_ok=True)
        staged_training.unlink(missing_ok=True)
        if lock_dir is not None:
            _release_lock(lock_dir)


def _compare_yolo_outputs(
    runtime_path: Path,
    training_weight_path: Path,
    input_size: int,
    rtol: float,
    atol: float,
) -> NumericalComparison:
    """Compare raw YOLO outputs through OpenCV DNN and PyTorch."""
    try:
        import cv2  # type: ignore
        import numpy as np  # type: ignore
        import torch  # type: ignore

        os.environ.setdefault(
            "YOLO_CONFIG_DIR",
            str((Path.cwd() / ".tmp" / "ultralytics").resolve()),
        )
        from ultralytics import YOLO  # type: ignore
    except (ImportError, OSError) as exc:
        raise RuntimePairError(
            "Numerical pair verification requires ultralytics, torch, numpy, "
            f"and opencv-python: {exc}"
        ) from exc

    rng = np.random.default_rng(20260716)
    input_tensor = rng.random(
        (1, 3, input_size, input_size), dtype=np.float32
    )
    training_model: Any = YOLO(str(training_weight_path)).model
    if training_model is None or not hasattr(training_model, "float"):
        raise RuntimePairError("PT checkpoint did not expose a runnable model.")
    model = training_model.float().eval()
    with torch.inference_mode():
        training_output = model(torch.from_numpy(input_tensor))
    if isinstance(training_output, (tuple, list)):
        training_output = training_output[0]
    if not hasattr(training_output, "detach"):
        raise RuntimePairError("PT model returned an unsupported output type.")
    training_array = training_output.detach().cpu().numpy()

    network = cv2.dnn.readNetFromONNX(str(runtime_path))
    network.setInput(input_tensor)
    runtime_array = network.forward()
    if isinstance(runtime_array, (tuple, list)):
        if len(runtime_array) != 1:
            raise RuntimePairError("ONNX model returned multiple unsupported outputs.")
        runtime_array = runtime_array[0]

    runtime_shape = tuple(int(value) for value in runtime_array.shape)
    training_shape = tuple(int(value) for value in training_array.shape)
    if runtime_shape != training_shape:
        return NumericalComparison(
            runtime_shape=runtime_shape,
            training_shape=training_shape,
            max_abs_error=float("inf"),
            mean_abs_error=float("inf"),
            p99_abs_error=float("inf"),
            passed=False,
            class_names=_ordered_class_names(getattr(model, "names", {})),
        )
    absolute_error = np.abs(runtime_array - training_array)
    return NumericalComparison(
        runtime_shape=runtime_shape,
        training_shape=training_shape,
        max_abs_error=float(absolute_error.max()),
        mean_abs_error=float(absolute_error.mean()),
        p99_abs_error=float(np.quantile(absolute_error, 0.99)),
        passed=bool(np.allclose(runtime_array, training_array, rtol=rtol, atol=atol)),
        class_names=_ordered_class_names(getattr(model, "names", {})),
    )


def _ordered_class_names(names: Any) -> tuple[str, ...]:
    if isinstance(names, Mapping):
        try:
            indexed = {int(key): str(value) for key, value in names.items()}
            return tuple(indexed[index] for index in sorted(indexed))
        except (AttributeError, KeyError, TypeError, ValueError):
            return ()
    if isinstance(names, Sequence) and not isinstance(names, (str, bytes)):
        return tuple(str(name) for name in names)
    return ()


def _validate_source_artifact(path: Path, suffix: str, label: str) -> None:
    if path.suffix.lower() != suffix:
        raise RuntimePairError(f"{label.capitalize()} must use the {suffix} extension.")
    if not path.is_file():
        raise RuntimePairError(f"{label.capitalize()} was not found: {path}")
    if path.stat().st_size <= 0:
        raise RuntimePairError(f"{label.capitalize()} is empty: {path}")


def _validate_verification_files(verification: PairVerification) -> None:
    if not verification.comparison.passed:
        raise RuntimePairError("Pair verification did not pass numerical comparison.")
    if (
        verification.comparison.runtime_shape
        != verification.comparison.training_shape
    ):
        raise RuntimePairError("Pair verification contains different output shapes.")
    if not _SHA256_RE.fullmatch(verification.runtime_sha256):
        raise RuntimePairError("Verified ONNX checksum is invalid.")
    if not _SHA256_RE.fullmatch(verification.training_weight_sha256):
        raise RuntimePairError("Verified PT checksum is invalid.")
    if sha256_file(verification.runtime_path) != verification.runtime_sha256:
        raise RuntimePairError("Verified ONNX source changed before deployment.")
    if (
        sha256_file(verification.training_weight_path)
        != verification.training_weight_sha256
    ):
        raise RuntimePairError("Verified PT source changed before deployment.")


def _stage_verified_copy(source: Path, destination: Path, expected_hash: str) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    if sha256_file(destination) != expected_hash:
        raise RuntimePairError(f"Staged artifact checksum mismatch: {source.name}")


def _load_yaml_mapping(path: Path, label: str) -> dict[str, Any]:
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except (OSError, UnicodeDecodeError, yaml.YAMLError) as exc:
        raise RuntimePairError(f"Unable to read {label}: {exc}") from exc
    if not isinstance(payload, dict):
        raise RuntimePairError(f"{label.capitalize()} must contain a YAML mapping.")
    return payload


def _next_version(weights_dir: Path, product: str, area: str) -> tuple[int, int, int]:
    prefix = f"{product}_{area}_v"
    versions: list[tuple[int, int, int]] = []
    for candidate in weights_dir.glob(f"{prefix}*.onnx"):
        match = _VERSION_RE.search(candidate.name)
        if match:
            versions.append(
                (
                    int(match.group(1)),
                    int(match.group(2)),
                    int(match.group(3)),
                )
            )
    if not versions:
        return (1, 0, 0)
    major, minor, patch = max(versions)
    return (major, minor, patch + 1)


def _acquire_lock(station_dir: Path, timeout: float) -> Path:
    lock_dir = station_dir / ".deploy.lock"
    deadline = time.monotonic() + max(timeout, 0.0)
    while True:
        try:
            lock_dir.mkdir()
            return lock_dir
        except FileExistsError:
            if time.monotonic() >= deadline:
                raise RuntimePairError(f"Timed out waiting for deploy lock: {lock_dir}")
            time.sleep(0.2)


def _release_lock(lock_dir: Path) -> None:
    try:
        lock_dir.rmdir()
    except FileNotFoundError:
        return


def _write_yaml_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(
            yaml.safe_dump(dict(payload), allow_unicode=True, sort_keys=False),
            encoding="utf-8",
        )
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _write_bytes_atomic(path: Path, payload: bytes) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_bytes(payload)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _validate_segment(value: str, label: str) -> None:
    if not value or value in {".", ".."} or any(
        not (character.isalnum() or character in "._-") for character in value
    ):
        raise RuntimePairError(f"Invalid {label}: {value!r}")


def _default_inference_models_dir() -> Path:
    try:
        return WorkspacePaths.discover().inference_models
    except WorkspaceConfigurationError as exc:
        raise RuntimePairError(
            f"Unable to resolve the inference models directory: {exc}"
        ) from exc


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Verify and safely deploy a paired ONNX/PT YOLO model."
    )
    parser.add_argument("--weights", type=Path, required=True, help="ONNX runtime")
    parser.add_argument(
        "--training-weights", type=Path, required=True, help="paired PT checkpoint"
    )
    parser.add_argument("--contract", type=Path, help="runtime_export_manifest.json")
    parser.add_argument("--product", default="Cable1")
    parser.add_argument("--area", default="A")
    parser.add_argument(
        "--inference-models-dir",
        type=Path,
        help="deployment directory; defaults to the discovered inference workspace",
    )
    parser.add_argument("--imgsz", type=int, default=_DEFAULT_INPUT_SIZE)
    parser.add_argument(
        "--deploy",
        action="store_true",
        help="publish after validation; otherwise only validate",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    try:
        args = build_arg_parser().parse_args(argv)
        verification = verify_runtime_pair(
            args.weights,
            args.training_weights,
            input_size=args.imgsz,
        )
        contract = (
            validate_export_contract(args.contract, verification)
            if args.contract is not None
            else None
        )
        comparison = verification.comparison
        print(
            "ONNX/PT pair verified: "
            f"shape={comparison.runtime_shape}, "
            f"max_abs_error={comparison.max_abs_error:.6g}, "
            f"mean_abs_error={comparison.mean_abs_error:.6g}"
        )
        if not args.deploy:
            print("Validation only; production files were not changed.")
            return 0
        inference_models_dir = (
            args.inference_models_dir
            if args.inference_models_dir is not None
            else _default_inference_models_dir()
        )
        deployment = deploy_runtime_pair(
            verification,
            inference_models_dir=inference_models_dir,
            product=args.product,
            area=args.area,
            export_contract=contract,
        )
        print(
            f"Deployed {args.product}/{args.area} v{deployment.version}: "
            f"{deployment.runtime_path.name} + {deployment.training_weight_path.name}"
        )
        print(f"Deployment manifest: {deployment.manifest_path}")
        return 0
    except RuntimePairError as exc:
        print(f"ERROR: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

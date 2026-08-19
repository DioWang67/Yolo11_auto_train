import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional, MutableMapping

import os

try:
    if os.environ.get("PYTEST_IS_RUNNING") == "1":
        raise ImportError("Bypass")
    from ultralytics import YOLO  # type: ignore
except ImportError:
    YOLO = None  # type: ignore

from picture_tool.utils.normalization import normalize_imgsz

SUPPORTED_RUNTIME_FORMATS = {"onnx", "openvino", "engine", "torchscript"}


class OnnxExporter:
    @staticmethod
    def is_enabled(config: MutableMapping[str, Any]) -> bool:
        """Return whether the pipeline requires a runtime export."""
        ycfg = config.get("yolo_training", {})
        if not isinstance(ycfg, dict):
            return False
        export_cfg, _ = _select_export_config(ycfg)
        return bool(
            isinstance(export_cfg, dict)
            and export_cfg.get("enabled", True) is not False
        )

    @staticmethod
    def ensure(
        config: MutableMapping[str, Any], run_dir: Path, logger: logging.Logger
    ) -> Optional[Path]:
        """Return a current validated export, rebuilding stale artifacts.

        A training task may be skipped when its dataset/config hash is unchanged.
        Deployment still calls this method so a missing or stale ONNX lineage
        contract cannot silently fall back to the PT checkpoint.
        """
        ycfg = config.get("yolo_training", {})
        if not isinstance(ycfg, dict):
            return None
        export_cfg, legacy_onnx_cfg = _select_export_config(ycfg)
        if not isinstance(export_cfg, dict) or export_cfg.get("enabled", True) is False:
            return None

        export_format = str(export_cfg.get("format") or "onnx").lower()
        if legacy_onnx_cfg:
            export_format = "onnx"
        weights_name = str(export_cfg.get("weights_name") or "best.pt")
        current = _resolve_current_export(
            run_dir,
            runtime_format=export_format,
            training_weight_name=weights_name,
        )
        if current is not None:
            logger.info("Reusing validated YOLO %s export: %s", export_format, current)
            return current
        logger.info(
            "YOLO %s export is missing or stale; rebuilding from %s.",
            export_format,
            weights_name,
        )
        return OnnxExporter.export(config, run_dir, logger)

    @staticmethod
    def export(
        config: MutableMapping[str, Any], run_dir: Path, logger: logging.Logger
    ) -> Optional[Path]:
        """Export trained YOLO weights to a configured runtime artifact.

        Legacy configs can keep using ``yolo_training.export_onnx``. Newer
        deployment configs may use ``yolo_training.export_runtime`` with
        ``format: onnx|openvino|engine|torchscript``. ONNX keeps the stricter
        validation path because it is the existing production behavior.
        """
        ycfg = config.get("yolo_training", {})
        if not isinstance(ycfg, dict):
            return None
        export_cfg, legacy_onnx_cfg = _select_export_config(ycfg)
        if not isinstance(export_cfg, dict) or export_cfg.get("enabled", True) is False:
            return None

        export_format = str(export_cfg.get("format") or "onnx").lower()
        if legacy_onnx_cfg:
            export_format = "onnx"
        if export_format not in SUPPORTED_RUNTIME_FORMATS:
            supported = ", ".join(sorted(SUPPORTED_RUNTIME_FORMATS))
            logger.error("YOLO export skipped: unsupported format %r.", export_format)
            raise ValueError(
                f"Unsupported YOLO export format: {export_format}. "
                f"Use one of: {supported}"
            )
        if bool(export_cfg.get("half", False)) and bool(export_cfg.get("int8", False)):
            raise ValueError(
                "Use either half precision or int8 quantization, not both."
            )

        cwd = Path.cwd()

        # 1. Fail-fast dependency checks
        import importlib.util

        if export_format == "onnx" and importlib.util.find_spec("onnx") is None:
            logger.error("ONNX export failed: 'onnx' package not found.")
            raise ImportError(
                "ONNX export requires package onnx. Install via: pip install onnx"
            )

        # 2. Config & Fallback logic
        try:
            if YOLO is None:
                logger.warning("ONNX export skipped: ultralytics is not available.")
                return None

            weights_name = str(export_cfg.get("weights_name") or "best.pt")
            weights_path = (run_dir / "weights" / weights_name).resolve()
            if not weights_path.exists():
                logger.warning(
                    "ONNX export skipped: unable to find weights at %s", weights_path
                )
                return None

            imgsz = normalize_imgsz(export_cfg.get("imgsz"))
            if imgsz is None:
                imgsz = normalize_imgsz(ycfg.get("imgsz"))
            if imgsz is None:
                imgsz = [640, 640]
            imgsz_arg: Any = (
                imgsz[0] if len(imgsz) == 2 and imgsz[0] == imgsz[1] else imgsz
            )

            device = str(export_cfg.get("device") or ycfg.get("device") or "cpu")
            half = bool(export_cfg.get("half", False))
            dynamic = bool(export_cfg.get("dynamic", False))
            simplify = bool(export_cfg.get("simplify", False))
            int8 = bool(export_cfg.get("int8", False))

            # Check onnxsim availability if simplify requested
            if export_format == "onnx" and simplify:
                import importlib.util

                if importlib.util.find_spec("onnxsim") is None:
                    logger.warning(
                        "ONNX export: simplify=True requested but 'onnxsim' not found. "
                        "Falling back to simplify=False."
                    )
                    simplify = False

            export_kwargs: Dict[str, Any] = {
                "format": export_format,
                "imgsz": imgsz_arg,
                "device": device,
                "dynamic": dynamic,
            }
            if half:
                export_kwargs["half"] = True
            if int8:
                export_kwargs["int8"] = True
            if export_format == "onnx":
                export_kwargs["simplify"] = simplify

            opset_val = export_cfg.get("opset")
            if export_format == "onnx" and opset_val is not None:
                try:
                    export_kwargs["opset"] = int(opset_val)
                except (TypeError, ValueError):
                    logger.warning("YOLO export: invalid opset %r, ignoring", opset_val)

            # 3. Execution with detailed logging
            logger.info("Starting YOLO %s export from %s", export_format, cwd)
            logger.info(f"Export args: model={weights_path}, kwargs={export_kwargs}")

            try:
                model = YOLO(str(weights_path))
                result_path = model.export(**export_kwargs)
                logger.info(f"Ultralytics export returned: {result_path}")
            except (RuntimeError, OSError, AttributeError) as exc:
                logger.error("YOLO %s export runtime error: %s", export_format, exc)
                raise

            # 4. Path Resolution Strategy
            candidates = []

            # Candidate A: Return value from export
            if result_path:
                try:
                    candidates.append(Path(str(result_path)).resolve())
                except (TypeError, ValueError, OSError):
                    pass

            # Candidate B: Derived from weights path
            candidates.extend(_derived_export_candidates(weights_path, export_format))

            # Candidate C: Search in weights dir
            weights_dir = weights_path.parent
            if weights_dir.exists():
                found_artifacts = sorted(
                    _iter_export_artifacts(
                        weights_dir,
                        weights_path.stem,
                        export_format,
                    ),
                    key=lambda p: p.stat().st_mtime,
                    reverse=True,
                )
                candidates.extend([p.resolve() for p in found_artifacts])

            export_path: Optional[Path] = None
            seen = set()
            logger.info("Resolving YOLO %s export path from candidates:", export_format)
            for cand in candidates:
                if cand in seen:
                    continue
                seen.add(cand)
                exists = _artifact_exists(cand)
                status = "FOUND" if exists else "MISSING/EMPTY"
                logger.info(f"  - {cand} [{status}]")
                if export_path is None and exists:
                    export_path = cand

            if not export_path:
                msg = (
                    f"YOLO {export_format} export appeared to succeed but output "
                    f"artifact not found. Searched: {[str(c) for c in seen]}"
                )
                logger.error(msg)
                raise FileNotFoundError(msg)

            logger.info(
                "Resolved valid YOLO %s export path: %s",
                export_format,
                export_path,
            )

            # 5. Validation using helper
            if export_format != "onnx":
                _write_export_contract(
                    run_dir,
                    runtime_path=export_path,
                    training_weight_path=weights_path,
                    runtime_format=export_format,
                )
                return export_path

            try:
                from picture_tool.utils.onnx_validation import (
                    validate_onnx_structure,
                    validate_onnx_runtime,
                )

                # Structural validation (Fatal)
                validate_onnx_structure(export_path)

                # Runtime smoke test (Strict/Fatal since we want robust pipelines)
                validate_onnx_runtime(export_path, imgsz=imgsz, device=device)

            except (RuntimeError, OSError, ValueError) as val_err:
                logger.error(f"ONNX validation failed: {val_err}")
                # Treat validation failure as fatal
                raise RuntimeError(
                    f"ONNX validation failed for {export_path}"
                ) from val_err

            _write_export_contract(
                run_dir,
                runtime_path=export_path,
                training_weight_path=weights_path,
                runtime_format=export_format,
            )
            return export_path

        except (ImportError, FileNotFoundError, RuntimeError, OSError) as e:
            logger.exception("YOLO export process failed: %s", e)
            return None


def _write_export_contract(
    run_dir: Path,
    *,
    runtime_path: Path,
    training_weight_path: Path,
    runtime_format: str,
) -> Path:
    """Record the exact PT-to-runtime lineage used by safe deployment."""
    resolved_run_dir = run_dir.resolve()
    resolved_runtime = runtime_path.resolve()
    resolved_training_weight = training_weight_path.resolve()
    if not resolved_runtime.is_relative_to(resolved_run_dir):
        raise RuntimeError("Exported runtime artifact is outside the training run.")
    if not resolved_training_weight.is_relative_to(resolved_run_dir):
        raise RuntimeError("Export source training weight is outside the training run.")
    if resolved_training_weight.suffix.lower() != ".pt":
        raise RuntimeError("Export source training weight must be a .pt checkpoint.")

    payload = {
        "schema_version": 1,
        "runtime_format": runtime_format,
        "runtime_file": resolved_runtime.relative_to(resolved_run_dir).as_posix(),
        "runtime_sha256": _sha256_artifact(resolved_runtime),
        "training_weight_file": resolved_training_weight.relative_to(
            resolved_run_dir
        ).as_posix(),
        "training_weight_sha256": _sha256_artifact(resolved_training_weight),
    }
    contract_path = resolved_run_dir / "runtime_export_manifest.json"
    temporary = contract_path.with_name(f".{contract_path.name}.tmp")
    try:
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        temporary.replace(contract_path)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
    return contract_path


def _sha256_artifact(path: Path) -> str:
    """Hash a file or directory export deterministically."""
    digest = hashlib.sha256()
    if path.is_file():
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
    if not path.is_dir():
        raise FileNotFoundError(path)
    files = sorted(candidate for candidate in path.rglob("*") if candidate.is_file())
    if not files:
        raise RuntimeError(f"Runtime export directory is empty: {path}")
    for candidate in files:
        digest.update(candidate.relative_to(path).as_posix().encode("utf-8"))
        with candidate.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def _select_export_config(
    ycfg: dict[str, Any],
) -> tuple[Optional[dict[str, Any]], bool]:
    """Select the runtime export configuration to apply.

    Args:
        ycfg: The ``yolo_training`` config block.

    Returns:
        A tuple of ``(config, is_legacy_onnx_config)``. Disabled
        ``export_runtime`` blocks intentionally fall back to ``export_onnx``
        for backward compatibility with existing default configs.
    """
    runtime_cfg = ycfg.get("export_runtime")
    if isinstance(runtime_cfg, dict) and runtime_cfg.get("enabled", False):
        return runtime_cfg, False

    onnx_cfg = ycfg.get("export_onnx")
    if isinstance(onnx_cfg, dict):
        return onnx_cfg, True

    return None, False


def _resolve_current_export(
    run_dir: Path,
    *,
    runtime_format: str,
    training_weight_name: str,
) -> Optional[Path]:
    """Resolve an export only when its immutable lineage contract is current."""
    resolved_run_dir = run_dir.resolve()
    contract_path = resolved_run_dir / "runtime_export_manifest.json"
    if not contract_path.is_file():
        return None
    try:
        payload = json.loads(contract_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict) or payload.get("schema_version") != 1:
            return None
        if str(payload.get("runtime_format") or "").lower() != runtime_format:
            return None
        runtime_path = (
            resolved_run_dir / str(payload.get("runtime_file") or "")
        ).resolve()
        training_weight = (
            resolved_run_dir / str(payload.get("training_weight_file") or "")
        ).resolve()
        expected_training_weight = (
            resolved_run_dir / "weights" / Path(training_weight_name).name
        ).resolve()
        if not runtime_path.is_relative_to(resolved_run_dir):
            return None
        if not training_weight.is_relative_to(resolved_run_dir):
            return None
        if training_weight != expected_training_weight:
            return None
        if not _artifact_exists(runtime_path) or not training_weight.is_file():
            return None
        if _sha256_artifact(runtime_path) != str(
            payload.get("runtime_sha256") or ""
        ).lower():
            return None
        if _sha256_artifact(training_weight) != str(
            payload.get("training_weight_sha256") or ""
        ).lower():
            return None
        return runtime_path
    except (
        OSError,
        UnicodeDecodeError,
        json.JSONDecodeError,
        ValueError,
        TypeError,
    ):
        return None


def _derived_export_candidates(weights_path: Path, export_format: str) -> list[Path]:
    if export_format == "onnx":
        return [weights_path.with_suffix(".onnx").resolve()]
    if export_format == "openvino":
        return [(weights_path.parent / f"{weights_path.stem}_openvino_model").resolve()]
    if export_format == "engine":
        return [weights_path.with_suffix(".engine").resolve()]
    if export_format == "torchscript":
        return [weights_path.with_suffix(".torchscript").resolve()]
    return []


def _iter_export_artifacts(
    weights_dir: Path,
    stem: str,
    export_format: str,
) -> list[Path]:
    if export_format == "onnx":
        return list(weights_dir.glob("*.onnx"))
    if export_format == "openvino":
        return [p for p in weights_dir.glob(f"{stem}*_openvino_model") if p.is_dir()]
    if export_format == "engine":
        return list(weights_dir.glob("*.engine"))
    if export_format == "torchscript":
        return list(weights_dir.glob("*.torchscript"))
    return []


def _artifact_exists(path: Path) -> bool:
    if path.is_dir():
        try:
            return any(path.iterdir())
        except OSError:
            return False
    return path.exists() and path.stat().st_size > 0

import logging
from pathlib import Path
from typing import Any, Dict, Optional, MutableMapping

try:
    from ultralytics import YOLO # type: ignore
except ImportError:
    YOLO = None # type: ignore

from picture_tool.utils.normalization import normalize_imgsz

class OnnxExporter:
    @staticmethod
    def export(config: MutableMapping[str, Any], run_dir: Path, logger: logging.Logger) -> Optional[Path]:
        """Export trained weights to ONNX when enabled, with robust error handling and validation."""
        ycfg = config.get("yolo_training", {})
        if not isinstance(ycfg, dict):
            return None
        export_cfg = ycfg.get("export_onnx")
        if not isinstance(export_cfg, dict) or export_cfg.get("enabled", True) is False:
            return None
            
        cwd = Path.cwd()

        # 1. Fail-fast dependency checks
        import importlib.util
        if importlib.util.find_spec("onnx") is None:
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
            
            # Check onnxsim availability if simplify requested
            if simplify:
                import importlib.util
                if importlib.util.find_spec("onnxsim") is None:
                    logger.warning(
                        "ONNX export: simplify=True requested but 'onnxsim' not found. "
                        "Falling back to simplify=False."
                    )
                    simplify = False

            export_kwargs: Dict[str, Any] = {
                "format": "onnx",
                "imgsz": imgsz_arg,
                "device": device,
                "half": half,
                "dynamic": dynamic,
                "simplify": simplify,
            }

            opset_val = export_cfg.get("opset")
            if opset_val is not None:
                try:
                    export_kwargs["opset"] = int(opset_val)
                except (TypeError, ValueError):
                    logger.warning("ONNX export: invalid opset %r, ignoring", opset_val)

            # 3. Execution with detailed logging
            logger.info(f"Starting ONNX export from {cwd}")
            logger.info(f"Export args: model={weights_path}, kwargs={export_kwargs}")
            
            try:
                model = YOLO(str(weights_path))
                result_path = model.export(**export_kwargs)
                logger.info(f"Ultralytics export returned: {result_path}")
            except Exception as exc:  # Catch RuntimeError/Torchexport errors
                logger.error(f"ONNX export runtime error: {exc}")
                raise

            # 4. Path Resolution Strategy
            candidates = []
            
            # Candidate A: Return value from export
            if result_path:
                try:
                    candidates.append(Path(str(result_path)).resolve())
                except Exception:
                    pass
                    
            # Candidate B: Derived from weights path
            derived = weights_path.with_suffix(".onnx")
            candidates.append(derived.resolve())
            
            # Candidate C: Search in weights dir
            weights_dir = weights_path.parent
            if weights_dir.exists():
                found_onnx = sorted(weights_dir.glob("*.onnx"), key=lambda p: p.stat().st_mtime, reverse=True)
                candidates.extend([p.resolve() for p in found_onnx])

            export_path: Optional[Path] = None
            seen = set()
            logger.info("Resolving ONNX path from candidates:")
            for cand in candidates:
                if cand in seen:
                    continue
                seen.add(cand)
                exists = cand.exists() and cand.stat().st_size > 0
                status = "FOUND" if exists else "MISSING/EMPTY"
                logger.info(f"  - {cand} [{status}]")
                if export_path is None and exists:
                    export_path = cand
            
            if not export_path:
                msg = f"ONNX export appeared to succeed but output file not found. Searched: {[str(c) for c in seen]}"
                logger.error(msg)
                raise FileNotFoundError(msg)

            logger.info(f"Resolved valid ONNX path: {export_path}")

            # 5. Validation using helper
            try:
                from picture_tool.utils.onnx_validation import (
                    validate_onnx_structure,
                    validate_onnx_runtime,
                )
                
                # Structural validation (Fatal)
                validate_onnx_structure(export_path)
                
                # Runtime smoke test (Strict/Fatal since we want robust pipelines)
                validate_onnx_runtime(export_path, imgsz=imgsz, device=device)
                
            except Exception as val_err:
                logger.error(f"ONNX validation failed: {val_err}")
                # Treat validation failure as fatal
                raise RuntimeError(f"ONNX validation failed for {export_path}") from val_err

            return export_path

        except Exception as e:
            logger.exception(f"ONNX export process failed: {e}")
            return None

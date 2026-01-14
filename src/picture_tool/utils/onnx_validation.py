import logging
import importlib.util
from pathlib import Path
from typing import Optional, Sequence, Union, Any

logger = logging.getLogger(__name__)


def _is_package_available(pkg_name: str) -> bool:
    return importlib.util.find_spec(pkg_name) is not None


def validate_onnx_structure(onnx_path: Path) -> None:
    """
    Validate the structure of an ONNX model using onnx.checker.

    Args:
        onnx_path: Path to the .onnx file.

    Raises:
        ImportError: If onnx is not installed.
        ValueError: If file does not exist or has 0 size.
        Exception: If onnx.checker fails (propagates validation errors).
    """
    if not _is_package_available("onnx"):
        raise ImportError(
            "ONNX validation requires 'onnx' package. Install via: pip install onnx"
        )

    import onnx  # type: ignore

    onnx_path = onnx_path.resolve()
    if not onnx_path.exists():
        raise FileNotFoundError(f"ONNX file not found at: {onnx_path}")
    if onnx_path.stat().st_size == 0:
        raise ValueError(f"ONNX file is empty (0 bytes): {onnx_path}")

    logger.info(f"Validating ONNX model structure: {onnx_path}")
    try:
        model = onnx.load(str(onnx_path))
        onnx.checker.check_model(model)
        logger.info("ONNX structural validation passed.")
    except (ImportError, RuntimeError, AttributeError, OSError) as e:
        logger.error(f"ONNX structural validation failed: {e}")
        raise


def validate_onnx_runtime(
    onnx_path: Path,
    imgsz: Optional[Union[int, Sequence[int]]] = None,
    device: str = "cpu",
) -> bool:
    """
    Perform a runtime smoke test using onnxruntime.

    Args:
        onnx_path: Path to the .onnx file.
        imgsz: Training image size (e.g. 640 or [640, 640]). Used for fallback if dynamic.
        device: Device to run on (cpu/cuda).

    Returns:
        bool: True if validation passed or skipped, False if failed (though mostly raises).

    Raises:
        RuntimeError: If runtime smoke test fails.
    """
    if not _is_package_available("onnxruntime"):
        logger.warning("onnxruntime not installed, skipping runtime smoke test.")
        return True

    import onnxruntime as ort  # type: ignore
    import numpy as np  # type: ignore

    logger.info(f"Starting ONNX runtime smoke test for: {onnx_path}")

    try:
        # Create session
        providers = ["CPUExecutionProvider"]
        if (
            device == "cuda"
            and "CUDAExecutionProvider" in ort.get_available_providers()
        ):
            providers.insert(0, "CUDAExecutionProvider")

        session = ort.InferenceSession(str(onnx_path), providers=providers)

        # Introspect input
        input_meta = session.get_inputs()[0]
        input_name = input_meta.name
        input_shape = input_meta.shape
        input_type = input_meta.type  # e.g. 'tensor(float)'

        # Determine strict shape
        # Handle dynamic dimensions (None or string)
        # Assuming format [Batch, Channel, Height, Width]

        if imgsz is None:
            config_h, config_w = 640, 640
        elif isinstance(imgsz, int):
            config_h, config_w = imgsz, imgsz
        elif isinstance(imgsz, Sequence):
            if len(imgsz) >= 2:
                config_h, config_w = int(imgsz[0]), int(imgsz[1])
            else:
                config_h, config_w = int(imgsz[0]), int(imgsz[0])
        else:
            config_h, config_w = 640, 640

        dummy_shape = []
        for idx, dim in enumerate(input_shape):
            if isinstance(dim, (int, float)):
                dummy_shape.append(int(dim))
            else:
                # Dynamic dimension fallback
                if idx == 0:  # Batch
                    dummy_shape.append(1)
                elif idx == 1:  # Channel
                    dummy_shape.append(3)
                elif idx == 2:  # Height
                    dummy_shape.append(config_h)
                elif idx == 3:  # Width
                    dummy_shape.append(config_w)
                else:
                    dummy_shape.append(1)

        # Handle dtype
        numpy_dtype: Any = np.float32
        if "float16" in input_type:
            numpy_dtype = np.float16
        elif "uint8" in input_type:
            numpy_dtype = np.uint8

        logger.info(
            f"ONNX Input: name='{input_name}', shape={input_shape}, type={input_type}"
        )
        logger.info(f"Smoke test input: shape={dummy_shape}, dtype={numpy_dtype}")

        dummy_input = np.zeros(dummy_shape, dtype=numpy_dtype)

        # Inspect outputs
        output_meta_list = session.get_outputs()
        output_info = []
        for out_meta in output_meta_list:
            output_info.append(
                f"name='{out_meta.name}', shape={out_meta.shape}, type={out_meta.type}"
            )
        logger.info(f"ONNX Outputs: {'; '.join(output_info)}")

        # Run inference
        outputs = session.run(None, {input_name: dummy_input})

        if not outputs:
            raise RuntimeError("ONNX inference returned no outputs.")

        logger.info("ONNX runtime smoke test passed successfully.")
        return True

    except (ImportError, FileNotFoundError, RuntimeError, OSError) as e:
        logger.error(f"ONNX runtime smoke test failed: {e}")
        raise RuntimeError(f"ONNX runtime smoke test failed: {e}") from e

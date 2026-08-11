from __future__ import annotations

import io
import logging
import threading
from contextlib import asynccontextmanager
from pathlib import Path


try:
    from fastapi import FastAPI, File, HTTPException, UploadFile  # type: ignore
    from fastapi.concurrency import run_in_threadpool
    import uvicorn
except ImportError:
    FastAPI = None  # type: ignore
    File = None  # type: ignore
    HTTPException = RuntimeError  # type: ignore
    UploadFile = object  # type: ignore
    run_in_threadpool = None  # type: ignore
    uvicorn = None  # type: ignore

try:
    from PIL import Image
except ImportError:
    Image = None  # type: ignore

import os
try:
    if os.environ.get("PYTEST_IS_RUNNING") == "1":
        raise ImportError("Bypass ultralytics during pytest")
    from ultralytics import YOLO  # type: ignore
except ImportError:
    YOLO = None  # type: ignore

MODEL_INSTANCE = None
_MODEL_LOCK = threading.Lock()  # Thread-safe model loading


def load_model(model_path: str):
    """Load YOLO model with thread-safe locking.
    
    Args:
        model_path: Path to model weights file
        
    Raises:
        RuntimeError: If ultralytics is not installed
        FileNotFoundError: If model file doesn't exist
        OSError: If model file cannot be read
    """
    global MODEL_INSTANCE
    
    # Validate path before checking dependencies
    model_file = Path(model_path)
    if not model_file.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
        
    if YOLO is None:
        raise RuntimeError("ultralytics not installed")
    
    try:
        new_model = YOLO(model_path)
    except (RuntimeError, OSError) as e:
        logging.error(f"Failed to load model from {model_path}: {e}")
        raise

    with _MODEL_LOCK:
        MODEL_INSTANCE = new_model
        logging.info(f"Model loaded from {model_path}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup logic
    default_model = Path("runs/detect/train/weights/best.pt")
    if MODEL_INSTANCE is not None:
        logging.info("Using the model loaded by the service entrypoint.")
    elif default_model.exists():
        try:
            load_model(str(default_model))
        except (RuntimeError, FileNotFoundError, OSError) as e:
            logging.warning(f"Failed to auto-load default model: {e}")
    else:
        logging.warning(
            f"Default model not found at {default_model}. Use /load_model endpoint."
        )

    yield
    # Shutdown logic (if any)


app = (
    FastAPI(title="YOLO11 Inference Service", lifespan=lifespan)
    if FastAPI is not None
    else None
)


def health_check():
    return {"status": "ok", "model_loaded": MODEL_INSTANCE is not None}


def api_load_model(path: str):
    """Load a YOLO model from specified path.
    
    Args:
        path: File path to model weights
        
    Returns:
        Success status and message
        
    Raises:
        HTTPException: 404 if file not found, 500 if loading fails
    """
    try:
        load_model(path)
        return {"status": "success", "message": f"Loaded {path}"}
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except (RuntimeError, OSError) as e:
        logging.error(f"Model loading failed: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to load model: {str(e)}")


async def predict(file: UploadFile, conf: float = 0.25):
    """Run YOLO inference on uploaded image.
    
    Args:
        file: Uploaded image file
        conf: Confidence threshold (0.0-1.0)
        
    Returns:
        Detection results with bounding boxes and confidence scores
        
    Raises:
        HTTPException: 503 if model not loaded, 400 for invalid input, 500 for inference errors
    """
    # Input validation
    if not 0.0 <= conf <= 1.0:
        raise HTTPException(
            status_code=400, 
            detail=f"Confidence threshold must be between 0.0 and 1.0, got {conf}"
        )
    
    if MODEL_INSTANCE is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    if Image is None:
        raise HTTPException(status_code=500, detail="Pillow not installed")
    if run_in_threadpool is None:
        raise HTTPException(status_code=500, detail="FastAPI threadpool unavailable")

    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except (OSError, ValueError) as e:
        logging.warning(f"Invalid image file {file.filename}: {e}")
        raise HTTPException(status_code=400, detail=f"Invalid image file: {str(e)}")

    try:
        # Inference (CPU/GPU Bound) offloaded to ThreadPool to prevent Event Loop blocking
        results = await run_in_threadpool(MODEL_INSTANCE, image, conf=conf)

        # Format response
        detections = []
        for r in results:
            for box in r.boxes:
                detections.append(
                    {
                        "class": int(box.cls),
                        "conf": float(box.conf),
                        "bbox": box.xyxy[0].tolist(),  # [x1, y1, x2, y2]
                    }
                )

        return {"filename": file.filename, "detections": detections}
    except (RuntimeError, AttributeError) as e:
        logging.error(f"Prediction failed for {file.filename}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")


if app is not None:
    app.get("/health")(health_check)
    app.post("/load_model")(api_load_model)

    async def predict_endpoint(file: UploadFile = File(...), conf: float = 0.25):
        return await predict(file=file, conf=conf)

    app.post("/predict")(predict_endpoint)


def main():
    if FastAPI is None:
        print(
            "FastAPI/Uvicorn not installed. Please pip install fastapi uvicorn python-multipart"
        )
        return
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--model", help="Path to initial model weights")
    args = parser.parse_args()

    if args.model:
        try:
            load_model(args.model)
        except (FileNotFoundError, RuntimeError, OSError) as exc:
            parser.error(f"Unable to load initial model: {exc}")

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()

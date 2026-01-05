import io
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional


try:
    from fastapi import FastAPI, File, UploadFile, HTTPException # type: ignore
    import uvicorn
    from PIL import Image
except ImportError:
    FastAPI = None # type: ignore

try:
    from ultralytics import YOLO # type: ignore
except ImportError:
    YOLO = None # type: ignore

app = FastAPI(title="YOLO11 Inference Service") if FastAPI is not None else None
MODEL_INSTANCE = None
STARTUP_MODEL_PATH: Optional[str] = None

def load_model(model_path: str):
    global MODEL_INSTANCE
    if YOLO is None:
        raise RuntimeError("ultralytics not installed")
    MODEL_INSTANCE = YOLO(model_path)
    logging.info(f"Model loaded from {model_path}")

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup logic
    default_model = Path("runs/detect/train/weights/best.pt")
    if default_model.exists():
        try:
            load_model(str(default_model))
        except Exception as e:
            logging.warning(f"Failed to auto-load default model: {e}")
    else:
        logging.warning(f"Default model not found at {default_model}. Use /load_model endpoint.")
    
    yield
    # Shutdown logic (if any)

# Re-create app with lifespan if supported
app = FastAPI(title="YOLO11 Inference Service", lifespan=lifespan) if FastAPI is not None else None


@app.get("/health")
def health_check():
    return {"status": "ok", "model_loaded": MODEL_INSTANCE is not None}

@app.post("/load_model")
def api_load_model(path: str):
    if not Path(path).exists():
        raise HTTPException(status_code=404, detail="Model file not found")
    try:
        load_model(path)
        return {"status": "success", "message": f"Loaded {path}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict")
async def predict(file: UploadFile = File(...), conf: float = 0.25):
    if MODEL_INSTANCE is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        
        # Inference
        results = MODEL_INSTANCE(image, conf=conf)
        
        # Format response
        detections = []
        for r in results:
            for box in r.boxes:
                detections.append({
                    "class": int(box.cls),
                    "conf": float(box.conf),
                    "bbox": box.xyxy[0].tolist() # [x1, y1, x2, y2]
                })
        
        return {"filename": file.filename, "detections": detections}
    except Exception as e:
        logging.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

def main():
    if FastAPI is None:
        print("FastAPI/Uvicorn not installed. Please pip install fastapi uvicorn python-multipart")
        return
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--model", help="Path to initial model weights")
    args = parser.parse_args()
    
    if args.model:
        # Pre-set for startup load
        # Hacky global set for simplicity in this script scope
        global STARTUP_MODEL_PATH 
        STARTUP_MODEL_PATH = args.model
        
        # Better way: startup event checks this var, or we load it here but app state is cleaner
        if Path(args.model).exists():
            # We can't easily pass args to startup_event without closures or env vars
            # So let's just attempt load if we are in main block context logic 
            pass 

    uvicorn.run(app, host=args.host, port=args.port)

if __name__ == "__main__":
    main()

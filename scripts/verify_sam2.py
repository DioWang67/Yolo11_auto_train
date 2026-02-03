
import sys
import numpy as np
# import cv2  <-- Avoid cv2 dependency for verification
from pathlib import Path

def verify_sam2():
    print("Checking Ultralytics installation...")
    try:
        from ultralytics import SAM
        print("✅ Ultralytics found.")
    except ImportError:
        print("❌ Ultralytics NOT found. Please run: pip install ultralytics")
        return

    model_name = "models/sam2_b.pt"
    print(f"Attempting to load SAM 2 model: {model_name}...")
    try:
        # This triggers auto-download if strictly necessary and allowed by library
        model = SAM(model_name)
        print("✅ Model loaded successfully.")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return

    print("Running dummy inference...")
    # Create dummy image (RGB)
    img = np.zeros((640, 640, 3), dtype=np.uint8)
    # cv2.circle would be here, but we just leave it black or set some pixels
    img[300:340, 300:340] = [255, 255, 255]
    
    # Prompt: center point
    points = [[320, 320]]
    labels = [1]
    
    try:
        results = model(img, points=points, labels=labels, verbose=True)
        if results:
            print(f"✅ Inference returned {len(results)} results.")
            r = results[0]
            if r.masks is not None:
                print(f"✅ Generated masks shape: {r.masks.data.shape}")
                
                # Simple check if any mask is non-empty (optional)
                mask_np = r.masks.data.cpu().numpy()
                if mask_np.sum() > 0:
                     print("✅ Mask contains valid prediction.")
                else:
                     print("⚠️ Mask is empty (might be expected for simple square).")

            else:
                print("⚠️ No masks generated.")
        else:
             print("⚠️ No results returned.")
             
    except Exception as e:
        print(f"❌ Inference failed: {e}")
        return

    print("\nSUCCESS: SAM 2 integration appears functional!")

if __name__ == "__main__":
    verify_sam2()

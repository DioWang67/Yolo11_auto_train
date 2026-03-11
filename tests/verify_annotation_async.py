import sys
import shutil
from pathlib import Path
from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import QTimer

# Add src to path
sys.path.append("src")

# Mock cv2 and other potential heavy dependencies

from unittest.mock import MagicMock  # noqa: E402

# Mock internal modules that trigger heavy imports
sys.modules["picture_tool.anomaly"] = MagicMock()
sys.modules["picture_tool.augment"] = MagicMock()
sys.modules["picture_tool.format"] = MagicMock()
sys.modules["picture_tool.main_pipeline"] = MagicMock()
sys.modules["picture_tool.split"] = MagicMock()
sys.modules["picture_tool.gui.pipeline_manager"] = MagicMock()

from picture_tool.gui.annotation_panel import AnnotationWorker  # noqa: E402
from picture_tool.gui.annotation_tracker import AnnotationTracker  # noqa: E402

def verify_async_worker():
    app = QApplication(sys.argv)
    
    # Setup dummy data
    test_dir = Path("temp_test_async")
    img_dir = test_dir / "images"
    lbl_dir = test_dir / "labels"
    
    if test_dir.exists():
        shutil.rmtree(test_dir)
    
    img_dir.mkdir(parents=True)
    lbl_dir.mkdir(parents=True)
    
    # Create dummy files
    for i in range(50):
        (img_dir / f"img_{i}.jpg").touch()
        if i % 2 == 0:
            (lbl_dir / f"img_{i}.txt").touch()
            
    print("Created 50 dummy images.")

    tracker = AnnotationTracker()
    worker = AnnotationWorker(tracker, img_dir, lbl_dir)
    
    def on_progress(current, total):
        print(f"Progress: {current}/{total}")
        
    def on_completed(stats):
        print("\nScan Completed!")
        print(f"Total: {stats['total_images']}")
        print(f"Annotated: {stats['annotated_images']}")
        print(f"Progress: {stats['progress_percent']:.1f}%")
        
        # Verify results
        assert stats['total_images'] == 50
        assert stats['annotated_images'] == 25
        print("VERIFICATION SUCCESS: Stats match expected values.")
        app.quit()
        
    def on_error(err):
        print(f"ERROR: {err}")
        app.quit()

    worker.progress_updated.connect(on_progress)
    worker.scan_completed.connect(on_completed)
    worker.error_occurred.connect(on_error)
    
    print("Starting worker...")
    worker.start()
    
    # Timeout after 5 seconds
    QTimer.singleShot(5000, app.quit)
    
    app.exec_()
    
    # Cleanup
    try:
        shutil.rmtree(test_dir)
    except Exception:
        pass

if __name__ == "__main__":
    verify_async_worker()

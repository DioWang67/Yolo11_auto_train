from pathlib import Path
from PyQt5.QtWidgets import QApplication

def load_stylesheet(app: QApplication, filename: str = "style.qss") -> None:
    """Load QSS stylesheet from resources directory."""
    try:
        # Determine path relative to this file
        current_dir = Path(__file__).parent.resolve()
        style_path = current_dir / "resources" / filename
        
        if style_path.exists():
            with open(style_path, "r", encoding="utf-8") as f:
                style_content = f.read()
                app.setStyleSheet(style_content)
        else:
            print(f"[WARNING] Stylesheet not found: {style_path}")
            
    except Exception as e:
        print(f"[ERROR] Failed to load stylesheet: {e}")

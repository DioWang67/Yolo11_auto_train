from pathlib import Path
from typing import Iterable, List


DEFAULT_IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")


def list_images(directory: Path, exts: Iterable[str] = DEFAULT_IMAGE_EXTS) -> List[str]:
    exts_set = {e.lower() if e.startswith(".") else f".{e.lower()}" for e in exts}
    if not directory.exists():
        return []
    files = [
        p.name
        for p in directory.iterdir()
        if p.is_file() and p.suffix.lower() in exts_set
    ]
    files.sort()
    return files

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np


def _check_import(name: str) -> Tuple[bool, str]:
    try:
        __import__(name)
        return True, ""
    except Exception as exc:
        return False, str(exc)


def _check_command(cmd: List[str]) -> Tuple[bool, str]:
    try:
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        return True, ""
    except Exception as exc:
        return False, str(exc)


def _create_demo_dataset(root: Path) -> Dict[str, str]:
    root.mkdir(parents=True, exist_ok=True)
    img_dir = root / "images"
    label_dir = root / "labels"
    img_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)
    for idx, color in enumerate([(0, 0, 255), (0, 255, 0), (255, 0, 0)]):
        img = np.zeros((320, 320, 3), dtype=np.uint8)
        cv2.rectangle(img, (40, 40), (280, 280), color, thickness=-1)
        img_path = img_dir / f"demo_{idx}.jpg"
        cv2.imwrite(str(img_path), img)
        # YOLO format: class x_center y_center width height (normalized)
        label_path = label_dir / f"demo_{idx}.txt"
        label_path.write_text("0 0.5 0.5 0.6 0.6\n", encoding="utf-8")
    return {"images": str(img_dir), "labels": str(label_dir)}


def run_doctor(create_demo: bool = False) -> int:
    results: Dict[str, Tuple[bool, str]] = {}
    results["python"] = (True, sys.version)
    for pkg in ["torch", "ultralytics", "yaml", "cv2"]:
        ok, msg = _check_import(pkg)
        results[pkg] = (ok, msg)
    results["ffmpeg"] = _check_command(["ffmpeg", "-version"])
    results["onnxruntime"] = _check_import("onnxruntime")

    print("\n[ picture-tool doctor ]")
    for name, (ok, msg) in results.items():
        status = "OK" if ok else "MISSING"
        extra = f" - {msg}" if msg else ""
        print(f"{name:12}: {status}{extra}")

    demo_info = {}
    if create_demo:
        demo_info = _create_demo_dataset(Path("data/demo_doctor"))
        print("\nCreated demo dataset under data/demo_doctor (images/labels).")

    missing = [k for k, v in results.items() if not v[0]]
    if missing:
        print(f"\nMissing/failed components: {missing}")
        return 1
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Environment checker for picture-tool.")
    parser.add_argument(
        "--create-demo",
        action="store_true",
        help="Generate a tiny synthetic dataset at data/demo_doctor for quick tests.",
    )
    args = parser.parse_args()
    code = run_doctor(create_demo=args.create_demo)
    sys.exit(code)


if __name__ == "__main__":
    main()

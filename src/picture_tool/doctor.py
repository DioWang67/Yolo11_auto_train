from __future__ import annotations

import argparse
import importlib.metadata
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Tuple

SUPPORTED_PYTHON = ((3, 10), (3, 11))
EXPECTED_DISTRIBUTIONS = {
    "torch": ("2.4.1", "2.5.1"),
    "torchvision": ("0.19.1", "0.20.1"),
    "onnxruntime": ("1.23.2",),
    "PyQt5": ("5.15.11",),
    "jsonargparse": ("4.34.0",),
    "numpy": ("1.26.4", "2.2.6"),
    "opencv-python": ("4.9.0.80", "4.10.0.84"),
    "ultralytics": ("8.3.156", "8.3.199"),
}
RUNTIME_PROFILES = (
    {
        "python": ((3, 10), (3, 11)),
        "torch": "2.4.1",
        "torchvision": "0.19.1",
        "numpy": "1.26.4",
        "opencv-python": "4.9.0.80",
        "ultralytics": "8.3.156",
    },
    {
        "python": ((3, 11),),
        "torch": "2.5.1",
        "torchvision": "0.20.1",
        "numpy": "2.2.6",
        "opencv-python": "4.10.0.84",
        "ultralytics": "8.3.199",
    },
)


def _check_import(name: str) -> Tuple[bool, str]:
    try:
        __import__(name)
        return True, ""
    except (ImportError, OSError, RuntimeError) as exc:
        return False, str(exc)


def _check_distribution_version(
    name: str,
    expected: str | Tuple[str, ...],
) -> Tuple[bool, str]:
    expected_versions = (expected,) if isinstance(expected, str) else expected
    try:
        actual = importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return False, f"distribution is not installed; expected={expected_versions}"
    return (
        actual in expected_versions,
        f"expected={expected_versions} actual={actual}",
    )


def _check_runtime_profile() -> Tuple[bool, str]:
    actual_python = sys.version_info[:2]
    try:
        versions = {
            name: importlib.metadata.version(name)
            for name in (
                "torch",
                "torchvision",
                "numpy",
                "opencv-python",
                "ultralytics",
            )
        }
    except importlib.metadata.PackageNotFoundError as exc:
        return False, f"missing distribution: {exc}"
    for index, profile in enumerate(RUNTIME_PROFILES, start=1):
        if actual_python not in profile["python"]:
            continue
        if all(versions[name] == profile[name] for name in versions):
            return True, f"profile={index}"
    return False, f"unsupported version combination: python={actual_python} {versions}"


def _check_command(cmd: List[str]) -> Tuple[bool, str]:
    try:
        subprocess.run(
            cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True
        )
        return True, ""
    except FileNotFoundError:
        return False, f"Command '{cmd[0]}' not found. Is it installed and in PATH?"
    except subprocess.CalledProcessError as exc:
        return False, f"Command '{cmd[0]}' failed with exit code {exc.returncode}: {exc}"
    except (OSError, RuntimeError) as exc:
        return False, f"Error running command '{cmd[0]}': {exc}"


def _create_demo_dataset(root: Path) -> Dict[str, str]:
    import cv2
    import numpy as np

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


def run_doctor(
    create_demo: bool = False,
    *,
    runtime_only: bool = False,
) -> int:
    results: Dict[str, Tuple[bool, str]] = {}
    results["python"] = (
        sys.version_info[:2] in SUPPORTED_PYTHON,
        (
            "expected=3.10 or 3.11 "
            f"actual={sys.version.split()[0]} executable={sys.executable}"
        ),
    )
    for pkg in [
        "torch",
        "torchvision",
        "ultralytics",
        "yaml",
        "cv2",
        "PyQt5.QtCore",
        "onnxruntime",
        "jsonargparse",
    ]:
        ok, msg = _check_import(pkg)
        results[pkg] = (ok, msg)
    for distribution, expected in EXPECTED_DISTRIBUTIONS.items():
        results[f"{distribution} version"] = _check_distribution_version(
            distribution,
            expected,
        )
    results["runtime profile"] = _check_runtime_profile()
    if not runtime_only:
        results["ffmpeg"] = _check_command(["ffmpeg", "-version"])

    print("\n[ picture-tool doctor ]")
    for name, (ok, msg) in results.items():
        status = "OK" if ok else "MISSING"
        extra = f" - {msg}" if msg else ""
        print(f"{name:12}: {status}{extra}")

    if create_demo:
        demo_root = Path("data/demo_doctor")
        _create_demo_dataset(demo_root)
        print("\nCreated demo dataset under data/demo_doctor (images/labels).")

    missing = [k for k, v in results.items() if not v[0]]
    if missing:
        print(f"\nMissing/failed components: {missing}")
        return 1
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Environment checker for picture-tool."
    )
    parser.add_argument(
        "--create-demo",
        action="store_true",
        help="Generate a tiny synthetic dataset at data/demo_doctor for quick tests.",
    )
    parser.add_argument(
        "--runtime-only",
        action="store_true",
        help="Check only components required to launch the training GUI.",
    )
    args = parser.parse_args()
    code = run_doctor(
        create_demo=args.create_demo,
        runtime_only=args.runtime_only,
    )
    sys.exit(code)


if __name__ == "__main__":
    main()

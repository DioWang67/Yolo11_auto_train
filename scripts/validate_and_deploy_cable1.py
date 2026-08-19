#!/usr/bin/env python
"""CLI wrapper for the guarded Cable1 ONNX/PT deployment workflow."""

from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from picture_tool.runtime_pair_deployment import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())

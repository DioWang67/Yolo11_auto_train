"""Top-level package bootstrap to expose nested modules as picture_tool.*"""
from __future__ import annotations

import importlib
import sys
from types import ModuleType

_SUBMODULES = [
    "utils",
    "anomaly",
    "augment",
    "color",
    "eval",
    "format",
    "infer",
    "position",
    "quality",
    "report",
    "split",
    "train",
]

__all__ = sorted(_SUBMODULES)


def _load_and_alias(name: str) -> ModuleType:
    module = importlib.import_module(f"{__name__}.picture_tool.{name}")
    alias = f"{__name__}.{name}"
    sys.modules[alias] = module
    globals()[name] = module
    return module

for _mod in _SUBMODULES:
    try:
        _load_and_alias(_mod)
    except ModuleNotFoundError as exc:
        print(f"[picture_tool] failed to import submodule {_mod}: {exc}")


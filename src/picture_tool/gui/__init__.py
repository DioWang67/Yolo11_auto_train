"""PyQt-based GUI utilities for the picture_tool project.

To avoid double-import warnings when launching with
``python -m picture_tool.gui.app``, objects that live in ``app`` are loaded
lazily via ``__getattr__``.
"""

from .pipeline_controller import PipelineControllerMixin
from .task_thread import WorkerThread

__all__ = [
    "PictureToolGUI",
    "PipelineControllerMixin",
    "WorkerThread",
    "main",
]


def __getattr__(name):
    if name in {"PictureToolGUI", "main"}:
        from . import app  # local import to avoid eager execution

        return getattr(app, name)
    raise AttributeError(f"module 'picture_tool.gui' has no attribute {name!r}")

"""PyQt-based GUI utilities for the picture_tool project.

To avoid double-import warnings when launching with
``python -m picture_tool.gui.app``, objects that live in ``app`` are loaded
lazily via ``__getattr__``.
"""

from .pipeline_manager import PipelineManager
from .task_thread import WorkerThread

__all__ = [
    "MainWindow",
    "PipelineManager",
    "WorkerThread",
    "main",
]


def __getattr__(name):
    if name == "MainWindow":
        from .main_window import MainWindow

        return MainWindow
    if name == "main":
        from . import app  # local import to avoid eager execution

        return getattr(app, name)
    raise AttributeError(f"module 'picture_tool.gui' has no attribute {name!r}")

"""Few-shot dataset bootstrapping: a POC, kept out of the AutoTrain core.

Nothing in :mod:`picture_tool.autotrain` imports this package. It consumes
the same profiles and produces a dataset the existing trainer accepts, so
the experiment can be deleted without leaving a mark on the path that is
already working.
"""

from __future__ import annotations

__all__ = ["profile", "evidence"]

"""Autonomous training: a parallel, opt-in path beside the production flow.

Nothing in the production inference project imports this package, and nothing
here writes into it. The subsystem reads production inspection records, keeps
its own candidate pool and dataset versions, trains *challenger* models, and
compares them against the deployed champion. It never deploys.

Imports here stay deliberately light. ``picture_tool.gui.__init__`` eagerly
pulls in ``main_pipeline``, and that cost has bitten this repository before;
submodules are imported by the callers that need them instead.
"""

from __future__ import annotations

__all__ = ["AutoTrainDisabledError", "AutoTrainError"]


class AutoTrainError(Exception):
    """Base error for the autonomous training subsystem."""


class AutoTrainDisabledError(AutoTrainError):
    """Raised when an entry point runs while ``enabled`` is false.

    This is a domain error, not a failure: it is the expected outcome of the
    feature flag being off, and callers report it rather than retrying.
    """

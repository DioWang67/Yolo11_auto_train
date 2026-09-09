"""Application-wide stylesheet installation."""

from __future__ import annotations

import logging

from PyQt5.QtWidgets import QApplication

from picture_tool.gui.theme import build_stylesheet

logger = logging.getLogger(__name__)


def load_stylesheet(app: QApplication) -> None:
    """Install the application stylesheet on ``app``.

    The sheet is rendered from :mod:`picture_tool.gui.theme` rather than read
    from a ``.qss`` file. The file-based version could not report a real
    failure: a missing stylesheet printed a warning to stdout -- where nobody
    running the GUI would see it -- and then left the application running with
    Qt's default styling, which looks like a plainer theme rather than like a
    fault. Building the sheet in-process removes that state entirely; there is
    no longer an I/O step that can half-succeed.
    """
    app.setStyleSheet(build_stylesheet())
    logger.debug("Application stylesheet installed.")

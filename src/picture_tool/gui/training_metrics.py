"""Real-time training metrics parser and widget.

Parses ultralytics epoch log lines and displays epoch progress + key
losses in a compact QWidget that auto-shows when training metrics are
detected in the log stream.
"""

from __future__ import annotations

import re
from typing import Optional

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QVBoxLayout,
    QWidget,
)


class TrainingMetricsParser:
    """Parses ultralytics YOLO training log lines for epoch metrics.

    Ultralytics v8/11 logs epoch summary lines in a tabular format::

        Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
          5/100      3.66G    0.6543    0.5432    0.1234       320        640

    This parser extracts epoch, total_epochs, and loss values from these
    lines, tolerating the ``%(asctime)s - %(levelname)s - %(message)s``
    wrapper added by the GUI log handler.
    """

    # Matches "5/100" or " 5/100" at start of actual message content
    _EPOCH_RE = re.compile(
        r"(\d+)/(\d+)\s+"        # epoch/total
        r"[\d.]+\w*\s+"          # GPU mem (e.g. 3.66G or 0)
        r"([\d.]+)\s+"           # box_loss
        r"([\d.]+)\s+"           # cls_loss
        r"([\d.]+)"              # dfl_loss
    )

    @staticmethod
    def parse_epoch_line(line: str) -> Optional[dict]:
        """Parse an ultralytics epoch log line.

        Returns:
            dict with keys ``epoch``, ``total_epochs``, ``box_loss``,
            ``cls_loss``, ``dfl_loss`` — or ``None`` if the line is not
            an epoch summary.
        """
        match = TrainingMetricsParser._EPOCH_RE.search(line)
        if not match:
            return None
        try:
            return {
                "epoch": int(match.group(1)),
                "total_epochs": int(match.group(2)),
                "box_loss": float(match.group(3)),
                "cls_loss": float(match.group(4)),
                "dfl_loss": float(match.group(5)),
            }
        except (ValueError, IndexError):
            return None


class TrainingMetricsWidget(QWidget):
    """Compact widget displaying real-time YOLO training metrics.

    Hidden by default; call :meth:`update_metrics` to show and populate.
    """

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setVisible(False)
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 4, 0, 4)
        layout.setSpacing(4)

        # Epoch progress bar
        self.epoch_bar = QProgressBar()
        self.epoch_bar.setTextVisible(True)
        self.epoch_bar.setFormat("Epoch %v / %m")
        self.epoch_bar.setMaximumHeight(20)
        layout.addWidget(self.epoch_bar)

        # Loss labels row
        loss_row = QHBoxLayout()
        loss_row.setSpacing(16)
        self.box_loss_label = QLabel("box: --")
        self.cls_loss_label = QLabel("cls: --")
        self.dfl_loss_label = QLabel("dfl: --")
        for lbl in (self.box_loss_label, self.cls_loss_label, self.dfl_loss_label):
            lbl.setStyleSheet("color: #b5b5b5; font-size: 9pt; font-family: Consolas;")
            loss_row.addWidget(lbl)
        loss_row.addStretch()
        layout.addLayout(loss_row)

    def update_metrics(self, metrics: dict) -> None:
        """Update displayed metrics from a parsed epoch dict."""
        epoch = metrics.get("epoch", 0)
        total = metrics.get("total_epochs", 1)
        self.epoch_bar.setMaximum(total)
        self.epoch_bar.setValue(epoch)
        self.box_loss_label.setText(f"box: {metrics.get('box_loss', 0):.4f}")
        self.cls_loss_label.setText(f"cls: {metrics.get('cls_loss', 0):.4f}")
        self.dfl_loss_label.setText(f"dfl: {metrics.get('dfl_loss', 0):.4f}")

    def reset(self) -> None:
        """Reset to initial hidden state."""
        self.setVisible(False)
        self.epoch_bar.setValue(0)
        self.box_loss_label.setText("box: --")
        self.cls_loss_label.setText("cls: --")
        self.dfl_loss_label.setText("dfl: --")

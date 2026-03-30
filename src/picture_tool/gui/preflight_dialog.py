"""Pre-flight check result dialog.

Shows ``PreflightIssue`` items before the pipeline runs:
* ERRORs are shown in red and block execution — only a "Close" button is shown.
* WARNINGs are shown in amber; a "Continue Anyway" + "Cancel" pair lets the
  user decide whether to proceed.
* If the issue list is empty the caller should not show this dialog at all.
"""

from __future__ import annotations

from typing import List

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from picture_tool.pipeline.preflight import PreflightIssue, Severity


_ICON = {Severity.ERROR: "✗", Severity.WARNING: "⚠"}
_COLOR = {Severity.ERROR: "#c0392b", Severity.WARNING: "#d68910"}
_BG = {Severity.ERROR: "#fdf2f2", Severity.WARNING: "#fefce8"}
_BORDER = {Severity.ERROR: "#e74c3c", Severity.WARNING: "#f1c40f"}


class PreflightDialog(QDialog):
    """Modal dialog that presents preflight issues and lets the user decide."""

    def __init__(self, issues: list[PreflightIssue], parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._issues = issues
        self._has_errors = any(i.is_blocking for i in issues)
        self._setup_ui()

    def _setup_ui(self) -> None:
        self.setWindowTitle("執行前檢查")
        self.setMinimumWidth(560)
        self.setWindowFlags(self.windowFlags() & ~Qt.WindowContextHelpButtonHint)

        outer = QVBoxLayout(self)
        outer.setSpacing(12)

        # ── Title ──────────────────────────────────────────────────────
        title_text = (
            "發現問題，無法執行" if self._has_errors
            else "發現警告，請確認後繼續"
        )
        title = QLabel(title_text)
        title.setFont(QFont("Microsoft JhengHei", 11, QFont.Bold))
        title.setStyleSheet(
            "color: #c0392b;" if self._has_errors else "color: #b7770d;"
        )
        outer.addWidget(title)

        # ── Scrollable issue list ───────────────────────────────────────
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setMaximumHeight(360)

        content = QWidget()
        content_layout = QVBoxLayout(content)
        content_layout.setSpacing(8)

        for issue in self._issues:
            content_layout.addWidget(self._make_issue_card(issue))

        content_layout.addStretch()
        scroll.setWidget(content)
        outer.addWidget(scroll)

        # ── Buttons ────────────────────────────────────────────────────
        if self._has_errors:
            # Only allow closing — execution is blocked
            btn_box = QDialogButtonBox(QDialogButtonBox.Close)
            btn_box.rejected.connect(self.reject)
        else:
            btn_box = QDialogButtonBox()
            continue_btn = btn_box.addButton("繼續執行", QDialogButtonBox.AcceptRole)
            cancel_btn = btn_box.addButton("取消", QDialogButtonBox.RejectRole)
            continue_btn.setStyleSheet(
                "QPushButton { background-color: #e67e22; color: white; "
                "border-radius: 4px; padding: 6px 16px; font-weight: bold; }"
                "QPushButton:hover { background-color: #ca6f1e; }"
            )
            btn_box.accepted.connect(self.accept)
            btn_box.rejected.connect(self.reject)

        outer.addWidget(btn_box)

    @staticmethod
    def _make_issue_card(issue: PreflightIssue) -> QFrame:
        sev = issue.severity
        card = QFrame()
        card.setStyleSheet(
            f"QFrame {{ background-color: {_BG[sev]}; "
            f"border: 1px solid {_BORDER[sev]}; border-radius: 4px; }}"
        )
        card.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)

        row = QHBoxLayout(card)
        row.setContentsMargins(10, 8, 10, 8)
        row.setSpacing(10)

        icon = QLabel(_ICON[sev])
        icon.setFont(QFont("Segoe UI", 12, QFont.Bold))
        icon.setStyleSheet(f"color: {_COLOR[sev]}; background: transparent; border: none;")
        icon.setFixedWidth(18)
        icon.setAlignment(Qt.AlignTop | Qt.AlignHCenter)
        row.addWidget(icon)

        col = QVBoxLayout()
        col.setSpacing(2)

        task_lbl = QLabel(f"[{issue.task}]")
        task_lbl.setFont(QFont("Consolas", 8))
        task_lbl.setStyleSheet(f"color: {_COLOR[sev]}; background: transparent; border: none;")
        col.addWidget(task_lbl)

        msg_lbl = QLabel(issue.message)
        msg_lbl.setWordWrap(True)
        msg_lbl.setStyleSheet("color: #1a1a1a; background: transparent; border: none; font-size: 9pt;")
        col.addWidget(msg_lbl)

        row.addLayout(col, stretch=1)
        return card

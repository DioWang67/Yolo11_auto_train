"""Pre-flight check result dialog.

Shows ``PreflightIssue`` items before the pipeline runs:
* ERRORs are shown in red and block execution — only a "Close" button is shown.
* WARNINGs are shown in amber; a "Continue Anyway" + "Cancel" pair lets the
  user decide whether to proceed.
* If the issue list is empty the caller should not show this dialog at all.
"""

from __future__ import annotations

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

from picture_tool.gui.theme import (
    FONT_MONO,
    FONT_SIZE_SMALL,
    RADIUS_INSET,
    STATUS_NG,
    STATUS_WARN,
    TEXT_PRIMARY,
    SemanticColor,
)
from picture_tool.pipeline.preflight import PreflightIssue, Severity


_ICON = {Severity.ERROR: "✗", Severity.WARNING: "⚠"}
#: One entry per severity instead of three parallel dicts keyed the same way:
#: a card's text, wash and border are only correct as a set, and three dicts
#: is three chances to update two of them.
_STATUS: dict[Severity, SemanticColor] = {
    Severity.ERROR: STATUS_NG,
    Severity.WARNING: STATUS_WARN,
}


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
        self.setWindowFlags(
            self.windowFlags()
            & ~Qt.WindowContextHelpButtonHint  # type: ignore[attr-defined]
        )

        outer = QVBoxLayout(self)
        outer.setSpacing(12)

        # ── Title ──────────────────────────────────────────────────────
        title_text = (
            "發現問題，無法執行" if self._has_errors
            else "發現警告，請確認後繼續"
        )
        title = QLabel(title_text)
        title.setFont(QFont("Microsoft JhengHei", 11, QFont.Bold))
        title_status = STATUS_NG if self._has_errors else STATUS_WARN
        title.setStyleSheet(f"color: {title_status.text};")
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
            btn_box.addButton("取消", QDialogButtonBox.RejectRole)
            if continue_btn is not None:
                # The affirmative choice in this dialog, so it takes the
                # application's primary action styling rather than a bespoke
                # amber fill. The warning itself is carried by the title and
                # the issue cards; a second amber surface on the button only
                # competed with them.
                continue_btn.setObjectName("primaryAction")
            btn_box.accepted.connect(self.accept)
            btn_box.rejected.connect(self.reject)

        outer.addWidget(btn_box)

    @staticmethod
    def _make_issue_card(issue: PreflightIssue) -> QFrame:
        status = _STATUS[issue.severity]
        card = QFrame()
        card.setStyleSheet(
            f"QFrame {{ background-color: {status.wash}; "
            f"border: 1px solid {status.border}; "
            f"border-radius: {RADIUS_INSET}; }}"
        )
        card.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)

        row = QHBoxLayout(card)
        row.setContentsMargins(10, 8, 10, 8)
        row.setSpacing(10)

        icon = QLabel(_ICON[issue.severity])
        icon.setFont(QFont("Segoe UI", 12, QFont.Bold))
        icon.setStyleSheet(
            f"color: {status.text}; background: transparent; border: none;"
        )
        icon.setFixedWidth(18)
        icon.setAlignment(
            Qt.AlignTop  # type: ignore[attr-defined]
            | Qt.AlignHCenter  # type: ignore[attr-defined]
        )
        row.addWidget(icon)

        col = QVBoxLayout()
        col.setSpacing(2)

        task_lbl = QLabel(f"[{issue.task}]")
        task_lbl.setFont(QFont(FONT_MONO, 8))
        task_lbl.setStyleSheet(
            f"color: {status.text}; background: transparent; border: none;"
        )
        col.addWidget(task_lbl)

        msg_lbl = QLabel(issue.message)
        msg_lbl.setWordWrap(True)
        msg_lbl.setStyleSheet(
            f"color: {TEXT_PRIMARY}; background: transparent; "
            f"border: none; font-size: {FONT_SIZE_SMALL};"
        )
        col.addWidget(msg_lbl)

        row.addLayout(col, stretch=1)
        return card

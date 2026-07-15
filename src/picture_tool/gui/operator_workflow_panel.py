"""Unified operator-facing presentation for the model update workflow."""

from __future__ import annotations

from dataclasses import dataclass, replace

from PyQt5 import QtCore
from PyQt5.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QVBoxLayout,
    QWidget,
)


OPERATOR_WORKFLOW_STEPS = (
    "接收資料",
    "補齊標註",
    "補訓模型",
    "品質驗證",
    "安全部署",
)


@dataclass(frozen=True)
class OperatorWorkflowPresentation:
    """Immutable UI copy for one operator workflow state."""

    step_index: int
    title: str
    detail: str
    progress: int
    is_terminal: bool = False
    is_success: bool = False


_STATE_PRESENTATIONS = {
    "queued": OperatorWorkflowPresentation(
        0, "正在接收補訓資料", "系統正在確認產品、站別與類別順序。", 5
    ),
    "waiting_annotation": OperatorWorkflowPresentation(
        1, "需要補齊標註", "標註工具關閉後會自動檢查並接續補訓。", 10
    ),
    "preparing_dataset": OperatorWorkflowPresentation(
        2, "正在準備訓練資料", "系統正在合併歷史資料並隔離訓練、驗證與測試集。", 20
    ),
    "training": OperatorWorkflowPresentation(
        2, "正在補訓模型", "這個步驟通常最久，完成後會自動驗證品質。", 55
    ),
    "evaluating": OperatorWorkflowPresentation(
        3, "正在驗證模型品質", "新模型必須在相同測試集通過門檻，才允許部署。", 85
    ),
    "deploying": OperatorWorkflowPresentation(
        4, "正在安全部署", "舊模型會保留；部署完成後推理端會自動載入新模型。", 95
    ),
    "deployed": OperatorWorkflowPresentation(
        4,
        "模型更新完成",
        "新模型已通過驗證並完成部署。",
        100,
        is_terminal=True,
        is_success=True,
    ),
    "failed": OperatorWorkflowPresentation(
        0, "模型更新未完成", "舊模型保持不變，請依畫面訊息處理。", 0, is_terminal=True
    ),
    "cancelled": OperatorWorkflowPresentation(
        0, "模型更新已停止", "舊模型保持不變。", 0, is_terminal=True
    ),
}


def build_operator_workflow_presentation(
    state: str,
    *,
    pending_count: int = 0,
    progress: int | None = None,
) -> OperatorWorkflowPresentation:
    """Return a stable operator presentation for a machine-readable job state."""
    normalized_state = str(state or "queued").strip().lower()
    presentation = _STATE_PRESENTATIONS.get(
        normalized_state, _STATE_PRESENTATIONS["queued"]
    )
    if normalized_state == "waiting_annotation":
        count = max(0, int(pending_count))
        presentation = replace(
            presentation,
            title=f"需要補齊 {count} 張標註" if count else "正在確認標註結果",
        )
    if progress is not None:
        presentation = replace(presentation, progress=max(0, min(100, int(progress))))
    return presentation


class OperatorWorkflowPanel(QFrame):
    """Persistent five-step facade shown throughout an operator update job."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._pending_count = 0
        self._last_active_step = 0
        self.setObjectName("OperatorWorkflowPanel")
        self.setStyleSheet(
            "QFrame#OperatorWorkflowPanel {"
            "background: #111820; border: 1px solid #2f3b49; border-radius: 10px;"
            "}"
        )

        layout = QVBoxLayout(self)
        layout.setContentsMargins(22, 18, 22, 18)
        layout.setSpacing(13)

        header_layout = QHBoxLayout()
        self.heading_label = QLabel("產線模型補訓")
        self.heading_label.setStyleSheet(
            "color: #f0f6fc; font-size: 17px; font-weight: 700;"
        )
        self.target_label = QLabel("尚未指定產品／站別")
        self.target_label.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        self.target_label.setStyleSheet(
            "color: #58a6ff; font-size: 13px; font-weight: 600;"
        )
        header_layout.addWidget(self.heading_label)
        header_layout.addStretch()
        header_layout.addWidget(self.target_label)
        layout.addLayout(header_layout)

        steps_layout = QHBoxLayout()
        steps_layout.setSpacing(7)
        self.step_labels: list[QLabel] = []
        for number, step_name in enumerate(OPERATOR_WORKFLOW_STEPS, start=1):
            label = QLabel(f"{number}  {step_name}")
            label.setAlignment(QtCore.Qt.AlignCenter)
            label.setMinimumHeight(34)
            self.step_labels.append(label)
            steps_layout.addWidget(label, 1)
        layout.addLayout(steps_layout)

        self.state_title_label = QLabel("正在接收補訓資料")
        self.state_title_label.setStyleSheet(
            "color: #f0f6fc; font-size: 16px; font-weight: 700;"
        )
        layout.addWidget(self.state_title_label)

        self.state_detail_label = QLabel("")
        self.state_detail_label.setWordWrap(True)
        self.state_detail_label.setStyleSheet("color: #b8c4d1; font-size: 12px;")
        layout.addWidget(self.state_detail_label)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setFormat("模型更新 %p%")
        self.progress_bar.setStyleSheet(
            "QProgressBar {background: #202b36; border: 0; border-radius: 5px; "
            "color: white; text-align: center; min-height: 18px;}"
            "QProgressBar::chunk {background: #2f81f7; border-radius: 5px;}"
        )
        layout.addWidget(self.progress_bar)
        self.set_state("queued")

    def configure_target(
        self,
        product: str,
        area: str,
        *,
        feedback_count: int,
        pending_count: int,
    ) -> None:
        """Set the stable job identity displayed for every workflow step."""
        self._pending_count = max(0, int(pending_count))
        self.target_label.setText(
            f"{product}／{area}  ·  本次 {max(0, int(feedback_count))} 張"
        )

    def set_state(
        self,
        state: str,
        *,
        message: str = "",
        progress: int | None = None,
        pending_count: int | None = None,
    ) -> None:
        """Render a machine-readable job state in operator language."""
        if pending_count is not None:
            self._pending_count = max(0, int(pending_count))
        presentation = build_operator_workflow_presentation(
            state,
            pending_count=self._pending_count,
            progress=progress,
        )
        normalized_state = str(state or "queued").strip().lower()
        if normalized_state in {"failed", "cancelled"}:
            presentation = replace(
                presentation,
                step_index=self._last_active_step,
                progress=self.progress_bar.value(),
            )
        else:
            self._last_active_step = presentation.step_index
        self.state_title_label.setText(presentation.title)
        self.state_detail_label.setText(str(message).strip() or presentation.detail)
        self.progress_bar.setValue(presentation.progress)
        self._render_steps(presentation)

    def _render_steps(self, presentation: OperatorWorkflowPresentation) -> None:
        for index, label in enumerate(self.step_labels):
            is_completed = index < presentation.step_index or (
                presentation.is_success and index <= presentation.step_index
            )
            is_active = index == presentation.step_index and not presentation.is_success
            if is_completed:
                style = (
                    "background: #1f6f3e; color: #d9fbe5; border: 1px solid #2ea44f; "
                    "border-radius: 6px; font-weight: 600;"
                )
            elif is_active:
                style = (
                    "background: #174b7a; color: #e6f2ff; border: 1px solid #58a6ff; "
                    "border-radius: 6px; font-weight: 700;"
                )
            else:
                style = (
                    "background: #202b36; color: #8b98a5; border: 1px solid #303b46; "
                    "border-radius: 6px;"
                )
            label.setStyleSheet(style)

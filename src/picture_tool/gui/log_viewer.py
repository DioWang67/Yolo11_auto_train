"""日誌和配置預覽組件

從 app.py 提取的日誌顯示功能，保持100%視覺和功能一致。
"""
from __future__ import annotations

from typing import List

from PyQt5 import QtGui
from PyQt5.QtWidgets import QTabWidget, QTextEdit, QWidget

from picture_tool.gui.theme import FONT_MONO, log_severity_status


class LogViewer(QWidget):
    """日誌顯示組件，包含 Execution Logs 和 Config Preview tabs"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._log_history: List[str] = []
        self._build_ui()

    def _build_ui(self) -> None:
        """建立 tab widget 和兩個 text edits（保持原始布局）"""
        # Tab container
        self.tabs = QTabWidget()

        # Tab 1: Execution Logs
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setFont(QtGui.QFont(FONT_MONO, 9))
        self.tabs.addTab(self.log_text, "Execution Logs")

        # Tab 2: Config Preview (YAML)
        self.config_text = QTextEdit()
        self.config_text.setReadOnly(True)
        self.config_text.setFont(QtGui.QFont(FONT_MONO, 9))
        self.tabs.addTab(self.config_text, "Config YAML")

        # NOTE: Layout 由 parent (MainWindow) 控制，這裡不設置

    def log_message(self, message: str) -> None:
        """添加日誌訊息（完全相同的邏輯）"""
        self._log_history.append(message)
        if self._should_display_log(message):
            self._render_log_message(message)
        # Auto-scroll to bottom
        sb = self.log_text.verticalScrollBar()
        if sb:
            sb.setValue(sb.maximum())

    def update_config_preview(self, config_yaml: str) -> None:
        """更新配置預覽內容"""
        self.config_text.setPlainText(config_yaml)

    def clear_logs(self) -> None:
        """清空日誌"""
        self.log_text.clear()
        self._log_history.clear()

    def _should_display_log(self, message: str) -> bool:
        """判斷是否顯示此日誌（與原始邏輯一致）"""
        # 保留 log_filter_combo 兼容性（如果未來添加）
        if not hasattr(self, "log_filter_combo"):
            return True
        mode = self.log_filter_combo.currentData()
        lower = message.lower()
        if mode == "issues":
            return "error" in lower or "warning" in lower
        return True

    def _render_log_message(self, message: str) -> None:
        """渲染日誌訊息，依嚴重度上色。

        顏色取自 theme 的語意色票（深字），而非原本的深色主題亮字。這是可
        讀性問題而非美觀問題：主控台已改為淺底，原本那組亮綠、亮紅與淺灰在
        白底上幾乎看不見，而看不見的錯誤訊息等於沒有錯誤訊息。
        """
        status = log_severity_status(message)
        self.log_text.append(
            f'<span style="color:{status.text};">{message}</span>'
        )

"""Entry point for the Picture Tool desktop GUI."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List

from PyQt5 import QtCore, QtGui
from PyQt5.QtWidgets import (
    QApplication,
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QPushButton,
    QProgressBar,
    QTextEdit,
    QVBoxLayout,
    QWidget,
    QCheckBox,
    QLineEdit,
    QTabWidget,
    QFrame,
    QSizePolicy
)

# 模擬 Mixin 以防導入失敗
try:
    from picture_tool.gui.pipeline_controller import PipelineControllerMixin
except ImportError:
    class PipelineControllerMixin:
        def _init_pipeline_controller(self): self.config = {}
        def load_default_config(self): pass
        def load_config(self): pass
        def start_pipeline(self): pass
        def stop_pipeline(self): pass

TASK_OPTIONS: List[tuple[str, str]] = [
    ("format_conversion", "Format Conversion"),
    ("anomaly_detection", "Anomaly Detection"),
    ("yolo_augmentation", "YOLO Augmentation"),
    ("image_augmentation", "Image Augmentation"),
    ("dataset_splitter", "Dataset Splitter"),
    ("yolo_train", "YOLO Training"),
    ("yolo_evaluation", "YOLO Evaluation"),
    ("position_validation", "Position Validation"),
    ("color_inspection", "顏色範本蒐集 (SAM)"),
    ("color_verification", "顏色批次驗證"),
    ("generate_report", "Generate Report"),
    ("dataset_lint", "Dataset Lint"),
    ("aug_preview", "Augmentation Preview"),
    ("batch_inference", "Batch Inference"),
]

# ------------------------------------------------------------------
# 現代化樣式表 (含 Tab 和 Frame 優化)
# ------------------------------------------------------------------
MODERN_STYLE = """
QMainWindow {
    background-color: #1e1e1e;
}
QWidget {
    font-family: "Segoe UI", sans-serif;
    font-size: 10pt;
    color: #cccccc;
}
/* 左側側邊欄背景 */
QWidget#SideBar {
    background-color: #252526;
    border-right: 1px solid #3e3e42;
}
/* 群組標題 */
QLabel.Header {
    font-weight: bold;
    font-size: 11pt;
    color: #ffffff;
    padding-bottom: 5px;
    border-bottom: 2px solid #007acc;
    margin-bottom: 10px;
}
/* 輸入框 */
QLineEdit {
    background-color: #3c3c3c;
    border: 1px solid #3e3e42;
    color: white;
    padding: 5px;
    border-radius: 3px;
}
/* 按鈕 */
QPushButton {
    background-color: #3c3c3c;
    border: none;
    color: white;
    padding: 8px 15px;
    border-radius: 3px;
}
QPushButton:hover {
    background-color: #4e4e4e;
}
QPushButton#PrimaryBtn {
    background-color: #007acc;
    font-weight: bold;
}
QPushButton#PrimaryBtn:hover {
    background-color: #0098ff;
}
QPushButton#DangerBtn {
    background-color: #ce3838;
}
QPushButton#DangerBtn:hover {
    background-color: #e84545;
}
/* 任務清單 */
QListWidget {
    background-color: #1e1e1e;
    border: 1px solid #3e3e42;
    border-radius: 4px;
}
QListWidget::item {
    padding: 5px;
}
/* Tab 頁籤 */
QTabWidget::pane {
    border: 1px solid #3e3e42;
    background: #1e1e1e;
}
QTabBar::tab {
    background: #2d2d2d;
    color: #999;
    padding: 8px 20px;
    border-top-left-radius: 4px;
    border-top-right-radius: 4px;
}
QTabBar::tab:selected {
    background: #1e1e1e;
    color: #fff;
    border-top: 2px solid #007acc;
}
/* 進度條 */
QProgressBar {
    border: none;
    background-color: #2d2d2d;
    height: 6px;
    text-align: center;
    border-radius: 3px;
}
QProgressBar::chunk {
    background-color: #007acc;
    border-radius: 3px;
}
"""

class PictureToolGUI(QMainWindow, PipelineControllerMixin):
    def __init__(self) -> None:
        super().__init__()
        self._init_pipeline_controller()
        self._log_history: List[str] = []
        self.task_checkboxes: Dict[str, QCheckBox] = {}
        self.task_status_items: Dict[str, QListWidgetItem] = {}

        self.setWindowTitle("Picture Tool Orchestrator")
        self.resize(1200, 800)
        self.setStyleSheet(MODERN_STYLE)

        self._build_ui()
        
        try:
            self.load_default_config()
        except Exception:
            pass

    def _build_ui(self) -> None:
        """建立左右分欄佈局"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # 主佈局：水平分割 (Left Sidebar | Right Dashboard)
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # --- 左側面板 (Control Panel) ---
        self.side_bar = QWidget()
        self.side_bar.setObjectName("SideBar") # 用於 CSS 定位
        self.side_bar.setFixedWidth(350) # 固定寬度，像工具列
        
        side_layout = QVBoxLayout(self.side_bar)
        side_layout.setContentsMargins(20, 20, 20, 20)
        side_layout.setSpacing(20)

        # 加入左側元件
        side_layout.addWidget(self._create_header_label("Configuration"))
        side_layout.addWidget(self._build_config_section())
        
        side_layout.addWidget(self._create_separator())
        
        side_layout.addWidget(self._create_header_label("Select Tasks"))
        side_layout.addWidget(self._build_task_grid()) # 任務勾選區
        
        side_layout.addStretch() # 彈簧，把按鈕推到底部
        
        side_layout.addWidget(self._create_separator())
        side_layout.addWidget(self._build_control_section()) # 開始/停止按鈕

        # --- 右側面板 (Dashboard) ---
        self.dashboard = QWidget()
        dash_layout = QVBoxLayout(self.dashboard)
        dash_layout.setContentsMargins(20, 20, 20, 20)
        dash_layout.setSpacing(15)

        # 右側上半部：狀態監控
        dash_layout.addWidget(self._create_header_label("Pipeline Status Queue"))
        self.status_list = QListWidget()
        self.status_list.setMaximumHeight(200) # 不佔滿整個畫面
        self.status_list.setAlternatingRowColors(True)
        dash_layout.addWidget(self.status_list)

        # 右側下半部：分頁顯示 (Logs / Config View)
        self.tabs = QTabWidget()
        
        # Tab 1: Logs
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setFont(QtGui.QFont("Consolas", 9))
        self.tabs.addTab(self.log_text, "Execution Logs")
        
        # Tab 2: Config Preview
        self.config_text = QTextEdit()
        self.config_text.setReadOnly(True)
        self.config_text.setFont(QtGui.QFont("Consolas", 9))
        self.tabs.addTab(self.config_text, "Config Preview (YAML)")

        dash_layout.addWidget(self.tabs)

        # 將左右面板加入主佈局
        main_layout.addWidget(self.side_bar)
        main_layout.addWidget(self.dashboard)

        self._rebuild_status_items()

    # ------------------------------------------------------------------
    # 左側組件構建
    # ------------------------------------------------------------------
    def _build_config_section(self) -> QWidget:
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0,0,0,0)
        layout.setSpacing(10)

        self.config_path_edit = QLineEdit()
        self.config_path_edit.setPlaceholderText("config.yaml path...")
        
        btn_layout = QHBoxLayout()
        browse_btn = QPushButton("📂")
        browse_btn.setToolTip("Browse File")
        browse_btn.setFixedWidth(40)
        browse_btn.clicked.connect(self.browse_config_file)

        reload_btn = QPushButton("Reload")
        reload_btn.clicked.connect(self.load_config)
        
        default_btn = QPushButton("Reset")
        default_btn.clicked.connect(self.load_default_config)

        row1 = QHBoxLayout()
        row1.addWidget(self.config_path_edit)
        row1.addWidget(browse_btn)

        row2 = QHBoxLayout()
        row2.addWidget(reload_btn)
        row2.addWidget(default_btn)

        layout.addLayout(row1)
        layout.addLayout(row2)
        return container

    def _build_task_grid(self) -> QWidget:
        """使用緊湊的網格顯示 Checkbox"""
        container = QWidget()
        grid = QGridLayout(container)
        grid.setContentsMargins(0,0,0,0)
        grid.setVerticalSpacing(8)
        grid.setHorizontalSpacing(5)

        for index, (task_key, label) in enumerate(TASK_OPTIONS):
            checkbox = QCheckBox(label)
            # 讓文字長度過長時顯示 ...
            checkbox.setToolTip(label)
            checkbox.setChecked(task_key in {"dataset_splitter", "yolo_train"})
            
            self.task_checkboxes[task_key] = checkbox
            # 2欄排列，比較適合側邊欄寬度
            grid.addWidget(checkbox, index // 2, index % 2)

        return container

    def _build_control_section(self) -> QWidget:
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0,0,0,0)
        layout.setSpacing(10)

        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(False) # 簡約風格

        self.status_label = QLabel("Ready to start")
        self.status_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.status_label.setStyleSheet("color: #888; font-size: 9pt;")

        btns_layout = QHBoxLayout()
        self.start_btn = QPushButton("RUN PIPELINE")
        self.start_btn.setObjectName("PrimaryBtn")
        self.start_btn.setMinimumHeight(45)
        self.start_btn.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
        self.start_btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.start_btn.clicked.connect(self.start_pipeline)

        self.stop_btn = QPushButton("STOP")
        self.stop_btn.setObjectName("DangerBtn")
        self.stop_btn.setMinimumHeight(45)
        self.stop_btn.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self.stop_pipeline)

        btns_layout.addWidget(self.start_btn, 3) # Start 佔 75%
        btns_layout.addWidget(self.stop_btn, 1)  # Stop 佔 25%

        layout.addWidget(self.status_label)
        layout.addWidget(self.progress_bar)
        layout.addLayout(btns_layout)

        return container

    # ------------------------------------------------------------------
    # UI Helpers (視覺裝飾)
    # ------------------------------------------------------------------
    def _create_header_label(self, text: str) -> QLabel:
        lbl = QLabel(text)
        lbl.setProperty("class", "Header") # 配合 QSS
        return lbl

    def _create_separator(self) -> QFrame:
        line = QFrame()
        line.setFrameShape(QFrame.Shape.HLine)
        line.setFrameShadow(QFrame.Shadow.Sunken)
        line.setStyleSheet("background-color: #3e3e42;")
        line.setFixedHeight(1)
        return line

    # ------------------------------------------------------------------
    # Mixin Hooks (功能實現)
    # ------------------------------------------------------------------
    def log_message(self, message: str) -> None:
        self._log_history.append(message)
        
        # 彩色日誌
        color = "#cccccc"
        if "ERROR" in message: color = "#ff6b6b"
        elif "WARNING" in message: color = "#cca700"
        elif "SUCCESS" in message: color = "#6BCB77"
        elif "INFO" in message: color = "#4D96FF"
        
        # 自動切換到 Log Tab (如果當前不在看 Config)
        # if self.tabs.currentIndex() != 0: self.tabs.setCurrentIndex(0)

        self.log_text.append(f'<span style="color:{color};">{message}</span>')
        sb = self.log_text.verticalScrollBar()
        if sb: sb.setValue(sb.maximum())

    def reset_task_statuses(self, tasks):
        self._rebuild_status_items(default_state="Pending...", only=tasks)

    def _set_task_status(self, task: str, message: str, color=None) -> None:
        item = self.task_status_items.get(task)
        if item:
            # 使用簡單的符號來表示狀態，讓列表更生動
            prefix = "⚪"
            if "running" in message.lower(): prefix = "🔵"
            elif "done" in message.lower(): prefix = "🟢"
            elif "error" in message.lower(): prefix = "🔴"
            
            item.setText(f"{prefix}  {TASK_OPTIONS_MAP.get(task, task)} \n      └─ {message}")

    def _validate_pipeline_configuration(self, tasks):
        if not hasattr(self, 'config'): return []
        missing = []
        if "pipeline" not in self.config: missing.append("pipeline")
        for t in tasks:
            if t in self.config and not isinstance(self.config[t], dict):
                missing.append(t)
        return [f"Missing config: {m}" for m in missing]

    def _rebuild_status_items(self, default_state: str = "Idle", only=None) -> None:
        if not hasattr(self, "status_list"): return
        self.status_list.clear()
        self.task_status_items.clear()
        
        targets = only if only else self.task_checkboxes.keys()
        for task in targets:
            # 這裡做了一個小改動：只顯示有勾選的任務在右側列表，或者全部顯示
            # 為了 Demo 效果，這裡只顯示有勾選的，保持介面乾淨
            if task in self.task_checkboxes and self.task_checkboxes[task].isChecked():
                label_text = TASK_OPTIONS_MAP.get(task, task)
                item = QListWidgetItem(f"⚪  {label_text} : {default_state}")
                item.setForeground(QtGui.QColor("#aaaaaa"))
                self.task_status_items[task] = item
                self.status_list.addItem(item)

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------
    def browse_config_file(self) -> None:
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Config", str(Path.cwd()), "YAML (*.yml *.yaml)")
        if file_path:
            self.config_path_edit.setText(file_path)
            self.load_config()

    def closeEvent(self, event) -> None:
        if hasattr(self, 'worker_thread') and self.worker_thread and self.worker_thread.isRunning():
            self.stop_pipeline()
            self.worker_thread.wait(1000)
        super().closeEvent(event)

# 輔助 Mapping，方便顯示中文名稱
TASK_OPTIONS_MAP = dict(TASK_OPTIONS)

def main() -> None:
    app = QApplication(sys.argv)
    font = QtGui.QFont("Segoe UI", 9)
    app.setFont(font)
    window = PictureToolGUI()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
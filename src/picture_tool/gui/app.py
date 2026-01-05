"""Entry point for the Picture Tool desktop GUI."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, TYPE_CHECKING

import yaml
from PyQt5 import QtCore, QtGui
from PyQt5.QtWidgets import (
    QApplication,
    QDialog,
    QFileDialog,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QComboBox,
    QPushButton,
    QProgressBar,
    QTextEdit,
    QVBoxLayout,
    QWidget,
    QCheckBox,
    QLineEdit,
    QTabWidget,
    QFrame,
    QSizePolicy,
    QScrollArea,
    QGroupBox,
    QMessageBox,
    QInputDialog,
    QColorDialog,
)

# 模擬 Mixin 以防導入失敗
if TYPE_CHECKING:
    from picture_tool.gui.pipeline_controller import PipelineControllerMixin
    from picture_tool.gui.labelimg_launcher import LabelImgLauncher
    from picture_tool.gui.annotation_tracker import AnnotationTracker
else:
    try:
        from picture_tool.gui.pipeline_controller import PipelineControllerMixin
        from picture_tool.gui.labelimg_launcher import LabelImgLauncher
        from picture_tool.gui.annotation_tracker import AnnotationTracker
        from picture_tool.gui.config_editor import ConfigEditor
        from picture_tool.gui.wizards import NewProjectWizard
    except ImportError:
        class PipelineControllerMixin:
            def _init_pipeline_controller(self): self.config = {}
            def load_default_config(self): pass
            def load_config(self): pass
            def start_pipeline(self): pass
            def stop_pipeline(self): pass
        from picture_tool.gui.config_editor import ConfigEditor
        class NewProjectWizard: pass # Mock

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
    ("artifact_bundle", "Artifact Bundle (Zip)"),
]

TASK_OPTIONS_MAP = {key: label for key, label in TASK_OPTIONS}
TASK_LABEL_TO_KEY = {label: key for key, label in TASK_OPTIONS}
TASK_DESCRIPTIONS: Dict[str, str] = {
    "format_conversion": "格式轉換，整理輸入圖檔格式。",
    "anomaly_detection": "瑕疵檢測基準生成。",
    "yolo_augmentation": "YOLO 標註資料增強。",
    "image_augmentation": "無標註圖像增強。",
    "dataset_splitter": "切分 train/val/test。",
    "yolo_train": "訓練 YOLO 模型。",
    "yolo_evaluation": "驗證模型表現。",
    "position_validation": "位置驗證輔助。",
    "color_inspection": "SAM 顏色範本蒐集。",
    "color_verification": "顏色批次驗證。",
    "generate_report": "彙整訓練/推理報告。",
    "dataset_lint": "資料品質檢查。",
    "aug_preview": "增強結果預覽。",
    "batch_inference": "批次推理輸出。",
}
DEFAULT_PRESETS = {
    "常用流程": ["dataset_splitter", "yolo_train", "yolo_evaluation", "generate_report"],
}

# ------------------------------------------------------------------
# 現代化樣式表 - 全面升級版
# ------------------------------------------------------------------
MODERN_STYLE = """
QMainWindow {
    background-color: #0d1117;
}
QWidget {
    font-family: "Segoe UI", "Microsoft JhengHei", sans-serif;
    font-size: 10pt;
    color: #c9d1d9;
}

/* 左側側邊欄背景 */
QWidget#SideBar {
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1, 
        stop:0 #161b22, stop:1 #0d1117);
    border-right: 2px solid #30363d;
}

/* 群組標題 */
QLabel.Header {
    font-weight: bold;
    font-size: 12pt;
    color: #ffffff;
    padding: 8px 0px;
    border-bottom: 2px solid;
    border-image: linear-gradient(90deg, #58a6ff 0%, #1f6feb 100%) 1;
    margin-bottom: 12px;
}

/* 輸入框 */
QLineEdit {
    background-color: #161b22;
    border: 2px solid #30363d;
    color: #c9d1d9;
    padding: 8px 12px;
    border-radius: 6px;
    selection-background-color: #1f6feb;
}
QLineEdit:focus {
    border: 2px solid #58a6ff;
    background-color: #0d1117;
}
QLineEdit:hover {
    border-color: #484f58;
}

/* 按鈕基礎樣式 */
QPushButton {
    background-color: #21262d;
    border: 1px solid #30363d;
    color: #c9d1d9;
    padding: 10px 20px;
    border-radius: 8px;
    font-weight: 500;
    min-width: 60px;
}
QPushButton:hover {
    background-color: #30363d;
    border-color: #484f58;
}
QPushButton:pressed {
    background-color: #161b22;
}
QPushButton:disabled {
    background-color: #161b22;
    color: #484f58;
    border-color: #21262d;
}

/* 主要按鈕 (漸層效果) */
QPushButton#PrimaryBtn {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0, 
        stop:0 #1f6feb, stop:1 #58a6ff);
    border: none;
    color: #ffffff;
    font-weight: 600;
    padding: 12px 28px;
    min-width: 100px;
}
QPushButton#PrimaryBtn:hover {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0, 
        stop:0 #388bfd, stop:1 #79c0ff);
}
QPushButton#PrimaryBtn:pressed {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0, 
        stop:0 #0969da, stop:1 #1f6feb);
}
QPushButton#PrimaryBtn:disabled {
    background: #21262d;
    color: #484f58;
}

/* 危險按鈕 */
QPushButton#DangerBtn {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0, 
        stop:0 #da3633, stop:1 #f85149);
    border: none;
    color: #ffffff;
    font-weight: 600;
    padding: 10px 20px;
    min-width: 80px;
}
QPushButton#DangerBtn:hover {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0, 
        stop:0 #ff6b6b, stop:1 #ff8787);
}
QPushButton#DangerBtn:pressed {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0, 
        stop:0 #b62324, stop:1 #da3633);
}

/* 成功按鈕 */
QPushButton#SuccessBtn {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0, 
        stop:0 #238636, stop:1 #2ea043);
    border: none;
    color: #ffffff;
    font-weight: 600;
    padding: 10px 20px;
    min-width: 80px;
}
QPushButton#SuccessBtn:hover {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0, 
        stop:0 #2ea043, stop:1 #3fb950);
}

/* 任務清單 */
QListWidget {
    background-color: #0d1117;
    border: 2px solid #30363d;
    border-radius: 8px;
    padding: 4px;
}
QListWidget::item {
    padding: 8px;
    border-radius: 6px;
    margin: 2px 0px;
}
QListWidget::item:hover {
    background-color: #161b22;
}
QListWidget::item:selected {
    background-color: #1f6feb;
    color: #ffffff;
}

/* Tab 頁籤 */
QTabWidget::pane {
    border: 2px solid #30363d;
    border-radius: 8px;
    background: #0d1117;
    top: -2px;
}
QTabBar::tab {
    background: #161b22;
    color: #8b949e;
    padding: 10px 24px;
    border-top-left-radius: 8px;
    border-top-right-radius: 8px;
    margin-right: 4px;
    font-weight: 500;
}
QTabBar::tab:selected {
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1, 
        stop:0 #0d1117, stop:1 #161b22);
    color: #58a6ff;
    border-top: 3px solid #58a6ff;
}
QTabBar::tab:hover:!selected {
    background: #21262d;
    color: #c9d1d9;
}

/* 下拉選單 */
QComboBox {
    background-color: #161b22;
    border: 2px solid #30363d;
    color: #c9d1d9;
    padding: 8px 12px;
    border-radius: 6px;
}
QComboBox:hover {
    border-color: #484f58;
}
QComboBox:focus {
    border-color: #58a6ff;
}
QComboBox::drop-down {
    border: none;
    width: 30px;
}
QComboBox::down-arrow {
    image: none;
    border-left: 5px solid transparent;
    border-right: 5px solid transparent;
    border-top: 6px solid #8b949e;
    margin-right: 8px;
}
QComboBox QAbstractItemView {
    background-color: #161b22;
    border: 2px solid #30363d;
    border-radius: 6px;
    color: #c9d1d9;
    selection-background-color: #1f6feb;
    selection-color: #ffffff;
    padding: 4px;
}

/* 進度條 */
QProgressBar {
    border: none;
    background-color: #161b22;
    height: 8px;
    text-align: center;
    border-radius: 4px;
}
QProgressBar::chunk {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0, 
        stop:0 #1f6feb, stop:0.5 #58a6ff, stop:1 #79c0ff);
    border-radius: 4px;
}

/* 文本區塊 */
QTextEdit {
    background-color: #0d1117;
    border: 2px solid #30363d;
    border-radius: 8px;
    color: #c9d1d9;
    padding: 8px;
    selection-background-color: #1f6feb;
}

/* CheckBox */
QCheckBox {
    spacing: 6px;
    color: #c9d1d9;
    padding: 4px;
}
QCheckBox::indicator {
    width: 16px;
    height: 16px;
    border-radius: 4px;
    border: 2px solid #30363d;
    background-color: #161b22;
}
QCheckBox::indicator:hover {
    border-color: #484f58;
}
QCheckBox::indicator:checked {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:1, 
        stop:0 #1f6feb, stop:1 #58a6ff);
    border-color: #1f6feb;
}
QCheckBox::indicator:checked:hover {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:1, 
        stop:0 #388bfd, stop:1 #79c0ff);
}

/* 分隔線 */
QFrame[frameShape="4"] { /* HLine */
    background-color: #30363d;
    max-height: 2px;
}

/* 滾動條 */
QScrollBar:vertical {
    background: #0d1117;
    width: 12px;
    border-radius: 6px;
}
QScrollBar::handle:vertical {
    background: #30363d;
    border-radius: 6px;
    min-height: 30px;
}
QScrollBar::handle:vertical:hover {
    background: #484f58;
}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
    height: 0px;
}
QScrollBar:horizontal {
    background: #0d1117;
    height: 12px;
    border-radius: 6px;
}
QScrollBar::handle:horizontal {
    background: #30363d;
    border-radius: 6px;
    min-width: 30px;
}
QScrollBar::handle:horizontal:hover {
    background: #484f58;
}
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {
    width: 0px;
}
"""

class PictureToolGUI(QMainWindow, PipelineControllerMixin):
    def __init__(self) -> None:
        super().__init__()
        self._init_pipeline_controller()
        self._log_history: List[str] = []
        self.task_checkboxes: Dict[str, QCheckBox] = {}
        self.task_status_items: Dict[str, QListWidgetItem] = {}
        self.presets: Dict[str, List[str]] = {}
        self.preset_source: Path | None = None
        self.presets = self._load_presets()
        
        # Annotation-related components
        self.labelimg_launcher = LabelImgLauncher()
        self.annotation_tracker = AnnotationTracker()
        self.annotation_classes: List[str] = []
        self.annotation_input_dir: Path | None = None
        self.annotation_output_dir: Path | None = None

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
        self.side_bar.setFixedWidth(420) # 增加寬度以容納文字
        
        side_layout = QVBoxLayout(self.side_bar)
        side_layout.setContentsMargins(20, 20, 20, 20)
        side_layout.setSpacing(20)

        # 加入左側元件
        side_layout.addWidget(self._create_header_label("Configuration"))
        side_layout.addWidget(self._create_hint_label("步驟：1) 選 config 2) 勾任務 3) RUN"))
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
        
        # Tab 3: Annotation Tool
        annotation_tab = self._build_annotation_tab()
        self.tabs.addTab(annotation_tab, "📝 圖像標註")

        # Tab 4: Config Editor (New)
        self.config_editor = ConfigEditor()
        self.tabs.addTab(self.config_editor, "⚙ 設定編輯器")

        dash_layout.addLayout(self._build_log_controls())
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
        
        browse_btn = QPushButton("📁 瀏覽")
        browse_btn.setToolTip("選擇設定檔 (yaml)")
        browse_btn.clicked.connect(self.browse_config_file)
        
        new_proj_btn = QPushButton("✨ 新專案")
        new_proj_btn.setToolTip("建立並初始化新專案資料夾")
        new_proj_btn.clicked.connect(self.launch_new_project_wizard)

        reload_btn = QPushButton("🔄 重載")
        reload_btn.clicked.connect(self.load_config)
        

        default_btn = QPushButton("↺ 重設")
        default_btn.clicked.connect(self.load_default_config)

        save_btn = QPushButton("💾 存擋")
        save_btn.clicked.connect(self.save_config)

        self.product_override_edit = QLineEdit()
        self.product_override_edit.setPlaceholderText("選填: 產品名稱 (如 Cable1)")
        self.product_override_edit.setToolTip("若填寫，將自動設定 data/raw/{產品}/images 為輸入並覆寫專案名稱。")

        row1 = QHBoxLayout()
        row1.addWidget(self.config_path_edit)
        row1.addWidget(browse_btn)

        row_prod = QHBoxLayout()
        row_prod.addWidget(QLabel("Product:"))
        row_prod.addWidget(self.product_override_edit)

        row2 = QHBoxLayout()
        row2.addWidget(new_proj_btn)
        row2.addWidget(reload_btn)
        row2.addWidget(default_btn)
        row2.addWidget(save_btn)

        layout.addLayout(row1)
        layout.addLayout(row_prod)
        layout.addLayout(row2)

        self.config_status_label = QLabel("尚未載入設定")
        self.config_status_label.setStyleSheet("color: #aaaaaa; font-size: 9pt;")
        layout.addWidget(self.config_status_label)
        return container

    def _build_task_grid(self) -> QWidget:
        """使用緊湊的網格顯示 Checkbox，並提供快捷控制"""
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)

        control_row = QHBoxLayout()
        select_all_btn = QPushButton("☑ 全選")
        select_all_btn.setToolTip("勾選全部任務")
        select_all_btn.clicked.connect(self._select_all_tasks)

        clear_all_btn = QPushButton("☐ 清空")
        clear_all_btn.setToolTip("清除所有勾選")
        clear_all_btn.clicked.connect(self._clear_all_tasks)

        control_row.addWidget(select_all_btn)
        control_row.addWidget(clear_all_btn)
        control_row.addStretch()
        layout.addLayout(control_row)

        preset_row = QHBoxLayout()
        self.preset_combo = QComboBox()
        self._populate_preset_combo()
        apply_preset_btn = QPushButton("✓ 套用")
        apply_preset_btn.setToolTip("依需求模式勾選任務")
        apply_preset_btn.clicked.connect(self._apply_selected_preset)
        reload_preset_btn = QPushButton("🔄")
        reload_preset_btn.setToolTip("重新讀取 configs/gui_presets.yaml")
        reload_preset_btn.clicked.connect(self._reload_presets)

        preset_row.addWidget(QLabel("預設模式："))
        preset_row.addWidget(self.preset_combo)
        preset_row.addWidget(apply_preset_btn)
        preset_row.addWidget(reload_preset_btn)
        preset_row.addStretch()
        layout.addLayout(preset_row)

        grid = QGridLayout()
        grid.setContentsMargins(0,0,0,0)
        grid.setVerticalSpacing(6)
        grid.setHorizontalSpacing(10)

        for index, (task_key, label) in enumerate(TASK_OPTIONS):
            checkbox = QCheckBox(label)
            checkbox.setToolTip(label)
            checkbox.setChecked(task_key in {"dataset_splitter", "yolo_train"})
            checkbox.setStatusTip(TASK_DESCRIPTIONS.get(task_key, label))
            checkbox.stateChanged.connect(self._on_tasks_changed)
            
            self.task_checkboxes[task_key] = checkbox
            # 2欄排列
            grid.addWidget(checkbox, index // 2, index % 2)

        layout.addLayout(grid)

        self.task_summary_label = QLabel("尚未選擇任務")
        self.task_summary_label.setStyleSheet("color: #aaaaaa; font-size: 9pt;")
        layout.addWidget(self.task_summary_label)

        self.task_feedback_label = QLabel("")
        self.task_feedback_label.setStyleSheet("color: #4D96FF; font-size: 9pt;")
        layout.addWidget(self.task_feedback_label)

        self._update_task_summary()
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

        self.run_summary_label = QLabel("將執行 0 項任務")
        self.run_summary_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.run_summary_label.setStyleSheet("color: #aaaaaa; font-size: 9pt;")

        btns_layout = QHBoxLayout()
        self.start_btn = QPushButton("▶ RUN PIPELINE")
        self.start_btn.setObjectName("PrimaryBtn")
        self.start_btn.setMinimumHeight(45)
        self.start_btn.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
        self.start_btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.start_btn.clicked.connect(self.start_pipeline)

        self.stop_btn = QPushButton("⏹ STOP")
        self.stop_btn.setObjectName("DangerBtn")
        self.stop_btn.setMinimumHeight(45)
        self.stop_btn.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self.stop_pipeline)

        btns_layout.addWidget(self.start_btn, 3) # Start 佔 75%
        btns_layout.addWidget(self.stop_btn, 1)  # Stop 佔 25%

        layout.addWidget(self.status_label)
        layout.addWidget(self.run_summary_label)
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

    def _create_hint_label(self, text: str) -> QLabel:
        lbl = QLabel(text)
        lbl.setStyleSheet("color: #b5b5b5; font-size: 9pt;")
        lbl.setWordWrap(True)
        return lbl

    # ------------------------------------------------------------------
    # Mixin Hooks (功能實現)
    # ------------------------------------------------------------------
    def load_default_config(self) -> None:
        """覆寫以附帶 UI 狀態更新"""
        PipelineControllerMixin.load_default_config(self)  # type: ignore[misc]
        self._update_config_status()
        if hasattr(self, "config_editor"):
            self.config_editor.set_config(self.config)

    def load_config(self) -> None:
        """覆寫以附帶 UI 狀態更新"""
        PipelineControllerMixin.load_config(self)  # type: ignore[misc]
        self._update_config_status()
        if hasattr(self, "config_editor"):
            self.config_editor.set_config(self.config)

    def log_message(self, message: str) -> None:
        self._log_history.append(message)
        if self._should_display_log(message):
            self._render_log_message(message)
        sb = self.log_text.verticalScrollBar()
        if sb:
            sb.setValue(sb.maximum())

    def reset_task_statuses(self, tasks):
        self._rebuild_status_items(default_state="Pending...", only=tasks)

    def _set_task_status(self, task: str, message: str, color=None) -> None:
        item = self.task_status_items.get(task)
        if item:
            # 使用簡單的符號來表示狀態，讓列表更生動
            prefix = "⚪"
            lower_msg = message.lower()

            if "running" in lower_msg:
                prefix = "🔵"
            elif "done" in lower_msg:
                prefix = "🟢"
            elif "error" in lower_msg:
                prefix = "🔴"

            item.setText(f"{prefix}  {TASK_OPTIONS_MAP.get(task, task)} \n      └─ {message}")

    def _validate_pipeline_configuration(self, tasks):
        issues = []
        path_text = ""
        if hasattr(self, "config_path_edit"):
            try:
                path_text = self.config_path_edit.text().strip()
                if path_text and not Path(path_text).exists():
                    issues.append(f"Config file not found: {path_text}")
            except Exception:
                pass
        cfg = getattr(self, "config", {})
        if not isinstance(cfg, dict):
            issues.append("Config not loaded or invalid format.")
        pipeline_cfg = cfg.get("pipeline") if isinstance(cfg, dict) else None
        if not isinstance(pipeline_cfg, dict):
            issues.append("`pipeline` section missing in config.")
        elif not pipeline_cfg.get("tasks"):
            issues.append("`pipeline.tasks` is empty; nothing to run.")
        return issues

    def _rebuild_status_items(self, default_state: str = "Idle", only=None) -> None:
        if not hasattr(self, "status_list"):
            return
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
        self._update_task_summary()

    def _build_log_controls(self) -> QHBoxLayout:
        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(8)
        self.log_filter_combo = QComboBox()
        self.log_filter_combo.addItem("全部訊息", userData="all")
        self.log_filter_combo.addItem("僅錯誤/警告", userData="issues")
        self.log_filter_combo.currentIndexChanged.connect(self._refresh_log_view)

        clear_btn = QPushButton("🗑 清空")
        clear_btn.clicked.connect(self._clear_logs)

        row.addWidget(self.log_filter_combo)
        row.addWidget(clear_btn)
        row.addStretch()
        return row

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------
    def browse_config_file(self) -> None:
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Config", str(Path.cwd()), "YAML (*.yml *.yaml)")
        if file_path:
            self.config_path_edit.setText(file_path)
            self.load_config()
            self._update_config_status()

    def launch_new_project_wizard(self) -> None:
        """Launch the wizard to create a new project."""
        dlg = NewProjectWizard(self)
        if dlg.exec_() == QDialog.Accepted:
            if hasattr(dlg, 'created_path') and dlg.created_path:
                self.config_path_edit.setText(str(dlg.created_path).replace('\\', '/'))
                self.load_config()
                QMessageBox.information(self, "Loaded", f"Loaded new project config from:\n{dlg.created_path}")

    def closeEvent(self, event) -> None:
        if hasattr(self, 'worker_thread') and self.worker_thread and self.worker_thread.isRunning():
            self.stop_pipeline()
            self.worker_thread.wait(1000)
        super().closeEvent(event)

    def _on_tasks_changed(self) -> None:
        self._rebuild_status_items()
        self._update_task_summary()

    def _select_all_tasks(self) -> None:
        for checkbox in self.task_checkboxes.values():
            checkbox.setChecked(True)
        self._rebuild_status_items()
        self._update_task_summary()
        self._show_task_feedback("已全選所有任務。")

    def _clear_all_tasks(self) -> None:
        for checkbox in self.task_checkboxes.values():
            checkbox.setChecked(False)
        self._rebuild_status_items()
        self._update_task_summary()
        self._show_task_feedback("已清除所有勾選。", color="#aaaaaa")

    def _apply_selected_preset(self) -> None:
        name = self.preset_combo.currentText() if hasattr(self, "preset_combo") else ""
        tasks = self.presets.get(name, [])
        normalized: List[str] = []
        for t in tasks:
            key = self._normalize_task_name(t)
            if key:
                normalized.append(key)
            else:
                self.log_message(f"[WARNING] 預設「{name}」包含未知任務：{t}")
        for key, checkbox in self.task_checkboxes.items():
            checkbox.setChecked(key in normalized)
        self._rebuild_status_items()
        self._update_task_summary()
        preset_labels = [TASK_OPTIONS_MAP.get(k, k) for k in normalized]
        src = f"(來源: {self.preset_source.name})" if self.preset_source else "(來源: 內建)"
        if preset_labels:
            self._show_task_feedback(f"已套用預設「{name}」：{', '.join(preset_labels)} {src}")
            self.log_message(f"[INFO] 套用預設 {name}: {', '.join(normalized)} {src}")
        else:
            self._show_task_feedback(f"預設「{name}」沒有有效任務", color="#cca700")

    def _update_task_summary(self) -> None:
        if not hasattr(self, "task_checkboxes"): 
            return
        selected_labels = [
            TASK_OPTIONS_MAP.get(key, key)
            for key, cb in self.task_checkboxes.items()
            if cb.isChecked()
        ]
        count = len(selected_labels)
        summary_text = "尚未選擇任務" if not selected_labels else f"將執行 {count} 項：{', '.join(selected_labels)}"
        if hasattr(self, "task_summary_label"):
            self.task_summary_label.setText(summary_text)
        if hasattr(self, "run_summary_label"):
            self.run_summary_label.setText(summary_text)

    def _show_task_feedback(self, message: str, color: str = "#4D96FF") -> None:
        if hasattr(self, "task_feedback_label"):
            self.task_feedback_label.setText(message)
            self.task_feedback_label.setStyleSheet(f"color: {color}; font-size: 9pt;")

    def _populate_preset_combo(self) -> None:
        if not hasattr(self, "preset_combo"):
            return
        self.preset_combo.clear()
        if not self.presets:
            self.presets = DEFAULT_PRESETS.copy()
            self.preset_source = None
        for name in self.presets.keys():
            self.preset_combo.addItem(name)

    def _reload_presets(self) -> None:
        self.presets = self._load_presets()
        self._populate_preset_combo()
        self._show_task_feedback("已重新載入預設設定檔。", color="#4D96FF")

    def _preset_candidate_paths(self) -> List[Path]:
        return [
            Path.cwd() / "configs" / "gui_presets.yaml",
            Path(__file__).resolve().parent.parent / "resources" / "gui_presets.yaml",
        ]

    def _normalize_task_name(self, name: str) -> str | None:
        trimmed = name.strip()
        if trimmed in self.task_checkboxes:
            return trimmed
        if trimmed in TASK_LABEL_TO_KEY:
            return TASK_LABEL_TO_KEY[trimmed]
        # case-insensitive label match
        for label, key in TASK_LABEL_TO_KEY.items():
            if trimmed.lower() == label.lower():
                return key
        return None

    def _load_presets(self) -> Dict[str, List[str]]:
        for path in self._preset_candidate_paths():
            if not path.exists():
                continue
            try:
                data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
            except Exception as exc:
                self.log_message(f"[WARNING] 無法讀取預設檔 {path}: {exc}")
                continue
            raw = data.get("presets") if isinstance(data, dict) else None
            if not isinstance(raw, dict):
                continue
            presets: Dict[str, List[str]] = {}
            for name, tasks in raw.items():
                if not isinstance(tasks, list):
                    continue
                presets[name] = [str(t) for t in tasks]
            if presets:
                self.preset_source = path
                return presets
        self.preset_source = None
        return DEFAULT_PRESETS.copy()

    def _update_config_status(self) -> None:
        if not hasattr(self, "config_status_label"): 
            return
        path_text = ""
        try:
            path_text = self.config_path_edit.text().strip()
        except Exception:
            path_text = ""
        if not path_text:
            msg = "未指定設定檔，將使用預設路徑"
            color = "#cca700"
        else:
            path = Path(path_text)
            if path.exists():
                msg = f"已載入：{path.name}"
                color = "#6BCB77"
            else:
                msg = f"找不到檔案：{path}"
                color = "#ff6b6b"
        self.config_status_label.setText(msg)
        self.config_status_label.setStyleSheet(f"color: {color}; font-size: 9pt;")

    def _render_log_message(self, message: str) -> None:
        color = "#cccccc"
        lower = message.lower()
        if "error" in lower:
            color = "#ff6b6b"
        elif "warning" in lower:
            color = "#cca700"
        elif "success" in lower:
            color = "#6BCB77"
        elif "info" in lower:
            color = "#4D96FF"
        self.log_text.append(f'<span style="color:{color};">{message}</span>')

    def _should_display_log(self, message: str) -> bool:
        if not hasattr(self, "log_filter_combo"):
            return True
        mode = self.log_filter_combo.currentData()
        lower = message.lower()
        if mode == "issues":
            return "error" in lower or "warning" in lower
        return True

    def _refresh_log_view(self) -> None:
        if not hasattr(self, "log_text"):
            return
        self.log_text.clear()
        for msg in self._log_history:
            if self._should_display_log(msg):
                self._render_log_message(msg)

    def _clear_logs(self) -> None:
        self._log_history.clear()
        if hasattr(self, "log_text"):
            self.log_text.clear()
    
    # ================================================================
    # Annotation Management Methods
    # ================================================================
    
    def _build_annotation_tab(self) -> QWidget:
        """Build the annotation management tab."""
        container = QWidget()
        main_layout = QHBoxLayout(container)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(15)
        
        # Left: Class Management
        left_panel = self._build_class_management_panel()
        left_panel.setMaximumWidth(300)
        
        # Middle: Progress and Statistics
        middle_panel = self._build_annotation_progress_panel()
        
        # Right: Actions and Settings
        right_panel = self._build_annotation_actions_panel()
        right_panel.setMaximumWidth(300)
        
        main_layout.addWidget(left_panel, 1)
        main_layout.addWidget(middle_panel, 2)
        main_layout.addWidget(right_panel, 1)
        
        return container
    
    def _build_class_management_panel(self) -> QWidget:
        """Build class management panel."""
        group = QGroupBox("類別管理")
        layout = QVBoxLayout(group)
        layout.setSpacing(10)
        
        # Class list
        self.annotation_class_list = QListWidget()
        self.annotation_class_list.setMaximumHeight(250)
        layout.addWidget(QLabel("類別列表："))
        layout.addWidget(self.annotation_class_list)
        
        # Buttons
        btn_layout = QGridLayout()
        
        add_btn = QPushButton("➕ 新增類別")
        add_btn.clicked.connect(self._add_annotation_class)
        
        edit_btn = QPushButton("✏️ 編輯")
        edit_btn.clicked.connect(self._edit_annotation_class)
        
        delete_btn = QPushButton("🗑️ 刪除")
        delete_btn.setObjectName("DangerBtn")
        delete_btn.clicked.connect(self._delete_annotation_class)
        
        import_btn = QPushButton("📥 從配置導入")
        import_btn.clicked.connect(self._import_classes_from_config)
        
        save_btn = QPushButton("💾 儲存類別")
        save_btn.setObjectName("SuccessBtn")
        save_btn.clicked.connect(self._save_annotation_classes)
        
        btn_layout.addWidget(add_btn, 0, 0)
        btn_layout.addWidget(edit_btn, 0, 1)
        btn_layout.addWidget(delete_btn, 1, 0)
        btn_layout.addWidget(import_btn, 1, 1)
        btn_layout.addWidget(save_btn, 2, 0, 1, 2)
        
        layout.addLayout(btn_layout)
        layout.addStretch()
        
        return group
    
    def _build_annotation_progress_panel(self) -> QWidget:
        """Build annotation progress panel."""
        group = QGroupBox("標註進度")
        layout = QVBoxLayout(group)
        layout.setSpacing(10)
        
        # Statistics
        self.annotation_stats_label = QLabel("尚未掃描")
        self.annotation_stats_label.setStyleSheet("font-size: 11pt; color: #c9d1d9;")
        layout.addWidget(self.annotation_stats_label)
        
        # Progress bar
        self.annotation_progress_bar = QProgressBar()
        self.annotation_progress_bar.setValue(0)
        layout.addWidget(self.annotation_progress_bar)
       
        # Class distribution
        layout.addWidget(QLabel("類別分佈："))
        self.annotation_class_dist = QTextEdit()
        self.annotation_class_dist.setReadOnly(True)
        self.annotation_class_dist.setMaximumHeight(150)
        self.annotation_class_dist.setFont(QtGui.QFont("Consolas", 9))
        layout.addWidget(self.annotation_class_dist)
        
        # Unannotated files
        layout.addWidget(QLabel("未標註圖片："))
        self.annotation_unannotated_list = QListWidget()
        self.annotation_unannotated_list.setMaximumHeight(200)
        layout.addWidget(self.annotation_unannotated_list)
        
        layout.addStretch()
        
        return group
    
    def _build_annotation_actions_panel(self) -> QWidget:
        """Build annotation actions panel."""
        group = QGroupBox("快速操作")
        layout = QVBoxLayout(group)
        layout.setSpacing(12)
        
        # Launch LabelImg button
        launch_btn = QPushButton("🚀 啟動 LabelImg")
        launch_btn.setObjectName("PrimaryBtn")
        launch_btn.setMinimumHeight(45)
        launch_btn.clicked.connect(self._launch_labelimg)
        layout.addWidget(launch_btn)
        
        # Validate annotations button
        validate_btn = QPushButton("📊 驗證標註")
        validate_btn.clicked.connect(self._validate_annotations)
        layout.addWidget(validate_btn)
        
        # Rescan button
        rescan_btn = QPushButton("🔄 重新掃描")
        rescan_btn.clicked.connect(self._scan_annotation_progress)
        layout.addWidget(rescan_btn)
        
        # Start augmentation button
        augment_btn = QPushButton("▶️ 完成，開始增強")
        augment_btn.setObjectName("SuccessBtn")
        augment_btn.clicked.connect(self._start_augmentation_from_annotation)
        layout.addWidget(augment_btn)
        
        layout.addWidget(self._create_separator())
        
        # Settings
        layout.addWidget(QLabel("⚙️ 設定"))
        
        # Input directory
        input_layout = QVBoxLayout()
        input_layout.addWidget(QLabel("輸入目錄："))
        self.annotation_input_edit = QLineEdit()
        self.annotation_input_edit.setPlaceholderText("選擇包含圖片的資料夾...")
        input_browse_btn = QPushButton("瀏覽...")
        input_browse_btn.clicked.connect(self._browse_annotation_input)
        
        input_row = QHBoxLayout()
        input_row.addWidget(self.annotation_input_edit)
        input_row.addWidget(input_browse_btn)
        input_layout.addLayout(input_row)
        layout.addLayout(input_layout)
        
        # Output directory
        output_layout = QVBoxLayout()
        output_layout.addWidget(QLabel("標註輸出目錄："))
        self.annotation_output_edit = QLineEdit()
        self.annotation_output_edit.setPlaceholderText("標註文件儲存位置...")
        output_browse_btn = QPushButton("瀏覽...")
        output_browse_btn.clicked.connect(self._browse_annotation_output)
        
        output_row = QHBoxLayout()
        output_row.addWidget(self.annotation_output_edit)
        output_row.addWidget(output_browse_btn)
        output_layout.addLayout(output_row)
        layout.addLayout(output_layout)
        
        layout.addStretch()
        
        return group
    
    # Class management methods
    def _add_annotation_class(self) -> None:
        """Add a new annotation class."""
        class_name, ok = QInputDialog.getText(
            self,
            "新增類別",
            "輸入類別名稱：",
        )
        if ok and class_name.strip():
            class_name = class_name.strip()
            if class_name in self.annotation_classes:
                QMessageBox.warning(self, "錯誤", f"類別 '{class_name}' 已存在！")
                return
            
            self.annotation_classes.append(class_name)
            self._refresh_class_list()
            self.log_message(f"[INFO] Added annotation class: {class_name}")
    
    def _edit_annotation_class(self) -> None:
        """Edit selected annotation class."""
        current_item = self.annotation_class_list.currentItem()
        if not current_item:
            QMessageBox.warning(self, "錯誤", "請先選擇要編輯的類別！")
            return
        
        old_name = current_item.text()
        new_name, ok = QInputDialog.getText(
            self,
            "編輯類別",
            "輸入新的類別名稱：",
            text=old_name,
        )
        if ok and new_name.strip():
            new_name = new_name.strip()
            if new_name != old_name and new_name in self.annotation_classes:
                QMessageBox.warning(self, "錯誤", f"類別 '{new_name}' 已存在！")
                return
            
            idx = self.annotation_classes.index(old_name)
            self.annotation_classes[idx] = new_name
            self._refresh_class_list()
            self.log_message(f"[INFO] Renamed class: {old_name} → {new_name}")
    
    def _delete_annotation_class(self) -> None:
        """Delete selected annotation class."""
        current_item = self.annotation_class_list.currentItem()
        if not current_item:
            QMessageBox.warning(self, "錯誤", "請先選擇要刪除的類別！")
            return
        
        class_name = current_item.text()
        reply = QMessageBox.question(
            self,
            "確認刪除",
            f"確定要刪除類別 '{class_name}' 嗎？",
            QMessageBox.Yes | QMessageBox.No,
        )
        if reply == QMessageBox.Yes:
            self.annotation_classes.remove(class_name)
            self._refresh_class_list()
            self.log_message(f"[INFO] Deleted annotation class: {class_name}")
    
    def _import_classes_from_config(self) -> None:
        """Import classes from yolo_training.class_names."""
        config = getattr(self, "config", {})
        yolo_cfg = config.get("yolo_training", {})
        class_names = yolo_cfg.get("class_names", [])
        
        if not class_names:
            QMessageBox.warning(
                self,
                "無法導入",
                "配置中沒有找到 yolo_training.class_names！",
            )
            return
        
        # Add classes that don't exist
        added = []
        for class_name in class_names:
            if class_name not in self.annotation_classes:
                self.annotation_classes.append(class_name)
                added.append(class_name)
        
        self._refresh_class_list()
        
        if added:
            QMessageBox.information(
                self,
                "導入成功",
                f"已導入 {len(added)} 個類別：\n" + ", ".join(added),
            )
            self.log_message(f"[INFO] Imported {len(added)} classes from config")
        else:
            QMessageBox.information(self, "完成", "所有類別已存在，無需導入。")
    
    def _save_annotation_classes(self) -> None:
        """Save classes to predefined_classes.txt."""
        if not self.annotation_classes:
            QMessageBox.warning(self, "錯誤", "沒有類別可以儲存！")
            return
        
        if not self.annotation_output_dir:
            QMessageBox.warning(
                self,
                "錯誤",
                "請先設定標註輸出目錄！",
            )
            return
        
        try:
            output_dir = Path(self.annotation_output_dir)
            classes_file = output_dir.parent / "predefined_classes.txt"
            classes_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(classes_file, "w", encoding="utf-8") as f:
                for class_name in self.annotation_classes:
                    f.write(f"{class_name}\n")
            
            QMessageBox.information(
                self,
                "儲存成功",
                f"類別列表已儲存到：\n{classes_file}",
            )
            self.log_message(f"[INFO] Saved {len(self.annotation_classes)} classes to {classes_file}")
        except Exception as e:
            QMessageBox.critical(self, "錯誤", f"儲存失敗：\n{e}")
            self.log_message(f"[ERROR] Failed to save classes: {e}")
    
    def _refresh_class_list(self) -> None:
        """Refresh the class list widget."""
        self.annotation_class_list.clear()
        for class_name in self.annotation_classes:
            self.annotation_class_list.addItem(class_name)
    
    # Directory browsing
    def _browse_annotation_input(self) -> None:
        """Browse for annotation input directory."""
        dir_path = QFileDialog.getExistingDirectory(
            self,
            "選擇輸入圖片資料夾",
            str(Path.cwd()),
        )
        if dir_path:
            self.annotation_input_dir = Path(dir_path)
            self.annotation_input_edit.setText(dir_path)
            self._scan_annotation_progress()
    
    def _browse_annotation_output(self) -> None:
        """Browse for annotation output directory."""
        dir_path = QFileDialog.getExistingDirectory(
            self,
            "選擇標註輸出資料夾",
            str(Path.cwd()),
        )
        if dir_path:
            self.annotation_output_dir = Path(dir_path)
            self.annotation_output_edit.setText(dir_path)
            self._scan_annotation_progress()
    
    # Progress tracking
    def _scan_annotation_progress(self) -> None:
        """Scan and update annotation progress."""
        if not self.annotation_input_dir or not self.annotation_output_dir:
            self.annotation_stats_label.setText("請設定輸入和輸出目錄")
            return
        
        stats = self.annotation_tracker.scan_directory(
            self.annotation_input_dir,
            self.annotation_output_dir,
        )
        
        # Update statistics label
        self.annotation_stats_label.setText(
            f"📊 總圖片：{stats['total_images']}  |  "
            f"✅ 已標註：{stats['annotated_images']} ({stats['progress_percent']:.1f}%)  |  "
            f"⏳ 未標註：{len(stats['unannotated_images'])}"
        )
        
        # Update progress bar
        self.annotation_progress_bar.setValue(int(stats['progress_percent']))
        
        # Update unannotated list
        self.annotation_unannotated_list.clear()
        for img_name in stats['unannotated_images'][:20]:  # Show max 20
            self.annotation_unannotated_list.addItem(img_name)
        if len(stats['unannotated_images']) > 20:
            self.annotation_unannotated_list.addItem(
                f"... 還有 {len(stats['unannotated_images']) - 20} 張"
            )
        
        # Update class distribution
        if self.annotation_classes and stats['annotated_images'] > 0:
            class_dist = self.annotation_tracker.get_class_distribution(
                self.annotation_output_dir,
                self.annotation_classes,
            )
            dist_text = "\n".join([
                f"{name}: {count}" for name, count in class_dist.items()
            ])
            self.annotation_class_dist.setText(dist_text)
        else:
            self.annotation_class_dist.setText("尚無標註資料")
        
        self.log_message(f"[INFO] Scanned annotations: {stats['annotated_images']}/{stats['total_images']}")
    
    # LabelImg integration
    def _launch_labelimg(self) -> None:
        """Launch LabelImg with current settings."""
        if not self.labelimg_launcher.is_installed():
            QMessageBox.critical(
                self,
                "LabelImg 未安裝",
                "請先安裝 LabelImg:\n\npip install labelImg",
            )
            return
        
        if not self.annotation_classes:
            QMessageBox.warning(
                self,
                "錯誤",
                "請先新增至少一個類別！",
            )
            return
        
        if not self.annotation_input_dir or not self.annotation_output_dir:
            QMessageBox.warning(
                self,
                "錯誤",
                "請先設定輸入和輸出目錄！",
            )
            return
        
        # Prepare environment
        success = self.labelimg_launcher.prepare_environment(
            self.annotation_classes,
            self.annotation_input_dir,
            self.annotation_output_dir,
        )
        
        if not success:
            QMessageBox.critical(self, "錯誤", "準備環境失敗！")
            return
        
        # Launch
        classes_file = self.annotation_output_dir.parent / "predefined_classes.txt"
        success = self.labelimg_launcher.launch(
            self.annotation_input_dir,
            self.annotation_output_dir,
            classes_file,
        )
        
        if success:
            QMessageBox.information(
                self,
                "已啟動",
                "LabelImg 已啟動！\n\n完成標註後關閉 LabelImg，然後點擊「重新掃描」查看進度。",
            )
            self.log_message("[INFO] Launched LabelImg")
        else:
            QMessageBox.critical(self, "錯誤", "啟動 LabelImg 失敗！")
    
    def _validate_annotations(self) -> None:
        """Validate annotation files."""
        if not self.annotation_output_dir or not self.annotation_classes:
            QMessageBox.warning(
                self,
                "錯誤",
                "請先設定輸出目錄並建立類別！",
            )
            return
        
        errors = self.annotation_tracker.validate_annotations(
            self.annotation_output_dir,
            len(self.annotation_classes),
        )
        
        if not errors:
            QMessageBox.information(
                self,
                "驗證成功",
                "所有標註文件格式正確！✅",
            )
            self.log_message("[INFO] All annotations validated successfully")
        else:
            error_text = "\n".join(errors[:10])  # Show max 10 errors
            if len(errors) > 10:
                error_text += f"\n\n... 還有 {len(errors) - 10} 個錯誤"
            
            QMessageBox.warning(
                self,
                f"發現 {len(errors)} 個錯誤",
                error_text,
            )
            self.log_message(f"[WARNING] Found {len(errors)} validation errors")
    
    def _start_augmentation_from_annotation(self) -> None:
        """Set augmentation input to annotation output and switch tab."""
        if not self.annotation_output_dir:
            QMessageBox.warning(
                self,
                "錯誤",  
                "請先設定標註輸出目錄！",
            )
            return
        
        reply = QMessageBox.question(
            self,
            "確認",
            f"將使用標註輸出目錄：\n{self.annotation_output_dir}\n\n作為圖像增強的輸入，繼續嗎？",
            QMessageBox.Yes | QMessageBox.No,
        )
        
        if reply == QMessageBox.Yes:
            # TODO: Set yolo_augmentation input directories in config
            QMessageBox.information(
                self,
                "完成",
                "請切換到主標籤頁勾選「YOLO Augmentation」任務並執行。",
            )
            self.log_message("[INFO] Ready to start augmentation from annotation output")

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

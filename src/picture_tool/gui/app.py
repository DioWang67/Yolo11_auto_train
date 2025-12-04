"""Entry point for the Picture Tool desktop GUI."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List

import yaml
from PyQt5 import QtCore, QtGui
from PyQt5.QtWidgets import (
    QApplication,
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

/* 下拉選單 (QComboBox) */
QComboBox {
    background-color: #1e1e1e;
    border: 1px solid #3e3e42;
    color: #e6e6e6;
    padding: 6px 8px;
    border-radius: 3px;
}
QComboBox QAbstractItemView {
    background-color: #252526;
    color: #e6e6e6;
    selection-background-color: #007acc;
    selection-color: #ffffff;
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

/* 文本區塊 (Logs / Config preview) */
QTextEdit {
    background-color: #1e1e1e;
    border: 1px solid #3e3e42;
    color: #e6e6e6;
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
        
        browse_btn = QPushButton("Browse")
        browse_btn.setToolTip("選擇設定檔 (yaml)")
        browse_btn.setFixedWidth(80)
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
        select_all_btn = QPushButton("全選")
        select_all_btn.setToolTip("勾選全部任務")
        select_all_btn.setFixedWidth(60)
        select_all_btn.clicked.connect(self._select_all_tasks)

        clear_all_btn = QPushButton("清空")
        clear_all_btn.setToolTip("清除所有勾選")
        clear_all_btn.setFixedWidth(60)
        clear_all_btn.clicked.connect(self._clear_all_tasks)

        control_row.addWidget(select_all_btn)
        control_row.addWidget(clear_all_btn)
        control_row.addStretch()
        layout.addLayout(control_row)

        preset_row = QHBoxLayout()
        self.preset_combo = QComboBox()
        self._populate_preset_combo()
        apply_preset_btn = QPushButton("套用預設")
        apply_preset_btn.setToolTip("依需求模式勾選任務")
        apply_preset_btn.clicked.connect(self._apply_selected_preset)
        reload_preset_btn = QPushButton("重新載入")
        reload_preset_btn.setFixedWidth(80)
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
        grid.setVerticalSpacing(8)
        grid.setHorizontalSpacing(5)

        for index, (task_key, label) in enumerate(TASK_OPTIONS):
            checkbox = QCheckBox(label)
            # 讓文字長度過長時顯示 ...
            checkbox.setToolTip(label)
            checkbox.setChecked(task_key in {"dataset_splitter", "yolo_train"})
            checkbox.setStatusTip(TASK_DESCRIPTIONS.get(task_key, label))
            checkbox.stateChanged.connect(self._on_tasks_changed)
            
            self.task_checkboxes[task_key] = checkbox
            # 2欄排列，比較適合側邊欄寬度
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

    def load_config(self) -> None:
        """覆寫以附帶 UI 狀態更新"""
        PipelineControllerMixin.load_config(self)  # type: ignore[misc]
        self._update_config_status()

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

        clear_btn = QPushButton("清空 Log")
        clear_btn.setFixedWidth(90)
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

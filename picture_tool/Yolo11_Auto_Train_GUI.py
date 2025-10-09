import sys
import os
from pathlib import Path
import csv
import yaml
from typing import Optional

if __name__ == "__main__" or __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, QWidget,
    QPushButton, QLabel, QTextEdit, QProgressBar, QGroupBox,
    QCheckBox, QLineEdit, QFileDialog, QSplitter, QTabWidget,
    QScrollArea, QGridLayout, QMessageBox, QToolButton, QMenu,
    QComboBox, QInputDialog, QListWidget, QListWidgetItem, QDialog, QDialogButtonBox, QFrame
)
from PyQt5.QtCore import Qt, QSize
from PyQt5.QtGui import QColor, QIcon, QPixmap

POSITION_TASK_LABEL = "位置檢查"
YOLO_TRAIN_LABEL = "YOLO訓練"

PRESET_CONFIG_PATH = Path(__file__).parent / "preset_config.yaml"

from picture_tool.gui.task_thread import WorkerThread


class CompactCheckBox(QCheckBox):
    def __init__(self, text, parent=None):
        super().__init__(text, parent)
        self.setStyleSheet("""
            QCheckBox {
                font-size: 9pt;
                color: #2c3e50;
                spacing: 5px;
                padding: 2px;
                margin: 1px;
            }
            QCheckBox::indicator {
                width: 14px;
                height: 14px;
                border: 1px solid #ced4da;
                border-radius: 3px;
                background-color: white;
            }
            QCheckBox::indicator:checked {
                background-color: #007bff;
                border-color: #0056b3;
            }
            QCheckBox::indicator:hover {
                border-color: #80bdff;
            }
        """)


class CompactButton(QPushButton):
    def __init__(self, text, color_theme="primary", parent=None):
        super().__init__(text, parent)
        self.color_theme = color_theme
        self._setup_style()

    def _setup_style(self):
        base_style = """
            QPushButton {
                border: none;
                padding: 6px 12px;
                border-radius: 4px;
                font-size: 9pt;
                font-weight: bold;
                min-height: 24px;
                text-align: center;
            }
            QPushButton:disabled {
                background-color: #6c757d;
                color: #dee2e6;
            }
        """
        color_styles = {
            "primary": """
                QPushButton {
                    background-color: #007bff;
                    color: white;
                }
                QPushButton:hover {
                    background-color: #0056b3;
                }
                QPushButton:pressed {
                    background-color: #004085;
                }
            """,
            "success": """
                QPushButton {
                    background-color: #28a745;
                    color: white;
                }
                QPushButton:hover {
                    background-color: #218838;
                }
                QPushButton:pressed {
                    background-color: #1e7e34;
                }
            """,
            "danger": """
                QPushButton {
                    background-color: #dc3545;
                    color: white;
                }
                QPushButton:hover {
                    background-color: #c82333;
                }
                QPushButton:pressed {
                    background-color: #bd2130;
                }
            """,
            "secondary": """
                QPushButton {
                    background-color: #6c757d;
                    color: white;
                }
                QPushButton:hover {
                    background-color: #5a6268;
                }
                QPushButton:pressed {
                    background-color: #545b62;
                }
            """
        }
        self.setStyleSheet(base_style + color_styles.get(self.color_theme, ""))


class PositionSettingsDialog(QDialog):
    def __init__(self, parent, settings: dict):
        super().__init__(parent)
        self.setWindowTitle("YOLO 位置檢查設定")
        self.setMinimumWidth(500)
        self._settings = settings or {}

        layout = QVBoxLayout(self)
        form = QGridLayout()
        form.setSpacing(8)

        # 基本設定
        self.product_edit = QLineEdit(self._settings.get("product") or "")
        self.area_edit = QLineEdit(self._settings.get("area") or "")
        
        form.addWidget(QLabel("產品:"), 0, 0)
        form.addWidget(self.product_edit, 0, 1)
        form.addWidget(QLabel("區域:"), 1, 0)
        form.addWidget(self.area_edit, 1, 1)

        # 模型參數
        imgsz_value = self._settings.get("imgsz")
        self.imgsz_edit = QLineEdit("" if imgsz_value in (None, "") else str(imgsz_value))
        conf_value = self._settings.get("conf")
        self.conf_edit = QLineEdit("" if conf_value in (None, "") else str(conf_value))
        tol_value = self._settings.get("tolerance_override")
        self.tolerance_edit = QLineEdit("" if tol_value in (None, "") else str(tol_value))

        form.addWidget(QLabel("影像尺寸:"), 2, 0)
        form.addWidget(self.imgsz_edit, 2, 1)
        form.addWidget(QLabel("信心閾值:"), 3, 0)
        form.addWidget(self.conf_edit, 3, 1)
        form.addWidget(QLabel("容差 (%):"), 4, 0)
        form.addWidget(self.tolerance_edit, 4, 1)

        # 分隔線
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setFrameShadow(QFrame.Sunken)
        form.addWidget(line, 5, 0, 1, 2)

        # 路徑設定
        self.sample_dir_edit = QLineEdit(self._settings.get("sample_dir") or "")
        sample_btn = CompactButton("瀏覽", "secondary")
        sample_btn.clicked.connect(self._browse_sample_dir)
        sample_layout = QHBoxLayout()
        sample_layout.addWidget(self.sample_dir_edit)
        sample_layout.addWidget(sample_btn)
        form.addWidget(QLabel("樣本資料夾:"), 6, 0)
        form.addLayout(sample_layout, 6, 1)

        self.config_path_edit = QLineEdit(self._settings.get("config_path") or "")
        config_btn = CompactButton("瀏覽", "secondary")
        config_btn.clicked.connect(self._browse_config_path)
        config_layout = QHBoxLayout()
        config_layout.addWidget(self.config_path_edit)
        config_layout.addWidget(config_btn)
        form.addWidget(QLabel("設定檔:"), 7, 0)
        form.addLayout(config_layout, 7, 1)

        self.output_dir_edit = QLineEdit(self._settings.get("output_dir") or "")
        output_btn = CompactButton("瀏覽", "secondary")
        output_btn.clicked.connect(self._browse_output_dir)
        output_layout = QHBoxLayout()
        output_layout.addWidget(self.output_dir_edit)
        output_layout.addWidget(output_btn)
        form.addWidget(QLabel("輸出資料夾:"), 8, 0)
        form.addLayout(output_layout, 8, 1)

        self.weights_path_edit = QLineEdit(self._settings.get("weights") or "")
        weights_btn = CompactButton("瀏覽", "secondary")
        weights_btn.clicked.connect(self._browse_weights_path)
        weights_layout = QHBoxLayout()
        weights_layout.addWidget(self.weights_path_edit)
        weights_layout.addWidget(weights_btn)
        form.addWidget(QLabel("權重檔:"), 9, 0)
        form.addLayout(weights_layout, 9, 1)

        layout.addLayout(form)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _browse_sample_dir(self):
        directory = QFileDialog.getExistingDirectory(
            self, "選擇樣本資料夾", self.sample_dir_edit.text() or ""
        )
        if directory:
            self.sample_dir_edit.setText(directory)

    def _browse_config_path(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "選擇位置設定檔", self.config_path_edit.text() or "",
            "YAML files (*.yaml *.yml)"
        )
        if file_path:
            self.config_path_edit.setText(file_path)

    def _browse_output_dir(self):
        directory = QFileDialog.getExistingDirectory(
            self, "選擇輸出資料夾", self.output_dir_edit.text() or ""
        )
        if directory:
            self.output_dir_edit.setText(directory)

    def _browse_weights_path(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "選擇權重檔", self.weights_path_edit.text() or "",
            "Model files (*.pt *.onnx *.engine);;All files (*.*)"
        )
        if file_path:
            self.weights_path_edit.setText(file_path)

    def get_settings(self):
        return {
            "product": self.product_edit.text().strip() or None,
            "area": self.area_edit.text().strip() or None,
            "imgsz": self._parse_int(self.imgsz_edit.text()),
            "tolerance_override": self._parse_float(self.tolerance_edit.text()),
            "conf": self._parse_float(self.conf_edit.text()),
            "sample_dir": self.sample_dir_edit.text().strip() or None,
            "config_path": self.config_path_edit.text().strip() or None,
            "output_dir": self.output_dir_edit.text().strip() or None,
            "weights": self.weights_path_edit.text().strip() or None,
        }

    @staticmethod
    def _parse_int(value: str):
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _parse_float(value: str):
        try:
            return float(value)
        except (TypeError, ValueError):
            return None


class PictureToolGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.config = {}
        self.worker_thread = None
        self.position_settings: dict = {}
        self.task_checkboxes: dict[str, QCheckBox] = {}
        self.task_presets: dict[str, list[str]] = {}
        self.preset_storage: dict = {'presets': {}}
        self.task_status_items: dict[str, QListWidgetItem] = {}
        self._status_icon_cache: dict[str, QIcon] = {}
        self.preset_config_path = PRESET_CONFIG_PATH
        self.init_ui()
        self._load_preset_storage()
        self.load_config()

    def init_ui(self):
        self.setWindowTitle("圖像處理工具")
        self.setGeometry(100, 100, 1400, 850)
        self.setMinimumSize(1000, 700)

        # 全局樣式
        self.setStyleSheet("""
            QMainWindow {
                background-color: #f5f6fa;
            }
            QGroupBox {
                font-size: 10pt;
                font-weight: bold;
                color: #2c3e50;
                border: 2px solid #e1e8ed;
                border-radius: 8px;
                margin-top: 10px;
                padding-top: 10px;
                background-color: white;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 12px;
                padding: 0 6px;
                background-color: white;
            }
            QTextEdit {
                border: 1px solid #e1e8ed;
                border-radius: 6px;
                background-color: #ffffff;
                font-family: 'Consolas', 'Monaco', monospace;
                font-size: 9pt;
                padding: 10px;
            }
            QProgressBar {
                border: 1px solid #e1e8ed;
                border-radius: 6px;
                background-color: #f8f9fa;
                text-align: center;
                font-size: 9pt;
                font-weight: bold;
                color: #2c3e50;
                height: 24px;
            }
            QProgressBar::chunk {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #28a745, stop:1 #20c997);
                border-radius: 5px;
            }
            QLineEdit {
                padding: 6px 10px;
                border: 1px solid #ced4da;
                border-radius: 5px;
                background-color: white;
                font-size: 9pt;
                min-height: 20px;
            }
            QLineEdit:focus {
                border: 2px solid #007bff;
            }
            QLabel {
                font-size: 9pt;
                color: #495057;
            }
            QScrollArea {
                border: none;
            }
        """)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_splitter = QSplitter(Qt.Horizontal)
        central_layout = QVBoxLayout(central_widget)
        central_layout.setContentsMargins(8, 8, 8, 8)
        central_layout.addWidget(main_splitter)

        # 左側面板
        left_scroll = QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        left_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        left_scroll.setMinimumWidth(320)
        left_scroll.setMaximumWidth(450)
        left_panel = self.create_left_panel()
        left_scroll.setWidget(left_panel)

        # 右側面板
        right_panel = self.create_right_panel()

        main_splitter.addWidget(left_scroll)
        main_splitter.addWidget(right_panel)
        main_splitter.setSizes([350, 900])

    def create_left_panel(self):
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setSpacing(10)
        left_layout.setContentsMargins(8, 8, 8, 8)

        # 1. 配置設定
        config_group = QGroupBox("⚙️ 配置設定")
        config_layout = QVBoxLayout(config_group)
        config_layout.setSpacing(8)

        try:
            default_cfg_path = str((Path(__file__).parent / "config.yaml").resolve())
        except Exception:
            default_cfg_path = os.path.join("picture_tool", "config.yaml")

        self.config_path_edit = QLineEdit(default_cfg_path)
        self.config_path_edit.setPlaceholderText("配置文件路徑")

        config_btn_layout = QHBoxLayout()
        config_btn_layout.setSpacing(6)
        config_browse_btn = CompactButton("📁 瀏覽", "primary")
        config_browse_btn.clicked.connect(self.browse_config_file)
        reload_config_btn = CompactButton("🔄 重載", "success")
        reload_config_btn.clicked.connect(self.load_config)
        config_btn_layout.addWidget(config_browse_btn)
        config_btn_layout.addWidget(reload_config_btn)

        config_layout.addWidget(self.config_path_edit)
        config_layout.addLayout(config_btn_layout)

        # 2. 任務選擇 (單一清單，移除重複)
        task_group = QGroupBox("📋 任務選擇")
        task_layout = QVBoxLayout(task_group)
        task_layout.setSpacing(6)

        # 任務清單 - 3列佈局更緊湊
        task_grid = QGridLayout()
        task_grid.setSpacing(4)
        task_grid.setHorizontalSpacing(8)

        self.task_checkboxes = {}
        tasks = [
            ("格式轉換", "🔄"),
            ("異常檢測", "🔍"),
            ("YOLO數據增強", "📊"),
            ("影像增強", "🎨"),
            ("數據分割", "✂️"),
            (YOLO_TRAIN_LABEL, "🚀"),
            ("YOLO評估", "📈"),
            ("生成報告", "📝"),
            ("數據檢查", "✅"),
            ("增強預覽", "👁️"),
            ("批次推論", "⚡"),
            ("LED QC 建模", "🔧"),
            ("LED QC 單張檢測", "🔬"),
            ("LED QC 批次檢測", "🔬"),
            ("LED QC 分析", "📊"),
            (POSITION_TASK_LABEL, "📍"),
        ]

        for index, (task, icon) in enumerate(tasks):
            checkbox = CompactCheckBox(f"{icon} {task}")
            self.task_checkboxes[task] = checkbox
            row = index // 3  # 改為 3 列
            col = index % 3
            task_grid.addWidget(checkbox, row, col)

        task_layout.addLayout(task_grid)

        select_layout = QHBoxLayout()
        select_layout.setSpacing(6)
        select_all_btn = CompactButton('全部選取', 'success')
        deselect_all_btn = CompactButton('全部清除', 'danger')
        self.preset_combo = QComboBox()
        self.preset_combo.setEditable(False)
        self.preset_combo.setMinimumContentsLength(10)
        self.preset_combo.setPlaceholderText('選擇流程')
        self.apply_preset_btn = CompactButton('套用流程', 'primary')

        select_all_btn.clicked.connect(self.select_all_tasks)
        deselect_all_btn.clicked.connect(self.deselect_all_tasks)
        self.apply_preset_btn.clicked.connect(self.apply_selected_preset)
        self.preset_combo.currentIndexChanged.connect(self._on_preset_selection_changed)

        select_layout.addWidget(select_all_btn)
        select_layout.addWidget(deselect_all_btn)
        select_layout.addWidget(self.preset_combo)
        select_layout.addWidget(self.apply_preset_btn)
        select_layout.addStretch()
        task_layout.addLayout(select_layout)

        manage_layout = QHBoxLayout()
        manage_layout.setSpacing(6)
        self.save_preset_btn = CompactButton('儲存流程', 'secondary')
        self.delete_preset_btn = CompactButton('刪除流程', 'danger')
        export_preset_btn = CompactButton('匯出流程', 'secondary')
        import_preset_btn = CompactButton('匯入流程', 'secondary')
        self.save_preset_btn.clicked.connect(self.save_selected_as_preset)
        self.delete_preset_btn.clicked.connect(self.delete_selected_preset)
        self.delete_preset_btn.setEnabled(False)
        export_preset_btn.clicked.connect(self.export_presets)
        import_preset_btn.clicked.connect(self.import_presets)
        manage_layout.addWidget(self.save_preset_btn)
        manage_layout.addWidget(self.delete_preset_btn)
        manage_layout.addWidget(export_preset_btn)
        manage_layout.addWidget(import_preset_btn)
        manage_layout.addStretch()
        task_layout.addLayout(manage_layout)

        status_group = QGroupBox('流程狀態')
        status_layout = QVBoxLayout(status_group)
        self.status_list = QListWidget()
        self.status_list.setIconSize(QSize(14, 14))
        self.status_list.setSelectionMode(QListWidget.NoSelection)
        self.status_list.setFocusPolicy(Qt.NoFocus)
        self.status_list.setMaximumHeight(200)
        status_layout.addWidget(self.status_list)
        reset_status_btn = CompactButton('重設狀態', 'secondary')
        reset_status_btn.clicked.connect(self.reset_task_statuses)
        status_layout.addWidget(reset_status_btn)
        task_layout.addWidget(status_group)
        self._populate_status_items()

        utility_layout = QHBoxLayout()
        utility_layout.setSpacing(6)
        self.quick_nav_btn = QToolButton()
        self.quick_nav_btn.setText('快速導覽')
        self.quick_nav_btn.setPopupMode(QToolButton.InstantPopup)
        self.quick_nav_btn.setToolButtonStyle(Qt.ToolButtonTextOnly)
        self.quick_nav_btn.setCursor(Qt.PointingHandCursor)
        self.quick_nav_btn.setStyleSheet("""
            QToolButton {
                background-color: #6c757d;
                color: white;
                border: none;
                padding: 6px 12px;
                border-radius: 4px;
                font-size: 9pt;
                font-weight: bold;
            }
            QToolButton:hover {
                background-color: #5a6268;
            }
            QToolButton:pressed {
                background-color: #545b62;
            }
            QToolButton::menu-indicator {
                image: none;
            }
        """)
        self.quick_nav_menu = QMenu(self.quick_nav_btn)
        self.quick_nav_menu.addAction('操作指南', self.show_quick_guide)
        self.quick_nav_btn.setMenu(self.quick_nav_menu)
        utility_layout.addWidget(self.quick_nav_btn)

        self.preflight_btn = CompactButton('啟動前檢查', 'primary')
        self.preflight_btn.clicked.connect(self.run_preflight_check)
        utility_layout.addWidget(self.preflight_btn)
        utility_layout.addStretch()
        task_layout.addLayout(utility_layout)

# 3. 位置檢查設定
        position_group = QGroupBox("📍 位置檢查")
        position_layout = QVBoxLayout(position_group)
        position_layout.setSpacing(8)

        # 狀態列
        status_row = QHBoxLayout()
        self.position_enable_cb = CompactCheckBox("✓ 啟用位置驗證")
        self.position_enable_cb.toggled.connect(self._on_position_enabled)
        self.position_summary_label = QLabel("未設定")
        self.position_summary_label.setStyleSheet("""
            QLabel {
                color: #6c757d;
                font-size: 9pt;
                font-style: italic;
                padding: 4px 8px;
                background-color: #f8f9fa;
                border-radius: 4px;
            }
        """)
        status_row.addWidget(self.position_enable_cb)
        status_row.addWidget(self.position_summary_label)
        status_row.addStretch()
        position_layout.addLayout(status_row)

        # 操作按鈕
        btn_row = QHBoxLayout()
        btn_row.setSpacing(6)
        self.position_settings_btn = CompactButton("⚙️ 詳細設定", "primary")
        self.position_build_btn = CompactButton("🔨 生成配置", "secondary")
        self.position_settings_btn.clicked.connect(self._show_position_settings_dialog)
        self.position_build_btn.clicked.connect(self._generate_position_reference)
        btn_row.addWidget(self.position_settings_btn)
        btn_row.addWidget(self.position_build_btn)
        position_layout.addLayout(btn_row)

        # 4. 訓練參數
        options_group = QGroupBox("🎛️ 訓練參數")
        options_layout = QVBoxLayout(options_group)
        options_layout.setSpacing(8)

        self.apply_overrides_cb = CompactCheckBox("✓ 啟用參數覆寫")
        options_layout.addWidget(self.apply_overrides_cb)

        # 參數網格
        param_grid = QGridLayout()
        param_grid.setSpacing(6)
        param_grid.setVerticalSpacing(8)

        self.override_device_edit = QLineEdit()
        self.override_device_edit.setPlaceholderText("0 或 cpu")
        self.override_epochs_edit = QLineEdit()
        self.override_epochs_edit.setPlaceholderText("例: 100")
        self.override_imgsz_edit = QLineEdit()
        self.override_imgsz_edit.setPlaceholderText("例: 640")
        self.override_batch_edit = QLineEdit()
        self.override_batch_edit.setPlaceholderText("例: 16")

        param_grid.addWidget(QLabel("🖥️ 裝置:"), 0, 0)
        param_grid.addWidget(self.override_device_edit, 0, 1)
        param_grid.addWidget(QLabel("🔁 訓練輪數:"), 1, 0)
        param_grid.addWidget(self.override_epochs_edit, 1, 1)
        param_grid.addWidget(QLabel("📏 影像尺寸:"), 2, 0)
        param_grid.addWidget(self.override_imgsz_edit, 2, 1)
        param_grid.addWidget(QLabel("📦 批次大小:"), 3, 0)
        param_grid.addWidget(self.override_batch_edit, 3, 1)

        options_layout.addLayout(param_grid)

        # 工具按鈕
        tool_btn_layout = QHBoxLayout()
        tool_btn_layout.setSpacing(6)
        detect_gpu_btn = CompactButton("🔍 偵測GPU", "primary")
        detect_gpu_btn.clicked.connect(self._detect_gpu)
        self.force_cb = CompactCheckBox("⚠️ 強制執行")
        tool_btn_layout.addWidget(detect_gpu_btn)
        tool_btn_layout.addWidget(self.force_cb)
        tool_btn_layout.addStretch()
        options_layout.addLayout(tool_btn_layout)

        # 5. 控制面板
        control_group = QGroupBox("🎮 執行控制")
        control_layout = QVBoxLayout(control_group)
        control_layout.setSpacing(10)

        # 執行按鈕
        btn_layout = QHBoxLayout()
        btn_layout.setSpacing(8)
        self.start_btn = CompactButton("▶️ 開始執行", "success")
        self.stop_btn = CompactButton("⏹️ 停止", "danger")
        self.stop_btn.setEnabled(False)
        self.start_btn.clicked.connect(self.start_pipeline)
        self.stop_btn.clicked.connect(self.stop_pipeline)
        
        # 放大按鈕
        self.start_btn.setMinimumHeight(36)
        self.stop_btn.setMinimumHeight(36)
        
        btn_layout.addWidget(self.start_btn)
        btn_layout.addWidget(self.stop_btn)
        control_layout.addLayout(btn_layout)

        # 進度條
        self.progress_bar = QProgressBar()
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setMinimumHeight(28)
        control_layout.addWidget(self.progress_bar)

        # 狀態標籤
        self.status_label = QLabel("🟢 就緒")
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setStyleSheet("""
            QLabel {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #d4edda, stop:1 #c3e6cb);
                color: #155724;
                padding: 10px;
                border: 2px solid #c3e6cb;
                border-radius: 6px;
                font-weight: bold;
                font-size: 10pt;
            }
        """)
        control_layout.addWidget(self.status_label)

        # 組裝左側面板
        left_layout.addWidget(config_group)
        left_layout.addWidget(task_group)
        left_layout.addWidget(position_group)
        left_layout.addWidget(options_group)
        left_layout.addWidget(control_group)
        left_layout.addStretch()

        self._sync_position_controls()
        return left_widget

    def create_right_panel(self):
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setSpacing(10)
        right_layout.setContentsMargins(8, 8, 8, 8)

        # 標題卡片

        title_card = QFrame()
        title_card.setStyleSheet("""
            QFrame {
                background-color: white;
                border: none;
                border-bottom: 3px solid #667eea;
                padding: 12px;
            }
        """)
        title_layout = QHBoxLayout(title_card)  # 改為橫向
        title_label = QLabel("🖼️ 圖像處理工具")
        title_label.setStyleSheet("""
            QLabel {
                font-size: 14pt;
                font-weight: bold;
                color: #2c3e50;
                background: transparent;
            }
        """)
        subtitle_label = QLabel("智能視覺處理與 YOLO 訓練平台")
        subtitle_label.setStyleSheet("""
            QLabel {
                font-size: 9pt;
                color: #6c757d;
                background: transparent;
                margin-left: 10px;
            }
        """)
        title_layout.addWidget(title_label)
        title_layout.addWidget(subtitle_label)
        title_layout.addStretch()

        # Tab Widget
        self.tab_widget = QTabWidget()
        self.tab_widget.setStyleSheet("""
            QTabWidget::pane {
                border: 2px solid #e1e8ed;
                border-radius: 8px;
                background-color: white;
                padding: 8px;
            }
            QTabBar::tab {
                background-color: #f8f9fa;
                color: #495057;
                padding: 10px 20px;
                margin-right: 4px;
                border: 2px solid #e1e8ed;
                border-bottom: none;
                border-top-left-radius: 8px;
                border-top-right-radius: 8px;
                font-size: 10pt;
                font-weight: bold;
                min-width: 100px;
            }
            QTabBar::tab:selected {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #007bff, stop:1 #0056b3);
                color: white;
                border-bottom: 2px solid white;
            }
            QTabBar::tab:hover:!selected {
                background-color: #e9ecef;
            }
        """)

        # 日誌頁
        log_widget = QWidget()
        log_layout = QVBoxLayout(log_widget)
        log_layout.setSpacing(8)

        self.log_text = QTextEdit()
        self.log_text.setPlainText(
            "🎉 歡迎使用圖像處理工具！\n"
            "📝 請選擇任務並點擊開始執行。\n"
            "💡 提示：可使用快速預設按鈕快速選擇常用任務組合。\n"
        )
        self.log_text.setStyleSheet("""
            QTextEdit {
                background-color: #2c3e50;
                color: #ecf0f1;
                border: none;
                border-radius: 8px;
                font-family: 'Consolas', 'Monaco', monospace;
                font-size: 9pt;
                padding: 12px;
                line-height: 1.5;
            }
        """)

        clear_log_btn = CompactButton("🗑️ 清空日誌", "danger")
        clear_log_btn.clicked.connect(self.clear_log)

        log_layout.addWidget(self.log_text)
        log_btn_layout = QHBoxLayout()
        log_btn_layout.addStretch()
        log_btn_layout.addWidget(clear_log_btn)
        log_layout.addLayout(log_btn_layout)

        # 配置預覽頁
        config_widget = QWidget()
        config_layout = QVBoxLayout(config_widget)
        config_layout.setSpacing(8)

        self.config_text = QTextEdit()
        self.config_text.setPlainText('請先載入設定檔…')
        self.config_text.setReadOnly(True)
        self.config_text.setStyleSheet("""
            QTextEdit {
                background-color: #f8f9fa;
                color: #2c3e50;
                border: none;
                border-radius: 8px;
                font-family: 'Consolas', 'Monaco', monospace;
                font-size: 9pt;
                padding: 12px;
            }
        """)
        config_layout.addWidget(self.config_text)

        preset_widget = QWidget()
        preset_layout = QVBoxLayout(preset_widget)
        preset_layout.setSpacing(8)

        self.preset_text = QTextEdit()
        self.preset_text.setPlainText('尚未載入流程設定…')
        self.preset_text.setReadOnly(True)
        self.preset_text.setStyleSheet("""
            QTextEdit {
                background-color: #f8f9fa;
                color: #2c3e50;
                border: none;
                border-radius: 8px;
                font-family: 'Consolas', 'Monaco', monospace;
                font-size: 9pt;
                padding: 12px;
            }
        """)
        preset_layout.addWidget(self.preset_text)

        metrics_widget = QWidget()
        metrics_layout = QVBoxLayout(metrics_widget)
        metrics_layout.setSpacing(8)

        self.metrics_text = QTextEdit()
        self.metrics_text.setReadOnly(True)
        self.metrics_text.setStyleSheet("""
            QTextEdit {
                background-color: #f8f9fa;
                color: #2c3e50;
                border: none;
                border-radius: 8px;
                font-family: 'Consolas', 'Monaco', monospace;
                font-size: 9pt;
                padding: 12px;
            }
        """)
        metrics_layout.addWidget(self.metrics_text)
        metrics_refresh_btn = CompactButton('重新整理指標', 'primary')
        metrics_refresh_btn.clicked.connect(self.refresh_metrics_dashboard)
        metrics_layout.addWidget(metrics_refresh_btn)

        # Tabs
        self.log_tab = log_widget
        self.tab_widget.addTab(self.log_tab, '執行紀錄')
        self.config_tab = config_widget
        self.tab_widget.addTab(self.config_tab, '設定預覽')
        self.preset_tab = preset_widget
        self.tab_widget.addTab(self.preset_tab, '流程設定')
        self.metrics_tab = metrics_widget
        self.tab_widget.addTab(self.metrics_tab, '重要指標')

        right_layout.addWidget(title_card)
        right_layout.addWidget(self.tab_widget)

        self._rebuild_quick_nav_menu()
        self.refresh_metrics_dashboard()
        return right_widget


    def _load_preset_storage(self) -> None:
        storage = {'presets': {}}
        config_path = getattr(self, 'preset_config_path', PRESET_CONFIG_PATH)
        try:
            if config_path.exists():
                with config_path.open('r', encoding='utf-8') as handle:
                    loaded = yaml.safe_load(handle) or {}
                if isinstance(loaded, dict) and isinstance(loaded.get('presets'), dict):
                    storage['presets'] = loaded['presets']
            else:
                config_path.parent.mkdir(parents=True, exist_ok=True)
        except Exception as exc:
            self.log_message(f"載入流程設定失敗：{exc}")
        self.preset_storage = storage
        self._rebuild_task_presets()
        self._update_preset_display()

    def _save_preset_storage(self) -> None:
        config_path = getattr(self, 'preset_config_path', PRESET_CONFIG_PATH)
        data = {'presets': self.preset_storage.get('presets', {})}
        try:
            config_path.parent.mkdir(parents=True, exist_ok=True)
            with config_path.open('w', encoding='utf-8') as handle:
                yaml.safe_dump(data, handle, allow_unicode=True, sort_keys=True)
        except Exception as exc:
            self.log_message(f"寫入流程設定失敗：{exc}")

    def _update_preset_display(self) -> None:
        if not hasattr(self, 'preset_text'):
            return
        try:
            display = yaml.safe_dump(
                {'presets': self.preset_storage.get('presets', {})},
                allow_unicode=True,
                sort_keys=True,
            )
        except Exception:
            display = '無法顯示流程設定。'
        self.preset_text.setPlainText(display or '尚無流程。')

    def _rebuild_task_presets(self) -> None:
        if not hasattr(self, 'preset_combo'):
            return
        self.task_presets = {}
        self.preset_combo.blockSignals(True)
        self.preset_combo.clear()
        presets = self.preset_storage.get('presets') if isinstance(self.preset_storage, dict) else {}
        if not isinstance(presets, dict):
            presets = {}
        has_presets = False
        for name in sorted(presets.keys()):
            labels = [label for label in presets.get(name, []) if label in self.task_checkboxes]
            if not labels:
                continue
            key = f"custom::{name}"
            self.task_presets[key] = labels
            self.preset_combo.addItem(f"{name} ({len(labels)})", key)
            has_presets = True
        if not has_presets:
            self.preset_combo.addItem('尚無流程', None)
            self.preset_combo.setEnabled(False)
            self.apply_preset_btn.setEnabled(False)
            self.delete_preset_btn.setEnabled(False)
            self.preset_combo.setCurrentIndex(0)
        else:
            self.preset_combo.setEnabled(True)
            self.apply_preset_btn.setEnabled(True)
            self.delete_preset_btn.setEnabled(False)
            self.preset_combo.setCurrentIndex(0)
        self.preset_combo.blockSignals(False)
        self._on_preset_selection_changed()

    def _on_preset_selection_changed(self) -> None:
        if not hasattr(self, 'preset_combo') or not hasattr(self, 'delete_preset_btn'):
            return
        key = self.preset_combo.currentData()
        can_delete = isinstance(key, str) and key.startswith('custom::')
        self.delete_preset_btn.setEnabled(bool(can_delete))

    def apply_selected_preset(self) -> None:
        if not hasattr(self, 'preset_combo'):
            return
        key = self.preset_combo.currentData()
        labels = self.task_presets.get(key, [])
        if not labels:
            if key is not None:
                self.log_message('流程內容為空或無效。')
            return
        for label, checkbox in self.task_checkboxes.items():
            checkbox.setChecked(label in labels)
        name = key.split('::', 1)[1] if isinstance(key, str) and '::' in key else str(key)
        self.log_message(f"已套用流程：{name}")

    def save_selected_as_preset(self) -> None:
        if not hasattr(self, 'task_checkboxes'):
            return
        selected = [label for label, checkbox in self.task_checkboxes.items() if checkbox.isChecked()]
        if not selected:
            QMessageBox.warning(self, '警告', '請先勾選至少一個任務再儲存流程。')
            return
        name, ok = QInputDialog.getText(self, '儲存流程', '流程名稱：')
        if not ok:
            return
        name = (name or '').strip()
        if not name:
            QMessageBox.warning(self, '警告', '流程名稱不可為空。')
            return
        if not isinstance(self.preset_storage, dict):
            self.preset_storage = {'presets': {}}
        presets = self.preset_storage.setdefault('presets', {})
        if name in presets:
            reply = QMessageBox.question(
                self,
                '覆寫流程',
                f"流程「{name}」已存在，是否覆寫？",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if reply != QMessageBox.Yes:
                return
        presets[name] = selected
        self._save_preset_storage()
        self._rebuild_task_presets()
        self._update_preset_display()
        self.update_config_display()
        target_key = f"custom::{name}"
        for index in range(self.preset_combo.count()):
            if self.preset_combo.itemData(index) == target_key:
                self.preset_combo.setCurrentIndex(index)
                break
        self.log_message(f"已儲存流程：{name}")

    def delete_selected_preset(self) -> None:
        if not hasattr(self, 'preset_combo'):
            return
        key = self.preset_combo.currentData()
        if not isinstance(key, str) or not key.startswith('custom::'):
            QMessageBox.information(self, '提醒', '請先在下拉選單選擇自訂流程再刪除。')
            return
        name = key.split('::', 1)[1]
        presets = self.preset_storage.get('presets') if isinstance(self.preset_storage, dict) else None
        if not isinstance(presets, dict) or name not in presets:
            QMessageBox.information(self, '提醒', '找不到要刪除的流程或已被移除。')
            return
        reply = QMessageBox.question(
            self,
            '刪除流程',
            f"確定要刪除流程「{name}」嗎？",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return
        presets.pop(name, None)
        self._save_preset_storage()
        self._rebuild_task_presets()
        self._update_preset_display()
        self.update_config_display()
        self.log_message(f"已刪除流程：{name}")


    def reset_task_statuses(self, selected: Optional[list[str]] = None) -> None:
        if not hasattr(self, 'status_list') or not self.task_status_items:
            return
        labels = list(self.task_status_items.keys())
        if isinstance(selected, list):
            selected_set = set(selected)
        else:
            selected_set = set(labels)
        pending_color = QColor('#6c757d')
        skipped_color = QColor('#adb5bd')
        for label in labels:
            if label in selected_set:
                self._set_task_status(label, '待執行', pending_color)
            else:
                self._set_task_status(label, '略過', skipped_color)

    def _populate_status_items(self) -> None:
        if not hasattr(self, 'status_list'):
            return
        self.status_list.clear()
        self.task_status_items.clear()
        for label in self.task_checkboxes.keys():
            item = QListWidgetItem()
            item.setData(Qt.UserRole, label)
            item.setFlags(Qt.ItemIsEnabled)
            self.status_list.addItem(item)
            self.task_status_items[label] = item
        self.reset_task_statuses()

    def _set_task_status(self, label: str, status: str, color: QColor) -> None:
        item = self.task_status_items.get(label)
        if not item:
            return
        item.setText(f"{label} - {status}")
        item.setIcon(self._get_status_icon(color))
        item.setData(Qt.UserRole + 1, status)
        background = QColor(color).lighter(170)
        item.setBackground(background)
        item.setForeground(QColor('#212529'))
        item.setToolTip(f"{label} 狀態：{status}")

    def _get_status_icon(self, color: QColor) -> QIcon:
        key = color.name()
        icon = self._status_icon_cache.get(key)
        if icon is None:
            pixmap = QPixmap(14, 14)
            pixmap.fill(color)
            icon = QIcon(pixmap)
            self._status_icon_cache[key] = icon
        return icon

    def _rebuild_quick_nav_menu(self) -> None:
        if not hasattr(self, 'quick_nav_menu') or self.quick_nav_menu is None:
            return
        if not hasattr(self, 'tab_widget'):
            return
        self.quick_nav_menu.clear()
        tab_entries = [
            ('執行紀錄', getattr(self, 'log_tab', None)),
            ('設定預覽', getattr(self, 'config_tab', None)),
            ('流程設定', getattr(self, 'preset_tab', None)),
            ('重要指標', getattr(self, 'metrics_tab', None)),
        ]
        for label, widget in tab_entries:
            if widget is None:
                continue
            action = self.quick_nav_menu.addAction(label)
            action.triggered.connect(lambda _, target=widget: self.tab_widget.setCurrentWidget(target))
        if tab_entries:
            self.quick_nav_menu.addSeparator()
        guide_action = self.quick_nav_menu.addAction('操作指南')
        guide_action.triggered.connect(self.show_quick_guide)

    def show_quick_guide(self) -> None:
        guide_text = (
            "建議流程：\n"
            "1. 準備資料：將原始影像放在 data/raw/images，標註放在 data/raw/labels。\n"
            "2. 依序執行格式轉換 → 增強 → 切割 → 訓練 → 評估。\n"
            "3. 完成後可執行批次推論與 LED QC 檢測。\n\n"
            "Sample dataset：data/sample_dataset（可自行建立或下載示例資料）。"
        )
        QMessageBox.information(self, '快速導覽', guide_text)

    def _validate_pipeline_configuration(self, selected_tasks: list[str]) -> list[str]:
        issues: list[str] = []

        config = self.config if isinstance(self.config, dict) else {}
        yolo_cfg = config.get('yolo_training') if isinstance(config, dict) else None
        if not isinstance(yolo_cfg, dict):
            yolo_cfg = {}
        pos_cfg = yolo_cfg.get('position_validation') if isinstance(yolo_cfg, dict) else None
        if not isinstance(pos_cfg, dict):
            pos_cfg = {}

        want_position_validation = POSITION_TASK_LABEL in selected_tasks
        train_selected = YOLO_TRAIN_LABEL in selected_tasks

        if want_position_validation and not pos_cfg.get('enabled'):
            issues.append('已選擇位置檢查任務，但未啟用位置檢查設定。請在左側啟用並填寫必填欄位。')

        if pos_cfg.get('enabled') and (want_position_validation or train_selected):
            missing_fields: list[str] = []
            if not pos_cfg.get('product'):
                missing_fields.append('產品')
            if not pos_cfg.get('area'):
                missing_fields.append('區域')
            if not (pos_cfg.get('config_path') or pos_cfg.get('config')):
                missing_fields.append('位置設定檔')
            if missing_fields:
                issues.append('定位檢查缺少必要欄位：' + '、'.join(missing_fields))

            config_path = pos_cfg.get('config_path')
            if isinstance(config_path, str) and config_path and not Path(config_path).exists():
                issues.append(f'位置設定檔不存在：{config_path}')

            sample_dir = pos_cfg.get('sample_dir')
            if isinstance(sample_dir, str) and sample_dir and not Path(sample_dir).exists():
                issues.append(f'定位樣本資料夾不存在：{sample_dir}')

        if train_selected:
            dataset_dir = yolo_cfg.get('dataset_dir') if isinstance(yolo_cfg, dict) else None
            if isinstance(dataset_dir, str) and dataset_dir and not Path(dataset_dir).exists():
                issues.append(f'YOLO 訓練資料集不存在：{dataset_dir}')

        return issues

    def run_preflight_check(self) -> None:
        selected_tasks = self.get_selected_tasks()
        if not selected_tasks:
            QMessageBox.information(self, '提示', '請先勾選至少一個任務再進行檢查。')
            return
        self._apply_position_settings()
        issues = self._validate_pipeline_configuration(selected_tasks)
        if issues:
            QMessageBox.warning(
                self,
                '啟動前檢查',
                '\n'.join(issues),
            )
        else:
            QMessageBox.information(
                self,
                '啟動前檢查',
                '所有檢查項目通過，準備就緒。',
            )

    def export_presets(self) -> None:
        if not isinstance(self.preset_storage, dict) or not self.preset_storage.get('presets'):
            QMessageBox.information(self, '提醒', '目前沒有可匯出的流程。')
            return
        file_path, _ = QFileDialog.getSaveFileName(
            self, '匯出流程設定', str(Path.cwd() / 'preset_export.yaml'), 'YAML Files (*.yaml *.yml)')
        if not file_path:
            return
        try:
            with open(file_path, 'w', encoding='utf-8') as fh:
                yaml.safe_dump({'presets': self.preset_storage.get('presets', {})}, fh, allow_unicode=True, sort_keys=True)
            message = f'流程設定已匯出至：\n{file_path}'
            QMessageBox.information(
                self,
                '匯出成功',
                message,
            )
        except Exception as exc:
            error_message = f'寫入檔案時發生錯誤：\n{exc}'
            QMessageBox.critical(
                self,
                '匯出失敗',
                error_message,
            )

    def import_presets(self) -> None:
        file_path, _ = QFileDialog.getOpenFileName(self, '匯入流程設定', '', 'YAML Files (*.yaml *.yml)')
        if not file_path:
            return
        try:
            with open(file_path, 'r', encoding='utf-8') as fh:
                data = yaml.safe_load(fh) or {}
            presets = data.get('presets') if isinstance(data, dict) else None
            if not isinstance(presets, dict):
                QMessageBox.warning(self, '匯入失敗', '檔案格式不符合預期。')
                return
            self.preset_storage.setdefault('presets', {}).update(presets)
            self._save_preset_storage()
            self._rebuild_task_presets()
            self._update_preset_display()
            QMessageBox.information(
                self,
                '匯入完成',
                '流程設定已更新。',
            )
        except Exception as exc:
            error_message = f'讀取檔案時發生錯誤：\n{exc}'
            QMessageBox.critical(
                self,
                '匯入失敗',
                error_message,
            )

    def refresh_metrics_dashboard(self) -> None:
        if not hasattr(self, 'metrics_text'):
            return
        summary_lines = []
        summary_lines.append('【YOLO 訓練指標】')
        yolo_stats = self._load_latest_yolo_metrics()
        if yolo_stats:
            summary_lines.extend(yolo_stats)
        else:
            summary_lines.append('尚未找到訓練結果 (runs/detect)。')
        summary_lines.append('')
        summary_lines.append('【LED QC 統計】')
        led_stats = self._load_latest_led_metrics()
        if led_stats:
            summary_lines.extend(led_stats)
        else:
            summary_lines.append('尚未找到 LED QC 報表 (reports/led_qc)。')
        self.metrics_text.setPlainText("\n".join(summary_lines))


    def _load_latest_yolo_metrics(self) -> list[str]:
        from datetime import datetime

        search_roots: list[Path] = []
        if isinstance(self.config, dict):
            project = self.config.get('yolo_training', {}).get('project')
            if isinstance(project, str) and project:
                search_roots.append(Path(project))
        for fallback in ('./runs/train', './runs/detect'):
            candidate = Path(fallback)
            if candidate not in search_roots:
                search_roots.append(candidate)

        candidates: list[Path] = []
        for root in search_roots:
            try:
                if root.exists():
                    candidates.extend(root.glob('**/results.csv'))
            except Exception:
                continue
        if not candidates:
            return []

        latest = max(candidates, key=lambda p: p.stat().st_mtime)
        try:
            with latest.open('r', encoding='utf-8') as fh:
                rows = list(csv.DictReader(fh))
        except Exception:
            return []
        if not rows:
            return []

        last = rows[-1]

        def pick(*keys: str) -> str:
            for key in keys:
                value = last.get(key)
                if value not in (None, '', 'nan'):
                    return str(value)
            return 'N/A'

        epoch = pick('epoch', 'Epoch', 'epochs')
        map50 = pick('metrics/mAP50', 'metrics/mAP_50', 'mAP50', 'map50')
        map50_95 = pick('metrics/mAP50-95', 'metrics/mAP_50_95', 'mAP50-95', 'map50-95')
        precision = pick('metrics/precision(B)', 'precision', 'metrics/precision')
        recall = pick('metrics/recall(B)', 'recall', 'metrics/recall')
        box_loss = pick('train/box_loss', 'box_loss', 'loss/box')
        cls_loss = pick('train/cls_loss', 'cls_loss', 'loss/cls')

        last_update = datetime.fromtimestamp(latest.stat().st_mtime).strftime('%Y-%m-%d %H:%M')
        metrics = [
            f"結果目錄：{self._format_relative_path(latest.parent)}",
            f"最後更新：{last_update}",
            f"最新 Epoch：{epoch}",
            f"mAP50：{map50}",
            f"mAP50-95：{map50_95}",
            f"Precision：{precision}",
            f"Recall：{recall}",
        ]
        if box_loss != 'N/A' or cls_loss != 'N/A':
            metrics.append(f"Box/Cls Loss：{box_loss} / {cls_loss}")
        return metrics

    def _load_latest_led_metrics(self) -> list[str]:
        from datetime import datetime

        led_config = self.config.get('led_qc_enhanced', {}) if isinstance(self.config, dict) else {}
        detect_dir_cfg = led_config.get('detect_dir', {}) if isinstance(led_config, dict) else {}
        led_dir = Path(detect_dir_cfg.get('out_dir') or './reports/led_qc/batch')
        fallback_dirs = [Path('./reports/led_qc'), Path('./reports/led_qc/batch')]

        search_roots: list[Path] = []
        for root in [led_dir, *fallback_dirs]:
            if root not in search_roots:
                search_roots.append(root)

        candidates: list[Path] = []
        for root in search_roots:
            if root and root.exists():
                candidates.extend(root.glob('**/*.csv'))
        if not candidates:
            return []

        latest = max(candidates, key=lambda p: p.stat().st_mtime)
        try:
            with latest.open('r', encoding='utf-8') as fh:
                rows = list(csv.DictReader(fh))
        except Exception:
            return []
        if not rows:
            return []

        total = len(rows)
        pass_markers = {'PASS', 'OK', '正常', '合格', '良品'}
        fail_markers = {'FAIL', 'NG', '異常', '不良', 'FAILURE'}

        anomalies = 0
        for row in rows:
            raw_status = None
            for key in ('color_status', 'status', 'result', 'decision'):
                if row.get(key):
                    raw_status = row.get(key)
                    break
            status = (raw_status or '').strip()
            normalized = status.upper()
            if not status:
                continue
            if normalized in pass_markers or status in pass_markers:
                continue
            if normalized in fail_markers or status in fail_markers or status:
                anomalies += 1

        pass_count = total - anomalies
        pass_rate = (pass_count / total * 100) if total else 0.0
        anomaly_rate = (anomalies / total * 100) if total else 0.0
        last_update = datetime.fromtimestamp(latest.stat().st_mtime).strftime('%Y-%m-%d %H:%M')

        metrics = [
            f"統計檔案：{self._format_relative_path(latest)}",
            f"最後更新：{last_update}",
            f"樣本總數：{total}",
        ]
        if total:
            metrics.append(f"疑似異常：{anomalies} ({anomaly_rate:.1f}%)")
            metrics.append(f"良品數：{pass_count}")
            metrics.append(f"通過率：{pass_rate:.1f}%")
        return metrics

    def _format_relative_path(self, path: Path) -> str:
        try:
            return str(path.resolve().relative_to(Path.cwd()))
        except Exception:
            return str(path)

    def _detect_gpu(self):
        try:
            import torch
            if torch.cuda.is_available():
                self.override_device_edit.setText("0")
                self.log_message(f"🎮 偵測到 GPU: {torch.cuda.get_device_name(0)}")
            else:
                self.override_device_edit.setText("cpu")
                self.log_message("⚠️ 未偵測到 GPU，將使用 CPU")
        except Exception as e:
            self.override_device_edit.setText("cpu")
            self.log_message(f"⚠️ GPU 偵測失敗: {e}")
            
    def _select_image_source(self, default_dir: str = "") -> tuple[list[Path], str]:
        """
        選擇圖片來源(單張或資料夾)
        
        Returns:
            tuple: (圖片路徑列表, 選擇的目錄路徑)
        """
        source_choice = QMessageBox(self)
        source_choice.setWindowTitle("選擇來源")
        source_choice.setText("要使用單張圖片還是資料夾?")
        image_button = source_choice.addButton("選擇圖片", QMessageBox.ActionRole)
        folder_button = source_choice.addButton("選擇資料夾", QMessageBox.ActionRole)
        cancel_button = source_choice.addButton(QMessageBox.Cancel)
        source_choice.exec_()
        
        if source_choice.clickedButton() == cancel_button:
            return [], ""
        
        images = []
        selected_dir = ""
        exts = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff', '.webp')
        
        if source_choice.clickedButton() == image_button:
            files, _ = QFileDialog.getOpenFileNames(
                self, "選擇參考圖片", default_dir,
                "Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.webp)"
            )
            if files:
                images = [Path(f) for f in files]
                selected_dir = str(images[0].parent)
        
        elif source_choice.clickedButton() == folder_button:
            folder = QFileDialog.getExistingDirectory(self, "選擇圖片資料夾", default_dir)
            if folder:
                folder_path = Path(folder)
                images = [p for p in sorted(folder_path.iterdir()) if p.suffix.lower() in exts]
                selected_dir = str(folder_path)
        
        return images, selected_dir
    
    def _generate_position_reference(self):
        self._apply_position_settings()
        settings = dict(getattr(self, "position_settings", {}) or {})
        product = settings.get("product")
        area = settings.get("area")
        if not product or not area:
            QMessageBox.warning(self, "警告", "請先在位置檢查設定中填寫產品與區域。")
            return

        tolerance = settings.get("tolerance_override")
        if tolerance is None:
            tolerance = 0.0
        
        imgsz_value = settings.get("imgsz")
        try:
            imgsz_int = int(imgsz_value) if imgsz_value else 640
        except ValueError:
            imgsz_int = 640

        conf_value = settings.get("conf")
        try:
            conf = float(conf_value) if conf_value else 0.25
        except ValueError:
            conf = 0.25

        # 選擇圖片來源
        default_sample_dir = settings.get("sample_dir") or settings.get("output_dir") or ""
        images, selected_dir = self._select_image_source(default_sample_dir)
        
        if not images:
            QMessageBox.information(self, "提示", "未選取任何圖片。")
            return
        
        # 更新設定中的樣本目錄
        if selected_dir:
            settings["sample_dir"] = selected_dir

        weights_path = settings.get("weights")
        chosen_weights: Optional[str] = None
        
        if weights_path and Path(weights_path).exists():
            reuse = QMessageBox.question(
                self, "使用既有權重",
                f"是否使用目前設定的權重？\n{weights_path}",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes
            )
            if reuse == QMessageBox.Yes:
                chosen_weights = weights_path
        
        if not chosen_weights:
            default_weights_dir = str(Path(weights_path).parent) if weights_path else ""
            chosen_weights, _ = QFileDialog.getOpenFileName(
                self, "選擇模型權重", default_weights_dir,
                "PyTorch Weights (*.pt);;All Files (*.*)"
            )
            if not chosen_weights:
                QMessageBox.information(self, "提示", "未選擇權重檔。")
                return
        
        settings["weights"] = chosen_weights
        weights_path = chosen_weights

        save_path, _ = QFileDialog.getSaveFileName(
            self, "輸出位置設定檔",
            f"{product}_{area}_position.yaml",
            "YAML Files (*.yaml *.yml)"
        )
        if not save_path:
            return
        save_path = Path(save_path)

        try:
            from ultralytics import YOLO
        except Exception as exc:
            QMessageBox.critical(self, "錯誤", f"無法載入 Ultralytics YOLO：{exc}")
            return

        try:
            from picture_tool.position.yolo_position_validator import convert_results_to_detections
        except Exception as exc:
            QMessageBox.critical(self, "錯誤", f"無法載入位置檢查工具：{exc}")
            return

        self.log_message(f"🔨 使用權重 {weights_path} 生成參考設定...")
        
        try:
            model = YOLO(weights_path)
        except Exception as exc:
            QMessageBox.critical(self, "錯誤", f"載入權重失敗：{exc}")
            return

        device_value = settings.get("device")
        if not device_value:
            device_value = self.config.get("yolo_training", {}).get("device") if isinstance(self.config, dict) else None
        device_str = str(device_value) if device_value else "cpu"

        boxes_by_class: dict[str, list[list[int]]] = {}
        for img_path in images:
            try:
                results = model(str(img_path), imgsz=imgsz_int, device=device_str, conf=conf)
            except Exception as exc:
                self.log_message(f"⚠️ 推論失敗 {img_path.name}: {exc}")
                continue
            
            for res in results:
                detections = convert_results_to_detections(res, imgsz_int)
                for det in detections:
                    cls = str(det.get("class"))
                    bbox = det.get("bbox")
                    if not isinstance(bbox, list) or len(bbox) != 4:
                        continue
                    boxes_by_class.setdefault(cls, []).append([int(v) for v in bbox])

        if not boxes_by_class:
            QMessageBox.warning(self, "提醒", "未偵測到任何物件，無法生成位置設定。")
            return

        def aggregate(boxes: list[list[int]]) -> dict[str, int]:
            x1 = min(b[0] for b in boxes)
            y1 = min(b[1] for b in boxes)
            x2 = max(b[2] for b in boxes)
            y2 = max(b[3] for b in boxes)
            return {"x1": int(x1), "y1": int(y1), "x2": int(x2), "y2": int(y2)}

        expected_boxes = {cls: aggregate(bxs) for cls, bxs in boxes_by_class.items()}
        position_config = {
            str(product): {
                str(area): {
                    "enabled": True,
                    "mode": "bbox",
                    "tolerance": float(tolerance or 0.0),
                    "expected_boxes": expected_boxes,
                }
            }
        }
        if imgsz_int:
            position_config[str(product)][str(area)]["imgsz"] = int(imgsz_int)

        try:
            with open(save_path, "w", encoding="utf-8") as fh:
                yaml.safe_dump(position_config, fh, allow_unicode=True, sort_keys=False)
        except Exception as exc:
            QMessageBox.critical(self, "錯誤", f"寫入設定檔失敗：{exc}")
            return

        if not hasattr(self, "position_settings"):
            self.position_settings = {}
        self.position_settings.update({
            "product": product,
            "area": area,
            "tolerance_override": tolerance,
            "imgsz": imgsz_int,
            "conf": conf,
            "config_path": str(save_path),
        })
        self._apply_position_settings()

        self.log_message(f"✅ 已產生位置設定：{save_path}")
        QMessageBox.information(self, "完成", f"位置設定已寫入：{save_path}")

    def _on_position_enabled(self, checked: bool):
        self._sync_position_controls()
        self.update_config_display()

    def _show_position_settings_dialog(self):
        dialog = PositionSettingsDialog(self, dict(self.position_settings))
        if dialog.exec_() == QDialog.Accepted:
            updated = dialog.get_settings()
            for key in ("config", "device"):
                if key not in updated:
                    updated[key] = self.position_settings.get(key) if hasattr(self, "position_settings") else None
            self.position_settings = updated
            self._sync_position_controls()
            self.update_config_display()

    def _sync_position_controls(self):
        if not hasattr(self, "position_settings_btn"):
            return
        
        settings = getattr(self, "position_settings", {}) if hasattr(self, "position_settings") else {}
        product = settings.get("product")
        area = settings.get("area")
        summary_core = " / ".join([str(v) for v in (product, area) if v]) or "未設定"
        is_enabled = bool(self.position_enable_cb.isChecked()) if hasattr(self, "position_enable_cb") else False
        
        if hasattr(self, "position_summary_label"):
            # 🔧 優化:快取當前狀態,避免重複設置
            if not hasattr(self, '_last_position_state'):
                self._last_position_state = {}
            
            current_state = (is_enabled, summary_core)
            if self._last_position_state.get('state') == current_state:
                return  # 狀態未變,跳過更新
            
            self._last_position_state['state'] = current_state
            
            if is_enabled and summary_core != "未設定":
                display_text = f"✅ {summary_core}"
                style = """
                    QLabel {
                        color: #155724;
                        background-color: #d4edda;
                        border: 1px solid #c3e6cb;
                        font-size: 9pt;
                        font-weight: bold;
                        padding: 4px 8px;
                        border-radius: 4px;
                    }
                """
            elif is_enabled:
                display_text = "⚠️ 已啟用但未設定"
                style = """
                    QLabel {
                        color: #856404;
                        background-color: #fff3cd;
                        border: 1px solid #ffeaa7;
                        font-size: 9pt;
                        padding: 4px 8px;
                        border-radius: 4px;
                    }
                """
            else:
                display_text = "⭕ 未啟用"
                style = """
                    QLabel {
                        color: #6c757d;
                        background-color: #f8f9fa;
                        border: 1px solid #dee2e6;
                        font-size: 9pt;
                        font-style: italic;
                        padding: 4px 8px;
                        border-radius: 4px;
                    }
                """
            self.position_summary_label.setText(display_text)
            self.position_summary_label.setStyleSheet(style)

        if hasattr(self, "position_enable_cb"):
            self.position_enable_cb.blockSignals(True)
            self.position_enable_cb.setChecked(is_enabled)
            self.position_enable_cb.blockSignals(False)
    def _populate_position_widgets(self):
        if not hasattr(self, "position_enable_cb"):
            return
        
        ycfg = self.config.get("yolo_training", {}) if isinstance(self.config, dict) else {}
        pos_cfg = ycfg.get("position_validation", {}) if isinstance(ycfg, dict) else {}
        
        self.position_settings = {
            "product": pos_cfg.get("product"),
            "area": pos_cfg.get("area"),
            "imgsz": pos_cfg.get("imgsz"),
            "tolerance_override": pos_cfg.get("tolerance_override"),
            "conf": pos_cfg.get("conf"),
            "sample_dir": pos_cfg.get("sample_dir"),
            "config_path": pos_cfg.get("config_path"),
            "output_dir": pos_cfg.get("output_dir"),
            "weights": pos_cfg.get("weights"),
            "config": pos_cfg.get("config"),
            "device": pos_cfg.get("device"),
        }
        
        if self.position_settings.get("imgsz") in (None, "", 0):
            self.position_settings["imgsz"] = ycfg.get("imgsz", 640)
        if self.position_settings.get("conf") in (None, ""):
            self.position_settings["conf"] = 0.25
        if self.position_settings.get("tolerance_override") in (None, ""):
            self.position_settings["tolerance_override"] = None
        
        enabled = bool(pos_cfg.get("enabled", False))
        self.position_enable_cb.blockSignals(True)
        self.position_enable_cb.setChecked(enabled)
        self.position_enable_cb.blockSignals(False)
        self._sync_position_controls()

    def _apply_position_settings(self):
        if not hasattr(self, "position_enable_cb"):
            return
        
        if not isinstance(self.config, dict):
            self.config = {}
        
        ycfg = self.config.get("yolo_training")
        if not isinstance(ycfg, dict):
            ycfg = {}
        
        settings = dict(getattr(self, "position_settings", {}) or {})

        def _ensure_int(value):
            if isinstance(value, int):
                return value
            try:
                return int(str(value))
            except (TypeError, ValueError):
                return None

        def _ensure_float(value):
            if isinstance(value, float):
                return value
            try:
                return float(str(value))
            except (TypeError, ValueError):
                return None

        pos_cfg = {
            "enabled": bool(self.position_enable_cb.isChecked()),
            "product": settings.get("product"),
            "area": settings.get("area"),
            "imgsz": _ensure_int(settings.get("imgsz")),
            "tolerance_override": _ensure_float(settings.get("tolerance_override")),
            "conf": _ensure_float(settings.get("conf")),
            "sample_dir": settings.get("sample_dir"),
            "config_path": settings.get("config_path"),
            "output_dir": settings.get("output_dir"),
            "weights": settings.get("weights"),
            "config": settings.get("config"),
            "device": settings.get("device"),
        }
        ycfg["position_validation"] = pos_cfg
        self.config["yolo_training"] = ycfg
        self._sync_position_controls()

    def browse_config_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "選擇配置文件", "", "YAML files (*.yaml *.yml)"
        )
        if file_path:
            self.config_path_edit.setText(file_path)
            self.load_config()

    def load_default_config(self):
        default_config = {
            "pipeline": {
                "log_file": "pipeline.log",
                "tasks": [
                    {"name": "format_conversion", "enabled": True},
                    {"name": "anomaly_detection", "enabled": True},
                    {"name": "yolo_augmentation", "enabled": True},
                    {"name": "image_augmentation", "enabled": True},
                    {"name": "dataset_splitter", "enabled": True},
                    {"name": "yolo_train", "enabled": False},
                    {"name": "yolo_evaluation", "enabled": False},
                    {"name": "generate_report", "enabled": True},
                    {"name": "dataset_lint", "enabled": True},
                    {"name": "aug_preview", "enabled": True},
                ],
            },
            "format_conversion": {
                "input_formats": [".bmp", ".tiff"],
                "output_format": ".png",
                "input_dir": "input/",
                "output_dir": "output/",
            },
            "yolo_training": {
                "dataset_dir": "./data/split",
                "class_names": [],
                "imgsz": 640,
                "position_validation": {
                    "enabled": False,
                    "product": None,
                    "area": None,
                    "imgsz": 640,
                    "sample_dir": None,
                    "config_path": None,
                    "output_dir": None,
                    "weights": None,
                    "conf": 0.25,
                    "tolerance_override": None,
                },
            },
        }
        self.config = default_config
        self._populate_position_widgets()
        self._load_preset_storage()
        self.update_config_display()
        self.refresh_metrics_dashboard()

    def load_config(self):
        config_path = self.config_path_edit.text()
        
        # 🔧 優化:更詳細的錯誤處理
        if not config_path:
            self.log_message("⚠️ 未指定配置文件路徑,載入預設配置")
            self.load_default_config()
            return
        
        try:
            if not os.path.exists(config_path):
                self.log_message(f"⚠️ 配置文件不存在: {config_path},載入預設配置")
                self.load_default_config()
                return
                
            with open(config_path, "r", encoding="utf-8") as f:
                loaded_config = yaml.safe_load(f)
                
            if not isinstance(loaded_config, dict):
                raise ValueError("配置文件格式錯誤:根節點必須是字典")
                
            self.config = loaded_config
            self.log_message(f"✅ 成功載入配置文件: {config_path}")
            self._populate_position_widgets()
            self._load_preset_storage()
            self.update_config_display()
            self.refresh_metrics_dashboard()
            
        except yaml.YAMLError as e:
            self.log_message(f"❌ YAML 解析錯誤: {str(e)}")
            QMessageBox.critical(self, "配置錯誤", f"配置文件格式錯誤:\n{str(e)}\n\n將載入預設配置")
            self.load_default_config()
        except Exception as e:
            self.log_message(f"❌ 載入配置文件失敗: {str(e)}")
            QMessageBox.critical(self, "載入錯誤", f"無法載入配置文件:\n{str(e)}\n\n將載入預設配置")
            self.load_default_config()

    def update_config_display(self):
        self._apply_position_settings()
        config_text = yaml.dump(
            self.config, default_flow_style=False, allow_unicode=True, indent=2
        )
        self.config_text.setPlainText(config_text)
        self._update_preset_display()

    def select_all_tasks(self):
        for checkbox in self.task_checkboxes.values():
            checkbox.setChecked(True)
        self.log_message("✅ 已選擇所有任務")

    def deselect_all_tasks(self):
        for checkbox in self.task_checkboxes.values():
            checkbox.setChecked(False)
        self.log_message("❌ 已取消所有任務")

    def get_selected_tasks(self):
        selected_tasks = []
        for task_name, checkbox in self.task_checkboxes.items():
            if checkbox.isChecked():
                selected_tasks.append(task_name)
        return selected_tasks

    def start_pipeline(self):
        selected_tasks = self.get_selected_tasks()
        if not selected_tasks:
            QMessageBox.warning(self, "警告", "請至少選擇一個任務！")
            return

        if self.worker_thread is not None:
            if self.worker_thread.isRunning():
                QMessageBox.warning(self, "警告", "已有任務在執行中!")
                return
            # 斷開舊連接並清理
            self.worker_thread.progress_updated.disconnect()
            self.worker_thread.task_started.disconnect()
            self.worker_thread.task_completed.disconnect()
            self.worker_thread.log_message.disconnect()
            self.worker_thread.finished_signal.disconnect()
            self.worker_thread.error_occurred.disconnect()
            self.worker_thread.deleteLater()
            self.worker_thread = None

        self.reset_task_statuses(selected_tasks)
        self._apply_position_settings()
        validation_errors = self._validate_pipeline_configuration(selected_tasks)
        if validation_errors:
            QMessageBox.warning(
                self,
                '啟動前檢查',
                '\n'.join(validation_errors),
            )
            return

        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.progress_bar.setValue(0)
        self.status_label.setText("🔄 執行中...")
        self.status_label.setStyleSheet("""
            QLabel {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #fff3cd, stop:1 #ffeaa7);
                color: #856404;
                padding: 10px;
                border: 2px solid #ffeaa7;
                border-radius: 6px;
                font-weight: bold;
                font-size: 10pt;
            }
        """)

        try:
            if hasattr(self, "apply_overrides_cb") and self.apply_overrides_cb.isChecked():
                yt = self.config.get("yolo_training", {}) or {}
                dev = self.override_device_edit.text().strip()
                ep = self.override_epochs_edit.text().strip()
                im = self.override_imgsz_edit.text().strip()
                bt = self.override_batch_edit.text().strip()
                if dev:
                    yt["device"] = dev
                if ep.isdigit():
                    yt["epochs"] = int(ep)
                if im.isdigit():
                    yt["imgsz"] = int(im)
                if bt.isdigit():
                    yt["batch"] = int(bt)
                self.config["yolo_training"] = yt
                
                ye = self.config.get("yolo_evaluation", {}) or {}
                if yt.get("device"):
                    ye["device"] = yt["device"]
                self.config["yolo_evaluation"] = ye
                
                bi = self.config.get("batch_inference", {}) or {}
                if yt.get("device"):
                    bi["device"] = yt["device"]
                self.config["batch_inference"] = bi
            
            pos_cfg: dict = {}
            if isinstance(self.config, dict):
                ycfg = self.config.get("yolo_training", {})
                if isinstance(ycfg, dict):
                    pos_cfg = ycfg.get("position_validation", {}) or {}
            
            want_position_validation = POSITION_TASK_LABEL in selected_tasks
            train_selected = YOLO_TRAIN_LABEL in selected_tasks
            
            if want_position_validation and not pos_cfg.get("enabled"):
                QMessageBox.warning(
                    self, "警告",
                    "已選擇位置檢查任務，但未啟用或設定位置檢查。請先在左側啟用並填寫必要欄位。"
                )
                self.on_pipeline_finished()
                return
            
            if pos_cfg.get("enabled") and (want_position_validation or train_selected):
                missing_fields = []
                if not pos_cfg.get("product"):
                    missing_fields.append("產品")
                if not pos_cfg.get("area"):
                    missing_fields.append("區域")
                if not (pos_cfg.get("config_path") or pos_cfg.get("config")):
                    missing_fields.append("位置設定檔")
                if missing_fields:
                    QMessageBox.warning(
                        self, "警告",
                        "定位檢查缺少必要欄位：" + "、".join(missing_fields)
                    )
                    self.on_pipeline_finished()
                    return
            
            if hasattr(self, "force_cb") and self.force_cb.isChecked():
                pl = self.config.get("pipeline", {}) or {}
                pl["force"] = True
                self.config["pipeline"] = pl
        except Exception:
            pass

        self.worker_thread = WorkerThread(
            selected_tasks, self.config, self.config_path_edit.text()
        )
        self.worker_thread.progress_updated.connect(self.update_progress)
        self.worker_thread.task_started.connect(self.on_task_started)
        self.worker_thread.task_completed.connect(self.on_task_completed)
        self.worker_thread.log_message.connect(self.log_message)
        self.worker_thread.finished_signal.connect(self.on_pipeline_finished)
        self.worker_thread.error_occurred.connect(self.on_error_occurred)

        self.worker_thread.start()
        self.log_message(f"🚀 開始執行管道，選中的任務: {', '.join(selected_tasks)}")

    def stop_pipeline(self):
        if self.worker_thread:
            self.worker_thread.cancel()
            if hasattr(self, 'task_status_items'):
                for label, item in self.task_status_items.items():
                    if '執行中' in item.text():
                        self._set_task_status(label, '已停止', QColor('#dc3545'))
            self.log_message("⏹️ 正在停止執行...")


    def update_progress(self, value):
        self.progress_bar.setValue(value)

    def on_task_started(self, task_name):
        self.log_message(f"⚡ 開始執行任務: {task_name}")
        if hasattr(self, 'task_status_items'):
            self._set_task_status(task_name, '執行中', QColor('#17a2b8'))

    def on_task_completed(self, task_name):
        self.log_message(f"✅ 任務完成: {task_name}")
        if hasattr(self, 'task_status_items'):
            self._set_task_status(task_name, '完成', QColor('#28a745'))

    def on_pipeline_finished(self):
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.status_label.setText("🟢 完成")
        self.status_label.setStyleSheet("""
            QLabel {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #d4edda, stop:1 #c3e6cb);
                color: #155724;
                padding: 10px;
                border: 2px solid #c3e6cb;
                border-radius: 6px;
                font-weight: bold;
                font-size: 10pt;
            }
        """)
        if hasattr(self, 'task_status_items'):
            for label, item in self.task_status_items.items():
                if '執行中' in item.text():
                    self._set_task_status(label, '完成', QColor('#28a745'))
        self.refresh_metrics_dashboard()
        self.log_message("🎉 所有任務執行完成！")

    def on_error_occurred(self, error_message):
        self.log_message(f'❌ 錯誤: {error_message}')
        detail = f'執行過程中發生錯誤：\n{error_message}'
        QMessageBox.critical(self, '錯誤', detail)
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.status_label.setText('🛑 發生錯誤')
        self.status_label.setStyleSheet("""
            QLabel {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #f8d7da, stop:1 #f1b0b7);
                color: #721c24;
                padding: 10px;
                border: 2px solid #f5c6cb;
                border-radius: 6px;
                font-weight: bold;
                font-size: 10pt;
            }
        """)
        if hasattr(self, 'task_status_items'):
            error_color = QColor('#e83e8c')
            for label, item in self.task_status_items.items():
                if '執行中' in item.text():
                    self._set_task_status(label, '錯誤', error_color)
        if self.worker_thread:
            try:
                self.worker_thread.wait(100)
            except Exception:
                pass
            self.worker_thread = None

    def log_message(self, message):
        from datetime import datetime
        timestamp = datetime.now().strftime("%H:%M:%S")
        formatted_message = f"[{timestamp}] {message}"
        
        # 🔧 優化:限制日誌行數,避免記憶體無限增長
        max_lines = 1000
        current_text = self.log_text.toPlainText()
        lines = current_text.split('\n')
        
        if len(lines) >= max_lines:
            # 保留最新的 80% 日誌
            keep_lines = int(max_lines * 0.8)
            self.log_text.setPlainText('\n'.join(lines[-keep_lines:]))
        
        self.log_text.append(formatted_message)
        
        # 優化:使用 QTimer 延遲滾動,避免頻繁更新
        if not hasattr(self, '_scroll_timer'):
            self._scroll_timer = None
        
        if self._scroll_timer:
            self._scroll_timer.stop()
        
        from PyQt5.QtCore import QTimer
        self._scroll_timer = QTimer()
        self._scroll_timer.setSingleShot(True)
        self._scroll_timer.timeout.connect(lambda: self.log_text.verticalScrollBar().setValue(
            self.log_text.verticalScrollBar().maximum()
        ))
        self._scroll_timer.start(100)  # 100ms 後滾動

    def clear_log(self):
        self.log_text.clear()
        self.log_message("🗑️ 日誌已清空")

    def closeEvent(self, event):
        if self.worker_thread and self.worker_thread.isRunning():
            reply = QMessageBox.question(
                self, "確認退出",
                "任務正在執行中，確定要退出嗎？",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No
            )
            if reply == QMessageBox.Yes:
                self.worker_thread.cancel()
                self.worker_thread.wait()
                event.accept()
            else:
                event.ignore()
        else:
            event.accept()


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("圖像處理工具")
    app.setApplicationVersion("2.0")
    app.setOrganizationName("ImageTool Corp")

    font = app.font()
    font.setPointSize(9)
    app.setFont(font)

    window = PictureToolGUI()
    window.show()

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
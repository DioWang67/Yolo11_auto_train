import sys
import os
from pathlib import Path

import yaml

if __name__ == "__main__" or __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QVBoxLayout,
    QHBoxLayout,
    QWidget,
    QPushButton,
    QLabel,
    QTextEdit,
    QProgressBar,
    QGroupBox,
    QCheckBox,
    QLineEdit,
    QFileDialog,
    QSplitter,
    QTabWidget,
    QScrollArea,
    QGridLayout,
    QMessageBox,
    QToolButton,
)
from PyQt5.QtCore import Qt

from picture_tool.gui.task_thread import WorkerThread


class CompactCheckBox(QCheckBox):
    def __init__(self, text, parent=None):
        super().__init__(text, parent)
        self.setStyleSheet(
            """
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
        """
        )


class CompactButton(QPushButton):
    """緊湊型按鈕"""

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

        if self.color_theme == "primary":
            color_style = """
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
            """
        elif self.color_theme == "success":
            color_style = """
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
            """
        elif self.color_theme == "danger":
            color_style = """
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
            """
        else:
            color_style = base_style

        self.setStyleSheet(base_style + color_style)


class PictureToolGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.config = {}
        self.worker_thread = None
        self.init_ui()
        self.load_config()

    def init_ui(self):
        self.setWindowTitle("圖像處理工具 - 響應式GUI")
        self.setGeometry(100, 100, 1200, 800)
        self.setMinimumSize(800, 600)

        # 設置全局樣式
        self.setStyleSheet(
            """
            QMainWindow {
                background-color: #f8f9fa;
            }
            QGroupBox {
                font-size: 10pt;
                font-weight: bold;
                color: #2c3e50;
                border: 2px solid #dee2e6;
                border-radius: 6px;
                margin: 8px 2px 2px 2px;
                padding-top: 8px;
                background-color: white;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 8px;
                padding: 0 5px 0 5px;
            }
            QTextEdit {
                border: 1px solid #dee2e6;
                border-radius: 4px;
                background-color: #ffffff;
                font-family: 'Consolas', monospace;
                font-size: 9pt;
                padding: 8px;
            }
            QProgressBar {
                border: 1px solid #dee2e6;
                border-radius: 4px;
                background-color: #f8f9fa;
                text-align: center;
                font-size: 9pt;
                font-weight: bold;
                color: #2c3e50;
                height: 20px;
            }
            QProgressBar::chunk {
                background-color: #28a745;
                border-radius: 3px;
            }
            QComboBox, QLineEdit {
                padding: 4px 8px;
                border: 1px solid #ced4da;
                border-radius: 4px;
                background-color: white;
                font-size: 9pt;
                min-height: 16px;
            }
            QComboBox:hover, QLineEdit:hover {
                border-color: #80bdff;
            }
            QComboBox:focus, QLineEdit:focus {
                border-color: #007bff;
            }
            QLabel {
                font-size: 9pt;
                color: #495057;
            }
        """
        )

        # 中央部件
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # 使用 QSplitter 來創建可調整的佈局
        main_splitter = QSplitter(Qt.Horizontal)
        central_layout = QVBoxLayout(central_widget)
        central_layout.setContentsMargins(5, 5, 5, 5)
        central_layout.addWidget(main_splitter)

        # 左側面板（使用滾動區域）
        left_scroll = QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        left_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        left_scroll.setMinimumWidth(280)
        left_scroll.setMaximumWidth(400)

        left_panel = self.create_left_panel()
        left_scroll.setWidget(left_panel)

        # 右側面板
        right_panel = self.create_right_panel()

        # 添加到分割器
        main_splitter.addWidget(left_scroll)
        main_splitter.addWidget(right_panel)

        # 設置初始比例
        main_splitter.setSizes([300, 700])

    def create_left_panel(self):
        """創建左側控制面板"""
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setSpacing(8)
        left_layout.setContentsMargins(5, 5, 5, 5)

        # 配置文件選擇 - 緊湊版
        config_group = QGroupBox("配置設定")
        config_layout = QVBoxLayout(config_group)
        config_layout.setSpacing(6)

        # 配置文件路徑
        try:
            default_cfg_path = str((Path(__file__).parent / "config.yaml").resolve())
        except Exception:
            default_cfg_path = os.path.join("picture_tool", "config.yaml")

        self.config_path_edit = QLineEdit(default_cfg_path)
        self.config_path_edit.setToolTip("配置文件路徑")

        config_btn_layout = QHBoxLayout()
        config_btn_layout.setSpacing(4)
        config_browse_btn = CompactButton("瀏覽", "primary")
        config_browse_btn.clicked.connect(self.browse_config_file)
        reload_config_btn = CompactButton("重載", "primary")
        reload_config_btn.clicked.connect(self.load_config)

        config_btn_layout.addWidget(config_browse_btn)
        config_btn_layout.addWidget(reload_config_btn)

        config_layout.addWidget(QLabel("配置文件:"))
        config_layout.addWidget(self.config_path_edit)
        config_layout.addLayout(config_btn_layout)

        # 任務選擇 - 緊湊版
        task_group = QGroupBox("任務選擇")
        task_layout = QVBoxLayout(task_group)
        task_layout.setSpacing(2)
        # 收合切換列（任務選擇）
        all_toggle_bar = QHBoxLayout()
        all_toggle_bar.addStretch()
        all_toggle_btn = QToolButton()
        all_toggle_btn.setCheckable(True)
        all_toggle_btn.setChecked(True)
        all_toggle_btn.setArrowType(Qt.DownArrow)
        all_toggle_btn.setToolTip("展開/收起")
        all_toggle_bar.addWidget(all_toggle_btn)
        task_layout.addLayout(all_toggle_bar)

        # 可收合容器
        all_tasks_content = QWidget()
        all_tasks_content_layout = QVBoxLayout(all_tasks_content)
        all_tasks_content_layout.setSpacing(2)

        # 任務複選框 - 使用網格佈局節省空間
        task_grid = QGridLayout()
        task_grid.setSpacing(2)
        task_grid.setContentsMargins(2, 2, 2, 2)

        self.task_checkboxes = {}
        tasks = [
            "格式轉換",
            "異常檢測",
            "YOLO數據增強",
            "圖像數據增強",
            "數據集分割",
            "YOLO訓練",
            "YOLO評估",
            "生成報告",
            "數據集檢查",
            "增強預覽",
            "批次推論",
            "LED QC 建模",
            "LED QC 單張檢測",
            "LED QC 批次檢測",
            "LED QC 分析",
        ]

        # 2列佈局
        for i, task in enumerate(tasks):
            checkbox = CompactCheckBox(task)
            self.task_checkboxes[task] = checkbox
            row = i // 2
            col = i % 2
            task_grid.addWidget(checkbox, row, col)

        all_tasks_content_layout.addLayout(task_grid)

        # 全選/取消全選按鈕
        select_layout = QHBoxLayout()
        select_layout.setSpacing(4)
        select_all_btn = CompactButton("全選", "success")
        deselect_all_btn = CompactButton("取消", "danger")
        select_all_btn.clicked.connect(self.select_all_tasks)
        deselect_all_btn.clicked.connect(self.deselect_all_tasks)
        select_layout.addWidget(select_all_btn)
        select_layout.addWidget(deselect_all_btn)
        all_tasks_content_layout.addLayout(select_layout)

        task_layout.addWidget(all_tasks_content)

        def _toggle_all(checked):
            all_tasks_content.setVisible(checked)
            all_toggle_btn.setArrowType(Qt.DownArrow if checked else Qt.RightArrow)

        all_toggle_btn.toggled.connect(_toggle_all)

        # YOLO 訓練群組（可收合）
        yolo_group = QGroupBox("YOLO 訓練")
        yolo_layout = QVBoxLayout(yolo_group)
        yolo_layout.setSpacing(2)

        yolo_toggle_bar = QHBoxLayout()
        yolo_toggle_bar.addStretch()
        yolo_toggle_btn = QToolButton()
        yolo_toggle_btn.setCheckable(True)
        yolo_toggle_btn.setChecked(True)
        yolo_toggle_btn.setArrowType(Qt.DownArrow)
        yolo_toggle_btn.setToolTip("展開/收起")
        yolo_toggle_bar.addWidget(yolo_toggle_btn)
        yolo_layout.addLayout(yolo_toggle_bar)

        yolo_content = QWidget()
        yolo_content_layout = QVBoxLayout(yolo_content)
        yolo_content_layout.setSpacing(2)
        yolo_grid = QGridLayout()
        yolo_grid.setSpacing(2)
        yolo_grid.setContentsMargins(2, 2, 2, 2)
        yolo_entries = [
            "YOLO數據增強",
            "數據分割",
            "YOLO訓練",
            "YOLO評估",
            "生成報告",
            "數據檢查",
            "增強預覽",
            "批次推論",
        ]
        for i, label_text in enumerate(yolo_entries):
            cb = CompactCheckBox(label_text)
            self.task_checkboxes[label_text] = cb
            row = i // 2
            col = i % 2
            yolo_grid.addWidget(cb, row, col)
        yolo_content_layout.addLayout(yolo_grid)

        yolo_select_layout = QHBoxLayout()
        yolo_select_layout.setSpacing(4)
        yolo_select_all = CompactButton("全選", "success")
        yolo_deselect_all = CompactButton("全不選", "danger")
        yolo_select_all.clicked.connect(self.select_all_tasks)
        yolo_deselect_all.clicked.connect(self.deselect_all_tasks)
        yolo_select_layout.addWidget(yolo_select_all)
        yolo_select_layout.addWidget(yolo_deselect_all)
        yolo_content_layout.addLayout(yolo_select_layout)
        yolo_layout.addWidget(yolo_content)

        def _toggle_yolo(checked):
            yolo_content.setVisible(checked)
            yolo_toggle_btn.setArrowType(Qt.DownArrow if checked else Qt.RightArrow)

        yolo_toggle_btn.toggled.connect(_toggle_yolo)

        # 訓練參數覆蓋 - 摺疊式設計
        options_group = QGroupBox("訓練參數")
        options_layout = QVBoxLayout(options_group)
        options_layout.setSpacing(4)
        # 收合切換列（訓練參數）
        opts_toggle_bar = QHBoxLayout()
        opts_toggle_bar.addStretch()
        opts_toggle_btn = QToolButton()
        opts_toggle_btn.setCheckable(True)
        opts_toggle_btn.setChecked(True)
        opts_toggle_btn.setArrowType(Qt.DownArrow)
        opts_toggle_btn.setToolTip("展開/收起")
        opts_toggle_bar.addWidget(opts_toggle_btn)
        options_layout.addLayout(opts_toggle_bar)

        self.apply_overrides_cb = CompactCheckBox("啟用參數覆蓋")
        options_layout.addWidget(self.apply_overrides_cb)

        # 參數輸入 - 2x2 網格
        param_grid = QGridLayout()
        param_grid.setSpacing(4)

        self.override_device_edit = QLineEdit()
        self.override_device_edit.setPlaceholderText("設備 (0/cpu)")
        self.override_epochs_edit = QLineEdit()
        self.override_epochs_edit.setPlaceholderText("輪數")
        self.override_imgsz_edit = QLineEdit()
        self.override_imgsz_edit.setPlaceholderText("圖片尺寸")
        self.override_batch_edit = QLineEdit()
        self.override_batch_edit.setPlaceholderText("批次大小")

        param_grid.addWidget(QLabel("Device:"), 0, 0)
        param_grid.addWidget(self.override_device_edit, 0, 1)
        param_grid.addWidget(QLabel("Epochs:"), 1, 0)
        param_grid.addWidget(self.override_epochs_edit, 1, 1)
        param_grid.addWidget(QLabel("ImgSz:"), 2, 0)
        param_grid.addWidget(self.override_imgsz_edit, 2, 1)
        param_grid.addWidget(QLabel("Batch:"), 3, 0)
        param_grid.addWidget(self.override_batch_edit, 3, 1)

        options_layout.addLayout(param_grid)

        # GPU偵測和強制選項
        option_buttons = QHBoxLayout()
        option_buttons.setSpacing(4)
        detect_btn = CompactButton("偵測GPU", "primary")

        def _detect_gpu():
            try:
                import torch

                self.override_device_edit.setText(
                    "0" if torch.cuda.is_available() else "cpu"
                )
            except Exception:
                self.override_device_edit.setText("cpu")

        detect_btn.clicked.connect(_detect_gpu)
        option_buttons.addWidget(detect_btn)

        self.force_cb = CompactCheckBox("強制重跑")
        option_buttons.addWidget(self.force_cb)
        options_layout.addLayout(option_buttons)
        # 收合切換：延後綁定，隱藏/顯示參數元件

        def _toggle_opts(checked):
            self.apply_overrides_cb.setVisible(checked)
            for w in [
                self.override_device_edit,
                self.override_epochs_edit,
                self.override_imgsz_edit,
                self.override_batch_edit,
            ]:
                w.setVisible(checked)
            for i in range(option_buttons.count()):
                item = option_buttons.itemAt(i)
                wid = item.widget()
                if wid:
                    wid.setVisible(checked)
            opts_toggle_btn.setArrowType(Qt.DownArrow if checked else Qt.RightArrow)

        opts_toggle_btn.toggled.connect(_toggle_opts)

        # 執行控制
        control_group = QGroupBox("執行控制")
        control_layout = QVBoxLayout(control_group)
        control_layout.setSpacing(6)

        # 執行按鈕
        control_btn_layout = QHBoxLayout()
        control_btn_layout.setSpacing(6)
        self.start_btn = CompactButton("▶ 開始", "success")
        self.stop_btn = CompactButton("⏹ 停止", "danger")
        self.stop_btn.setEnabled(False)

        self.start_btn.clicked.connect(self.start_pipeline)
        self.stop_btn.clicked.connect(self.stop_pipeline)

        control_btn_layout.addWidget(self.start_btn)
        control_btn_layout.addWidget(self.stop_btn)
        control_layout.addLayout(control_btn_layout)

        # 進度條
        self.progress_bar = QProgressBar()
        self.progress_bar.setTextVisible(True)
        control_layout.addWidget(self.progress_bar)

        # 狀態標籤
        self.status_label = QLabel("就緒")
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setStyleSheet(
            """
            QLabel {
                background-color: #d4edda;
                color: #155724;
                padding: 6px;
                border: 1px solid #c3e6cb;
                border-radius: 4px;
                font-weight: bold;
            }
        """
        )
        control_layout.addWidget(self.status_label)

        # 添加到主佈局
        left_layout.addWidget(config_group)
        # 插入 YOLO 訓練群組於任務選擇之前
        left_layout.addWidget(yolo_group)
        left_layout.addWidget(task_group)
        left_layout.addWidget(options_group)
        left_layout.addWidget(control_group)
        left_layout.addStretch()

        return left_widget

    def create_right_panel(self):
        """創建右側內容面板"""
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setSpacing(8)
        right_layout.setContentsMargins(5, 5, 5, 5)

        # 標題
        title_label = QLabel("圖像處理工具管理")
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet(
            """
            QLabel {
                font-size: 14pt;
                font-weight: bold;
                color: #2c3e50;
                padding: 10px;
                background-color: #e9ecef;
                border: 1px solid #dee2e6;
                border-radius: 6px;
            }
        """
        )

        # Tab Widget - 響應式設計
        self.tab_widget = QTabWidget()
        self.tab_widget.setStyleSheet(
            """
            QTabWidget::pane {
                border: 1px solid #dee2e6;
                border-radius: 4px;
                background-color: white;
                padding: 5px;
            }
            QTabBar::tab {
                background-color: #f8f9fa;
                color: #495057;
                padding: 8px 16px;
                margin-right: 2px;
                border: 1px solid #dee2e6;
                border-top-left-radius: 4px;
                border-top-right-radius: 4px;
                font-size: 9pt;
                font-weight: bold;
            }
            QTabBar::tab:selected {
                background-color: #007bff;
                color: white;
                border-bottom: none;
            }
            QTabBar::tab:hover:!selected {
                background-color: #e9ecef;
            }
        """
        )

        # 日誌標籤頁
        log_widget = QWidget()
        log_layout = QVBoxLayout(log_widget)
        log_layout.setSpacing(5)

        self.log_text = QTextEdit()
        self.log_text.setPlainText(
            "歡迎使用圖像處理工具！\n請選擇任務並點擊開始執行。\n"
        )
        self.log_text.setStyleSheet(
            """
            QTextEdit {
                background-color: #f8f9fa;
                color: #2c3e50;
                border: 1px solid #dee2e6;
                border-radius: 4px;
                font-family: 'Consolas', monospace;
                font-size: 9pt;
                padding: 8px;
            }
        """
        )

        clear_log_btn = CompactButton("清空日誌", "danger")
        clear_log_btn.clicked.connect(self.clear_log)

        log_layout.addWidget(self.log_text)

        # 按鈕佈局
        log_btn_layout = QHBoxLayout()
        log_btn_layout.addStretch()
        log_btn_layout.addWidget(clear_log_btn)
        log_layout.addLayout(log_btn_layout)

        # 配置預覽標籤頁
        config_widget = QWidget()
        config_layout = QVBoxLayout(config_widget)
        config_layout.setSpacing(5)

        self.config_text = QTextEdit()
        self.config_text.setPlainText("請載入配置文件...")
        self.config_text.setStyleSheet(
            """
            QTextEdit {
                background-color: #ffffff;
                border: 1px solid #dee2e6;
                border-radius: 4px;
                font-family: 'Consolas', monospace;
                font-size: 9pt;
                padding: 8px;
            }
        """
        )

        config_layout.addWidget(self.config_text)

        # 添加標籤頁
        self.tab_widget.addTab(log_widget, "📋 執行日誌")
        self.tab_widget.addTab(config_widget, "⚙️ 配置預覽")

        # 添加到右側佈局
        right_layout.addWidget(title_label)
        right_layout.addWidget(self.tab_widget)

        return right_widget

    def browse_config_file(self):
        """瀏覽配置文件"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "選擇配置文件", "", "YAML files (*.yaml *.yml)"
        )
        if file_path:
            self.config_path_edit.setText(file_path)
            self.load_config()

    def load_default_config(self):
        """載入默認配置"""
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
        }
        self.config = default_config
        self.update_config_display()

    def load_config(self):
        """載入配置文件"""
        config_path = self.config_path_edit.text()
        try:
            if os.path.exists(config_path):
                with open(config_path, "r", encoding="utf-8") as f:
                    self.config = yaml.safe_load(f)
                self.log_message(f"成功載入配置文件: {config_path}")
                self.update_config_display()
            else:
                self.log_message(f"配置文件不存在: {config_path}")
                self.load_default_config()
        except Exception as e:
            self.log_message(f"載入配置文件失敗: {str(e)}")
            self.load_default_config()

    def update_config_display(self):
        """更新配置顯示"""
        config_text = yaml.dump(
            self.config, default_flow_style=False, allow_unicode=True, indent=2
        )
        self.config_text.setPlainText(config_text)

    def select_all_tasks(self):
        """全選任務"""
        for checkbox in self.task_checkboxes.values():
            checkbox.setChecked(True)

    def deselect_all_tasks(self):
        """取消全選任務"""
        for checkbox in self.task_checkboxes.values():
            checkbox.setChecked(False)

    def get_selected_tasks(self):
        """獲取選中的任務"""
        selected_tasks = []
        for task_name, checkbox in self.task_checkboxes.items():
            if checkbox.isChecked():
                selected_tasks.append(task_name)
        return selected_tasks

    def start_pipeline(self):
        """開始執行管道"""
        selected_tasks = self.get_selected_tasks()
        if not selected_tasks:
            QMessageBox.warning(self, "警告", "請至少選擇一個任務！")
            return

        # 更新UI狀態
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.progress_bar.setValue(0)
        self.status_label.setText("執行中...")
        self.status_label.setStyleSheet(
            """
            QLabel {
                background-color: #fff3cd;
                color: #856404;
                padding: 6px;
                border: 1px solid #ffeaa7;
                border-radius: 4px;
                font-weight: bold;
            }
        """
        )

        # 應用覆蓋與強制選項
        try:
            if (
                hasattr(self, "apply_overrides_cb")
                and self.apply_overrides_cb.isChecked()
            ):
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
        self.log_message(f"開始執行管道，選中的任務: {', '.join(selected_tasks)}")

    def stop_pipeline(self):
        """停止執行管道"""
        if self.worker_thread:
            self.worker_thread.cancel()
            self.log_message("正在停止執行...")

    def update_progress(self, value):
        """更新進度條"""
        self.progress_bar.setValue(value)

    def on_task_started(self, task_name):
        """任務開始時的回調"""
        self.log_message(f"⚡ 開始執行任務: {task_name}")

    def on_task_completed(self, task_name):
        """任務完成時的回調"""
        self.log_message(f"✅ 任務完成: {task_name}")

    def on_pipeline_finished(self):
        """管道執行完成時的回調"""
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.status_label.setText("完成")
        self.status_label.setStyleSheet(
            """
            QLabel {
                background-color: #d4edda;
                color: #155724;
                padding: 6px;
                border: 1px solid #c3e6cb;
                border-radius: 4px;
                font-weight: bold;
            }
        """
        )
        self.log_message("🎉 所有任務執行完成！")

    def on_error_occurred(self, error_message):
        """錯誤發生時的回調"""
        self.log_message(f"❌ 錯誤: {error_message}")
        QMessageBox.critical(self, "錯誤", f"執行過程中發生錯誤:\n{error_message}")
        self.on_pipeline_finished()

    def log_message(self, message):
        """添加日誌訊息"""
        from datetime import datetime

        timestamp = datetime.now().strftime("%H:%M:%S")
        formatted_message = f"[{timestamp}] {message}"
        self.log_text.append(formatted_message)
        # 自動滾動到底部
        scrollbar = self.log_text.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def clear_log(self):
        """清空日誌"""
        self.log_text.clear()
        self.log_message("日誌已清空")

    def closeEvent(self, event):
        """窗口關閉事件"""
        if self.worker_thread and self.worker_thread.isRunning():
            reply = QMessageBox.question(
                self,
                "確認退出",
                "任務正在執行中，確定要退出嗎？",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
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
    """主函數"""
    app = QApplication(sys.argv)

    # 設置應用程式
    app.setApplicationName("圖像處理工具")
    app.setApplicationVersion("1.0")
    app.setOrganizationName("ImageTool Corp")

    # 設置全局字體 - 使用系統預設字體
    font = app.font()
    font.setPointSize(9)
    app.setFont(font)

    # 創建主窗口
    window = PictureToolGUI()
    window.show()

    # 運行應用程式
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()

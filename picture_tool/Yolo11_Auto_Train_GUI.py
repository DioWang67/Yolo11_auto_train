import sys
from pathlib import Path
import yaml
from typing import Optional

if __name__ == "__main__" or __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, QWidget,
    QLabel, QTextEdit, QProgressBar, QGroupBox,
    QCheckBox, QLineEdit, QFileDialog, QSplitter, QTabWidget,
    QScrollArea, QGridLayout, QMessageBox, QListWidgetItem, QDialog, QFrame
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor, QIcon

POSITION_TASK_LABEL = "位置檢查"
YOLO_TRAIN_LABEL = "YOLO訓練"

PRESET_CONFIG_PATH = Path(__file__).parent / "preset_config.yaml"

from picture_tool.gui.custom_widgets import CompactCheckBox, CompactButton
from picture_tool.gui.position_settings_dialog import PositionSettingsDialog
from picture_tool.gui.preset_manager import PresetManagerMixin, DEFAULT_PRESET_CONFIG_PATH
from picture_tool.gui.pipeline_validation import PipelineValidationMixin
from picture_tool.gui.layout_builder import LayoutBuilderMixin
from picture_tool.gui.metrics_dashboard import MetricsDashboardMixin
from picture_tool.gui.pipeline_controller import PipelineControllerMixin


class PictureToolGUI(
    PipelineControllerMixin,
    LayoutBuilderMixin,
    PresetManagerMixin,
    PipelineValidationMixin,
    MetricsDashboardMixin,
    QMainWindow,
):
    def __init__(self):
        super().__init__()
        self._init_pipeline_controller()
        self.position_settings: dict = {}
        self.task_checkboxes: dict[str, QCheckBox] = {}
        self.task_presets: dict[str, list[str]] = {}
        self.preset_storage: dict = {'presets': {}}
        self.task_status_items: dict[str, QListWidgetItem] = {}
        self._status_icon_cache: dict[str, QIcon] = {}
        self.POSITION_TASK_LABEL = POSITION_TASK_LABEL
        self.YOLO_TRAIN_LABEL = YOLO_TRAIN_LABEL
        self.preset_config_path = DEFAULT_PRESET_CONFIG_PATH
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

        left_layout.addWidget(self._build_config_section())
        left_layout.addWidget(self._build_task_section())
        left_layout.addWidget(self._build_position_section())
        left_layout.addWidget(self._build_training_options_section())
        left_layout.addWidget(self._build_control_section())
        left_layout.addStretch()

        self._sync_position_controls()
        return left_widget

    def _build_position_section(self) -> QGroupBox:
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

        return position_group

    def _build_training_options_section(self) -> QGroupBox:
        options_group = QGroupBox("🎛️ 訓練參數")
        options_layout = QVBoxLayout(options_group)
        options_layout.setSpacing(8)

        self.apply_overrides_cb = CompactCheckBox("✓ 啟用參數覆寫")
        options_layout.addWidget(self.apply_overrides_cb)

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

        tool_btn_layout = QHBoxLayout()
        tool_btn_layout.setSpacing(6)
        detect_gpu_btn = CompactButton("🔍 偵測GPU", "primary")
        detect_gpu_btn.clicked.connect(self._detect_gpu)
        self.force_cb = CompactCheckBox("⚠️ 強制執行")
        tool_btn_layout.addWidget(detect_gpu_btn)
        tool_btn_layout.addWidget(self.force_cb)
        tool_btn_layout.addStretch()
        options_layout.addLayout(tool_btn_layout)

        return options_group

    def _build_control_section(self) -> QGroupBox:
        control_group = QGroupBox("🎮 執行控制")
        control_layout = QVBoxLayout(control_group)
        control_layout.setSpacing(10)

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

        return control_group

    def create_right_panel(self):
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setSpacing(10)
        right_layout.setContentsMargins(8, 8, 8, 8)

        right_layout.addWidget(self._build_title_card())
        right_layout.addWidget(self._build_tab_widget())

        self._rebuild_quick_nav_menu()
        self.refresh_metrics_dashboard()
        return right_widget

    def _build_title_card(self) -> QFrame:
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
        return title_card

    def _build_tab_widget(self) -> QTabWidget:
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

        self.log_tab = self._create_log_tab()
        self.tab_widget.addTab(self.log_tab, '執行紀錄')
        self.config_tab = self._create_config_tab()
        self.tab_widget.addTab(self.config_tab, '設定預覽')
        self.preset_tab = self._create_preset_tab()
        self.tab_widget.addTab(self.preset_tab, '流程設定')
        self.metrics_tab = self._create_metrics_tab()
        self.tab_widget.addTab(self.metrics_tab, '重要指標')

        return self.tab_widget

    def _create_log_tab(self) -> QWidget:
        log_widget = QWidget()
        log_layout = QVBoxLayout(log_widget)
        log_layout.setSpacing(8)

        self.log_text = QTextEdit()
        self.log_text.setPlainText(
            "🎉 歡迎使用圖像處理工具！\n"
            "📝 請選擇任務並點擊開始執行。\n"
            "💡 提示：可使用快速預設按鈕快速套用常用流程。\n"
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

        return log_widget

    def _create_config_tab(self) -> QWidget:
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
        return config_widget

    def _create_preset_tab(self) -> QWidget:
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
        self.bind_preset_controls(text=self.preset_text)
        return preset_widget

    def _create_metrics_tab(self) -> QWidget:
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
        self.bind_metrics_display(self.metrics_text)
        metrics_refresh_btn = CompactButton('重新整理指標', 'primary')
        metrics_refresh_btn.clicked.connect(self.refresh_metrics_dashboard)
        metrics_layout.addWidget(metrics_refresh_btn)

        return metrics_widget

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


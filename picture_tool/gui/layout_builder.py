"""Layout helper mixin for Picture Tool GUI."""

from __future__ import annotations

from typing import Dict

from PyQt5.QtCore import Qt, QSize
from PyQt5.QtGui import QIcon, QColor, QPixmap
from PyQt5.QtWidgets import (
    QGroupBox,
    QVBoxLayout,
    QHBoxLayout,
    QGridLayout,
    QComboBox,
    QToolButton,
    QMenu,
    QListWidget,
    QListWidgetItem,
    QLineEdit,
)

from picture_tool.gui.custom_widgets import CompactButton, CompactCheckBox


class LayoutBuilderMixin:
    """Provide shared layout helpers for the Qt GUI."""

    _status_icon_cache: Dict[str, QIcon]

    def _build_config_section(self) -> QGroupBox:
        config_group = QGroupBox(" 設定檔位址")
        config_layout = QVBoxLayout(config_group)
        config_layout.setSpacing(8)

        default_cfg_path = str(self._default_config_path())

        if not hasattr(self, "config_path_edit"):
            self.config_path_edit = QLineEdit()
        self.config_path_edit.setText(default_cfg_path)
        self.config_path_edit.setPlaceholderText("設定檔路徑")

        config_btn_layout = QHBoxLayout()
        config_btn_layout.setSpacing(6)
        config_browse_btn = CompactButton(" 瀏覽", "primary")
        config_browse_btn.clicked.connect(self.browse_config_file)
        reload_config_btn = CompactButton(" 重新載入", "success")
        reload_config_btn.clicked.connect(self.load_config)
        config_btn_layout.addWidget(config_browse_btn)
        config_btn_layout.addWidget(reload_config_btn)

        config_layout.addWidget(self.config_path_edit)
        config_layout.addLayout(config_btn_layout)
        return config_group

    def _build_task_section(self) -> QGroupBox:
        task_group = QGroupBox(" 任務選擇")
        task_layout = QVBoxLayout(task_group)
        task_layout.setSpacing(6)

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
            (self.YOLO_TRAIN_LABEL, "🚀"),
            ("YOLO評估", "📈"),
            ("生成報告", "📝"),
            ("數據檢查", "✅"),
            ("增強預覽", "👁️"),
            ("批次推論", "⚡"),
            ("LED QC 建模", "🔧"),
            ("LED QC 單張檢測", "🔬"),
            ("LED QC 批次檢測", "🔬"),
            ("LED QC 分析", "📊"),
            (self.POSITION_TASK_LABEL, "📍"),
        ]

        for index, (task, icon) in enumerate(tasks):
            checkbox = CompactCheckBox(f"{icon} {task}")
            self.task_checkboxes[task] = checkbox
            row = index // 3
            col = index % 3
            task_grid.addWidget(checkbox, row, col)

        task_layout.addLayout(task_grid)

        select_layout = QHBoxLayout()
        select_layout.setSpacing(6)
        select_all_btn = CompactButton(" 全部勾選", "success")
        deselect_all_btn = CompactButton(" 全部取消", "danger")
        self.preset_combo = QComboBox()
        self.preset_combo.setEditable(False)
        self.preset_combo.setMinimumContentsLength(10)
        self.preset_combo.setPlaceholderText("載入預設")
        self.apply_preset_btn = CompactButton(" 套用預設", "primary")

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
        self.save_preset_btn = CompactButton(" 儲存預設", "secondary")
        self.delete_preset_btn = CompactButton(" 刪除預設", "danger")
        export_preset_btn = CompactButton(" 匯出預設", "secondary")
        import_preset_btn = CompactButton(" 匯入預設", "secondary")
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

        self.bind_preset_controls(
            combo=self.preset_combo,
            apply_button=self.apply_preset_btn,
            delete_button=self.delete_preset_btn,
        )

        status_group = QGroupBox(" 任務狀態")
        status_layout = QVBoxLayout(status_group)
        self.status_list = QListWidget()
        self.status_list.setIconSize(QSize(14, 14))
        self.status_list.setSelectionMode(QListWidget.NoSelection)
        self.status_list.setFocusPolicy(Qt.NoFocus)
        self.status_list.setMaximumHeight(200)
        status_layout.addWidget(self.status_list)
        task_layout.addWidget(status_group)

        self._populate_status_items()

        utility_layout = QHBoxLayout()
        utility_layout.setSpacing(6)
        self.quick_nav_btn = QToolButton()
        self.quick_nav_btn.setText(" 快速導覽")
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
        self.quick_nav_menu.addAction("操作說明", self.show_quick_guide)
        self.quick_nav_btn.setMenu(self.quick_nav_menu)
        utility_layout.addWidget(self.quick_nav_btn)

        self.preflight_btn = CompactButton(" 預檢", "primary")
        self.preflight_btn.clicked.connect(self.run_preflight_check)
        utility_layout.addWidget(self.preflight_btn)
        utility_layout.addStretch()
        task_layout.addLayout(utility_layout)

        return task_group

    def _populate_status_items(self) -> None:
        self.status_list.clear()
        self.task_status_items = {}
        for name in self.task_checkboxes.keys():
            item = QListWidgetItem(self._get_status_icon("idle"), f"{name} - 待命")
            self.task_status_items[name] = item
            self.status_list.addItem(item)

    def _get_status_icon(self, status) -> QIcon:
        if not hasattr(self, "_status_icon_cache"):
            self._status_icon_cache = {}
        if isinstance(status, QColor):
            key = status.name()
            color_hex = key
        else:
            key = str(status)
            color_hex = {
                "idle": "#6c757d",
                "running": "#17a2b8",
                "success": "#28a745",
                "error": "#dc3545",
                "warning": "#ffc107",
            }.get(key, "#6c757d" if not key.startswith("#") else key)
            if key.startswith("#"):
                color_hex = key
        if key in self._status_icon_cache:
            return self._status_icon_cache[key]
        pixmap = QPixmap(16, 16)
        pixmap.fill(QColor(color_hex))
        icon = QIcon(pixmap)
        self._status_icon_cache[key] = icon
        return icon

"""Entry point for the Picture Tool desktop GUI."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import yaml
from PyQt5 import QtCore, QtGui
from PyQt5.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QComboBox,
    QPushButton,
    QProgressBar,
    QVBoxLayout,
    QWidget,
    QFrame,
    QSizePolicy,
    QMessageBox,
)

# Import new components
try:
    from picture_tool.gui.pipeline_manager import PipelineManager
    from picture_tool.gui.config_editor import ConfigEditor
    from picture_tool.gui.wizards import NewProjectWizard
    from picture_tool.exceptions import ConfigurationError
    from picture_tool.gui.log_viewer import LogViewer
    from picture_tool.gui.annotation_panel import AnnotationPanel
    from picture_tool.gui.config_panel import ConfigPanel
    from picture_tool.gui.color_panel import ColorPanel
    from picture_tool.gui.task_control_panel import TaskControlPanel
    from picture_tool.gui.style_manager import load_stylesheet
    from picture_tool.gui.training_metrics import (
        TrainingMetricsParser,
        TrainingMetricsWidget,
    )
    from picture_tool.gui.constants import (
        TASK_OPTIONS_MAP,
    )
except ImportError:
    # Fallback mocks if components unavailable
    class PipelineManager:  # type: ignore
        def __init__(self, parent=None):
            self.config = {}
        def start_pipeline(self, tasks, config_path=None, product_id=None):
            pass
        def stop_pipeline(self):
            pass

    class ConfigEditor:  # type: ignore
        pass

    class NewProjectWizard:  # type: ignore
        pass

    class LabelImgLauncher:  # type: ignore
        pass

    class AnnotationTracker:  # type: ignore
        pass


# ------------------------------------------------------------------
class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        
        # Initialize PipelineManager (replaces Mixin)
        self.manager = PipelineManager(parent=self)
        self.manager.pipeline_finished.connect(self.on_pipeline_finished)
        self.manager.error_occurred.connect(self.on_pipeline_error)
        self.manager.task_started.connect(self.on_task_started)
        self.manager.task_completed.connect(self.on_task_completed)
        self.manager.progress_updated.connect(self.on_progress_updated)
        self.manager.log_message.connect(self.log_message)
        
        # Initialize LogViewer for logs and config preview
        self.log_viewer = LogViewer(parent=self)
        self._log_history: List[str] = []
        
        # Initialize ConfigEditor before _build_ui
        self.config_editor = ConfigEditor()
        
        # Backward compatibility: alias log_viewer components
        # This allows existing code to reference self.tabs, self.log_text, self.config_text
        self.tabs = None  # Will be set after _build_ui
        self.log_text = None
        self.config_text = None
        
        self.task_status_items: Dict[str, QListWidgetItem] = {}
        
        
        

        # Annotation-related components (Moved to AnnotationPanel)
        # self.labelimg_launcher = LabelImgLauncher()
        # self.annotation_tracker = AnnotationTracker()
        # self.annotation_classes: List[str] = []
        # self.annotation_input_dir: Path | None = None
        # self.annotation_output_dir: Path | None = None

        self.setWindowTitle("Picture Tool Orchestrator")
        self.resize(1200, 800)
        # Load external stylesheet
        try:
             from PyQt5.QtWidgets import QApplication
             app = QApplication.instance()
             if app:
                 load_stylesheet(app)  # type: ignore[arg-type]
        except ImportError:
             pass

        self._build_ui()

        try:
            if hasattr(self, "config_panel"):
                self.config_panel.load_default_config()
        except (ConfigurationError, OSError, yaml.YAMLError):
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
        self.side_bar.setObjectName("SideBar")  # 用於 CSS 定位
        self.side_bar.setFixedWidth(420)  # 恢復至穩定的 420px

        side_layout = QVBoxLayout(self.side_bar)
        side_layout.setContentsMargins(20, 20, 20, 20) 
        side_layout.setSpacing(20) 

        # 加入左側元件
        side_layout.addWidget(self._create_header_label("Configuration"))
        side_layout.addWidget(
            self._create_hint_label("步驟：1) 選 config 2) 勾任務 3) RUN")
        )
        self.config_panel = ConfigPanel(self.manager, parent=self)
        self.config_panel.config_loaded.connect(self.on_config_loaded)
        self.config_panel.log_message.connect(self.log_message)
        side_layout.addWidget(self.config_panel)

        side_layout.addWidget(self._create_separator())

        side_layout.addWidget(self._create_header_label("Select Tasks"))
        self.task_control = TaskControlPanel(parent=self)
        self.task_control.tasks_changed.connect(self.on_tasks_changed)
        self.task_control.log_message.connect(self.log_message)
        side_layout.addWidget(self.task_control)  # 任務勾選區

        side_layout.addStretch()  # 彈簧，把按鈕推到底部

        side_layout.addWidget(self._create_separator())
        side_layout.addWidget(self._build_control_section())  # 開始/停止按鈕

        # --- 右側面板 (Dashboard) ---
        self.dashboard = QWidget()
        dash_layout = QVBoxLayout(self.dashboard)
        dash_layout.setContentsMargins(20, 20, 20, 20)
        dash_layout.setSpacing(15)

        # 右側上半部：訓練指標 (hidden until metrics detected)
        self.training_metrics = TrainingMetricsWidget()
        dash_layout.addWidget(self.training_metrics)

        # 狀態監控
        dash_layout.addWidget(self._create_header_label("Pipeline Status Queue"))
        self.status_list = QListWidget()
        self.status_list.setMaximumHeight(200)  # 不佔滿整個畫面
        self.status_list.setAlternatingRowColors(True)
        dash_layout.addWidget(self.status_list)

        # 右側下半部：使用 LogViewer 的 tabs（保持視覺一致）
        # LogViewer 包含 Execution Logs 和 Config Preview tabs
        
        # Tab 3: Annotation Tool
        self.annotation_panel = AnnotationPanel(manager=self.manager, parent=self)
        self.annotation_panel.message_logged.connect(self.log_message)
        self.log_viewer.tabs.addTab(self.annotation_panel, "📝 圖像標註")

        # Tab 4: Config Editor
        self.log_viewer.tabs.addTab(self.config_editor, "⚙ 設定編輯器")

        # Tab 5: Color Verification
        self.color_panel = ColorPanel(manager=self.manager, parent=self)
        self.color_panel.log_message.connect(self.log_message)
        self.log_viewer.tabs.addTab(self.color_panel, "🎨 顏色驗證")

        dash_layout.addLayout(self._build_log_controls())
        dash_layout.addWidget(self.log_viewer.tabs, 2)  # 使用 LogViewer 的 tabs

        main_layout.addWidget(self.side_bar)
        main_layout.addWidget(self.dashboard)
        
        # Set up backward compatibility aliases
        self.tabs = self.log_viewer.tabs
        self.log_text = self.log_viewer.log_text
        self.config_text = self.log_viewer.config_text

        self._rebuild_status_items()

    # ------------------------------------------------------------------
    # 左側組件構建
    # ------------------------------------------------------------------
    def _build_control_section(self) -> QWidget:
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)

        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(False)  # 簡約風格

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

        btns_layout.addWidget(self.start_btn, 3)  # Start 佔 75%
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
        lbl.setProperty("class", "Header")  # 配合 QSS
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
    # Pipeline Controller Integration (replaces Mixin hooks)
    # ------------------------------------------------------------------
    def load_config(self) -> None:
        """Load config from path and update UI"""
        path = self.config_path_edit.text().strip() if hasattr(self, "config_path_edit") else None
        self.manager.load_config(path)
    
        # Update path display with the actually loaded config path
        if hasattr(self, "config_path_edit") and self.manager.current_config_path:
            self.config_path_edit.setText(str(self.manager.current_config_path))
    
        # self._update_config_status()  # Method was removed during refactoring
        if hasattr(self, "config_editor"):
            self.config_editor.set_config(self.manager.config)
    
        # Update Config Preview tab
        if self.config_text is not None:
            try:
                config_yaml = yaml.dump(self.manager.config, allow_unicode=True, sort_keys=False)
                self.config_text.setPlainText(config_yaml)
            except (yaml.YAMLError, TypeError, ValueError) as e:
                self.config_text.setPlainText(f"Error displaying config: {e}")

    def on_config_loaded(self, config: dict) -> None:
        """Handle config loaded signal from ConfigPanel."""
        if hasattr(self, "config_editor"):
            self.config_editor.set_config(config)
    
        # Update Config Preview in LogViewer
        if self.config_text is not None:
            try:
                config_yaml = yaml.dump(config, allow_unicode=True, sort_keys=False)
                self.config_text.setPlainText(config_yaml)
            except Exception as e:
                self.config_text.setPlainText(f"Error displaying config: {e}")

    def start_pipeline(self) -> None:
        """Start the pipeline with selected tasks"""
        # Collect selected tasks
        selected_tasks = self.task_control.get_selected_tasks()
    
        if not selected_tasks:
            self.log_message("[WARNING] No tasks selected.")
            return
    
        # Get overrides
        config_path = self.config_panel.get_config_path()
        product = self.config_panel.get_product_override()
    
        # Validate product override if placeholders are present
        if not product:
            import json
            cfg_dump = json.dumps(self.manager.config)
            if "/project/" in cfg_dump or "./data/project" in cfg_dump or "./runs/project" in cfg_dump:
                QMessageBox.warning(
                    self, 
                    "未填寫產品名稱", 
                    "偵測到設定檔包含路徑佔位符 (project)，請先於左側「Product」欄位輸入產品名稱 (如 Cable1) 以進行自動路徑對齊。"
                )
                return
    
        # Update UI state
        if hasattr(self, "start_btn"):
            self.start_btn.setEnabled(False)
        if hasattr(self, "stop_btn"):
            self.stop_btn.setEnabled(True)
    
        # Reset status items
        self.reset_task_statuses(selected_tasks)
    
        # Start via manager
        self.manager.start_pipeline(selected_tasks, config_path, product)

    def stop_pipeline(self) -> None:
        """Stop the running pipeline"""
        self.manager.stop_pipeline()
    
        # Update UI state
        if hasattr(self, "start_btn"):
            self.start_btn.setEnabled(True)
        if hasattr(self, "stop_btn"):
            self.stop_btn.setEnabled(False)

    def log_message(self, message: str) -> None:
        """Log message wrapper - delegates to LogViewer and extracts metrics."""
        self._log_history.append(message)
        self.log_viewer.log_message(message)
        # Extract training metrics from ultralytics epoch lines
        if hasattr(self, "training_metrics"):
            metrics = TrainingMetricsParser.parse_epoch_line(message)
            if metrics:
                self.training_metrics.setVisible(True)
                self.training_metrics.update_metrics(metrics)

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

            item.setText(
                f"{prefix}  {TASK_OPTIONS_MAP.get(task, task)} \n      └─ {message}"
            )

    def _validate_pipeline_configuration(self, tasks):
        issues = []
        path_text = ""
        if hasattr(self, "config_path_edit"):
            try:
                path_text = self.config_panel.get_config_path() or ""
                if path_text and not Path(path_text).exists():
                    issues.append(f"Config file not found: {path_text}")
            except (OSError, ValueError):
                pass
        cfg = self.manager.config if hasattr(self, "manager") else {}
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

        # Update status list based on provided list or current selection
        targets = only if only is not None else self.task_control.get_selected_tasks()
    
        for task in targets:
            label_text = TASK_OPTIONS_MAP.get(task, task)
            item = QListWidgetItem(f"⚪  {label_text} : {default_state}")
            item.setForeground(QtGui.QColor("#aaaaaa"))
            self.task_status_items[task] = item
            self.status_list.addItem(item)

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
    def closeEvent(self, event) -> None:
        if (
            hasattr(self, "manager")
            and self.manager.worker_thread
            and self.manager.worker_thread.isRunning()
        ):
            self.stop_pipeline()
            self.manager.worker_thread.wait(1000)
        super().closeEvent(event)

    def on_tasks_changed(self, tasks: list) -> None:
        """Handle tasks changed signal from TaskControlPanel."""
        # Resolve dependency chain for preview
        try:
            ordered, auto_added = self.manager.resolve_task_chain(tasks)
            self.task_control.show_dependency_chain(ordered, auto_added)
            # Rebuild status list using the full resolved chain
            self._rebuild_status_items(only=ordered)
        except Exception:
            self.task_control.show_dependency_chain([], set())
            self._rebuild_status_items(only=tasks)

    # ------------------------------------------------------------------
    # Signal Handlers
    # ------------------------------------------------------------------
    def on_pipeline_finished(self):
        """Handle pipeline completion."""
        self.log_message("[SUCCESS] Pipeline finished successfully.")
        self._reset_ui_state()
        if hasattr(self, "progress_bar"):
            self.progress_bar.setValue(100)
            self.status_label.setText("Pipeline Completed")
        if hasattr(self, "training_metrics"):
            self.training_metrics.reset()

    def on_pipeline_error(self, message: str):
        """Handle pipeline error."""
        self.log_message(f"[ERROR] Pipeline failed: {message}")
        self._reset_ui_state()
        if hasattr(self, "status_label"):
            self.status_label.setText("Pipeline Error")
        if hasattr(self, "training_metrics"):
            self.training_metrics.reset()

    def on_task_started(self, task_name: str):
        """Handle task start."""
        self._set_task_status(task_name, "Running...", color="#4D96FF")
        if hasattr(self, "status_label"):
            self.status_label.setText(f"Running: {task_name}")

    def on_task_completed(self, task_name: str):
        """Handle task completion."""
        self._set_task_status(task_name, "Done", color="#6BCB77")

    def on_progress_updated(self, value: int):
        """Handle progress update."""
        if hasattr(self, "progress_bar"):
            self.progress_bar.setValue(value)

    def _reset_ui_state(self):
        """Re-enable controls after run."""
        if hasattr(self, "start_btn"):
            self.start_btn.setEnabled(True)
        if hasattr(self, "stop_btn"):
            self.stop_btn.setEnabled(False)

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
        if self.log_text is not None:
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
        if self.log_text is None:
            return
        self.log_text.clear()
        for msg in self._log_history:
            if self._should_display_log(msg):
                self._render_log_message(msg)

    def _clear_logs(self) -> None:
        self._log_history.clear()
        if self.log_text is not None:
            self.log_text.clear()




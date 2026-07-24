"""Entry point for the Picture Tool desktop GUI."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List

import yaml
from picture_tool.operator_acceptance import (
    OperatorAcceptanceError,
    load_operator_acceptance_summary,
)
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
    QScrollArea,
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
    from picture_tool.gui.operator_workflow_panel import OperatorWorkflowPanel
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
_YOLO_EPOCH_LIFECYCLE_PATTERN = re.compile(
    r"\[YOLO Lifecycle\]\s+(Starting|Finished) Epoch\s+(\d+)/(\d+)",
    re.IGNORECASE,
)


def _operator_epoch_status(message: str) -> tuple[str, int] | None:
    """Translate a YOLO lifecycle log into operator-facing progress."""
    match = _YOLO_EPOCH_LIFECYCLE_PATTERN.search(message)
    if match is None:
        return None
    phase, epoch_text, total_text = match.groups()
    epoch = int(epoch_text)
    total = int(total_text)
    if total <= 0 or epoch <= 0 or epoch > total:
        return None
    completed_epochs = epoch if phase.lower() == "finished" else epoch - 1
    # Training occupies 40-77% of the full prepare/train/evaluate/deploy flow.
    overall_progress = 40 + round(37 * completed_epochs / total)
    return f"模型訓練中：Epoch {epoch}/{total}", overall_progress


def _operator_error_message(message: str) -> str:
    """Translate expected safety-gate failures into line-leader guidance."""
    normalized = message.lower()
    if "operator_training_preflight_failed" in normalized:
        issues: list[str] = []
        if "operator_feedback_not_actionable" in normalized:
            issues.append(
                "本批只有原本就辨識正確的照片；資料已加入樣本庫，但要累積足夠的"
                "補框、修正類別、刪除錯框或確認無目標照片才會開始補訓。"
            )
        if "deployed_training_pair_missing" in normalized:
            issues.append(
                "目前產線 ONNX 找不到同版本的 PT 訓練權重，無法保證從現行模型繼續學習。"
                "請通知工程人員執行 ONNX/PT 成對驗證與部署工具；通過後即可補訓。"
            )
        if not issues:
            issues.append("補訓前安全檢查未通過，請通知工程人員查看工作紀錄。")
        return "補訓尚未開始，資料已安全保存，產線模型不會變更。\n\n" + "\n".join(
            f"• {issue}" for issue in issues
        )
    if any(
        token in normalized
        for token in (
            "split_underrepresented",
            "class_underrepresented",
            "not enough independent",
            "at least three independent",
        )
    ):
        return (
            "補標資料已安全保存，但目前樣本數不足，這次不會更新產線模型。\n"
            "請繼續收集並補標漏檢圖片；資料達到安全門檻後再送出補訓。"
        )
    if "deployment quality gate failed" in normalized:
        raw_failures = message.split(":", 1)[1].strip() if ":" in message else ""
        detail = ""
        if raw_failures:
            detail = "\n\n未通過項目：\n" + "\n".join(
                f"• {item.strip()}" for item in raw_failures.split(";") if item.strip()
            )
        return (
            "新模型未通過產線品質比較，因此未部署，現場仍使用原本模型。\n"
            "請查看下列指標，補充對應類別的漏檢／錯框照片後再重試。"
            f"{detail}"
        )
    if "deployed_training_pair_missing" in normalized:
        return (
            "產線模型缺少同版本 PT 訓練權重，因此補訓尚未開始。\n"
            "資料已保存且產線模型不會變更；請通知工程人員執行 ONNX/PT "
            "成對驗證與部署工具。"
        )
    if (
        "incumbent baseline" in normalized
        or "deployed model is unavailable" in normalized
    ):
        return (
            "找不到目前產線模型，無法安全比較新舊模型，因此未部署。\n"
            "請通知工程人員檢查模型檔案。"
        )
    if "class contract" in normalized or "class order" in normalized:
        return (
            "檢測類別資料不一致，已停止補訓且不會部署。\n"
            "請用目前產線模型重新檢測該圖片後再送出。"
        )
    return message


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
        self._operator_handoff_target = None
        self._operator_handoff = None
        self._operator_training_lock = None
        self._operator_cancel_requested = False
        self._operator_mode_enabled = False
        self._background_mode = False
        self._close_when_pipeline_stops = False
        self._operator_heartbeat_failures = 0
        self._operator_heartbeat_timer = QtCore.QTimer(self)
        self._operator_heartbeat_timer.setInterval(5000)
        self._operator_heartbeat_timer.timeout.connect(
            self._refresh_operator_job_lease
        )

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
        self.setMinimumSize(960, 600)
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

    def set_background_mode(self, enabled: bool) -> None:
        """Run operator handoffs without showing the orchestration window."""
        self._background_mode = bool(enabled)

    def _request_background_exit(self) -> None:
        """End a hidden worker process after its operator job becomes terminal."""
        if not self._background_mode:
            return
        application = QtCore.QCoreApplication.instance()
        if application is not None:
            QtCore.QTimer.singleShot(0, application.quit)

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
        self.side_bar.setMinimumWidth(480)
        self.side_bar.setMaximumWidth(520)

        side_outer = QVBoxLayout(self.side_bar)
        side_outer.setContentsMargins(0, 0, 0, 0)
        side_outer.setSpacing(0)

        # 上半部：可滾動區域（Configuration + Select Tasks）
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setFrameShape(QFrame.NoFrame)
        scroll_area.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        scroll_area.setObjectName("SideBarScroll")

        scroll_content = QWidget()
        scroll_content.setObjectName("SideBarScrollContent")
        side_layout = QVBoxLayout(scroll_content)
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

        side_layout.addStretch()
        scroll_area.setWidget(scroll_content)
        side_outer.addWidget(scroll_area, 1)  # 滾動區佔滿剩餘空間

        # 下半部：固定控制區（RUN/STOP 按鈕），不會被擠壓
        side_outer.addWidget(self._create_separator())
        control_section = self._build_control_section()
        control_section.setContentsMargins(20, 10, 20, 15)
        side_outer.addWidget(control_section)

        # --- 右側面板 (Dashboard) ---
        self.dashboard = QWidget()
        dash_layout = QVBoxLayout(self.dashboard)
        dash_layout.setContentsMargins(20, 20, 20, 20)
        dash_layout.setSpacing(15)

        self.operator_workflow_panel = OperatorWorkflowPanel(parent=self)
        self.operator_workflow_panel.setVisible(False)
        dash_layout.addWidget(self.operator_workflow_panel)

        # 右側上半部：訓練指標 (hidden until metrics detected)
        self.training_metrics = TrainingMetricsWidget()
        dash_layout.addWidget(self.training_metrics)

        # 狀態監控
        self.pipeline_status_header = self._create_header_label("Pipeline Status Queue")
        dash_layout.addWidget(self.pipeline_status_header)
        self.status_list = QListWidget()
        self.status_list.setMaximumHeight(200)  # 不佔滿整個畫面
        self.status_list.setAlternatingRowColors(True)
        dash_layout.addWidget(self.status_list)

        # 右側下半部：使用 LogViewer 的 tabs（保持視覺一致）
        # LogViewer 包含 Execution Logs 和 Config Preview tabs

        # Tab 3: Annotation Tool
        self.annotation_panel = AnnotationPanel(manager=self.manager, parent=self)
        self.annotation_panel.message_logged.connect(self.log_message)
        self.annotation_panel.pending_ready.connect(self.apply_operator_handoff)
        self.log_viewer.tabs.addTab(self.annotation_panel, "📝 圖像標註")

        # Tab 4: Config Editor
        self.log_viewer.tabs.addTab(self.config_editor, "⚙ 設定編輯器")

        # Tab 5: Color Verification
        self.color_panel = ColorPanel(manager=self.manager, parent=self)
        self.color_panel.log_message.connect(self.log_message)
        self.log_viewer.tabs.addTab(self.color_panel, "🎨 顏色驗證")

        self.log_controls_container = QWidget()
        self.log_controls_container.setLayout(self._build_log_controls())
        dash_layout.addWidget(self.log_controls_container)
        dash_layout.addWidget(self.log_viewer.tabs, 2)  # 使用 LogViewer 的 tabs

        main_layout.addWidget(self.side_bar, 0)  # 固定寬度，不伸展
        main_layout.addWidget(self.dashboard, 1)  # 填滿剩餘空間

        # Set up backward compatibility aliases
        self.tabs = self.log_viewer.tabs
        self.log_text = self.log_viewer.log_text
        self.config_text = self.log_viewer.config_text

        # 統一設定所有 tab bar 屬性（必須在 QSS load 之後、所有 tab 加完之後）
        for tab_widget in (
            self.log_viewer.tabs,
            self.config_editor.tabs,
            self.color_panel.tabs,
        ):
            tab_widget.setElideMode(QtCore.Qt.ElideNone)
            tab_widget.tabBar().setExpanding(False)
            tab_widget.tabBar().setUsesScrollButtons(True)

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
        path = (
            self.config_path_edit.text().strip()
            if hasattr(self, "config_path_edit")
            else None
        )
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
                config_yaml = yaml.dump(
                    self.manager.config, allow_unicode=True, sort_keys=False
                )
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

    def start_pipeline(self) -> bool:
        """Start the pipeline with selected tasks"""
        # Collect selected tasks
        selected_tasks = self.task_control.get_selected_tasks()

        if not selected_tasks:
            self.log_message("[WARNING] No tasks selected.")
            return False

        # Get overrides
        config_path = self.config_panel.get_config_path()
        product = self.config_panel.get_product_override()

        if product:
            try:
                from picture_tool.path_resolver import parse_project_area_override

                parse_project_area_override(product)
            except ValueError as exc:
                QMessageBox.warning(
                    self,
                    "Invalid Product",
                    f"Product format must be 'Product' or 'Product,Area'.\n\n{exc}",
                )
                return False

        # Validate product override if placeholders are present
        if not product:
            import json

            cfg_dump = json.dumps(self.manager.config)
            if (
                "/project/" in cfg_dump
                or "./data/project" in cfg_dump
                or "./runs/project" in cfg_dump
            ):
                QMessageBox.warning(
                    self,
                    "未填寫產品名稱",
                    "偵測到設定檔包含路徑佔位符 (project)，請先於左側「Product」欄位輸入產品名稱 (如 Cable1) 以進行自動路徑對齊。",
                )
                return

        # ── Preflight checks ──────────────────────────────────────────
        try:
            from picture_tool.pipeline.preflight import PreflightChecker
            from picture_tool.gui.preflight_dialog import PreflightDialog
            from picture_tool.path_resolver import resolve_project_paths

            # Apply product path substitution before preflight so checks
            # see the actual resolved paths (e.g. Cable1) instead of
            # the 'project' placeholder that lives in the raw config.
            preflight_config = (
                resolve_project_paths(self.manager.config, product)
                if product
                else self.manager.config
            )
            issues = PreflightChecker().run(selected_tasks, preflight_config)
            if issues:
                if self._background_mode:
                    for issue in issues:
                        self.log_message(
                            f"[PREFLIGHT {issue.severity.value.upper()}] "
                            f"{issue.task}: {issue.message}"
                        )
                    return False
                dlg = PreflightDialog(issues, parent=self)
                if not dlg.exec_():
                    return False  # User cancelled or errors block execution
        except Exception as exc:
            self.log_message(f"[WARNING] Preflight check failed: {exc}")

        # Update UI state
        if hasattr(self, "start_btn"):
            self.start_btn.setEnabled(False)
        if hasattr(self, "stop_btn"):
            self.stop_btn.setEnabled(True)

        # Reset status items
        self.reset_task_statuses(selected_tasks)

        # Start via manager
        self.manager.start_pipeline(selected_tasks, config_path, product)
        return True

    def apply_operator_handoff(self, handoff_path: str | Path) -> None:
        """Load an inference handoff and prepare one-click train/deploy.

        Args:
            handoff_path: Validated JSON contract exported by the inference GUI.
        """
        handoff = None
        try:
            from picture_tool.gui.operator_handoff import (
                OperatorHandoffError,
                apply_handoff_to_config,
                load_operator_handoff,
                materialize_job_dataset_snapshot,
            )
            from picture_tool.operator_job import (
                acquire_target_training_lock,
            )

            training_root = Path.cwd()
            handoff = load_operator_handoff(handoff_path, training_root=training_root)
            self._operator_handoff = handoff
            target = handoff.selected_target
            specific_config = (
                training_root / "configs" / f"{target.product.lower()}_pipeline.yaml"
            )
            config_path = (
                specific_config
                if specific_config.exists()
                else training_root / "configs" / "default_pipeline.yaml"
            )
            self.config_panel.config_path_edit.setText(str(config_path))
            self.config_panel.load_config()
            prepared = apply_handoff_to_config(self.manager.config, handoff)
            self.manager.update_config(prepared)
            self.config_panel.set_product_override(target.product, target.area)
            self.config_editor.set_config(prepared)
            feedback_count = len(target.sample_ids) or (
                target.ready_count + target.pending_count
            )
            self.operator_workflow_panel.configure_target(
                target.product,
                target.area,
                feedback_count=feedback_count,
                pending_count=target.pending_count,
                training_summary=(
                    f"{handoff.training_options.epochs} Epochs／"
                    f"增強 {handoff.training_options.augmentations_per_image}／"
                    f"Batch {handoff.training_options.batch}／"
                    f"{handoff.training_options.imgsz}px"
                ),
            )
            self._set_operator_mode(pending=target.pending_count > 0)
            if target.pending_count > 0:
                class_names = list(
                    (prepared.get("yolo_training", {}) or {}).get("class_names") or []
                )
                self.annotation_panel.configure_operator_pending(
                    target.dataset_root,
                    class_names,
                    handoff.path,
                )
                self.log_viewer.tabs.setCurrentWidget(self.annotation_panel)
                self._operator_handoff_target = None
                self._operator_handoff = handoff
                self._publish_operator_status(
                    state="waiting_annotation",
                    message=f"等待完成 {target.pending_count} 張補標",
                    pending_count=target.pending_count,
                    progress=0,
                )
                self.status_label.setText(
                    f"待標註：{target.product}/{target.area}（{target.pending_count} 張）"
                )
                self.status_label.setStyleSheet(
                    "color: #F0AD4E; font-size: 10pt; font-weight: bold;"
                )
                self._start_operator_job_heartbeat()
                return
            self._operator_training_lock = acquire_target_training_lock(
                handoff.data_root,
                product=target.product,
                area=target.area,
                job_id=handoff.job_id or handoff.path.stem,
            )
            self._publish_operator_status(
                state="preparing_dataset",
                message="正在建立固定的訓練資料快照",
                pending_count=0,
                progress=10,
                error="",
            )
            handoff = materialize_job_dataset_snapshot(handoff)
            target = handoff.selected_target
            prepared = apply_handoff_to_config(self.manager.config, handoff)
            self.manager.update_config(prepared)
            self.config_editor.set_config(prepared)
            if not self.task_control.apply_workflow("YOLO: train and deploy"):
                raise OperatorHandoffError(
                    "The 'YOLO: train and deploy' workflow is unavailable."
                )

            self._operator_handoff_target = target
            self._operator_handoff = handoff
            self._operator_cancel_requested = False
            self._start_operator_job_heartbeat()
            if self._operator_cancel_requested:
                self._publish_operator_status(
                    state="cancelled",
                    message="模型更新已在啟動前安全停止",
                )
                self._release_operator_training_lock()
                self._stop_operator_job_heartbeat()
                self._operator_handoff_target = None
                self._operator_handoff = None
                self._operator_cancel_requested = False
                self._request_background_exit()
                return
            self.config_panel.setEnabled(False)
            self.task_control.setEnabled(False)
            self.status_label.setText(f"待開始：{target.product}/{target.area}")
            self.status_label.setStyleSheet(
                "color: #6BCB77; font-size: 10pt; font-weight: bold;"
            )
            self.start_btn.setText("開始訓練並部署")
            if not self.start_pipeline():
                raise OperatorHandoffError("Training was not started.")
        except (OSError, RuntimeError, ValueError, yaml.YAMLError) as exc:
            operator_message = _operator_error_message(str(exc))
            if handoff is not None:
                self._publish_operator_status(
                    state="failed",
                    message=operator_message.splitlines()[0],
                    error=operator_message,
                )
            self._release_operator_training_lock()
            self._stop_operator_job_heartbeat()
            self._operator_handoff_target = None
            self._operator_handoff = None
            if self._background_mode:
                self.log_message(f"[ERROR] {operator_message}")
                self._request_background_exit()
            else:
                QMessageBox.critical(self, "無法接收訓練資料", operator_message)

    def _publish_operator_status(self, *, state: str, message: str, **values) -> None:
        """Publish the current operator job state without masking pipeline errors."""
        if self._operator_mode_enabled:
            self.operator_workflow_panel.set_state(
                state,
                message=message,
                progress=values.get("progress"),
                pending_count=values.get("pending_count"),
            )
        handoff = self._operator_handoff
        if handoff is None:
            return
        try:
            from picture_tool.operator_job import update_job_status

            update_job_status(
                handoff.status_path,
                state=state,
                message=message,
                **values,
            )
        except RuntimeError as exc:
            self.log_message(f"[WARNING] Unable to update operator status: {exc}")

    def _start_operator_job_heartbeat(self) -> None:
        """Publish an immediate lease and keep it fresh while this window owns a job."""
        if not self._operator_heartbeat_timer.isActive():
            self._operator_heartbeat_timer.start()
        self._refresh_operator_job_lease()

    def _stop_operator_job_heartbeat(self) -> None:
        self._operator_heartbeat_timer.stop()
        self._operator_heartbeat_failures = 0

    def _refresh_operator_job_lease(self) -> None:
        """Refresh persisted liveness and consume a cooperative cancel request."""
        handoff = self._operator_handoff
        if handoff is None:
            self._stop_operator_job_heartbeat()
            return
        try:
            from picture_tool.operator_job import refresh_operator_job_lease

            request = refresh_operator_job_lease(
                handoff.status_path,
                job_id=handoff.job_id or handoff.path.stem,
                lock=self._operator_training_lock,
            )
            self._operator_heartbeat_failures = 0
        except RuntimeError as exc:
            self._operator_heartbeat_failures += 1
            if self._operator_heartbeat_failures in {1, 3}:
                self.log_message(
                    "[WARNING] Unable to refresh operator heartbeat: "
                    f"{exc}"
                )
            return
        if request is not None and not self._operator_cancel_requested:
            self._handle_operator_cancel_request(
                request.request_id,
                request.requested_at,
            )

    def _handle_operator_cancel_request(
        self,
        request_id: str,
        requested_at: str,
    ) -> None:
        """Acknowledge one idempotent request before stopping at a safe point."""
        self._operator_cancel_requested = True
        self._publish_operator_status(
            state="cancelling",
            message="已收到安全停止要求，正在結束目前工作",
            handled_control_request_id=request_id,
            cancel_requested_at=requested_at,
        )
        if self._operator_handoff_target is None:
            self._publish_operator_status(
                state="cancelled",
                message="模型更新已安全停止",
                handled_control_request_id=request_id,
            )
            self._stop_operator_job_heartbeat()
            self._operator_handoff = None
            self._operator_cancel_requested = False
            self._request_background_exit()
            return
        self._request_operator_pipeline_stop()

    def _release_operator_training_lock(self) -> None:
        """Release the product/station lifecycle lock when a job terminates."""
        lock = self._operator_training_lock
        self._operator_training_lock = None
        if lock is None:
            return
        try:
            from picture_tool.operator_job import release_target_training_lock

            release_target_training_lock(lock)
        except RuntimeError as exc:
            self.log_message(f"[WARNING] Unable to release training lock: {exc}")

    def _set_operator_mode(self, *, pending: bool) -> None:
        """Reduce the training GUI to the controls required by an operator."""
        self._operator_mode_enabled = True
        self.setWindowTitle("產線模型補訓")
        self.side_bar.setVisible(False)
        self.operator_workflow_panel.setVisible(True)
        self.training_metrics.setVisible(False)
        self.pipeline_status_header.setVisible(False)
        self.status_list.setVisible(False)
        self.log_controls_container.setVisible(False)
        self.log_viewer.tabs.setVisible(pending)
        self.log_viewer.tabs.tabBar().setVisible(False)
        if pending:
            annotation_index = self.log_viewer.tabs.indexOf(self.annotation_panel)
            for index in range(self.log_viewer.tabs.count()):
                self.log_viewer.tabs.setTabVisible(index, index == annotation_index)
            self.log_viewer.tabs.setCurrentIndex(annotation_index)

    def stop_pipeline(self) -> None:
        """Request a non-blocking, cooperative pipeline stop."""
        if self._operator_handoff_target is not None:
            if self._operator_cancel_requested:
                return
            self._operator_cancel_requested = True
            self._publish_operator_status(
                state="cancelling",
                message="正在安全停止模型更新",
            )
            self._request_operator_pipeline_stop()
            if hasattr(self, "start_btn"):
                self.start_btn.setEnabled(False)
        else:
            self.manager.stop_pipeline()
            if hasattr(self, "start_btn"):
                self.start_btn.setEnabled(True)
        if hasattr(self, "stop_btn"):
            self.stop_btn.setEnabled(False)

    def _request_operator_pipeline_stop(self) -> None:
        request_stop = getattr(self.manager, "request_pipeline_stop", None)
        if callable(request_stop):
            request_stop()
            return
        self.manager.stop_pipeline()

    def log_message(self, message: str) -> None:
        """Log message wrapper - delegates to LogViewer and extracts metrics."""
        self._log_history.append(message)
        self.log_viewer.log_message(message)
        if self._operator_handoff_target is not None:
            epoch_status = _operator_epoch_status(message)
            if epoch_status is not None:
                status_message, progress = epoch_status
                self._publish_operator_status(
                    state="training",
                    message=status_message,
                    current_task="yolo_train",
                    progress=progress,
                )
        # Extract training metrics from ultralytics epoch lines
        if hasattr(self, "training_metrics") and not self._operator_mode_enabled:
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
            if self._operator_handoff_target is not None:
                self._close_when_pipeline_stops = True
                self.stop_pipeline()
                event.ignore()
                return
            self.manager.stop_pipeline()
        if self._operator_handoff is not None:
            try:
                from picture_tool.operator_job import clear_operator_job_process

                clear_operator_job_process(
                    self._operator_handoff.status_path,
                    job_id=(
                        self._operator_handoff.job_id
                        or self._operator_handoff.path.stem
                    ),
                )
            except RuntimeError as exc:
                self.log_message(
                    f"[WARNING] Unable to clear operator process lease: {exc}"
                )
        self._stop_operator_job_heartbeat()
        super().closeEvent(event)

    def on_tasks_changed(self, tasks: list) -> None:
        """Handle tasks changed signal from TaskControlPanel."""
        if self._operator_mode_enabled:
            # The operator facade does not render dependency details. Resolving
            # them here imports the training registry (and potentially Torch)
            # on the UI thread, delaying the transition after annotation.
            self._rebuild_status_items(only=tasks)
            return
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
        operator_target = self._operator_handoff_target
        operator_handoff = self._operator_handoff
        operator_was_cancelled = self._operator_cancel_requested
        self._operator_handoff_target = None
        if operator_target is not None:
            if operator_was_cancelled:
                self._publish_operator_status(
                    state="cancelled",
                    message="模型更新已停止",
                )
            else:
                self._publish_operator_status(
                    state="deployed",
                    message="模型訓練、驗證與部署已完成",
                    progress=100,
                    pending_count=0,
                )
            self._release_operator_training_lock()
            self._stop_operator_job_heartbeat()
            self._operator_handoff = None
            self._operator_cancel_requested = False
        self.log_message(
            "[INFO] Pipeline stopped at a safe point."
            if operator_was_cancelled
            else "[SUCCESS] Pipeline finished successfully."
        )
        self._reset_ui_state()
        if hasattr(self, "progress_bar"):
            if not operator_was_cancelled:
                self.progress_bar.setValue(100)
            self.status_label.setText(
                "Pipeline Stopped"
                if operator_was_cancelled
                else "Pipeline Completed"
            )
        if hasattr(self, "training_metrics"):
            self.training_metrics.reset()
        if operator_target is not None:
            self.config_panel.setEnabled(True)
            self.task_control.setEnabled(True)
            self.start_btn.setText("▶ RUN PIPELINE")
            if operator_was_cancelled:
                self.log_message("[INFO] Operator retraining stopped safely.")
                if self._background_mode:
                    self._request_background_exit()
            else:
                acceptance_message = (
                    f"{operator_target.product}/{operator_target.area} 訓練與部署已完成。"
                )
                if operator_handoff is not None:
                    try:
                        acceptance = load_operator_acceptance_summary(
                            operator_handoff.inference_models_dir,
                            product=operator_target.product,
                            area=operator_target.area,
                        )
                        acceptance_message = acceptance.to_operator_text()
                    except OperatorAcceptanceError as exc:
                        self.log_message(
                            "[WARNING] Deployment finished but acceptance evidence "
                            f"could not be loaded: {exc}"
                        )
                        acceptance_message += (
                            "\n\n無法讀取完整離線驗收證據，請勿直接認定產線驗收完成。\n"
                            f"{exc}"
                        )
                if self._background_mode:
                    self.log_message(f"[SUCCESS] {acceptance_message}")
                    self._request_background_exit()
                else:
                    QMessageBox.information(
                        self,
                        "訓練與部署驗收摘要",
                        acceptance_message,
                    )
            if self._close_when_pipeline_stops:
                self._close_when_pipeline_stops = False
                QtCore.QTimer.singleShot(0, self.close)

    def on_pipeline_error(self, message: str):
        """Handle pipeline error."""
        operator_target = self._operator_handoff_target
        operator_was_cancelled = self._operator_cancel_requested
        self._operator_handoff_target = None
        operator_message = _operator_error_message(message)
        if operator_target is not None:
            self._publish_operator_status(
                state="cancelled" if operator_was_cancelled else "failed",
                message=(
                    "模型更新已停止"
                    if operator_was_cancelled
                    else operator_message.splitlines()[0]
                ),
                error="" if operator_was_cancelled else operator_message,
            )
            self._release_operator_training_lock()
            self._stop_operator_job_heartbeat()
            self._operator_handoff = None
            self._operator_cancel_requested = False
        self.log_message(f"[ERROR] Pipeline failed: {message}")
        self._reset_ui_state()
        if hasattr(self, "status_label"):
            self.status_label.setText("Pipeline Error")
        if hasattr(self, "training_metrics"):
            self.training_metrics.reset()
        if operator_target is not None:
            self.config_panel.setEnabled(True)
            self.task_control.setEnabled(True)
            self.start_btn.setText("▶ RUN PIPELINE")
            if self._background_mode:
                self._request_background_exit()
            elif not operator_was_cancelled:
                QMessageBox.critical(
                    self,
                    "訓練未完成",
                    f"{operator_target.product}/{operator_target.area} 未部署。\n\n"
                    f"{operator_message}",
                )
            if self._close_when_pipeline_stops:
                self._close_when_pipeline_stops = False
                QtCore.QTimer.singleShot(0, self.close)

    def on_task_started(self, task_name: str):
        """Handle task start."""
        self._set_task_status(task_name, "Running...", color="#4D96FF")
        operator_states = {
            "yolo_augmentation": ("preparing_dataset", "正在進行資料增強", 15),
            "dataset_lint": ("preparing_dataset", "正在檢查訓練資料", 25),
            "dataset_readiness": ("preparing_dataset", "正在檢查資料量", 20),
            "dataset_splitter": ("preparing_dataset", "正在切分訓練資料", 30),
            "yolo_train": ("training", "模型訓練中", 40),
            "position_validation": ("evaluating", "正在驗證物件位置", 78),
            "yolo_evaluation": ("evaluating", "正在驗證模型品質", 84),
            "generate_report": ("evaluating", "正在產生訓練報告", 87),
            "batch_inference": ("evaluating", "正在測試推論結果", 90),
            "qc_summary": ("evaluating", "正在彙整品質報告", 93),
            "deploy": ("deploying", "正在部署新模型", 95),
        }
        if task_name in operator_states:
            state, message, progress = operator_states[task_name]
            self._publish_operator_status(
                state=state,
                message=message,
                current_task=task_name,
                progress=progress,
            )
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

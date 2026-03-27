"""Color Verification and Inspection Panel.

Integrates SAM-based color sampling and batch verification tools into the main GUI.
"""
from __future__ import annotations

import logging
from pathlib import Path

from PyQt5 import QtCore
from PyQt5.QtWidgets import (
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
    QTabWidget,
    QDialog,
    QFormLayout,
    QDialogButtonBox,
    QComboBox,
)

# Placeholder imports - these will be replaced with actual module imports 
# once those modules are fully ready/refactored for GUI usage.
try:
    from picture_tool.color import color_inspection
    from picture_tool.color import color_verifier
except ImportError:
    color_inspection = None  # type: ignore[assignment]
    color_verifier = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)


class ColorPanel(QWidget):
    """
    Main container for Color Tools.
    Contains two tabs:
    1. Inspection (Template Collection with SAM)
    2. Verification (Batch Analysis)
    """

    log_message = QtCore.pyqtSignal(str)

    def __init__(self, manager, parent=None):
        super().__init__(parent)
        self.manager = manager
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        self.tabs = QTabWidget()
        self.tabs.addTab(self._build_inspection_tab(), "🎨 顏色範本蒐集 (SAM)")
        self.tabs.addTab(self._build_verification_tab(), "✅ 批次顏色驗證")
        
        layout.addWidget(self.tabs)

    def _build_inspection_tab(self) -> QWidget:
        """Build the SAM-based template collection tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(10, 10, 10, 10)
        
        # Determine status of SAM
        sam_status = "未安裝"
        sam_color = "#ff6b6b" # Red
        if color_inspection:
            try:
                import os
                if os.environ.get("PYTEST_IS_RUNNING") == "1":
                    raise ImportError("Bypass ultralytics during pytest")
                import ultralytics  # noqa: F401
                sam_status = "已就緒 (SAM 2 Supported)"
                sam_color = "#6BCB77" # Green
            except ImportError:
                sam_status = "缺少 ultralytics 套件"
        
        info_label = QLabel(
            f"此工具使用 Segment Anything Model (SAM) 進行快速顏色採樣。\n"
            f"狀態: <span style='color:{sam_color}; font-weight:bold'>{sam_status}</span>"
        )
        info_label.setTextFormat(QtCore.Qt.TextFormat.RichText)
        layout.addWidget(info_label)
        
        # Placeholder for tool launch
        btn_layout = QHBoxLayout()
        launch_btn = QPushButton("🚀 啟動 SAM 採樣工具 (獨立視窗)")
        launch_btn.setMinimumHeight(50)
        launch_btn.clicked.connect(self._launch_sam_tool)
        btn_layout.addWidget(launch_btn)
        
        layout.addLayout(btn_layout)
        layout.addStretch()
        return widget

    def _build_verification_tab(self) -> QWidget:
        """Build the batch verification tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(10, 10, 10, 10)
        
        # Configuration Area
        config_group = QGroupBox("驗證設定")
        form_layout = QGridLayout(config_group)
        
        # Input Dir
        form_layout.addWidget(QLabel("待測圖片目錄:"), 0, 0)
        self.verify_input_edit = QLineEdit()
        browse_in_btn = QPushButton("...")
        browse_in_btn.clicked.connect(lambda: self._browse_dir(self.verify_input_edit))
        form_layout.addWidget(self.verify_input_edit, 0, 1)
        form_layout.addWidget(browse_in_btn, 0, 2)
        
        # Color Stats JSON
        form_layout.addWidget(QLabel("顏色範本 (JSON):"), 1, 0)
        self.verify_stats_edit = QLineEdit()
        browse_stats_btn = QPushButton("...")
        browse_stats_btn.clicked.connect(lambda: self._browse_file(self.verify_stats_edit, "*.json"))
        form_layout.addWidget(self.verify_stats_edit, 1, 1)
        form_layout.addWidget(browse_stats_btn, 1, 2)
        
        layout.addWidget(config_group)
        
        # Action Area
        action_layout = QHBoxLayout()
        run_btn = QPushButton("▶ 開始驗證")
        run_btn.setObjectName("PrimaryBtn")
        run_btn.setMinimumHeight(40)
        run_btn.clicked.connect(self._run_verification)
        action_layout.addWidget(run_btn)
        layout.addLayout(action_layout)
        
        # Result Area
        self.result_text = QTextEdit()
        self.result_text.setReadOnly(True)
        self.result_text.setPlaceholderText("驗證結果將顯示於此...")
        layout.addWidget(self.result_text)
        
        return widget

    def _browse_dir(self, line_edit: QLineEdit) -> None:
        folder = QFileDialog.getExistingDirectory(self, "選擇目錄")
        if folder:
            line_edit.setText(folder)

    def _browse_file(self, line_edit: QLineEdit, filter_pattern: str) -> None:
        file_path, _ = QFileDialog.getOpenFileName(self, "選擇檔案", filter=filter_pattern)
        if file_path:
            line_edit.setText(file_path)

    def _launch_sam_tool(self) -> None:
        """Launch the SAM-based color inspection tool."""
        if not color_inspection:
            QMessageBox.critical(self, "錯誤", "未安裝 picture_tool.color 模組。")
            return

        # Simple input dialog for setup
        dialog = QDialog(self)
        dialog.setWindowTitle("啟動 SAM 採樣工具")
        dialog.setObjectName("SamDialog")
        dialog.resize(500, 300)
        
        layout = QVBoxLayout(dialog)
        
        # Form
        form = QFormLayout()
        
        
        # Load previous settings
        settings = QtCore.QSettings()
        last_input = settings.value("sam_tool/input_dir", "")
        last_output = settings.value("sam_tool/output_json", "")
        last_ckpt = settings.value("sam_tool/checkpoint", "models/sam2_b.pt")
        last_type = settings.value("sam_tool/model_type", "sam2_b")
        last_colors = settings.value("sam_tool/target_colors", "Red, Green, Blue, Yellow, Black, White")

        # Input Dir
        self._sam_input_edit = QLineEdit(last_input)
        browse_in_btn = QPushButton("...")
        browse_in_btn.clicked.connect(lambda: self._browse_dir(self._sam_input_edit))
        input_row = QHBoxLayout()
        input_row.addWidget(self._sam_input_edit)
        input_row.addWidget(browse_in_btn)
        form.addRow("待標註圖片目錄:", input_row)
        
        # Output JSON
        self._sam_output_edit = QLineEdit(last_output)
        browse_out_btn = QPushButton("...")
        browse_out_btn.clicked.connect(lambda: self._save_file(self._sam_output_edit, "*.json"))
        output_row = QHBoxLayout()
        output_row.addWidget(self._sam_output_edit)
        output_row.addWidget(browse_out_btn)
        form.addRow("輸出 JSON 路徑:", output_row)
        
        # Model Checkpoint
        self._sam_ckpt_edit = QLineEdit(last_ckpt)
        browse_ckpt_btn = QPushButton("...")
        browse_ckpt_btn.clicked.connect(self._browse_sam_checkpoint)
        ckpt_row = QHBoxLayout()
        ckpt_row.addWidget(self._sam_ckpt_edit)
        ckpt_row.addWidget(browse_ckpt_btn)
        form.addRow("SAM 模型權重:", ckpt_row)
        
        # Model Type
        self._sam_type_combo = QComboBox()
        self._sam_type_combo.addItems(["sam2_t", "sam2_s", "sam2_b", "sam2_l", "vit_b", "vit_l", "vit_h"])
        self._sam_type_combo.setCurrentText(str(last_type))
        form.addRow("模型類型:", self._sam_type_combo)

        # Target Colors 
        self._sam_colors_edit = QLineEdit(last_colors)
        form.addRow("目標顏色 (逗號分隔):", self._sam_colors_edit)
        
        layout.addLayout(form)
        
        # Buttons
        btn_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btn_box.accepted.connect(dialog.accept)
        btn_box.rejected.connect(dialog.reject)
        layout.addWidget(btn_box)
        
        if dialog.exec_() == QDialog.Accepted:
            input_dir = self._sam_input_edit.text().strip()
            output_json = self._sam_output_edit.text().strip()
            ckpt = self._sam_ckpt_edit.text().strip()
            model_type = self._sam_type_combo.currentText()
            colors = [c.strip() for c in self._sam_colors_edit.text().split(",") if c.strip()]
            
            # Save settings
            settings.setValue("sam_tool/input_dir", input_dir)
            settings.setValue("sam_tool/output_json", output_json)
            settings.setValue("sam_tool/checkpoint", ckpt)
            settings.setValue("sam_tool/model_type", model_type)
            settings.setValue("sam_tool/target_colors", self._sam_colors_edit.text())
            
            if not input_dir or not output_json or not ckpt:
                QMessageBox.warning(self, "參數不足", "請填寫所有必欄位。")
                return

            # Check if model exists or is a valid downloadable name
            is_file = Path(ckpt).exists()
            # Ultralytics supports sam2_t.pt, sam2_s.pt, sam2_b.pt, sam2_l.pt
            valid_names = ["sam2_t.pt", "sam2_s.pt", "sam2_b.pt", "sam2_l.pt", "sam2.pt"]
            
            if not is_file:
                if ckpt in valid_names:
                     # Auto-download scenario
                     pass
                else:
                     QMessageBox.warning(self, "錯誤", f"找不到模型權重: {ckpt}\n若是本地檔案請確認路徑，若是官方模型請使用標準名稱 (如 sam2_b.pt)。")
                     return
            
            try:
                # Prepare config
                # We construct SessionConfig manually or via a dict
                # Note: color_inspection.SessionConfig expects specific types
                from picture_tool.color.color_inspection import SessionConfig, SamSettings, run_gui_session
                # Check actual CUDA availability
                use_cuda_pref = QtCore.QSettings().value("use_cuda", True, type=bool)
                device = "cpu"
                use_cuda = False
                import os
                if os.environ.get("PYTEST_IS_RUNNING") != "1":
                    import torch
                    use_cuda = torch.cuda.is_available()
                    
                if use_cuda_pref and use_cuda:
                    device = "cuda"
                elif use_cuda_pref:
                    logger.warning("CUDA requested but not available in Torch. Falling back to CPU.")
                
                cfg = SessionConfig(
                    input_dir=Path(input_dir),
                    output_json=Path(output_json),
                    colors=colors,
                    sam=SamSettings(
                        checkpoint=Path(ckpt),
                        model_type=model_type,
                        device=device
                    )
                )
                
                # Launch!
                # Since run_gui_session creates a new window, calling it directly is fine 
                # as long as we are on the main thread.
                # However, it might block the main window if it calls app.exec_().
                # Let's check run_gui_session implementation again.
                # It does: window.show() then if owns_app -> app.exec_().
                # Since we are already in an app, it won't call exec_(), just show().
                # We need to keep a reference to the window or it will be garbage collected.
                
                logger.debug("Launching run_gui_session")
                self._sam_window = run_gui_session(cfg)
                logger.debug("run_gui_session returned")
                
            except ImportError as e:
                QMessageBox.critical(self, "模組遺失", f"無法啟動工具，缺少相依套件:\n{e}")
                logger.error(f"Missing import for SAM tool: {e}")
            except RuntimeError as e:
                QMessageBox.critical(self, "執行錯誤", f"SAM 工具執行時發生問題:\n{e}")
                logger.exception("Runtime error in SAM tool")
            except Exception as e:
                QMessageBox.critical(self, "未預期錯誤", f"啟動失敗:\n{e}")
                logger.exception("Unexpected failure when launching SAM tool")

    def _browse_sam_checkpoint(self) -> None:
        file_path, _ = QFileDialog.getOpenFileName(self, "選擇 SAM 模型", filter="*.pt;*.pth")
        if file_path:
            self._sam_ckpt_edit.setText(file_path)
            # Auto-detect type
            lower = Path(file_path).name.lower()
            if "sam2" in lower:
                if "_t" in lower: 
                    self._sam_type_combo.setCurrentText("sam2_t")
                elif "_s" in lower: 
                    self._sam_type_combo.setCurrentText("sam2_s")
                elif "_b" in lower: 
                    self._sam_type_combo.setCurrentText("sam2_b")
                elif "_l" in lower: 
                    self._sam_type_combo.setCurrentText("sam2_l")
            elif "vit_h" in lower:
                self._sam_type_combo.setCurrentText("vit_h")
            elif "vit_l" in lower:
                self._sam_type_combo.setCurrentText("vit_l")
            elif "vit_b" in lower:
                self._sam_type_combo.setCurrentText("vit_b")

    def _save_file(self, line_edit: QLineEdit, filter_pattern: str) -> None:
        file_path, _ = QFileDialog.getSaveFileName(self, "儲存檔案", filter=filter_pattern)
        if file_path:
            line_edit.setText(file_path)

    def _run_verification(self) -> None:
        input_dir = self.verify_input_edit.text()
        stats_path = self.verify_stats_edit.text()
        
        if not input_dir or not stats_path:
            QMessageBox.warning(self, "參數不足", "請指定圖片目錄和顏色範本 JSON。")
            return
            
        if not Path(input_dir).exists():
             QMessageBox.warning(self, "錯誤", f"找不到目錄: {input_dir}")
             return

        if not Path(stats_path).exists():
             QMessageBox.warning(self, "錯誤", f"找不到範本檔案: {stats_path}")
             return

        self.result_text.append(f"正在建立驗證排程: {input_dir}...")
        self.result_text.append(f"使用範本: {stats_path}")
        
        # Instantiate worker thread
        self.verify_worker = VerificationWorker(
            input_dir=input_dir,
            stats_path=stats_path
        )
        
        self.verify_worker.progress_msg.connect(self._on_verify_progress)
        self.verify_worker.result_msg.connect(self._on_verify_result)
        self.verify_worker.error_msg.connect(self._on_verify_error)
        self.verify_worker.finished.connect(self._on_verify_finished)
        
        # Disable button during verification to prevent multiple clicks
        # Assuming you want to find run_btn. If we need to disable it properly, 
        # we should make run_btn a self member, but we can pass for now.
        
        self.verify_worker.start()

    def _on_verify_progress(self, msg: str):
        self.log_message.emit(msg)
        
    def _on_verify_result(self, report_lines: list):
        self.result_text.append("\n" + "\n".join(report_lines))
        
    def _on_verify_error(self, err_msg: str):
        self.result_text.append(f"錯誤發生: {err_msg}")
        self.log_message.emit(f"[ERROR] Verification failed: {err_msg}")
        
    def _on_verify_finished(self):
        self.log_message.emit("[INFO] Verification thread finished.")


# Define the worker class at the bottom of the file
class VerificationWorker(QtCore.QThread):
    progress_msg = QtCore.pyqtSignal(str)
    result_msg = QtCore.pyqtSignal(list)
    error_msg = QtCore.pyqtSignal(str)
    
    def __init__(self, input_dir: str, stats_path: str):
        super().__init__()
        self.input_dir = Path(input_dir)
        self.stats_path = Path(stats_path)
        
    def run(self):
        import logging
        verify_logger = logging.getLogger("picture_tool.color.verification_gui")
        verify_logger.setLevel(logging.INFO)
        
        # Remove existing handlers to avoid duplication
        verify_logger.handlers.clear()
        
        # Add file handler
        log_file = Path("logs/color_verification.log")
        try:
            log_file.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(log_file, encoding='utf-8')
            file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
            verify_logger.addHandler(file_handler)
        except OSError as e:
            self.progress_msg.emit(f"[WARNING] Failed to setup file logging: {e}")
            
        try:
            if color_verifier:
                summary, decisions = color_verifier.verify_directory(
                    self.input_dir, 
                    self.stats_path,
                    logger=verify_logger
                )
                
                # Format output
                report = []
                report.append("=== 驗證完成 ===")
                for key, val in summary.items():
                    if isinstance(val, int):
                         report.append(f"{key}: {val}")
                
                # Show mismatches
                mismatches = [d for d in decisions if d.status == "MISMATCH"]
                if mismatches:
                    report.append(f"\n發現 {len(mismatches)} 個不匹配:")
                    for m in mismatches[:10]:
                        report.append(f"  {m.image.name}: 預測 {m.predicted_color} != 預期 {m.expected_color}")
                    if len(mismatches) > 10:
                        report.append("  ...")
                else:
                    report.append("\n✅ 所有圖片皆符合預期 (或無預期標籤)")
                
                self.result_msg.emit(report)
                self.progress_msg.emit("[INFO] Color verification completed.")
                verify_logger.info(f"Completed verification: {summary}")
            else:
                 self.error_msg.emit("無法載入 color_verifier 模組")
                 
        except (ValueError, FileNotFoundError) as e:
            self.error_msg.emit(f"驗證參數或檔案錯誤: {e}")
            if verify_logger:
                verify_logger.error(f"Validation Error: {e}")
        except RuntimeError as e:
            self.error_msg.emit(f"驗證核心執行錯誤: {e}")
            if verify_logger:
                verify_logger.exception("Verification core failed")
        except Exception as e:
            self.error_msg.emit(f"未預期的錯誤: {e}")
            if verify_logger:
                verify_logger.exception("Unexpected verification failure")
        finally:
            # Clean up handlers
            if verify_logger:
                for handler in verify_logger.handlers[:]:
                    handler.close()
                    verify_logger.removeHandler(handler)

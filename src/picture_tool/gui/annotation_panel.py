"""Annotation Panel Module.

Extracted from app.py to handle all annotation-related UI and logic.
Maintains 100% visual consistency with the original implementation.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from PyQt5 import QtCore, QtGui
from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import (
    QFileDialog,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QListWidget,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from picture_tool.gui.annotation_tracker import AnnotationTracker
from picture_tool.gui.labelimg_launcher import LabelImgLauncher
from picture_tool.pending_annotations import (
    PendingAnnotationError,
    configure_pending_workspace,
    promote_completed_pending,
    record_label_verification,
)


class AnnotationPanel(QWidget):
    """
    Manages the Annotation Tab UI and logic.
    Propagates logs via `message_logged` signal to the main window.
    """
    message_logged = pyqtSignal(str)
    pending_ready = pyqtSignal(str)

    def __init__(self, manager, parent=None):
        super().__init__(parent)
        self.manager = manager
        
        # Initialize components specific to this panel
        self.labelimg_launcher = LabelImgLauncher()
        self.annotation_tracker = AnnotationTracker()
        
        # State
        self.annotation_input_dir: Optional[Path] = None
        self.annotation_output_dir: Optional[Path] = None
        self.annotation_classes: List[str] = []
        self.operator_dataset_root: Optional[Path] = None
        self.operator_handoff_path: Optional[Path] = None
        self.operator_class_names: tuple[str, ...] = ()
        self.operator_mode_enabled = False
        self.operator_auto_launch_used = False
        self.labelimg_poll_timer = QtCore.QTimer(self)
        self.labelimg_poll_timer.setInterval(750)
        self.labelimg_poll_timer.timeout.connect(self._poll_operator_labelimg)
        
        # Build UI
        self._build_ui()
        self._load_annotation_settings()

    def _build_ui(self) -> None:
        """Build the annotation management tab (exact copy of original layout)."""
        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(15)

        # Left: Class Management
        self.class_management_panel = self._build_class_management_panel()
        self.class_management_panel.setMaximumWidth(300)

        # Middle: Progress and Statistics
        middle_panel = self._build_annotation_progress_panel()

        # Right: Actions and Settings
        right_panel = self._build_annotation_actions_panel()
        right_panel.setMaximumWidth(300)

        main_layout.addWidget(self.class_management_panel, 1)
        main_layout.addWidget(middle_panel, 2)
        main_layout.addWidget(right_panel, 1)

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

        import_file_btn = QPushButton("匯入標籤檔")
        import_file_btn.clicked.connect(self._import_classes_from_file_dialog)

        save_btn = QPushButton("💾 儲存類別")
        save_btn.setObjectName("SuccessBtn")
        save_btn.clicked.connect(self._save_annotation_classes)

        btn_layout.addWidget(add_btn, 0, 0)
        btn_layout.addWidget(edit_btn, 0, 1)
        btn_layout.addWidget(delete_btn, 1, 0)
        btn_layout.addWidget(import_btn, 1, 1)
        btn_layout.addWidget(import_file_btn, 2, 0, 1, 2)
        btn_layout.addWidget(save_btn, 3, 0, 1, 2)

        layout.addLayout(btn_layout)
        layout.addStretch()

        return group

    def _build_annotation_progress_panel(self) -> QWidget:
        """Build annotation progress panel."""
        group = QGroupBox("標註進度")
        layout = QVBoxLayout(group)
        layout.setSpacing(10)

        self.operator_instruction_label = QLabel(
            "漏檢補標只需 4 步：1. 拖曳框住漏檢物件；2. 選擇正確類別；"
            "3. 按 Ctrl+S 儲存；4. 關閉標註工具，系統會自動檢查。"
        )
        self.operator_instruction_label.setWordWrap(True)
        self.operator_instruction_label.setStyleSheet(
            "font-size: 12pt; font-weight: bold; padding: 12px; "
            "background: #243447; color: white;"
        )
        self.operator_instruction_label.setVisible(False)
        layout.addWidget(self.operator_instruction_label)

        # Statistics
        self.annotation_stats_label = QLabel("尚未掃描")
        self.annotation_stats_label.setStyleSheet("font-size: 11pt; color: #c9d1d9;")
        layout.addWidget(self.annotation_stats_label)

        # Progress bar
        self.annotation_progress_bar = QProgressBar()
        self.annotation_progress_bar.setValue(0)
        layout.addWidget(self.annotation_progress_bar)

        # Class distribution
        self.annotation_class_dist_label = QLabel("類別分佈：")
        layout.addWidget(self.annotation_class_dist_label)
        self.annotation_class_dist = QTextEdit()
        self.annotation_class_dist.setReadOnly(True)
        self.annotation_class_dist.setMaximumHeight(150)
        self.annotation_class_dist.setFont(QtGui.QFont("Consolas", 9))
        layout.addWidget(self.annotation_class_dist)

        # Unannotated files
        self.annotation_unannotated_label = QLabel("未標註圖片：")
        layout.addWidget(self.annotation_unannotated_label)
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
        self.launch_annotation_btn = QPushButton("開始標註")
        self.launch_annotation_btn.setObjectName("PrimaryBtn")
        self.launch_annotation_btn.setMinimumHeight(48)
        self.launch_annotation_btn.clicked.connect(self._launch_labelimg)
        layout.addWidget(self.launch_annotation_btn)

        # Validate annotations button
        self.validate_annotations_btn = QPushButton("驗證標註")
        self.validate_annotations_btn.clicked.connect(self._validate_annotations)
        layout.addWidget(self.validate_annotations_btn)

        self.complete_pending_btn = QPushButton("我已儲存，重新檢查")
        self.complete_pending_btn.setObjectName("SuccessBtn")
        self.complete_pending_btn.setEnabled(False)
        self.complete_pending_btn.clicked.connect(
            lambda _checked=False: self._complete_operator_pending()
        )
        layout.addWidget(self.complete_pending_btn)

        # Rescan button
        self.rescan_annotation_btn = QPushButton("重新掃描")
        self.rescan_annotation_btn.clicked.connect(self._scan_annotation_progress)
        layout.addWidget(self.rescan_annotation_btn)

        # Start augmentation button
        self.augment_annotation_btn = QPushButton("完成後，開始擴增")
        self.augment_annotation_btn.setObjectName("SuccessBtn")
        self.augment_annotation_btn.clicked.connect(
            self._start_augmentation_from_annotation
        )
        layout.addWidget(self.augment_annotation_btn)

        layout.addWidget(self._create_separator())

        # Settings
        self.annotation_settings_label = QLabel("設定")
        layout.addWidget(self.annotation_settings_label)

        # Input directory
        input_layout = QVBoxLayout()
        self.annotation_input_label = QLabel("輸入目錄：")
        input_layout.addWidget(self.annotation_input_label)
        self.annotation_input_edit = QLineEdit()
        self.annotation_input_edit.setPlaceholderText("選擇包含圖片的資料夾...")
        self.annotation_input_browse_btn = QPushButton("瀏覽...")
        self.annotation_input_browse_btn.clicked.connect(
            self._browse_annotation_input
        )

        input_row = QHBoxLayout()
        input_row.addWidget(self.annotation_input_edit)
        input_row.addWidget(self.annotation_input_browse_btn)
        input_layout.addLayout(input_row)
        layout.addLayout(input_layout)

        # Output directory
        output_layout = QVBoxLayout()
        self.annotation_output_label = QLabel("標註輸出目錄：")
        output_layout.addWidget(self.annotation_output_label)
        self.annotation_output_edit = QLineEdit()
        self.annotation_output_edit.setPlaceholderText("標註文件儲存位置...")
        self.annotation_output_browse_btn = QPushButton("瀏覽...")
        self.annotation_output_browse_btn.clicked.connect(
            self._browse_annotation_output
        )

        output_row = QHBoxLayout()
        output_row.addWidget(self.annotation_output_edit)
        output_row.addWidget(self.annotation_output_browse_btn)
        output_layout.addLayout(output_row)
        layout.addLayout(output_layout)

        layout.addStretch()

        return group

    def set_operator_mode(self, enabled: bool) -> None:
        """Show only the controls required by the inference-to-training workflow.

        Args:
            enabled: Hide engineering controls when true.
        """
        self.operator_mode_enabled = enabled
        self.class_management_panel.setVisible(not enabled)
        self.operator_instruction_label.setVisible(enabled)
        self.complete_pending_btn.setVisible(not enabled)
        self.launch_annotation_btn.setText(
            "開啟／繼續標註" if enabled else "開始標註"
        )
        for widget in (
            self.annotation_class_dist_label,
            self.annotation_class_dist,
            self.annotation_unannotated_label,
            self.annotation_unannotated_list,
            self.validate_annotations_btn,
            self.rescan_annotation_btn,
            self.augment_annotation_btn,
            self.annotation_settings_label,
            self.annotation_input_label,
            self.annotation_input_edit,
            self.annotation_input_browse_btn,
            self.annotation_output_label,
            self.annotation_output_edit,
            self.annotation_output_browse_btn,
        ):
            widget.setVisible(not enabled)

    def configure_operator_pending(
        self,
        dataset_root: str | Path,
        class_names: List[str],
        handoff_path: str | Path,
    ) -> None:
        """Load an inference handoff's pending queue without path entry.

        Args:
            dataset_root: Product/station dataset root.
            class_names: Ordered class contract validated by the handoff.
            handoff_path: JSON handoff updated after promotion.

        Raises:
            PendingAnnotationError: If paths or classes are invalid.
        """
        root = Path(dataset_root).expanduser().resolve()
        if root != self.operator_dataset_root:
            self.operator_auto_launch_used = False
        images_dir, labels_dir, _classes_file = configure_pending_workspace(
            root, [str(name) for name in class_names]
        )
        self.operator_dataset_root = root
        self.operator_handoff_path = Path(handoff_path).expanduser().resolve()
        self.operator_class_names = tuple(str(name) for name in class_names)
        self.annotation_input_dir = images_dir
        self.annotation_output_dir = labels_dir
        self.annotation_classes = [str(name) for name in class_names]
        self.annotation_input_edit.setText(str(images_dir))
        self.annotation_output_edit.setText(str(labels_dir))
        self._refresh_class_list()
        self.set_operator_mode(True)
        self._scan_annotation_progress()
        if not self.operator_auto_launch_used:
            self.operator_instruction_label.setText(
                "第 2 步：標註工具將自動開啟。拖曳框住漏檢物件、選擇類別，"
                "按 Ctrl+S 儲存；關閉工具後，系統會自動檢查並繼續補訓。"
            )
            self.operator_auto_launch_used = True
            QtCore.QTimer.singleShot(300, self._launch_labelimg)
        else:
            self.operator_instruction_label.setText(
                "仍有影像尚未完成。按「開啟／繼續標註」完成後續處理。"
            )

    def _complete_operator_pending(self, *, automatic: bool = False) -> None:
        """Validate saved labels, promote completed rows, and resume handoff."""
        if self.operator_dataset_root is None or self.operator_handoff_path is None:
            QMessageBox.warning(self, "無待標註案件", "目前沒有推理系統交付的待標註案件。")
            return
        if tuple(self.annotation_classes) != self.operator_class_names:
            QMessageBox.critical(
                self,
                "類別設定已變更",
                "推理模型的類別名稱或順序不可在本次補標流程中修改。請重新載入訓練資料。",
            )
            return
        try:
            report = promote_completed_pending(
                self.operator_dataset_root,
                self.annotation_classes,
                self.operator_handoff_path,
            )
        except (OSError, PendingAnnotationError) as exc:
            QMessageBox.critical(self, "標註資料未加入", str(exc))
            return
        self._scan_annotation_progress()
        if report.promoted_count == 0:
            if automatic:
                self.complete_pending_btn.setVisible(True)
                self.complete_pending_btn.setEnabled(True)
                self.operator_instruction_label.setText(
                    f"尚有 {report.remaining_count} 張未完成。"
                    "若已按 Ctrl+S，請按「重新檢查已儲存標註」；"
                    "否則再開啟標註工具。"
                )
                self.complete_pending_btn.setText("重新檢查已儲存標註")
                return
            QMessageBox.warning(
                self,
                "尚未完成",
                f"沒有可加入的完整標註；仍有 {report.remaining_count} 張待處理。",
            )
            return
        if not automatic:
            QMessageBox.information(
                self,
                "標註已加入",
                f"本次加入 {report.promoted_count} 張；"
                f"仍待標註 {report.remaining_count} 張。",
            )
        self.complete_pending_btn.setVisible(False)
        self.pending_ready.emit(str(report.handoff_path))

    def _poll_operator_labelimg(self) -> None:
        """Continue the operator workflow after the annotation window closes."""
        if self.labelimg_launcher.is_running():
            return
        self.labelimg_poll_timer.stop()
        self.launch_annotation_btn.setEnabled(True)
        exit_error = self.labelimg_launcher.process_exit_error()
        if exit_error:
            self.operator_instruction_label.setText(
                "標註工具未正常開啟，請通知工程人員查看錯誤紀錄。"
            )
            QMessageBox.critical(self, "標註工具錯誤", exit_error)
            return
        try:
            if self.annotation_output_dir is None:
                raise PendingAnnotationError("標註輸出目錄不存在。")
            for label_path in self.labelimg_launcher.completed_label_paths():
                record_label_verification(label_path, self.annotation_output_dir)
        except (OSError, PendingAnnotationError) as exc:
            self.operator_instruction_label.setText("無法確認標註儲存結果。")
            QMessageBox.critical(self, "標註確認失敗", str(exc))
            return
        self.operator_instruction_label.setText("正在檢查標註結果…")
        self._complete_operator_pending(automatic=True)

    def _create_separator(self) -> QFrame:
        """Create a horizontal line separator."""
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setFrameShadow(QFrame.Sunken)
        return line

    # ================================================================
    # Class Management Methods
    # ================================================================

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
            self.message_logged.emit(f"[INFO] Added annotation class: {class_name}")

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
            self.message_logged.emit(f"[INFO] Renamed class: {old_name} → {new_name}")

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
            self.message_logged.emit(f"[INFO] Deleted annotation class: {class_name}")

    def _import_classes_from_config(self) -> None:
        """Import classes from yolo_training.class_names."""
        config = self.manager.config if hasattr(self.manager, "config") else {}
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
            self.message_logged.emit(f"[INFO] Imported {len(added)} classes from config")
        else:
            QMessageBox.information(self, "完成", "所有類別已存在，無需導入。")

    def _import_classes_from_file_dialog(self) -> None:
        """Import LabelImg class names from a user-selected text file."""
        start_dir = self._default_class_file_dialog_dir()
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "選擇標籤檔",
            str(start_dir),
            "Label files (*.txt *.names);;All files (*.*)",
        )
        if not file_path:
            return

        try:
            imported_classes = self._read_classes_file(Path(file_path))
        except (OSError, ValueError) as e:
            QMessageBox.warning(self, "匯入失敗", str(e))
            self.message_logged.emit(f"[ERROR] Failed to import classes from file: {e}")
            return

        added = self._merge_annotation_classes(imported_classes)
        self._refresh_class_list()
        self._save_annotation_setting("last_classes_file", file_path)

        if added:
            QMessageBox.information(
                self,
                "匯入完成",
                f"已匯入 {len(added)} 個新標籤：\n" + ", ".join(added),
            )
            self.message_logged.emit(
                f"[INFO] Imported {len(added)} classes from {file_path}"
            )
        else:
            QMessageBox.information(self, "完成", "檔案中的標籤都已存在，無需匯入。")

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
            self.message_logged.emit(
                f"[INFO] Saved {len(self.annotation_classes)} classes to {classes_file}"
            )
        except (OSError, ValueError) as e:
            QMessageBox.critical(self, "錯誤", f"儲存失敗：\n{e}")
            self.message_logged.emit(f"[ERROR] Failed to save classes: {e}")

    def _refresh_class_list(self) -> None:
        """Refresh the class list widget."""
        self.annotation_class_list.clear()
        for class_name in self.annotation_classes:
            self.annotation_class_list.addItem(class_name)

    def _read_classes_file(self, class_file: Path) -> List[str]:
        """Read class names from a LabelImg-compatible text file.

        Args:
            class_file: Text file containing one class name per line.

        Returns:
            A de-duplicated class-name list preserving file order.

        Raises:
            ValueError: If the file is missing or contains no valid class names.
            OSError: If the file cannot be read.
        """
        if not class_file.exists():
            raise ValueError(f"標籤檔不存在：{class_file}")
        if not class_file.is_file():
            raise ValueError(f"標籤路徑不是檔案：{class_file}")

        classes: List[str] = []
        seen = set()
        for raw_line in class_file.read_text(encoding="utf-8-sig").splitlines():
            class_name = raw_line.strip()
            if not class_name or class_name.startswith("#"):
                continue
            if class_name in seen:
                continue
            seen.add(class_name)
            classes.append(class_name)

        if not classes:
            raise ValueError(f"標籤檔沒有有效標籤：{class_file}")
        return classes

    def _merge_annotation_classes(self, class_names: List[str]) -> List[str]:
        """Merge class names into the panel state without duplicating labels.

        Args:
            class_names: Candidate class names to append.

        Returns:
            Class names that were newly added.
        """
        added: List[str] = []
        for class_name in class_names:
            if class_name not in self.annotation_classes:
                self.annotation_classes.append(class_name)
                added.append(class_name)
        return added

    # ================================================================
    # Directory Browsing
    # ================================================================

    def _browse_annotation_input(self) -> None:
        """Browse for annotation input directory."""
        dir_path = QFileDialog.getExistingDirectory(
            self,
            "選擇輸入圖片資料夾",
            str(self.annotation_input_dir or self._saved_path("input_dir") or Path.cwd()),
        )
        if dir_path:
            self.annotation_input_dir = Path(dir_path)
            self.annotation_input_edit.setText(dir_path)
            self._save_annotation_setting("input_dir", dir_path)
            self._scan_annotation_progress()

    def _browse_annotation_output(self) -> None:
        """Browse for annotation output directory."""
        dir_path = QFileDialog.getExistingDirectory(
            self,
            "選擇標註輸出資料夾",
            str(self.annotation_output_dir or self._saved_path("output_dir") or Path.cwd()),
        )
        if dir_path:
            self.annotation_output_dir = Path(dir_path)
            self.annotation_output_edit.setText(dir_path)
            self._save_annotation_setting("output_dir", dir_path)
            self._scan_annotation_progress()

    def _load_annotation_settings(self) -> None:
        """Load remembered annotation paths into the panel state."""
        input_dir = self._saved_path("input_dir")
        if input_dir is not None:
            self.annotation_input_dir = input_dir
            self.annotation_input_edit.setText(str(input_dir))

        output_dir = self._saved_path("output_dir")
        if output_dir is not None:
            self.annotation_output_dir = output_dir
            self.annotation_output_edit.setText(str(output_dir))

    def _saved_path(self, key: str) -> Optional[Path]:
        """Return a remembered annotation path when it still exists.

        Args:
            key: Setting key suffix under ``annotation/``.

        Returns:
            Existing path, or None when unset/missing.
        """
        value = QtCore.QSettings().value(f"annotation/{key}", "")
        if not value:
            return None
        path = Path(str(value))
        return path if path.exists() else None

    def _save_annotation_setting(self, key: str, value: str) -> None:
        """Persist one annotation setting value."""
        QtCore.QSettings().setValue(f"annotation/{key}", value)

    def _default_class_file_dialog_dir(self) -> Path:
        """Return a useful start directory for class-file import."""
        last_file = self._saved_path("last_classes_file")
        if last_file is not None:
            return last_file.parent
        if self.annotation_output_dir is not None:
            return self.annotation_output_dir.parent
        return Path.cwd()

    # ================================================================
    # Progress Tracking
    # ================================================================

    # ================================================================
    # Async Methods
    # ================================================================

    def _scan_annotation_progress(self) -> None:
        """Scan and update annotation progress asynchronously."""
        if not self.annotation_input_dir or not self.annotation_output_dir:
            self.annotation_stats_label.setText("請設定輸入和輸出目錄")
            return

        # Disable buttons to prevent re-entrancy
        self._set_ui_locked(True)
        self.annotation_stats_label.setText("掃描中...")
        self.annotation_progress_bar.setValue(0)

        # Create and start worker
        self.worker = AnnotationWorker(
            self.annotation_tracker,
            self.annotation_input_dir,
            self.annotation_output_dir,
        )
        self.worker.progress_updated.connect(self._on_scan_progress)
        self.worker.scan_completed.connect(self._on_scan_completed)
        self.worker.error_occurred.connect(self._on_scan_error)
        self.worker.start()

    def _on_scan_progress(self, current: int, total: int) -> None:
        """Handle progress updates from worker."""
        if total > 0:
            percent = int((current / total) * 100)
            self.annotation_progress_bar.setValue(percent)
            self.annotation_stats_label.setText(f"掃描中... ({current}/{total})")

    def _on_scan_completed(self, stats: dict) -> None:
        """Handle successful scan completion."""
        self._set_ui_locked(False)
        
        # Update statistics label
        self.annotation_stats_label.setText(
            f"📊 總圖片：{stats['total_images']}  |  "
            f"✅ 已標註：{stats['annotated_images']} ({stats['progress_percent']:.1f}%)  |  "
            f"⏳ 未標註：{len(stats['unannotated_images'])}"
        )

        # Update progress bar
        self.annotation_progress_bar.setValue(int(stats["progress_percent"]))

        # Update unannotated list
        self.annotation_unannotated_list.clear()
        for img_name in stats["unannotated_images"][:20]:  # Show max 20
            self.annotation_unannotated_list.addItem(img_name)
        if len(stats["unannotated_images"]) > 20:
            self.annotation_unannotated_list.addItem(
                f"... 還有 {len(stats['unannotated_images']) - 20} 張"
            )

        # Update class distribution
        if self.annotation_classes and stats["annotated_images"] > 0 and self.annotation_output_dir is not None:
            class_dist = self.annotation_tracker.get_class_distribution(
                self.annotation_output_dir,
                self.annotation_classes,
            )
            dist_text = "\n".join(
                [f"{name}: {count}" for name, count in class_dist.items()]
            )
            self.annotation_class_dist.setText(dist_text)
        else:
            self.annotation_class_dist.setText("尚無標註資料")

        self.message_logged.emit(
            f"[INFO] Scanned annotations: {stats['annotated_images']}/{stats['total_images']}"
        )

    def _on_scan_error(self, error_msg: str) -> None:
        """Handle scan errors."""
        self._set_ui_locked(False)
        self.annotation_stats_label.setText("掃描失敗")
        QMessageBox.warning(self, "掃描錯誤", f"掃描過程中發生錯誤：\n{error_msg}")
        self.message_logged.emit(f"[ERROR] Scan failed: {error_msg}")

    def _set_ui_locked(self, locked: bool) -> None:
        """Lock/unlock UI during async operations."""
        # Find buttons directly to lock them (simplified approach)
        # Ideally, we should have references to these buttons as class attributes
        pass  # TODO: Implement granular button locking if needed

    # ================================================================
    # LabelImg Integration
    # ================================================================

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
        if self.operator_mode_enabled:
            self.complete_pending_btn.setVisible(False)
        classes_file = self.annotation_output_dir.parent / "predefined_classes.txt"
        success = self.labelimg_launcher.launch(
            self.annotation_input_dir,
            self.annotation_output_dir,
            classes_file,
        )

        if success:
            if self.operator_mode_enabled:
                self.launch_annotation_btn.setEnabled(False)
                self.operator_instruction_label.setText(
                    "標註工具已開啟：拖曳框選漏檢物件 → 選類別 → Ctrl+S。"
                    "全部完成後關閉工具，系統會自動檢查。"
                )
                self.labelimg_poll_timer.start()
            else:
                QMessageBox.information(
                    self,
                    "已啟動",
                    "LabelImg 已啟動！\n\n完成標註後關閉 LabelImg，然後點擊「重新掃描」查看進度。",
                )
            self.message_logged.emit("[INFO] Launched LabelImg")
        else:
            error_detail = self.labelimg_launcher.last_error or "請查看應用程式 log。"
            QMessageBox.critical(self, "錯誤", f"啟動 LabelImg 失敗！\n\n{error_detail}")
            self.message_logged.emit(f"[ERROR] Failed to launch LabelImg: {error_detail}")

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
            self.message_logged.emit("[INFO] All annotations validated successfully")
        else:
            error_text = "\n".join(errors[:10])  # Show max 10 errors
            if len(errors) > 10:
                error_text += f"\n\n... 還有 {len(errors) - 10} 個錯誤"

            QMessageBox.warning(
                self,
                f"發現 {len(errors)} 個錯誤",
                error_text,
            )
            self.message_logged.emit(f"[WARNING] Found {len(errors)} validation errors")

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
            self.message_logged.emit(
                "[INFO] Ready to start augmentation from annotation output"
            )


class AnnotationWorker(QtCore.QThread):
    """Background worker for annotation scanning."""
    progress_updated = pyqtSignal(int, int)  # current, total
    scan_completed = pyqtSignal(dict)
    error_occurred = pyqtSignal(str)

    def __init__(self, tracker, input_dir, output_dir):
        super().__init__()
        self.tracker = tracker
        self.input_dir = input_dir
        self.output_dir = output_dir

    def run(self):
        try:
            # Use a wrapper to bridge functional callback to signal emission
            def _progress_bridge(current, total):
                self.progress_updated.emit(current, total)

            stats = self.tracker.scan_directory(
                self.input_dir,
                self.output_dir,
                progress_callback=_progress_bridge
            )
            self.scan_completed.emit(stats)
        except Exception as e:
            self.error_occurred.emit(str(e))





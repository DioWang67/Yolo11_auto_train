# Annotation management methods (to be appended to PictureToolGUI class)

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
def _add_annotation_class(self):
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

def _edit_annotation_class(self):
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

def _delete_annotation_class(self):
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

def _import_classes_from_config(self):
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

def _save_annotation_classes(self):
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

def _refresh_class_list(self):
    """Refresh the class list widget."""
    self.annotation_class_list.clear()
    for class_name in self.annotation_classes:
        self.annotation_class_list.addItem(class_name)

# Directory browsing
def _browse_annotation_input(self):
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

def _browse_annotation_output(self):
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
def _scan_annotation_progress(self):
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
def _launch_labelimg(self):
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

def _validate_annotations(self):
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

def _start_augmentation_from_annotation(self):
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

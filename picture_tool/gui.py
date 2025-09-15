import sys
import os
import yaml
import threading
import time
from pathlib import Path
from PyQt5.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, 
                             QWidget, QPushButton, QLabel, QTextEdit, QProgressBar,
                             QGroupBox, QCheckBox, QComboBox, QLineEdit, QFileDialog,
                             QSplitter, QTabWidget, QScrollArea, QFrame, QGridLayout,
                             QMessageBox, QListWidget, QListWidgetItem, QSpacerItem,
                             QSizePolicy)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer, QPropertyAnimation, QEasingCurve
from PyQt5.QtGui import QFont, QIcon, QPalette, QColor, QPixmap, QLinearGradient

# 假設這些是您原始代碼中的模組
try:
    from picture_tool.format import convert_format
    from picture_tool.anomaly import process_anomaly_detection
    from picture_tool.augment import YoloDataAugmentor, ImageAugmentor
    from picture_tool.split import split_dataset
    from picture_tool.train.yolo_trainer import train_yolo
    from picture_tool.eval.yolo_evaluator import evaluate_yolo
    from picture_tool.report.report_generator import generate_report
    from picture_tool.quality.dataset_linter import lint_dataset, preview_dataset
except ImportError:
    # 如果無法導入，創建模擬函數
    def convert_format(config): print("執行格式轉換")
    def process_anomaly_detection(config): print("執行異常檢測")
    def split_dataset(config): print("執行數據集分割")
    def train_yolo(config): print("執行YOLO訓練")
    def evaluate_yolo(config): print("執行YOLO評估")
    def generate_report(config): print("生成報告")
    def lint_dataset(config): print("執行數據集檢查")
    def preview_dataset(config): print("預覽數據集")
    
    class YoloDataAugmentor:
        def __init__(self): pass
        def process_dataset(self): print("YOLO數據增強")
    
    class ImageAugmentor:
        def __init__(self): pass
        def process_dataset(self): print("圖像數據增強")

class WorkerThread(QThread):
    """工作線程用於執行任務"""
    progress_updated = pyqtSignal(int)
    task_started = pyqtSignal(str)
    task_completed = pyqtSignal(str)
    log_message = pyqtSignal(str)
    finished_signal = pyqtSignal()
    error_occurred = pyqtSignal(str)

    def __init__(self, tasks, config):
        super().__init__()
        self.tasks = tasks
        self.config = config
        self.is_cancelled = False

        # 任務處理器字典
        self.task_handlers = {
            "格式轉換": self.run_format_conversion,
            "異常檢測": self.run_anomaly_detection,
            "YOLO數據增強": self.run_yolo_augmentation,
            "圖像數據增強": self.run_image_augmentation,
            "數據集分割": self.run_dataset_splitter,
            "YOLO訓練": self.run_yolo_train,
            "YOLO評估": self.run_yolo_evaluation,
            "生成報告": self.run_generate_report,
            "數據集檢查": self.run_dataset_lint,
            "增強預覽": self.run_aug_preview,
        }

    def cancel(self):
        self.is_cancelled = True

    def run(self):
        try:
            total_tasks = len(self.tasks)
            for i, task in enumerate(self.tasks):
                if self.is_cancelled:
                    break
                
                self.task_started.emit(task)
                self.log_message.emit(f"開始執行任務: {task}")
                
                # 執行任務
                handler = self.task_handlers.get(task)
                if handler:
                    handler()
                    self.log_message.emit(f"任務 {task} 完成")
                    self.task_completed.emit(task)
                else:
                    self.log_message.emit(f"未知任務: {task}")
                
                # 更新進度
                progress = int((i + 1) / total_tasks * 100)
                self.progress_updated.emit(progress)
                
                # 模擬任務執行時間
                time.sleep(1)
            
            if not self.is_cancelled:
                self.log_message.emit("所有任務執行完成！")
            else:
                self.log_message.emit("任務執行已取消")
            
            self.finished_signal.emit()
        except Exception as e:
            self.error_occurred.emit(str(e))

    # 任務處理方法
    def run_format_conversion(self):
        if 'format_conversion' in self.config:
            convert_format(self.config['format_conversion'])

    def run_anomaly_detection(self):
        if 'anomaly_detection' in self.config:
            process_anomaly_detection(self.config)

    def run_yolo_augmentation(self):
        augmentor = YoloDataAugmentor()
        if hasattr(augmentor, 'config'):
            augmentor.config = self.config.get('yolo_augmentation', {})
        augmentor.process_dataset()

    def run_image_augmentation(self):
        augmentor = ImageAugmentor()
        if hasattr(augmentor, 'config'):
            augmentor.config = self.config.get('image_augmentation', {})
        augmentor.process_dataset()

    def run_dataset_splitter(self):
        split_dataset(self.config)

    def run_yolo_train(self):
        train_yolo(self.config)

    def run_yolo_evaluation(self):
        evaluate_yolo(self.config)

    def run_generate_report(self):
        generate_report(self.config)

    def run_dataset_lint(self):
        lint_dataset(self.config)

    def run_aug_preview(self):
        preview_dataset(self.config)

class ModernCheckBox(QCheckBox):
    """現代化的複選框樣式"""
    def __init__(self, text, parent=None):
        super().__init__(text, parent)
        self.setStyleSheet("""
            QCheckBox {
                font-size: 14px;
                color: #2c3e50;
                spacing: 8px;
                padding: 5px;
            }
            QCheckBox::indicator {
                width: 18px;
                height: 18px;
                border-radius: 9px;
                border: 2px solid #3498db;
                background-color: white;
            }
            QCheckBox::indicator:checked {
                background-color: #3498db;
                border: 2px solid #2980b9;
            }
            QCheckBox::indicator:checked:hover {
                background-color: #2980b9;
            }
            QCheckBox::indicator:hover {
                border: 2px solid #2980b9;
            }
        """)

class ModernButton(QPushButton):
    """現代化的按鈕樣式"""
    def __init__(self, text, color_theme="primary", parent=None):
        super().__init__(text, parent)
        self.color_theme = color_theme
        self._setup_style()

    def _setup_style(self):
        if self.color_theme == "primary":
            style = """
                QPushButton {
                    background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                                              stop: 0 #3498db, stop: 1 #2980b9);
                    color: white;
                    border: none;
                    padding: 12px 24px;
                    border-radius: 25px;
                    font-size: 14px;
                    font-weight: bold;
                    min-width: 120px;
                }
                QPushButton:hover {
                    background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                                              stop: 0 #2980b9, stop: 1 #2471a3);
                }
                QPushButton:pressed {
                    background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                                              stop: 0 #2471a3, stop: 1 #1f5582);
                }
                QPushButton:disabled {
                    background: #bdc3c7;
                    color: #7f8c8d;
                }
            """
        elif self.color_theme == "success":
            style = """
                QPushButton {
                    background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                                              stop: 0 #27ae60, stop: 1 #229954);
                    color: white;
                    border: none;
                    padding: 12px 24px;
                    border-radius: 25px;
                    font-size: 14px;
                    font-weight: bold;
                    min-width: 120px;
                }
                QPushButton:hover {
                    background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                                              stop: 0 #229954, stop: 1 #1e8449);
                }
                QPushButton:pressed {
                    background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                                              stop: 0 #1e8449, stop: 1 #196f3d);
                }
            """
        elif self.color_theme == "danger":
            style = """
                QPushButton {
                    background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                                              stop: 0 #e74c3c, stop: 1 #c0392b);
                    color: white;
                    border: none;
                    padding: 12px 24px;
                    border-radius: 25px;
                    font-size: 14px;
                    font-weight: bold;
                    min-width: 120px;
                }
                QPushButton:hover {
                    background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                                              stop: 0 #c0392b, stop: 1 #a93226);
                }
                QPushButton:pressed {
                    background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                                              stop: 0 #a93226, stop: 1 #922b21);
                }
            """
        self.setStyleSheet(style)

class PictureToolGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.config = {}
        self.worker_thread = None
        self.init_ui()
        self.load_default_config()

    def init_ui(self):
        self.setWindowTitle('圖像處理工具 - 現代化GUI')
        self.setGeometry(100, 100, 1200, 800)
        self.setMinimumSize(1000, 700)
        
        # 設置現代化樣式
        self.setStyleSheet("""
            QMainWindow {
                background-color: #f8f9fa;
            }
            QGroupBox {
                font-size: 16px;
                font-weight: bold;
                color: #2c3e50;
                border: 2px solid #e9ecef;
                border-radius: 10px;
                margin-top: 1ex;
                padding-top: 15px;
                background-color: white;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 8px 0 8px;
                background-color: white;
            }
            QTextEdit {
                border: 1px solid #dee2e6;
                border-radius: 8px;
                background-color: #ffffff;
                font-family: 'Consolas', 'Monaco', monospace;
                font-size: 12px;
                padding: 8px;
            }
            QProgressBar {
                border: 2px solid #dee2e6;
                border-radius: 15px;
                background-color: #f8f9fa;
                text-align: center;
                font-weight: bold;
                color: #2c3e50;
            }
            QProgressBar::chunk {
                background: qlineargradient(x1: 0, y1: 0, x2: 1, y2: 0,
                                          stop: 0 #3498db, stop: 1 #2980b9);
                border-radius: 13px;
            }
            QComboBox, QLineEdit {
                padding: 8px;
                border: 2px solid #dee2e6;
                border-radius: 8px;
                background-color: white;
                font-size: 14px;
            }
            QComboBox:focus, QLineEdit:focus {
                border-color: #3498db;
            }
        """)

        # 中央部件
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # 主佈局
        main_layout = QHBoxLayout(central_widget)
        main_layout.setSpacing(20)
        main_layout.setContentsMargins(20, 20, 20, 20)

        # 左側控制面板
        left_panel = self.create_left_panel()
        
        # 右側內容面板
        right_panel = self.create_right_panel()

        # 添加到主佈局
        main_layout.addWidget(left_panel, 1)
        main_layout.addWidget(right_panel, 2)

    def create_left_panel(self):
        """創建左側控制面板"""
        left_widget = QWidget()
        left_widget.setMaximumWidth(400)
        left_layout = QVBoxLayout(left_widget)
        left_layout.setSpacing(20)

        # 配置文件選擇
        config_group = QGroupBox("配置設定")
        config_layout = QVBoxLayout(config_group)
        
        config_file_layout = QHBoxLayout()
        self.config_path_edit = QLineEdit("config.yaml")
        config_browse_btn = ModernButton("瀏覽", "primary")
        config_browse_btn.clicked.connect(self.browse_config_file)
        config_file_layout.addWidget(QLabel("配置文件："))
        config_file_layout.addWidget(self.config_path_edit)
        config_file_layout.addWidget(config_browse_btn)
        
        config_layout.addLayout(config_file_layout)
        
        reload_config_btn = ModernButton("重新載入配置", "primary")
        reload_config_btn.clicked.connect(self.load_config)
        config_layout.addWidget(reload_config_btn)

        # 任務選擇
        task_group = QGroupBox("任務選擇")
        task_layout = QVBoxLayout(task_group)
        
        # 任務複選框
        self.task_checkboxes = {}
        tasks = [
            "格式轉換", "異常檢測", "YOLO數據增強", 
            "圖像數據增強", "數據集分割", "YOLO訓練",
            "YOLO評估", "生成報告", "數據集檢查", "增強預覽"
        ]
        
        for task in tasks:
            checkbox = ModernCheckBox(task)
            self.task_checkboxes[task] = checkbox
            task_layout.addWidget(checkbox)

        # 全選/取消全選
        select_buttons_layout = QHBoxLayout()
        select_all_btn = ModernButton("全選", "success")
        deselect_all_btn = ModernButton("取消全選", "danger")
        select_all_btn.clicked.connect(self.select_all_tasks)
        deselect_all_btn.clicked.connect(self.deselect_all_tasks)
        select_buttons_layout.addWidget(select_all_btn)
        select_buttons_layout.addWidget(deselect_all_btn)
        task_layout.addLayout(select_buttons_layout)

        # 執行控制
        control_group = QGroupBox("執行控制")
        control_layout = QVBoxLayout(control_group)
        
        self.start_btn = ModernButton("🚀 開始執行", "success")
        self.stop_btn = ModernButton("⏹️ 停止執行", "danger")
        self.stop_btn.setEnabled(False)
        
        self.start_btn.clicked.connect(self.start_pipeline)
        self.stop_btn.clicked.connect(self.stop_pipeline)
        
        control_layout.addWidget(self.start_btn)
        control_layout.addWidget(self.stop_btn)

        # 進度顯示
        self.progress_bar = QProgressBar()
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: 2px solid #dee2e6;
                border-radius: 15px;
                background-color: #f8f9fa;
                text-align: center;
                font-weight: bold;
                color: #2c3e50;
                height: 30px;
            }
            QProgressBar::chunk {
                background: qlineargradient(x1: 0, y1: 0, x2: 1, y2: 0,
                                          stop: 0 #27ae60, stop: 1 #229954);
                border-radius: 13px;
            }
        """)
        control_layout.addWidget(self.progress_bar)

        # 狀態標籤
        self.status_label = QLabel("就緒")
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setStyleSheet("""
            QLabel {
                background-color: #e8f5e8;
                color: #27ae60;
                padding: 10px;
                border-radius: 8px;
                font-weight: bold;
                font-size: 14px;
            }
        """)
        control_layout.addWidget(self.status_label)

        # 添加到左側佈局
        left_layout.addWidget(config_group)
        left_layout.addWidget(task_group)
        left_layout.addWidget(control_group)
        left_layout.addStretch()

        return left_widget

    def create_right_panel(self):
        """創建右側內容面板"""
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)

        # 標題
        title_label = QLabel("圖像處理工具管道")
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet("""
            QLabel {
                font-size: 28px;
                font-weight: bold;
                color: #2c3e50;
                padding: 20px;
                background: qlineargradient(x1: 0, y1: 0, x2: 1, y2: 0,
                                          stop: 0 #3498db, stop: 1 #2980b9);
                color: white;
                border-radius: 15px;
                margin-bottom: 20px;
            }
        """)

        # Tab Widget
        self.tab_widget = QTabWidget()
        self.tab_widget.setStyleSheet("""
            QTabWidget::pane {
                border: 2px solid #dee2e6;
                border-radius: 8px;
                background-color: white;
                padding: 10px;
            }
            QTabBar::tab {
                background-color: #f8f9fa;
                color: #495057;
                padding: 12px 20px;
                margin-right: 2px;
                border-top-left-radius: 8px;
                border-top-right-radius: 8px;
                font-weight: bold;
            }
            QTabBar::tab:selected {
                background-color: #3498db;
                color: white;
            }
            QTabBar::tab:hover:!selected {
                background-color: #e9ecef;
            }
        """)

        # 日誌標籤頁
        log_widget = QWidget()
        log_layout = QVBoxLayout(log_widget)
        
        self.log_text = QTextEdit()
        self.log_text.setPlainText("歡迎使用圖像處理工具！\n請選擇任務並點擊開始執行。\n")
        self.log_text.setStyleSheet("""
            QTextEdit {
                background-color: #2c3e50;
                color: #ecf0f1;
                border: none;
                font-family: 'Consolas', 'Monaco', monospace;
                font-size: 13px;
                line-height: 1.4;
            }
        """)
        
        clear_log_btn = ModernButton("清空日誌", "danger")
        clear_log_btn.clicked.connect(self.clear_log)
        
        log_layout.addWidget(self.log_text)
        log_layout.addWidget(clear_log_btn)
        
        # 配置預覽標籤頁
        config_widget = QWidget()
        config_layout = QVBoxLayout(config_widget)
        
        self.config_text = QTextEdit()
        self.config_text.setPlainText("請載入配置文件...")
        
        config_layout.addWidget(self.config_text)
        
        # 添加標籤頁
        self.tab_widget.addTab(log_widget, "📝 執行日誌")
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
                ]
            },
            "format_conversion": {
                "input_formats": [".bmp", ".tiff"],
                "output_format": ".png",
                "input_dir": "input/",
                "output_dir": "output/"
            }
        }
        self.config = default_config
        self.update_config_display()

    def load_config(self):
        """載入配置文件"""
        config_path = self.config_path_edit.text()
        try:
            if os.path.exists(config_path):
                with open(config_path, 'r', encoding='utf-8') as f:
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
        config_text = yaml.dump(self.config, default_flow_style=False, 
                               allow_unicode=True, indent=2)
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
        self.status_label.setStyleSheet("""
            QLabel {
                background-color: #fff3cd;
                color: #856404;
                padding: 10px;
                border-radius: 8px;
                font-weight: bold;
                font-size: 14px;
            }
        """)

        # 創建並啟動工作線程
        self.worker_thread = WorkerThread(selected_tasks, self.config)
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
        self.status_label.setStyleSheet("""
            QLabel {
                background-color: #e8f5e8;
                color: #27ae60;
                padding: 10px;
                border-radius: 8px;
                font-weight: bold;
                font-size: 14px;
            }
        """)
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

    def clear_log(self):
        """清空日誌"""
        self.log_text.clear()
        self.log_message("日誌已清空")

    def closeEvent(self, event):
        """窗口關閉事件"""
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
    """主函數"""
    app = QApplication(sys.argv)
    
    # 設置應用程式圖標和名稱
    app.setApplicationName("圖像處理工具")
    app.setApplicationVersion("1.0")
    app.setOrganizationName("ImageTool Corp")
    
    # 設置全局字體
    font = QFont("Microsoft YaHei UI", 10)
    app.setFont(font)
    
    # 創建主窗口
    window = PictureToolGUI()
    window.show()
    
    # 運行應用程式
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
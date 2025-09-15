import sys
import logging
import argparse
import yaml
import os
from pathlib import Path
from typing import Dict, Any
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFileDialog, QCheckBox, QTextEdit,
    QMessageBox, QLineEdit, QGroupBox, QDialog, QFormLayout,
    QDialogButtonBox, QTabWidget, QSplitter, QProgressBar,
    QScrollArea, QFrame, QSpinBox, QDoubleSpinBox, QComboBox,
    QSlider, QTreeWidget, QTreeWidgetItem, QToolBar, QAction,
    QStatusBar, QMenuBar, QMenu
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer, QObject
from PyQt5.QtGui import QIcon, QFont, QPalette, QColor, QPixmap

from picture_tool.pipeline import load_config, setup_logging, run_pipeline


class QTextEditLogger(QObject, logging.Handler):
    """將日誌輸出導向 QTextEdit，支援不同級別的顏色顯示。

    若需透過訊號傳遞 ``QTextCursor`` 等型別，請先呼叫 ``qRegisterMetaType`` 進行註冊。
    """

    log_signal = pyqtSignal(str)

    def __init__(self, widget: QTextEdit):
        QObject.__init__(self)
        logging.Handler.__init__(self)
        self.widget = widget
        self.log_signal.connect(self.widget.append)

    def emit(self, record: logging.LogRecord) -> None:
        msg = self.format(record)
        # 根據日誌級別設定顏色
        if record.levelno >= logging.ERROR:
            color_msg = f'<span style="color: red;">{msg}</span>'
        elif record.levelno >= logging.WARNING:
            color_msg = f'<span style="color: orange;">{msg}</span>'
        elif record.levelno >= logging.INFO:
            color_msg = f'<span style="color: white;">{msg}</span>'
        else:
            color_msg = f'<span style="color: gray;">{msg}</span>'

        self.log_signal.emit(color_msg)


class PipelineWorker(QThread):
    """在後台執行流程的工作線程"""
    progress = pyqtSignal(int)
    status = pyqtSignal(str)
    finished = pyqtSignal(bool, str)
    
    def __init__(self, selected_tasks, config, logger, args):
        super().__init__()
        self.selected_tasks = selected_tasks
        self.config = config
        self.logger = logger
        self.args = args
        
    def run(self):
        try:
            self.status.emit("開始執行流程...")
            run_pipeline(self.selected_tasks, self.config, self.logger, self.args)
            self.finished.emit(True, "流程執行完成")
        except Exception as e:
            self.finished.emit(False, str(e))


class AdvancedConfigDialog(QDialog):
    """進階配置編輯器，支援所有配置選項"""
    
    def __init__(self, config_path: str, parent=None):
        super().__init__(parent)
        self.config_path = config_path
        self.config = load_config(config_path)
        self.init_ui()
        
    def init_ui(self):
        self.setWindowTitle("進階配置編輯器")
        self.resize(800, 700)
        
        layout = QVBoxLayout(self)
        
        # 使用標籤頁組織不同的配置區域
        tab_widget = QTabWidget()
        
        # 基本設定標籤
        basic_tab = self.create_basic_tab()
        tab_widget.addTab(basic_tab, "基本設定")
        
        # 格式轉換標籤
        format_tab = self.create_format_conversion_tab()
        tab_widget.addTab(format_tab, "格式轉換")
        
        # 圖像增強標籤
        augmentation_tab = self.create_augmentation_tab()
        tab_widget.addTab(augmentation_tab, "圖像增強")
        
        # 異常檢測標籤
        anomaly_tab = self.create_anomaly_detection_tab()
        tab_widget.addTab(anomaly_tab, "異常檢測")
        
        # YOLO 增強標籤
        yolo_tab = self.create_yolo_augmentation_tab()
        tab_widget.addTab(yolo_tab, "YOLO 增強")
        
        # 資料集分割標籤
        split_tab = self.create_dataset_split_tab()
        tab_widget.addTab(split_tab, "資料集分割")
        
        layout.addWidget(tab_widget)
        
        # 按鈕
        button_box = QDialogButtonBox(
            QDialogButtonBox.Save | QDialogButtonBox.Cancel | QDialogButtonBox.RestoreDefaults
        )
        button_box.accepted.connect(self.save_config)
        button_box.rejected.connect(self.reject)
        button_box.button(QDialogButtonBox.RestoreDefaults).clicked.connect(self.restore_defaults)
        layout.addWidget(button_box)
        
    def create_basic_tab(self):
        widget = QWidget()
        layout = QFormLayout(widget)
        
        # 日誌檔案設定
        self.log_file_edit = QLineEdit(self.config["pipeline"].get("log_file", "pipeline.log"))
        layout.addRow("日誌檔案:", self.log_file_edit)
        
        # 任務啟用狀態
        tasks_group = QGroupBox("任務啟用狀態")
        tasks_layout = QVBoxLayout()
        self.task_checkboxes = {}
        
        for task in self.config["pipeline"]["tasks"]:
            cb = QCheckBox(task["name"])
            cb.setChecked(task.get("enabled", True))
            self.task_checkboxes[task["name"]] = cb
            tasks_layout.addWidget(cb)
            
        tasks_group.setLayout(tasks_layout)
        layout.addRow(tasks_group)
        
        return widget
    
    def create_format_conversion_tab(self):
        widget = QWidget()
        layout = QFormLayout(widget)
        
        fc = self.config.get("format_conversion", {})
        
        # 建立瀏覽按鈕的輔助函數
        def create_dir_selector(initial_path=""):
            container = QWidget()
            h_layout = QHBoxLayout(container)
            h_layout.setContentsMargins(0, 0, 0, 0)
            
            line_edit = QLineEdit(initial_path)
            browse_btn = QPushButton("瀏覽...")
            browse_btn.clicked.connect(lambda: self.browse_directory(line_edit))
            
            h_layout.addWidget(line_edit)
            h_layout.addWidget(browse_btn)
            return container, line_edit
        
        # 輸入資料夾
        input_container, self.fc_input_dir = create_dir_selector(fc.get("input_dir", ""))
        layout.addRow("輸入資料夾:", input_container)
        
        # 輸出資料夾
        output_container, self.fc_output_dir = create_dir_selector(fc.get("output_dir", ""))
        layout.addRow("輸出資料夾:", output_container)
        
        # 輸入格式
        self.fc_input_formats = QLineEdit(", ".join(fc.get("input_formats", [])))
        layout.addRow("輸入格式:", self.fc_input_formats)
        
        # 輸出格式
        self.fc_output_format = QComboBox()
        self.fc_output_format.addItems([".png", ".jpg", ".jpeg", ".bmp", ".tiff"])
        self.fc_output_format.setCurrentText(fc.get("output_format", ".png"))
        layout.addRow("輸出格式:", self.fc_output_format)
        
        # 品質設定
        self.fc_quality = QSpinBox()
        self.fc_quality.setRange(1, 100)
        self.fc_quality.setValue(fc.get("quality", 95))
        layout.addRow("輸出品質:", self.fc_quality)
        
        return widget
    
    def create_augmentation_tab(self):
        widget = QScrollArea()
        content = QWidget()
        layout = QVBoxLayout(content)
        
        aug_config = self.config.get("image_augmentation", {})
        
        # 基本設定
        basic_group = QGroupBox("基本設定")
        basic_layout = QFormLayout()
        
        # 輸入/輸出目錄
        input_dir = aug_config.get("input", {}).get("image_dir", "")
        output_dir = aug_config.get("output", {}).get("image_dir", "")
        
        input_container, self.aug_input_dir = self.create_dir_selector_widget(input_dir)
        basic_layout.addRow("輸入目錄:", input_container)
        
        output_container, self.aug_output_dir = self.create_dir_selector_widget(output_dir)
        basic_layout.addRow("輸出目錄:", output_container)
        
        # 增強參數
        aug_params = aug_config.get("augmentation", {})
        
        self.aug_num_images = QSpinBox()
        self.aug_num_images.setRange(1, 10000)
        self.aug_num_images.setValue(aug_params.get("num_images", 200))
        basic_layout.addRow("生成圖片數量:", self.aug_num_images)
        
        self.aug_num_workers = QSpinBox()
        self.aug_num_workers.setRange(1, 16)
        self.aug_num_workers.setValue(aug_config.get("processing", {}).get("num_workers", 2))
        basic_layout.addRow("工作線程數:", self.aug_num_workers)
        
        # 操作數量範圍
        num_ops = aug_params.get("num_operations", [2, 4])
        self.num_ops_min = QSpinBox()
        self.num_ops_min.setRange(1, 10)
        self.num_ops_min.setValue(num_ops[0])
        self.num_ops_max = QSpinBox()
        self.num_ops_max.setRange(1, 10)
        self.num_ops_max.setValue(num_ops[1])
        basic_layout.addRow("最小操作數量:", self.num_ops_min)
        basic_layout.addRow("最大操作數量:", self.num_ops_max)
        
        basic_group.setLayout(basic_layout)
        layout.addWidget(basic_group)
        
        # 增強操作設定
        ops_group = QGroupBox("增強操作設定")
        ops_layout = QVBoxLayout()
        
        operations = aug_params.get("operations", {})
        
        # 模糊
        blur_group = QGroupBox("模糊")
        blur_layout = QFormLayout()
        blur_kernel = operations.get("blur", {}).get("kernel", [1, 3])
        self.blur_min = QSpinBox()
        self.blur_min.setRange(1, 10)
        self.blur_min.setValue(blur_kernel[0])
        self.blur_max = QSpinBox()
        self.blur_max.setRange(1, 10)
        self.blur_max.setValue(blur_kernel[1])
        blur_layout.addRow("最小核心大小:", self.blur_min)
        blur_layout.addRow("最大核心大小:", self.blur_max)
        blur_group.setLayout(blur_layout)
        ops_layout.addWidget(blur_group)
        
        # 對比度
        contrast_group = QGroupBox("對比度")
        contrast_layout = QFormLayout()
        contrast_range = operations.get("contrast", {}).get("range", [0.95, 1.05])
        self.contrast_min = QDoubleSpinBox()
        self.contrast_min.setRange(0.1, 2.0)
        self.contrast_min.setSingleStep(0.01)
        self.contrast_min.setValue(contrast_range[0])
        self.contrast_max = QDoubleSpinBox()
        self.contrast_max.setRange(0.1, 2.0)
        self.contrast_max.setSingleStep(0.01)
        self.contrast_max.setValue(contrast_range[1])
        contrast_layout.addRow("最小對比度:", self.contrast_min)
        contrast_layout.addRow("最大對比度:", self.contrast_max)
        contrast_group.setLayout(contrast_layout)
        ops_layout.addWidget(contrast_group)
        
        # 翻轉
        flip_group = QGroupBox("翻轉")
        flip_layout = QFormLayout()
        flip_prob = operations.get("flip", {}).get("probability", 0)
        self.flip_prob = QDoubleSpinBox()
        self.flip_prob.setRange(0.0, 1.0)
        self.flip_prob.setSingleStep(0.01)
        self.flip_prob.setValue(flip_prob)
        flip_layout.addRow("翻轉機率:", self.flip_prob)
        flip_group.setLayout(flip_layout)
        ops_layout.addWidget(flip_group)
        
        # 色調
        hue_group = QGroupBox("色調")
        hue_layout = QFormLayout()
        hue_range = operations.get("hue", {}).get("range", [-5, 5])
        self.hue_min = QSpinBox()
        self.hue_min.setRange(-180, 180)
        self.hue_min.setValue(hue_range[0])
        self.hue_max = QSpinBox()
        self.hue_max.setRange(-180, 180)
        self.hue_max.setValue(hue_range[1])
        hue_layout.addRow("最小色調調整:", self.hue_min)
        hue_layout.addRow("最大色調調整:", self.hue_max)
        hue_group.setLayout(hue_layout)
        ops_layout.addWidget(hue_group)
        
        # 亮度調整
        multiply_group = QGroupBox("亮度調整")
        multiply_layout = QFormLayout()
        multiply_range = operations.get("multiply", {}).get("range", [0.95, 1.05])
        self.multiply_min = QDoubleSpinBox()
        self.multiply_min.setRange(0.1, 2.0)
        self.multiply_min.setSingleStep(0.01)
        self.multiply_min.setValue(multiply_range[0])
        self.multiply_max = QDoubleSpinBox()
        self.multiply_max.setRange(0.1, 2.0)
        self.multiply_max.setSingleStep(0.01)
        self.multiply_max.setValue(multiply_range[1])
        multiply_layout.addRow("最小亮度因子:", self.multiply_min)
        multiply_layout.addRow("最大亮度因子:", self.multiply_max)
        multiply_group.setLayout(multiply_layout)
        ops_layout.addWidget(multiply_group)
        
        # 噪聲
        noise_group = QGroupBox("噪聲")
        noise_layout = QFormLayout()
        noise_scale = operations.get("noise", {}).get("scale", [1, 3])
        self.noise_min = QSpinBox()
        self.noise_min.setRange(0, 10)
        self.noise_min.setValue(noise_scale[0])
        self.noise_max = QSpinBox()
        self.noise_max.setRange(0, 10)
        self.noise_max.setValue(noise_scale[1])
        noise_layout.addRow("最小噪聲尺度:", self.noise_min)
        noise_layout.addRow("最大噪聲尺度:", self.noise_max)
        noise_group.setLayout(noise_layout)
        ops_layout.addWidget(noise_group)
        
        # 透視變換
        perspective_group = QGroupBox("透視變換")
        perspective_layout = QFormLayout()
        perspective_scale = operations.get("perspective", {}).get("scale", [0.01, 0.02])
        self.perspective_min = QDoubleSpinBox()
        self.perspective_min.setRange(0.0, 0.5)
        self.perspective_min.setSingleStep(0.01)
        self.perspective_min.setValue(perspective_scale[0])
        self.perspective_max = QDoubleSpinBox()
        self.perspective_max.setRange(0.0, 0.5)
        self.perspective_max.setSingleStep(0.01)
        self.perspective_max.setValue(perspective_scale[1])
        perspective_layout.addRow("最小透視尺度:", self.perspective_min)
        perspective_layout.addRow("最大透視尺度:", self.perspective_max)
        perspective_group.setLayout(perspective_layout)
        ops_layout.addWidget(perspective_group)
        
        # 旋轉
        rotate_group = QGroupBox("旋轉")
        rotate_layout = QFormLayout()
        rotate_angle = operations.get("rotate", {}).get("angle", [0, 1])
        self.rotate_min = QSpinBox()
        self.rotate_min.setRange(-180, 180)
        self.rotate_min.setValue(rotate_angle[0])
        self.rotate_max = QSpinBox()
        self.rotate_max.setRange(-180, 180)
        self.rotate_max.setValue(rotate_angle[1])
        rotate_layout.addRow("最小旋轉角度:", self.rotate_min)
        rotate_layout.addRow("最大旋轉角度:", self.rotate_max)
        rotate_group.setLayout(rotate_layout)
        ops_layout.addWidget(rotate_group)
        
        # 縮放
        scale_group = QGroupBox("縮放")
        scale_layout = QFormLayout()
        scale_range = operations.get("scale", {}).get("range", [0.95, 1.05])
        self.scale_min = QDoubleSpinBox()
        self.scale_min.setRange(0.1, 2.0)
        self.scale_min.setSingleStep(0.01)
        self.scale_min.setValue(scale_range[0])
        self.scale_max = QDoubleSpinBox()
        self.scale_max.setRange(0.1, 2.0)
        self.scale_max.setSingleStep(0.01)
        self.scale_max.setValue(scale_range[1])
        scale_layout.addRow("最小縮放因子:", self.scale_min)
        scale_layout.addRow("最大縮放因子:", self.scale_max)
        scale_group.setLayout(scale_layout)
        ops_layout.addWidget(scale_group)
        
        ops_group.setLayout(ops_layout)
        layout.addWidget(ops_group)
        
        widget.setWidget(content)
        widget.setWidgetResizable(True)
        return widget
    
    def create_anomaly_detection_tab(self):
        widget = QWidget()
        layout = QFormLayout(widget)
        
        anomaly_config = self.config.get("anomaly_detection", {})
        
        # 基本設定
        threshold_container, self.anomaly_threshold = self.create_slider_widget(
            anomaly_config.get("threshold", 70), 0, 100
        )
        layout.addRow("閾值:", threshold_container)
        
        # 資料夾設定
        ref_container, self.anomaly_ref_folder = self.create_dir_selector_widget(
            anomaly_config.get("reference_folder", "")
        )
        layout.addRow("參考資料夾:", ref_container)
        
        test_container, self.anomaly_test_folder = self.create_dir_selector_widget(
            anomaly_config.get("test_folder", "")
        )
        layout.addRow("測試資料夾:", test_container)
        
        output_container, self.anomaly_output_folder = self.create_dir_selector_widget(
            anomaly_config.get("output_folder", "")
        )
        layout.addRow("輸出資料夾:", output_container)
        
        # 支援格式
        input_formats = ", ".join(anomaly_config.get("input_formats", [".png", ".jpg", ".jpeg"]))
        self.anomaly_input_formats = QLineEdit(input_formats)
        layout.addRow("支援格式:", self.anomaly_input_formats)
        
        return widget
    
    def create_yolo_augmentation_tab(self):
        widget = QScrollArea()
        content = QWidget()
        layout = QVBoxLayout(content)
        
        yolo_config = self.config.get("yolo_augmentation", {})
        
        # 輸入設定
        input_group = QGroupBox("輸入設定")
        input_layout = QFormLayout()
        
        input_config = yolo_config.get("input", {})
        img_container, self.yolo_img_dir = self.create_dir_selector_widget(
            input_config.get("image_dir", "")
        )
        input_layout.addRow("圖片目錄:", img_container)
        
        label_container, self.yolo_label_dir = self.create_dir_selector_widget(
            input_config.get("label_dir", "")
        )
        input_layout.addRow("標籤目錄:", label_container)
        
        input_group.setLayout(input_layout)
        layout.addWidget(input_group)
        
        # 輸出設定
        output_group = QGroupBox("輸出設定")
        output_layout = QFormLayout()
        
        output_config = yolo_config.get("output", {})
        out_img_container, self.yolo_out_img_dir = self.create_dir_selector_widget(
            output_config.get("image_dir", "")
        )
        output_layout.addRow("圖片輸出目錄:", out_img_container)
        
        out_label_container, self.yolo_out_label_dir = self.create_dir_selector_widget(
            output_config.get("label_dir", "")
        )
        output_layout.addRow("標籤輸出目錄:", out_label_container)
        
        output_group.setLayout(output_layout)
        layout.addWidget(output_group)
        
        # 基本增強參數
        params_group = QGroupBox("基本增強參數")
        params_layout = QFormLayout()
        
        aug_params = yolo_config.get("augmentation", {})
        
        self.yolo_num_images = QSpinBox()
        self.yolo_num_images.setRange(1, 1000)
        self.yolo_num_images.setValue(aug_params.get("num_images", 20))
        params_layout.addRow("生成圖片數量:", self.yolo_num_images)
        
        self.yolo_num_workers = QSpinBox()
        self.yolo_num_workers.setRange(1, 16)
        self.yolo_num_workers.setValue(yolo_config.get("processing", {}).get("num_workers", 2))
        params_layout.addRow("工作線程數:", self.yolo_num_workers)
        
        num_ops = aug_params.get("num_operations", [3, 5])
        self.num_ops_min = QSpinBox()
        self.num_ops_min.setRange(1, 10)
        self.num_ops_min.setValue(num_ops[0])
        self.num_ops_max = QSpinBox()
        self.num_ops_max.setRange(1, 10)
        self.num_ops_max.setValue(num_ops[1])
        params_layout.addRow("最小操作數量:", self.num_ops_min)
        params_layout.addRow("最大操作數量:", self.num_ops_max)
        
        params_group.setLayout(params_layout)
        layout.addWidget(params_group)
        
        # 增強操作設定
        ops_group = QGroupBox("增強操作設定")
        ops_layout = QVBoxLayout()
        
        operations = aug_params.get("operations", {})
        
        # 模糊
        blur_group = QGroupBox("模糊")
        blur_layout = QFormLayout()
        blur_kernel = operations.get("blur", {}).get("kernel", [1, 1])
        self.blur_min = QSpinBox()
        self.blur_min.setRange(0, 10)
        self.blur_min.setValue(blur_kernel[0])
        self.blur_max = QSpinBox()
        self.blur_max.setRange(0, 10)
        self.blur_max.setValue(blur_kernel[1])
        blur_layout.addRow("最小核心大小:", self.blur_min)
        blur_layout.addRow("最大核心大小:", self.blur_max)
        blur_group.setLayout(blur_layout)
        ops_layout.addWidget(blur_group)
        
        # 對比度
        contrast_group = QGroupBox("對比度")
        contrast_layout = QFormLayout()
        contrast_range = operations.get("contrast", {}).get("range", [1, 1])
        self.contrast_min = QDoubleSpinBox()
        self.contrast_min.setRange(0.1, 2.0)
        self.contrast_min.setSingleStep(0.01)
        self.contrast_min.setValue(contrast_range[0])
        self.contrast_max = QDoubleSpinBox()
        self.contrast_max.setRange(0.1, 2.0)
        self.contrast_max.setSingleStep(0.01)
        self.contrast_max.setValue(contrast_range[1])
        contrast_layout.addRow("最小對比度:", self.contrast_min)
        contrast_layout.addRow("最大對比度:", self.contrast_max)
        contrast_group.setLayout(contrast_layout)
        ops_layout.addWidget(contrast_group)
        
        # 翻轉
        flip_group = QGroupBox("翻轉")
        flip_layout = QFormLayout()
        flip_prob = operations.get("flip", {}).get("probability", 0)
        self.flip_prob = QDoubleSpinBox()
        self.flip_prob.setRange(0.0, 1.0)
        self.flip_prob.setSingleStep(0.01)
        self.flip_prob.setValue(flip_prob)
        flip_layout.addRow("翻轉機率:", self.flip_prob)
        flip_group.setLayout(flip_layout)
        ops_layout.addWidget(flip_group)
        
        # 色調
        hue_group = QGroupBox("色調")
        hue_layout = QFormLayout()
        hue_range = operations.get("hue", {}).get("range", [0, 0])
        self.hue_min = QSpinBox()
        self.hue_min.setRange(-180, 180)
        self.hue_min.setValue(hue_range[0])
        self.hue_max = QSpinBox()
        self.hue_max.setRange(-180, 180)
        self.hue_max.setValue(hue_range[1])
        hue_layout.addRow("最小色調調整:", self.hue_min)
        hue_layout.addRow("最大色調調整:", self.hue_max)
        hue_group.setLayout(hue_layout)
        ops_layout.addWidget(hue_group)
        
        # 亮度調整
        multiply_group = QGroupBox("亮度調整")
        multiply_layout = QFormLayout()
        multiply_range = operations.get("multiply", {}).get("range", [0.9, 1.1])
        self.multiply_min = QDoubleSpinBox()
        self.multiply_min.setRange(0.1, 2.0)
        self.multiply_min.setSingleStep(0.01)
        self.multiply_min.setValue(multiply_range[0])
        self.multiply_max = QDoubleSpinBox()
        self.multiply_max.setRange(0.1, 2.0)
        self.multiply_max.setSingleStep(0.01)
        self.multiply_max.setValue(multiply_range[1])
        multiply_layout.addRow("最小亮度因子:", self.multiply_min)
        multiply_layout.addRow("最大亮度因子:", self.multiply_max)
        multiply_group.setLayout(multiply_layout)
        ops_layout.addWidget(multiply_group)
        
        # 噪聲
        noise_group = QGroupBox("噪聲")
        noise_layout = QFormLayout()
        noise_scale = operations.get("noise", {}).get("scale", [0, 0])
        self.noise_min = QSpinBox()
        self.noise_min.setRange(0, 10)
        self.noise_min.setValue(noise_scale[0])
        self.noise_max = QSpinBox()
        self.noise_max.setRange(0, 10)
        self.noise_max.setValue(noise_scale[1])
        noise_layout.addRow("最小噪聲尺度:", self.noise_min)
        noise_layout.addRow("最大噪聲尺度:", self.noise_max)
        noise_group.setLayout(noise_layout)
        ops_layout.addWidget(noise_group)
        
        # 透視變換
        perspective_group = QGroupBox("透視變換")
        perspective_layout = QFormLayout()
        perspective_scale = operations.get("perspective", {}).get("scale", [0, 0])
        self.perspective_min = QDoubleSpinBox()
        self.perspective_min.setRange(0.0, 0.5)
        self.perspective_min.setSingleStep(0.01)
        self.perspective_min.setValue(perspective_scale[0])
        self.perspective_max = QDoubleSpinBox()
        self.perspective_max.setRange(0.0, 0.5)
        self.perspective_max.setSingleStep(0.01)
        self.perspective_max.setValue(perspective_scale[1])
        perspective_layout.addRow("最小透視尺度:", self.perspective_min)
        perspective_layout.addRow("最大透視尺度:", self.perspective_max)
        perspective_group.setLayout(perspective_layout)
        ops_layout.addWidget(perspective_group)
        
        # 旋轉
        rotate_group = QGroupBox("旋轉")
        rotate_layout = QFormLayout()
        rotate_angle = operations.get("rotate", {}).get("angle", [0, 0])
        self.rotate_min = QSpinBox()
        self.rotate_min.setRange(-180, 180)
        self.rotate_min.setValue(rotate_angle[0])
        self.rotate_max = QSpinBox()
        self.rotate_max.setRange(-180, 180)
        self.rotate_max.setValue(rotate_angle[1])
        rotate_layout.addRow("最小旋轉角度:", self.rotate_min)
        rotate_layout.addRow("最大旋轉角度:", self.rotate_max)
        rotate_group.setLayout(rotate_layout)
        ops_layout.addWidget(rotate_group)
        
        # 縮放
        scale_group = QGroupBox("縮放")
        scale_layout = QFormLayout()
        scale_range = operations.get("scale", {}).get("range", [1, 1])
        self.scale_min = QDoubleSpinBox()
        self.scale_min.setRange(0.1, 2.0)
        self.scale_min.setSingleStep(0.01)
        self.scale_min.setValue(scale_range[0])
        self.scale_max = QDoubleSpinBox()
        self.scale_max.setRange(0.1, 2.0)
        self.scale_max.setSingleStep(0.01)
        self.scale_max.setValue(scale_range[1])
        scale_layout.addRow("最小縮放因子:", self.scale_min)
        scale_layout.addRow("最大縮放因子:", self.scale_max)
        scale_group.setLayout(scale_layout)
        ops_layout.addWidget(scale_group)
        
        ops_group.setLayout(ops_layout)
        layout.addWidget(ops_group)
        
        widget.setWidget(content)
        widget.setWidgetResizable(True)
        return widget
    
    def create_dataset_split_tab(self):
        widget = QWidget()
        layout = QFormLayout(widget)
        
        split_config = self.config.get("train_test_split", {})
        
        # 輸入設定
        input_group = QGroupBox("輸入設定")
        input_layout = QFormLayout()
        
        input_config = split_config.get("input", {})
        img_container, self.split_img_dir = self.create_dir_selector_widget(
            input_config.get("image_dir", "")
        )
        input_layout.addRow("圖片目錄:", img_container)
        
        label_container, self.split_label_dir = self.create_dir_selector_widget(
            input_config.get("label_dir", "")
        )
        input_layout.addRow("標籤目錄:", label_container)
        
        input_group.setLayout(input_layout)
        layout.addRow(input_group)
        
        # 輸出設定
        output_config = split_config.get("output", {})
        out_container, self.split_output_dir = self.create_dir_selector_widget(
            output_config.get("output_dir", "")
        )
        layout.addRow("輸出目錄:", out_container)

        # 支援格式
        input_formats = ", ".join(split_config.get("input_formats", [".png", ".jpg", ".jpeg"]))
        self.split_input_formats = QLineEdit(input_formats)
        layout.addRow("支援格式:", self.split_input_formats)

        # 標籤格式
        self.split_label_format = QLineEdit(split_config.get("label_format", ".txt"))
        layout.addRow("標籤格式:", self.split_label_format)

        # 分割比例
        split_group = QGroupBox("分割比例")
        split_layout = QFormLayout()
        
        ratios = split_config.get("split_ratios", {})
        
        self.train_ratio = QDoubleSpinBox()
        self.train_ratio.setRange(0.1, 0.8)
        self.train_ratio.setSingleStep(0.05)
        self.train_ratio.setValue(ratios.get("train", 0.7))
        split_layout.addRow("訓練集比例:", self.train_ratio)
        
        self.val_ratio = QDoubleSpinBox()
        self.val_ratio.setRange(0.1, 0.4)
        self.val_ratio.setSingleStep(0.05)
        self.val_ratio.setValue(ratios.get("val", 0.15))
        split_layout.addRow("驗證集比例:", self.val_ratio)
        
        self.test_ratio = QDoubleSpinBox()
        self.test_ratio.setRange(0.1, 0.4)
        self.test_ratio.setSingleStep(0.05)
        self.test_ratio.setValue(ratios.get("test", 0.15))
        split_layout.addRow("測試集比例:", self.test_ratio)
        
        split_group.setLayout(split_layout)
        layout.addRow(split_group)
        
        return widget
    
    def create_dir_selector_widget(self, initial_path=""):
        """建立目錄選擇器小工具"""
        container = QWidget()
        layout = QHBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        
        line_edit = QLineEdit(initial_path)
        browse_btn = QPushButton("瀏覽...")
        browse_btn.clicked.connect(lambda: self.browse_directory(line_edit))
        
        layout.addWidget(line_edit)
        layout.addWidget(browse_btn)
        return container, line_edit
    
    def create_slider_widget(self, initial_value, min_val, max_val):
        """建立滑動條小工具"""
        container = QWidget()
        layout = QHBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        
        slider = QSlider(Qt.Horizontal)
        slider.setRange(min_val, max_val)
        slider.setValue(initial_value)
        
        spinbox = QSpinBox()
        spinbox.setRange(min_val, max_val)
        spinbox.setValue(initial_value)
        
        # 連接滑動條和數字框
        slider.valueChanged.connect(spinbox.setValue)
        spinbox.valueChanged.connect(slider.setValue)
        
        layout.addWidget(slider)
        layout.addWidget(spinbox)
        return container, spinbox
    
    def browse_directory(self, line_edit):
        """瀏覽目錄對話框"""
        directory = QFileDialog.getExistingDirectory(self, "選擇資料夾", line_edit.text())
        if directory:
            line_edit.setText(directory)
    
    def save_config(self):
        """儲存配置到檔案"""
        try:
            # 更新基本設定
            self.config["pipeline"]["log_file"] = self.log_file_edit.text()
            
            # 更新任務啟用狀態
            for task in self.config["pipeline"]["tasks"]:
                if task["name"] in self.task_checkboxes:
                    task["enabled"] = self.task_checkboxes[task["name"]].isChecked()
            
            # 更新格式轉換設定
            fc = self.config.setdefault("format_conversion", {})
            fc["input_dir"] = self.fc_input_dir.text()
            fc["output_dir"] = self.fc_output_dir.text()
            fc["input_formats"] = [s.strip() for s in self.fc_input_formats.text().split(",") if s.strip()]
            fc["output_format"] = self.fc_output_format.currentText()
            fc["quality"] = self.fc_quality.value()
            
            # 更新圖像增強設定
            aug = self.config.setdefault("image_augmentation", {})
            aug.setdefault("input", {})["image_dir"] = self.aug_input_dir.text()
            aug.setdefault("output", {})["image_dir"] = self.aug_output_dir.text()
            aug.setdefault("augmentation", {})["num_images"] = self.aug_num_images.value()
            aug.setdefault("processing", {})["num_workers"] = self.aug_num_workers.value()

            # 更新異常檢測設定
            anomaly = self.config.setdefault("anomaly_detection", {})
            anomaly["threshold"] = self.anomaly_threshold.value()
            anomaly["reference_folder"] = self.anomaly_ref_folder.text()
            anomaly["test_folder"] = self.anomaly_test_folder.text()
            anomaly["output_folder"] = self.anomaly_output_folder.text()
            anomaly["input_formats"] = [
                s.strip() for s in self.anomaly_input_formats.text().split(",") if s.strip()
            ]

            # 更新 YOLO 增強設定
            yolo = self.config.setdefault("yolo_augmentation", {})
            yolo.setdefault("input", {})["image_dir"] = self.yolo_img_dir.text()
            yolo.setdefault("input", {})["label_dir"] = self.yolo_label_dir.text()
            yolo.setdefault("output", {})["image_dir"] = self.yolo_out_img_dir.text()
            yolo.setdefault("output", {})["label_dir"] = self.yolo_out_label_dir.text()
            yolo.setdefault("processing", {})["num_workers"] = self.yolo_num_workers.value()
            yolo_aug = yolo.setdefault("augmentation", {})
            yolo_aug["num_images"] = self.yolo_num_images.value()
            yolo_aug["num_operations"] = [self.num_ops_min.value(), self.num_ops_max.value()]
            yolo_ops = yolo_aug.setdefault("operations", {})
            yolo_ops["blur"] = {"kernel": [self.blur_min.value(), self.blur_max.value()]}
            yolo_ops["contrast"] = {"range": [self.contrast_min.value(), self.contrast_max.value()]}
            yolo_ops["flip"] = {"probability": self.flip_prob.value()}
            yolo_ops["hue"] = {"range": [self.hue_min.value(), self.hue_max.value()]}
            yolo_ops["multiply"] = {"range": [self.multiply_min.value(), self.multiply_max.value()]}
            yolo_ops["noise"] = {"scale": [self.noise_min.value(), self.noise_max.value()]}
            yolo_ops["perspective"] = {"scale": [self.perspective_min.value(), self.perspective_max.value()]}
            yolo_ops["rotate"] = {"angle": [self.rotate_min.value(), self.rotate_max.value()]}
            yolo_ops["scale"] = {"range": [self.scale_min.value(), self.scale_max.value()]}

            # 更新資料集分割設定
            split = self.config.setdefault("train_test_split", {})
            split.setdefault("input", {})["image_dir"] = self.split_img_dir.text()
            split.setdefault("input", {})["label_dir"] = self.split_label_dir.text()
            split.setdefault("output", {})["output_dir"] = self.split_output_dir.text()
            split["input_formats"] = [
                s.strip() for s in self.split_input_formats.text().split(",") if s.strip()
            ]
            split["label_format"] = self.split_label_format.text()
            ratios = split.setdefault("split_ratios", {})
            ratios["train"] = self.train_ratio.value()
            ratios["val"] = self.val_ratio.value()
            ratios["test"] = self.test_ratio.value()
            
            # 儲存到檔案
            with open(self.config_path, "w", encoding="utf-8") as f:
                yaml.safe_dump(self.config, f, allow_unicode=True, default_flow_style=False, indent=2)

            QMessageBox.information(self, "成功", "配置已成功儲存！")
            parent = self.parent()
            if parent and hasattr(parent, "reload_config"):
                parent.reload_config()
            self.accept()

        except Exception as e:
            QMessageBox.critical(self, "錯誤", f"儲存配置時發生錯誤：{e}")
    
    def restore_defaults(self):
        """恢復預設值"""
        reply = QMessageBox.question(
            self, "確認", "是否要恢復所有設定為預設值？",
            QMessageBox.Yes | QMessageBox.No
        )
        if reply == QMessageBox.Yes:
            # 這裡可以載入預設配置
            pass


class VisionSuiteMainWindow(QMainWindow):
    """主視窗類別，提供完整的 GUI 體驗"""
    
    def __init__(self):
        super().__init__()
        self.config_path = "config.yaml"
        self.worker = None
        self.init_ui()
        self.setup_logging()
        self.apply_theme()
        
    def init_ui(self):
        self.setWindowTitle("VisionSuite - 圖像處理流程工具")
        self.setGeometry(100, 100, 1200, 800)
        
        # 建立選單欄
        self.create_menu_bar()
        
        # 建立工具欄
        self.create_tool_bar()
        
        # 建立中央小工具
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # 主佈局使用分割器
        main_splitter = QSplitter(Qt.Horizontal)
        central_widget.setLayout(QHBoxLayout())
        central_widget.layout().addWidget(main_splitter)
        
        # 左側控制面板
        control_panel = self.create_control_panel()
        main_splitter.addWidget(control_panel)
        
        # 右側日誌和狀態面板
        log_panel = self.create_log_panel()
        main_splitter.addWidget(log_panel)
        
        # 設定分割器比例
        main_splitter.setSizes([400, 800])
        
        # 建立狀態欄
        self.create_status_bar()
        
        # 載入預設配置
        self.load_default_config()
    
    def create_menu_bar(self):
        """建立選單欄"""
        menubar = self.menuBar()
        
        # 檔案選單
        file_menu = menubar.addMenu("檔案(&F)")
        
        open_config_action = QAction("開啟配置檔(&O)", self)
        open_config_action.setShortcut("Ctrl+O")
        open_config_action.triggered.connect(self.open_config)
        file_menu.addAction(open_config_action)
        
        save_config_action = QAction("儲存配置檔(&S)", self)
        save_config_action.setShortcut("Ctrl+S")
        save_config_action.triggered.connect(self.save_current_config)
        file_menu.addAction(save_config_action)
        
        file_menu.addSeparator()
        
        exit_action = QAction("結束(&X)", self)
        exit_action.setShortcut("Ctrl+Q")
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # 工具選單
        tools_menu = menubar.addMenu("工具(&T)")
        
        advanced_config_action = QAction("進階配置編輯器(&A)", self)
        advanced_config_action.triggered.connect(self.open_advanced_config)
        tools_menu.addAction(advanced_config_action)
        
        clear_log_action = QAction("清除日誌(&C)", self)
        clear_log_action.triggered.connect(self.clear_log)
        tools_menu.addAction(clear_log_action)
        
        # 幫助選單
        help_menu = menubar.addMenu("幫助(&H)")
        
        about_action = QAction("關於(&A)", self)
        about_action.triggered.connect(self.show_about)
        help_menu.addAction(about_action)
    
    def create_tool_bar(self):
        """建立工具欄"""
        toolbar = QToolBar()
        self.addToolBar(toolbar)
        
        # 執行按鈕
        run_action = QAction("執行流程", self)
        run_action.triggered.connect(self.execute_pipeline)
        toolbar.addAction(run_action)
        
        toolbar.addSeparator()
        
        # 停止按鈕
        stop_action = QAction("停止執行", self)
        stop_action.triggered.connect(self.stop_execution)
        toolbar.addAction(stop_action)
        
        toolbar.addSeparator()
        
        # 配置按鈕
        config_action = QAction("配置編輯", self)
        config_action.triggered.connect(self.open_advanced_config)
        toolbar.addAction(config_action)
    
    def create_control_panel(self):
        """建立左側控制面板"""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        
        # 配置檔案區域
        config_group = QGroupBox("配置檔案")
        config_layout = QVBoxLayout()
        
        # 配置檔案顯示和選擇
        config_file_layout = QHBoxLayout()
        self.config_label = QLabel("config.yaml")
        self.config_label.setStyleSheet("font-weight: bold; padding: 5px;")
        
        select_config_btn = QPushButton("選擇配置檔")
        select_config_btn.clicked.connect(self.open_config)
        
        config_file_layout.addWidget(QLabel("當前配置:"))
        config_file_layout.addWidget(self.config_label, 1)
        config_file_layout.addWidget(select_config_btn)
        config_layout.addLayout(config_file_layout)
        
        # 快速設定按鈕
        quick_buttons_layout = QHBoxLayout()
        edit_config_btn = QPushButton("編輯配置")
        edit_config_btn.clicked.connect(self.open_advanced_config)
        
        reload_config_btn = QPushButton("重新載入")
        reload_config_btn.clicked.connect(self.reload_config)
        
        quick_buttons_layout.addWidget(edit_config_btn)
        quick_buttons_layout.addWidget(reload_config_btn)
        config_layout.addLayout(quick_buttons_layout)
        
        config_group.setLayout(config_layout)
        layout.addWidget(config_group)
        
        # 任務選擇區域
        task_group = QGroupBox("任務選擇")
        task_layout = QVBoxLayout()
        
        # 任務樹狀檢視
        self.task_tree = QTreeWidget()
        self.task_tree.setHeaderLabels(["任務", "狀態", "說明"])
        self.task_tree.setRootIsDecorated(True)
        
        # 建立任務項目
        self.task_items = {}
        self.create_task_tree_items()
        
        task_layout.addWidget(self.task_tree)
        
        # 任務操作按鈕
        task_btn_layout = QHBoxLayout()
        select_all_btn = QPushButton("全選")
        select_all_btn.clicked.connect(self.select_all_tasks)
        
        clear_all_btn = QPushButton("全清")
        clear_all_btn.clicked.connect(self.clear_all_tasks)
        
        invert_btn = QPushButton("反選")
        invert_btn.clicked.connect(self.invert_task_selection)
        
        task_btn_layout.addWidget(select_all_btn)
        task_btn_layout.addWidget(clear_all_btn)
        task_btn_layout.addWidget(invert_btn)
        task_layout.addLayout(task_btn_layout)
        
        task_group.setLayout(task_layout)
        layout.addWidget(task_group)
        
        # 執行控制區域
        execution_group = QGroupBox("執行控制")
        execution_layout = QVBoxLayout()
        
        # 進度條
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        execution_layout.addWidget(self.progress_bar)
        
        # 執行按鈕
        exec_btn_layout = QHBoxLayout()
        self.run_btn = QPushButton("執行流程")
        self.run_btn.clicked.connect(self.execute_pipeline)
        self.run_btn.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                font-weight: bold;
                padding: 8px 16px;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QPushButton:disabled {
                background-color: #cccccc;
            }
        """)
        
        self.stop_btn = QPushButton("停止執行")
        self.stop_btn.clicked.connect(self.stop_execution)
        self.stop_btn.setEnabled(False)
        self.stop_btn.setStyleSheet("""
            QPushButton {
                background-color: #f44336;
                color: white;
                font-weight: bold;
                padding: 8px 16px;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #da190b;
            }
            QPushButton:disabled {
                background-color: #cccccc;
            }
        """)
        
        exec_btn_layout.addWidget(self.run_btn)
        exec_btn_layout.addWidget(self.stop_btn)
        execution_layout.addLayout(exec_btn_layout)
        
        execution_group.setLayout(execution_layout)
        layout.addWidget(execution_group)
        
        # 加入彈性空間
        layout.addStretch()
        
        return panel
    
    def create_log_panel(self):
        """建立右側日誌面板"""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        
        # 日誌控制區域
        log_control_layout = QHBoxLayout()
        
        log_label = QLabel("執行日誌:")
        log_label.setFont(QFont("Arial", 12, QFont.Bold))
        
        clear_log_btn = QPushButton("清除日誌")
        clear_log_btn.clicked.connect(self.clear_log)
        
        save_log_btn = QPushButton("儲存日誌")
        save_log_btn.clicked.connect(self.save_log)
        
        log_control_layout.addWidget(log_label)
        log_control_layout.addStretch()
        log_control_layout.addWidget(save_log_btn)
        log_control_layout.addWidget(clear_log_btn)
        
        layout.addLayout(log_control_layout)
        
        # 日誌顯示區域
        self.log_display = QTextEdit()
        self.log_display.setReadOnly(True)
        self.log_display.setFont(QFont("Consolas", 10))
        self.log_display.setStyleSheet("""
            QTextEdit {
                background-color: #1e1e1e;
                color: #dcdcdc;
                border: 1px solid #3c3c3c;
                border-radius: 4px;
                padding: 8px;
            }
        """)
        
        layout.addWidget(self.log_display)
        
        return panel
    
    def create_status_bar(self):
        """建立狀態欄"""
        self.statusBar = QStatusBar()
        self.setStatusBar(self.statusBar)
        
        # 狀態標籤
        self.status_label = QLabel("就緒")
        self.statusBar.addWidget(self.status_label)
        
        # 配置檔案狀態
        self.config_status_label = QLabel(f"配置: {self.config_path}")
        self.statusBar.addPermanentWidget(self.config_status_label)
        
        # 時間顯示
        self.time_label = QLabel()
        self.statusBar.addPermanentWidget(self.time_label)
        
        # 更新時間的定時器
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_time)
        self.timer.start(1000)  # 每秒更新一次
    
    def create_task_tree_items(self):
        """建立任務樹狀結構"""
        # 預處理組
        preprocess_item = QTreeWidgetItem(["預處理", "", "圖像預處理相關任務"])
        self.task_tree.addTopLevelItem(preprocess_item)
        
        format_item = QTreeWidgetItem(["格式轉換", "已停用", "轉換圖像格式"])
        format_item.setCheckState(0, Qt.Unchecked)
        preprocess_item.addChild(format_item)
        self.task_items["format_conversion"] = format_item
        
        # 增強組
        augmentation_item = QTreeWidgetItem(["資料增強", "", "圖像資料增強相關任務"])
        self.task_tree.addTopLevelItem(augmentation_item)
        
        img_aug_item = QTreeWidgetItem(["圖像增強", "已啟用", "一般圖像增強"])
        img_aug_item.setCheckState(0, Qt.Checked)
        augmentation_item.addChild(img_aug_item)
        self.task_items["image_augmentation"] = img_aug_item
        
        yolo_aug_item = QTreeWidgetItem(["YOLO 增強", "已停用", "YOLO 標籤增強"])
        yolo_aug_item.setCheckState(0, Qt.Unchecked)
        augmentation_item.addChild(yolo_aug_item)
        self.task_items["yolo_augmentation"] = yolo_aug_item
        
        # 檢測組
        detection_item = QTreeWidgetItem(["異常檢測", "", "異常檢測相關任務"])
        self.task_tree.addTopLevelItem(detection_item)
        
        anomaly_item = QTreeWidgetItem(["異常檢測", "已停用", "圖像異常檢測"])
        anomaly_item.setCheckState(0, Qt.Unchecked)
        detection_item.addChild(anomaly_item)
        self.task_items["anomaly_detection"] = anomaly_item
        
        # 資料集組
        dataset_item = QTreeWidgetItem(["資料集處理", "", "資料集相關處理"])
        self.task_tree.addTopLevelItem(dataset_item)
        
        split_item = QTreeWidgetItem(["資料集分割", "已停用", "分割訓練/測試資料集"])
        split_item.setCheckState(0, Qt.Unchecked)
        dataset_item.addChild(split_item)
        self.task_items["dataset_splitter"] = split_item
        
        # 展開所有項目
        self.task_tree.expandAll()
    
    def setup_logging(self):
        """設定日誌系統"""
        self.text_logger = QTextEditLogger(self.log_display)
        self.text_logger.setFormatter(
            logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        )
    
    def apply_theme(self):
        """套用主題樣式"""
        self.setStyleSheet("""
            QMainWindow {
                background-color: #f0f0f0;
            }
            QGroupBox {
                font-weight: bold;
                border: 2px solid #cccccc;
                border-radius: 5px;
                margin-top: 10px;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
                background-color: #f0f0f0;
            }
            QPushButton {
                padding: 6px 12px;
                border: 1px solid #cccccc;
                border-radius: 3px;
                background-color: #ffffff;
            }
            QPushButton:hover {
                background-color: #e6e6e6;
                border: 1px solid #999999;
            }
            QPushButton:pressed {
                background-color: #d4edda;
            }
            QTreeWidget {
                border: 1px solid #cccccc;
                border-radius: 3px;
                background-color: #ffffff;
            }
            QTreeWidget::item {
                padding: 2px;
            }
            QTreeWidget::item:selected {
                background-color: #007acc;
                color: white;
            }
        """)
    
    def load_default_config(self):
        """載入預設配置"""
        try:
            if os.path.exists(self.config_path):
                self.reload_config()
            else:
                self.status_label.setText("配置檔案不存在，請選擇配置檔案")
        except Exception as e:
            QMessageBox.warning(self, "警告", f"載入配置時發生錯誤：{e}")
    
    def reload_config(self):
        """重新載入配置檔案"""
        try:
            config = load_config(self.config_path)
            
            # 更新任務狀態
            task_configs = {
                t["name"]: t.get("enabled", True) 
                for t in config["pipeline"]["tasks"]
            }
            
            for task_name, item in self.task_items.items():
                enabled = task_configs.get(task_name, False)
                item.setCheckState(0, Qt.Checked if enabled else Qt.Unchecked)
                item.setText(1, "已啟用" if enabled else "已停用")
            
            self.config_status_label.setText(f"配置: {self.config_path}")
            self.status_label.setText("配置已重新載入")
            
        except Exception as e:
            QMessageBox.warning(self, "警告", f"重新載入配置時發生錯誤：{e}")
    
    def open_config(self):
        """開啟配置檔案"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "選擇配置檔案", "", "YAML Files (*.yaml *.yml);;All Files (*)"
        )
        
        if file_path:
            self.config_path = file_path
            self.config_label.setText(os.path.basename(file_path))
            self.reload_config()
    
    def save_current_config(self):
        """儲存當前配置"""
        file_path, _ = QFileDialog.getSaveFileName(
            self, "儲存配置檔案", self.config_path, "YAML Files (*.yaml *.yml)"
        )
        
        if file_path:
            try:
                config = load_config(self.config_path)
                
                # 更新任務啟用狀態
                for task in config["pipeline"]["tasks"]:
                    if task["name"] in self.task_items:
                        item = self.task_items[task["name"]]
                        task["enabled"] = item.checkState(0) == Qt.Checked
                
                # 儲存檔案
                with open(file_path, "w", encoding="utf-8") as f:
                    yaml.safe_dump(config, f, allow_unicode=True, default_flow_style=False, indent=2)
                
                self.status_label.setText(f"配置已儲存至 {file_path}")
                QMessageBox.information(self, "成功", "配置檔案已成功儲存！")
                
            except Exception as e:
                QMessageBox.critical(self, "錯誤", f"儲存配置時發生錯誤：{e}")
    
    def open_advanced_config(self):
        """開啟進階配置編輯器"""
        if not os.path.exists(self.config_path):
            QMessageBox.warning(self, "警告", "請先選擇一個有效的配置檔案")
            return
        
        dialog = AdvancedConfigDialog(self.config_path, self)
        if dialog.exec_():
            self.reload_config()
    
    def select_all_tasks(self):
        """選擇所有任務"""
        for item in self.task_items.values():
            item.setCheckState(0, Qt.Checked)
            item.setText(1, "已啟用")
    
    def clear_all_tasks(self):
        """清除所有任務選擇"""
        for item in self.task_items.values():
            item.setCheckState(0, Qt.Unchecked)
            item.setText(1, "已停用")
    
    def invert_task_selection(self):
        """反選任務"""
        for item in self.task_items.values():
            current_state = item.checkState(0)
            new_state = Qt.Unchecked if current_state == Qt.Checked else Qt.Checked
            item.setCheckState(0, new_state)
            item.setText(1, "已啟用" if new_state == Qt.Checked else "已停用")
    
    def execute_pipeline(self):
        """執行流程"""
        # 獲取選中的任務
        selected_tasks = []
        for task_name, item in self.task_items.items():
            if item.checkState(0) == Qt.Checked:
                selected_tasks.append(task_name)
        
        if not selected_tasks:
            QMessageBox.warning(self, "警告", "請至少選擇一個任務")
            return
        
        if not os.path.exists(self.config_path):
            QMessageBox.warning(self, "警告", "配置檔案不存在")
            return
        
        try:
            # 清除日誌
            self.log_display.clear()
            
            # 設定UI狀態
            self.run_btn.setEnabled(False)
            self.stop_btn.setEnabled(True)
            self.progress_bar.setVisible(True)
            self.progress_bar.setRange(0, 0)  # 無限進度條
            
            # 載入配置和設定日誌
            config = load_config(self.config_path)
            logger = setup_logging(config["pipeline"]["log_file"])
            logger.addHandler(self.text_logger)
            
            # 建立參數
            args = argparse.Namespace(
                config=self.config_path,
                input_format=None,
                output_format=None,
            )
            
            # 建立工作線程
            self.worker = PipelineWorker(selected_tasks, config, logger, args)
            self.worker.progress.connect(self.update_progress)
            self.worker.status.connect(self.update_status)
            self.worker.finished.connect(self.execution_finished)
            
            # 開始執行
            self.worker.start()
            self.status_label.setText("正在執行流程...")
            
        except Exception as e:
            self.execution_finished(False, str(e))
    
    def stop_execution(self):
        """停止執行"""
        if self.worker and self.worker.isRunning():
            self.worker.terminate()
            self.worker.wait()
            self.execution_finished(False, "執行已被使用者中止")
    
    def update_progress(self, value):
        """更新進度"""
        if self.progress_bar.maximum() > 0:
            self.progress_bar.setValue(value)
    
    def update_status(self, message):
        """更新狀態"""
        self.status_label.setText(message)
    
    def execution_finished(self, success, message):
        """執行完成"""
        # 恢復UI狀態
        self.run_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.progress_bar.setVisible(False)
        
        # 顯示結果
        if success:
            self.status_label.setText("流程執行完成")
            QMessageBox.information(self, "完成", message)
        else:
            self.status_label.setText("流程執行失敗")
            QMessageBox.critical(self, "錯誤", message)
        
        # 清理工作線程
        if self.worker:
            self.worker.deleteLater()
            self.worker = None
    
    def clear_log(self):
        """清除日誌"""
        self.log_display.clear()
        self.status_label.setText("日誌已清除")
    
    def save_log(self):
        """儲存日誌"""
        file_path, _ = QFileDialog.getSaveFileName(
            self, "儲存日誌", "pipeline_log.txt", "Text Files (*.txt);;All Files (*)"
        )
        
        if file_path:
            try:
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(self.log_display.toPlainText())
                self.status_label.setText(f"日誌已儲存至 {file_path}")
                QMessageBox.information(self, "成功", "日誌檔案已成功儲存！")
            except Exception as e:
                QMessageBox.critical(self, "錯誤", f"儲存日誌時發生錯誤：{e}")
    
    def update_time(self):
        """更新時間顯示"""
        from datetime import datetime
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.time_label.setText(current_time)
    
    def show_about(self):
        """顯示關於對話框"""
        QMessageBox.about(
            self, "關於 VisionSuite",
            """
            <h3>VisionSuite 圖像處理流程工具</h3>
            <p>版本: 2.0</p>
            <p>一個功能強大的圖像處理和機器學習資料預處理工具。</p>
            
            <p><b>主要功能：</b></p>
            <ul>
                <li>圖像格式轉換</li>
                <li>圖像資料增強</li>
                <li>YOLO 標籤增強</li>
                <li>異常檢測</li>
                <li>資料集分割</li>
            </ul>
            
            <p><b>特色：</b></p>
            <ul>
                <li>直觀的圖形化界面</li>
                <li>進階配置編輯器</li>
                <li>即時日誌顯示</li>
                <li>多線程執行支援</li>
            </ul>
            """
        )
    
    def closeEvent(self, event):
        """視窗關閉事件"""
        if self.worker and self.worker.isRunning():
            reply = QMessageBox.question(
                self, "確認結束", "流程正在執行中，是否要強制結束？",
                QMessageBox.Yes | QMessageBox.No
            )
            if reply == QMessageBox.Yes:
                self.worker.terminate()
                self.worker.wait()
                event.accept()
            else:
                event.ignore()
        else:
            event.accept()


def main():
    """主函數"""
    app = QApplication(sys.argv)
    
    # 設定應用程式屬性
    app.setApplicationName("VisionSuite")
    app.setApplicationVersion("2.0")
    app.setOrganizationName("VisionSuite Team")
    
    # 建立主視窗
    window = VisionSuiteMainWindow()
    window.show()
    
    # 啟動事件迴圈
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()

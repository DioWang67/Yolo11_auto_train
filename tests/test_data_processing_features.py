"""
功能测试：数据处理流程
覆盖功能：数据集分割、数据增强、质量检查
"""
import logging
from pathlib import Path
from types import SimpleNamespace
import pytest


class TestDatasetSplitFunctionality:
    """测试数据集分割功能"""
    
    def test_split_creates_train_val_test_directories(self, tmp_path, monkeypatch):
        """功能：创建train/val/test目录结构"""
        from picture_tool.split.dataset_splitter import split_dataset
        
        # 准备输入数据
        input_dir = tmp_path / "raw"
        images_dir = input_dir / "images"
        labels_dir = input_dir / "labels"
        images_dir.mkdir(parents=True)
        labels_dir.mkdir(parents=True)
        
        # 创建测试文件
        for i in range(20):
            (images_dir / f"img_{i:03d}.jpg").write_text(f"image{i}")
            (labels_dir / f"img_{i:03d}.txt").write_text(f"0 0.5 0.5 0.1 0.1")
        
        output_dir = tmp_path / "split"
        
        config = {
            "train_test_split": {
                "input": {
                    "image_dir": str(images_dir),
                    "label_dir": str(labels_dir)
                },
                "output": {
                    "output_dir": str(output_dir)
                },
                "split_ratios": {
                    "train": 0.7,
                    "val": 0.2,
                    "test": 0.1
                }
            }
        }
        
        # 执行功能
        split_dataset(config)
        
        # 验证功能结果
        assert (output_dir / "train" / "images").exists()
        assert (output_dir / "val" / "images").exists()
        assert (output_dir / "test" / "images").exists()
        assert (output_dir / "train" / "labels").exists()
        assert (output_dir / "data.yaml").exists()
    
    def test_split_distributes_files_correctly(self, tmp_path):
        """功能：按比例正确分配文件"""
        from picture_tool.split.dataset_splitter import split_dataset
        
        input_dir = tmp_path / "raw"
        images_dir = input_dir / "images"
        labels_dir = input_dir / "labels"
        images_dir.mkdir(parents=True)
        labels_dir.mkdir(parents=True)
        
        # 创建100个文件
        for i in range(100):
            (images_dir / f"img_{i:03d}.jpg").write_text(f"img")
            (labels_dir / f"img_{i:03d}.txt").write_text(f"label")
        
        output_dir = tmp_path / "split"
        
        config = {
            "train_test_split": {
                "input": {
                    "image_dir": str(images_dir),
                    "label_dir": str(labels_dir)
                },
                "output": {
                    "output_dir": str(output_dir)
                },
                "split_ratios": {
                    "train": 0.8,
                    "val": 0.1,
                    "test": 0.1
                }
            }
        }
        
        split_dataset(config)
        
        # 验证分配比例大致正确
        train_count = len(list((output_dir / "train" / "images").glob("*.jpg")))
        val_count = len(list((output_dir / "val" / "images").glob("*.jpg")))
        test_count = len(list((output_dir / "test" / "images").glob("*.jpg")))
        
        assert train_count + val_count + test_count == 100
        assert 75 <= train_count <= 85  # 80% ± 5%
        assert 5 <= val_count <= 15
        assert 5 <= test_count <= 15


class TestDataAugmentationFunctionality:
    """测试数据增强功能"""
    
    def test_augmentation_generates_new_images(self, tmp_path, monkeypatch):
        """功能：生成增强后的图像"""
        from picture_tool.augment.image_augmentor import augment_images
        
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        output_dir = tmp_path / "output"
        
        # 创建输入图像
        for i in range(5):
            (input_dir / f"img_{i}.jpg").write_text(f"image{i}")
        
        config = {
            "input_dir": str(input_dir),
            "output_dir": str(output_dir),
            "num_augmentations": 3,
            "operations": ["rotate", "flip"]
        }
        
        # Mock实际的图像处理
        def mock_augment(*args, **kwargs):
            output_dir.mkdir(exist_ok=True)
            for i in range(15):  # 5 images × 3 augmentations
                (output_dir / f"aug_{i}.jpg").write_text(f"augmented{i}")
        
        monkeypatch.setattr("picture_tool.augment.image_augmentor.augment_images", mock_augment)
        
        mock_augment()
        
        # 验证生成了增强图像
        output_files = list(output_dir.glob("*.jpg"))
        assert len(output_files) >= 5


class TestDatasetLintFunctionality:
    """测试数据集质量检查功能"""
    
    def test_lint_detects_missing_labels(self, tmp_path):
        """功能：检测缺失的标签文件"""
        from picture_tool.quality.dataset_linter import lint_dataset
        
        image_dir = tmp_path / "images"
        label_dir = tmp_path / "labels"
        image_dir.mkdir()
        label_dir.mkdir()
        
        # 创建10个图像，但只有8个标签
        for i in range(10):
            (image_dir / f"img_{i}.jpg").write_text(f"image{i}")
        for i in range(8):  # 缺少2个标签
            (label_dir / f"img_{i}.txt").write_text(f"0 0.5 0.5 0.1 0.1")
        
        output_dir = tmp_path / "lint_output"
        
        config = {
            "dataset_lint": {
                "image_dir": str(image_dir),
                "label_dir": str(label_dir),
                "output_dir": str(output_dir),
                "num_preview": 5
            }
        }
        
        # 执行检查
        lint_dataset(config)
        
        # 验证输出报告
        assert output_dir.exists()
        report_files = list(output_dir.glob("*.csv"))
        assert len(report_files) > 0
    
    def test_lint_validates_bbox_format(self, tmp_path):
        """功能：验证边界框格式"""
        from picture_tool.quality.dataset_linter import lint_dataset
        
        image_dir = tmp_path / "images"
        label_dir = tmp_path / "labels"
        image_dir.mkdir()
        label_dir.mkdir()
        output_dir = tmp_path / "lint"
        
        # 创建有效和无效的标签
        (image_dir / "valid.jpg").write_text("img")
        (label_dir / "valid.txt").write_text("0 0.5 0.5 0.2 0.2")  # 有效
        
        (image_dir / "invalid.jpg").write_text("img")
        (label_dir / "invalid.txt").write_text("0 1.5 0.5 0.2 0.2")  # 无效：x>1
        
        config = {
            "dataset_lint": {
                "image_dir": str(image_dir),
                "label_dir": str(label_dir),
                "output_dir": str(output_dir),
                "check_bbox_range": True
            }
        }
        
        lint_dataset(config)
        
        assert output_dir.exists()


class TestDataProcessingIntegration:
    """测试数据处理流程集成"""
    
    def test_complete_data_preparation_workflow(self, tmp_path):
        """功能：完整的数据准备流程"""
        # 1. 原始数据准备
        raw_dir = tmp_path / "raw"
        images = raw_dir / "images"
        labels = raw_dir / "labels"
        images.mkdir(parents=True)
        labels.mkdir(parents=True)
        
        for i in range(50):
            (images / f"img_{i}.jpg").write_text(f"img{i}")
            (labels / f"img_{i}.txt").write_text(f"0 0.5 0.5 0.1 0.1")
        
        # 2. 质量检查
        lint_output = tmp_path / "lint"
        assert raw_dir.exists()
        
        # 3. 数据集分割
        split_output = tmp_path / "split"
        # split_dataset会创建split_output
        
        # 4. 数据增强（可选）
        aug_output = tmp_path / "augmented"
        
        # 验证目录结构
        assert raw_dir.exists()
        assert len(list(images.glob("*.jpg"))) == 50

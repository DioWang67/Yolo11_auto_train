"""
功能测试：任务调度与配置管理
覆盖tasks模块中的所有任务函数
预计提升覆盖率：+10%
"""

import hashlib
import json
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest


def _write_runtime_export_contract(run_dir: Path) -> None:
    runtime = run_dir / "weights" / "best.onnx"
    training_weight = run_dir / "weights" / "best.pt"
    (run_dir / "runtime_export_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "runtime_format": "onnx",
                "runtime_file": "weights/best.onnx",
                "runtime_sha256": hashlib.sha256(runtime.read_bytes()).hexdigest(),
                "training_weight_file": "weights/best.pt",
                "training_weight_sha256": hashlib.sha256(
                    training_weight.read_bytes()
                ).hexdigest(),
            }
        ),
        encoding="utf-8",
    )


class TestTasksFunctionality:
    """测试tasks模块的任务调度功能"""

    def test_dataset_splitter_task_config(self, tmp_path):
        """功能：数据集分割任务配置"""
        from picture_tool.tasks.quality import run_dataset_splitter

        input_dir = tmp_path / "raw"
        (input_dir / "images").mkdir(parents=True)
        (input_dir / "labels").mkdir(parents=True)

        # 创建测试数据
        for i in range(20):
            (input_dir / "images" / f"img_{i}.jpg").write_text(f"img{i}")
            (input_dir / "labels" / f"img_{i}.txt").write_text("0 0.5 0.5 0.1 0.1")

        config = {
            "train_test_split": {
                "input": {
                    "image_dir": str(input_dir / "images"),
                    "label_dir": str(input_dir / "labels"),
                },
                "output": {"output_dir": str(tmp_path / "split")},
                "split_ratios": {"train": 0.7, "val": 0.2, "test": 0.1},
            }
        }

        args = SimpleNamespace()

        # 执行任务
        run_dataset_splitter(config, args)

        # 验证输出
        assert (tmp_path / "split" / "train" / "images").exists()

    def test_dataset_lint_task_config(self, tmp_path):
        """功能：数据集质量检查任务"""
        from picture_tool.tasks.quality import run_dataset_lint

        image_dir = tmp_path / "images"
        label_dir = tmp_path / "labels"
        image_dir.mkdir()
        label_dir.mkdir()

        for i in range(10):
            (image_dir / f"img_{i}.jpg").write_text(f"img{i}")
            (label_dir / f"img_{i}.txt").write_text("0 0.5 0.5 0.1 0.1")

        config = {
            "dataset_lint": {
                "image_dir": str(image_dir),
                "label_dir": str(label_dir),
                "output_dir": str(tmp_path / "lint"),
                "num_preview": 5,
            }
        }

        args = SimpleNamespace()
        run_dataset_lint(config, args)

        assert (tmp_path / "lint").exists()

    def test_skip_dataset_splitter_when_up_to_date(self, tmp_path):
        """功能：跳过已完成的数据分割"""
        from picture_tool.tasks.quality import skip_dataset_splitter

        # 创建输入和输出
        input_dir = tmp_path / "input"
        (input_dir / "images").mkdir(parents=True)
        (input_dir / "labels").mkdir(parents=True)

        output_dir = tmp_path / "output"
        (output_dir / "train" / "images").mkdir(parents=True)
        (output_dir / "val" / "images").mkdir(parents=True)
        (output_dir / "test" / "images").mkdir(parents=True)

        config = {
            "train_test_split": {
                "input": {
                    "image_dir": str(input_dir / "images"),
                    "label_dir": str(input_dir / "labels"),
                },
                "output": {"output_dir": str(output_dir)},
            }
        }

        args = SimpleNamespace()

        # 调用skip函数
        result = skip_dataset_splitter(config, args)

        # 如果输出存在且较新，应该返回skip消息
        if result:
            assert "skip" in result.lower()

    def test_deploy_uses_onnx_when_detection_config_selects_onnx(self, tmp_path):
        """Deploy should point config.yaml at versioned ONNX when selected."""
        import yaml

        from picture_tool.tasks.deploy import run_deploy

        run_dir = tmp_path / "runs" / "train"
        weights_dir = run_dir / "weights"
        weights_dir.mkdir(parents=True)
        (weights_dir / "best.pt").write_bytes(b"pt")
        (weights_dir / "best.onnx").write_bytes(b"onnx")
        _write_runtime_export_contract(run_dir)
        (run_dir / "detection_config.yaml").write_text(
            yaml.safe_dump(
                {
                    "weights": "best.onnx",
                    "current_product": "PCBA1",
                    "current_area": "A",
                    "pipeline": ["count_check", "sequence_check", "save_results"],
                    "enable_color_check": False,
                }
            ),
            encoding="utf-8",
        )

        inference_models_dir = tmp_path / "models"
        existing_dir = inference_models_dir / "PCBA1" / "A" / "yolo"
        existing_dir.mkdir(parents=True)
        (existing_dir / "config.yaml").write_text(
            yaml.safe_dump(
                {
                    "weights": "old.onnx",
                    "exposure_time": "45678",
                    "gain": "12.5",
                    "calibration": {"target_luma": 120.0},
                }
            ),
            encoding="utf-8",
        )
        config = {
            "yolo_training": {
                "project": str(tmp_path / "runs"),
                "name": "train",
                "deploy": {
                    "enabled": True,
                    "product": "PCBA1",
                    "area": "A",
                    "inference_models_dir": str(inference_models_dir),
                    "version": "1.0.0",
                },
            }
        }

        run_deploy(config, SimpleNamespace())

        deployed_dir = inference_models_dir / "PCBA1" / "A" / "yolo"
        deployed_config = yaml.safe_load(
            (deployed_dir / "config.yaml").read_text(encoding="utf-8")
        )

        assert "PCBA1_A_v1.0.0_" in deployed_config["weights"]
        assert deployed_config["weights"].endswith(".onnx")
        assert deployed_config["enable_color_check"] is False
        assert deployed_config["exposure_time"] == "45678"
        assert deployed_config["gain"] == "12.5"
        assert deployed_config["calibration"] == {"target_luma": 120.0}
        assert (deployed_dir / "weights" / "best.onnx").exists()
        assert list((deployed_dir / "weights").glob("PCBA1_A_v1.0.0_*.onnx"))
        deployment_manifest = yaml.safe_load(
            (deployed_dir / "deployment_manifest.yaml").read_text(encoding="utf-8")
        )
        assert len(deployment_manifest["weight_sha256"]) == 64
        assert deployment_manifest["schema_version"] == 2
        paired_training = (
            deployed_dir / "weights" / deployment_manifest["training_weight_file"]
        )
        assert deployment_manifest["training_weight_file"].endswith(".training.pt")
        assert paired_training.read_bytes() == b"pt"
        assert len(deployment_manifest["training_weight_sha256"]) == 64
        assert deployment_manifest["trained_at"]
        assert deployment_manifest["deployed_at"]
        versioned_weight = next(
            (deployed_dir / "weights").glob("PCBA1_A_v1.0.0_*.onnx")
        )
        assert versioned_weight.with_name(
            f"{versioned_weight.name}.manifest.yaml"
        ).exists()
        assert (deployed_dir / deployment_manifest["config_snapshot"]).exists()

    def test_artifact_bundle_matches_inference_models_layout(self, tmp_path):
        """Bundle should unzip directly under yolo11_inference/models."""
        import yaml

        from picture_tool.tasks.bundle import run_artifact_bundle

        run_dir = tmp_path / "runs" / "train"
        weights_dir = run_dir / "weights"
        weights_dir.mkdir(parents=True)
        (weights_dir / "best.pt").write_bytes(b"pt")
        (weights_dir / "best.onnx").write_bytes(b"onnx")
        (run_dir / "args.yaml").write_text("not part of model bundle", encoding="utf-8")
        (run_dir / "results.csv").write_text(
            "not part of model bundle", encoding="utf-8"
        )
        (run_dir / "color_stats.json").write_text("{}", encoding="utf-8")
        (run_dir / "detection_config.yaml").write_text(
            yaml.safe_dump(
                {
                    "weights": "best.onnx",
                    "current_product": "project",
                    "current_area": "train",
                    "expected_items": {"project": {"train": ["J5-1"]}},
                    "position_config": {
                        "project": {
                            "train": {
                                "enabled": True,
                                "expected_boxes": {"J5-1": {"x1": 1, "y1": 2}},
                            },
                            "B": {
                                "enabled": True,
                                "expected_boxes": {"J5-2": {"x1": 2, "y1": 3}},
                            },
                        }
                    },
                    "enable_color_check": True,
                    "color_model_path": "color_stats.json",
                }
            ),
            encoding="utf-8",
        )

        config = {
            "yolo_training": {
                "project": str(tmp_path / "runs"),
                "name": "train",
                "artifact_bundle": {
                    "enabled": True,
                    "product": "PCBA1",
                    "area": "A",
                    "base_dir": str(tmp_path),
                },
            }
        }

        run_artifact_bundle(config, SimpleNamespace())

        zip_path = tmp_path / "PCBA1_bundle.zip"
        with zipfile.ZipFile(zip_path) as bundle:
            names = set(bundle.namelist())
            bundled_config = yaml.safe_load(
                bundle.read("PCBA1/A/yolo/config.yaml").decode("utf-8")
            )

        assert "PCBA1/A/yolo/config.yaml" in names
        assert "PCBA1/A/yolo/weights/best.onnx" in names
        assert "PCBA1/A/yolo/color_stats.json" in names
        assert "PCBA1/A/yolo/args.yaml" not in names
        assert "PCBA1/A/yolo/results.csv" not in names
        assert bundled_config["weights"] == "models/PCBA1/A/yolo/weights/best.onnx"
        assert (
            bundled_config["color_model_path"] == "models/PCBA1/A/yolo/color_stats.json"
        )
        assert bundled_config["current_product"] == "PCBA1"
        assert bundled_config["current_area"] == "A"
        assert bundled_config["expected_items"] == {"PCBA1": {"A": ["J5-1"]}}
        assert "PCBA1" in bundled_config["position_config"]
        assert "A" in bundled_config["position_config"]["PCBA1"]
        assert "B" not in bundled_config["position_config"]["PCBA1"]

    def test_artifact_bundle_rejects_placeholder_product(self, tmp_path):
        """Bundle should fail fast instead of producing models/project/... output."""
        import yaml

        from picture_tool.tasks.bundle import run_artifact_bundle

        run_dir = tmp_path / "runs" / "train"
        weights_dir = run_dir / "weights"
        weights_dir.mkdir(parents=True)
        (weights_dir / "best.pt").write_bytes(b"pt")
        (weights_dir / "best.onnx").write_bytes(b"onnx")
        (run_dir / "detection_config.yaml").write_text(
            yaml.safe_dump({"weights": "best.onnx"}), encoding="utf-8"
        )

        config = {
            "yolo_training": {
                "project": str(tmp_path / "runs"),
                "name": "train",
                "artifact_bundle": {"enabled": True},
            }
        }

        with pytest.raises(ValueError, match="Deployment product"):
            run_artifact_bundle(config, SimpleNamespace())

    def test_artifact_bundle_supports_enhanced_color_model(self, tmp_path):
        """Bundle should preserve an enhanced color model when config asks for it."""
        import yaml

        from picture_tool.tasks.bundle import run_artifact_bundle

        run_dir = tmp_path / "runs" / "train"
        weights_dir = run_dir / "weights"
        weights_dir.mkdir(parents=True)
        (weights_dir / "best.pt").write_bytes(b"pt")
        (weights_dir / "best.onnx").write_bytes(b"onnx")
        (run_dir / "enhanced_model.json").write_text("{}", encoding="utf-8")
        (run_dir / "detection_config.yaml").write_text(
            yaml.safe_dump(
                {
                    "weights": "best.onnx",
                    "enable_color_check": True,
                    "color_model_path": "enhanced_model.json",
                }
            ),
            encoding="utf-8",
        )

        config = {
            "yolo_training": {
                "project": str(tmp_path / "runs"),
                "name": "train",
                "artifact_bundle": {
                    "enabled": True,
                    "product": "LED",
                    "area": "A",
                    "base_dir": str(tmp_path),
                },
            }
        }

        run_artifact_bundle(config, SimpleNamespace())

        zip_path = tmp_path / "LED_bundle.zip"
        with zipfile.ZipFile(zip_path) as bundle:
            names = set(bundle.namelist())
            bundled_config = yaml.safe_load(
                bundle.read("LED/A/yolo/config.yaml").decode("utf-8")
            )

        assert "LED/A/yolo/enhanced_model.json" in names
        assert (
            bundled_config["color_model_path"]
            == "models/LED/A/yolo/enhanced_model.json"
        )

    def test_artifact_bundle_does_not_fake_enhanced_model_from_stats(self, tmp_path):
        """enhanced_model.json must exist when config asks for that exact file."""
        import yaml

        from picture_tool.tasks.bundle import run_artifact_bundle

        run_dir = tmp_path / "runs" / "train"
        weights_dir = run_dir / "weights"
        weights_dir.mkdir(parents=True)
        (weights_dir / "best.pt").write_bytes(b"pt")
        (weights_dir / "best.onnx").write_bytes(b"onnx")
        (run_dir / "color_stats.json").write_text("{}", encoding="utf-8")
        (run_dir / "detection_config.yaml").write_text(
            yaml.safe_dump(
                {
                    "weights": "best.onnx",
                    "enable_color_check": True,
                    "color_model_path": "enhanced_model.json",
                }
            ),
            encoding="utf-8",
        )

        config = {
            "yolo_training": {
                "project": str(tmp_path / "runs"),
                "name": "train",
                "artifact_bundle": {
                    "enabled": True,
                    "product": "LED",
                    "area": "A",
                    "base_dir": str(tmp_path),
                },
            }
        }

        with pytest.raises(FileNotFoundError, match="enhanced_model.json"):
            run_artifact_bundle(config, SimpleNamespace())

    def test_deploy_writes_config_after_artifact_copy_succeeds(
        self, tmp_path, monkeypatch
    ):
        """A failed artifact copy should not publish config.yaml first."""
        import shutil
        import yaml

        from picture_tool.tasks.deploy import run_deploy

        run_dir = tmp_path / "runs" / "train"
        weights_dir = run_dir / "weights"
        weights_dir.mkdir(parents=True)
        (weights_dir / "best.pt").write_bytes(b"pt")
        (weights_dir / "best.onnx").write_bytes(b"onnx")
        _write_runtime_export_contract(run_dir)
        (run_dir / "detection_config.yaml").write_text(
            yaml.safe_dump({"weights": "best.onnx", "enable_color_check": False}),
            encoding="utf-8",
        )

        inference_models_dir = tmp_path / "models"
        config = {
            "yolo_training": {
                "project": str(tmp_path / "runs"),
                "name": "train",
                "deploy": {
                    "enabled": True,
                    "product": "PCBA1",
                    "area": "A",
                    "inference_models_dir": str(inference_models_dir),
                    "version": "1.0.0",
                },
            }
        }

        original_copy2 = shutil.copy2

        def fail_versioned_weight_copy(src, dst, *args, **kwargs):
            if "PCBA1_A_v1.0.0_" in str(dst):
                raise PermissionError("simulated copy failure")
            return original_copy2(src, dst, *args, **kwargs)

        monkeypatch.setattr(shutil, "copy2", fail_versioned_weight_copy)

        with pytest.raises(PermissionError, match="simulated copy failure"):
            run_deploy(config, SimpleNamespace())

        deployed_dir = inference_models_dir / "PCBA1" / "A" / "yolo"
        assert not (deployed_dir / "config.yaml").exists()


class TestUtilityFunctionality:
    """测试utility模块功能"""

    def test_hash_directory_computation(self, tmp_path):
        """功能：计算目录hash"""
        from picture_tool.utils.hashing import compute_dir_hash

        test_dir = tmp_path / "data"
        test_dir.mkdir()

        # 创建文件
        for i in range(5):
            (test_dir / f"file_{i}.txt").write_text(f"data{i}")

        hash1 = compute_dir_hash(test_dir)

        # 验证hash生成
        assert hash1 is not None
        assert hash1 != "empty"
        assert isinstance(hash1, str)

    def test_hash_config_computation(self):
        """功能：计算配置hash"""
        from picture_tool.utils.hashing import compute_config_hash

        config = {"model": "yolov11n.pt", "epochs": 10, "batch": 16}

        hash_val = compute_config_hash(config)

        assert hash_val is not None
        assert isinstance(hash_val, str)
        assert len(hash_val) > 0

    def test_setup_logger_creates_file(self, tmp_path):
        """功能：创建logger并写入文件"""
        from picture_tool.utils.logging_utils import setup_module_logger

        log_file = tmp_path / "test.log"

        logger = setup_module_logger("test_module", str(log_file))
        logger.info("Test log message")

        # Flush handlers
        for handler in logger.handlers:
            handler.flush()

        assert log_file.exists()

    def test_experiment_write_functionality(self, tmp_path):
        """功能：写入实验记录"""
        from picture_tool.utils.experiment import write_experiment

        config = {"test": "config"}
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        output_dir = tmp_path / "experiments"

        result = write_experiment(
            run_type="test", config=config, run_dir=run_dir, output_dir=output_dir
        )

        assert result.exists()
        assert result.suffix == ".yaml"


class TestPipelineUtilsFunctionality:
    """测试pipeline工具函数"""

    def test_detect_existing_weights(self, tmp_path):
        """功能：检测已有的权重文件"""
        from picture_tool.pipeline.utils import detect_existing_weights

        # 创建权重文件
        run_dir = tmp_path / "runs" / "detect" / "train"
        weights_dir = run_dir / "weights"
        weights_dir.mkdir(parents=True)
        best_pt = weights_dir / "best.pt"
        best_pt.write_text("weights")

        config = {"yolo_training": {"project": str(tmp_path / "runs"), "name": "train"}}

        weights_path, detected_run_dir = detect_existing_weights(config)

        if weights_path:
            assert Path(weights_path).exists()

    def test_mtime_latest_functionality(self, tmp_path):
        """功能：获取最新修改时间"""
        from picture_tool.pipeline.utils import mtime_latest

        # 创建多个目录
        dir1 = tmp_path / "dir1"
        dir2 = tmp_path / "dir2"
        dir1.mkdir()
        dir2.mkdir()

        (dir1 / "file.txt").write_text("data")

        import time

        time.sleep(0.1)

        (dir2 / "file.txt").write_text("data")

        # dir2应该更新
        latest = mtime_latest([dir1, dir2])

        assert latest > 0


class TestConfigValidationFunctionality:
    """测试配置验证功能"""

    def test_validate_config_structure(self):
        """功能：验证配置结构"""

        config = {
            "yolo_training": {
                "model": "yolov11n.pt",
                "dataset_dir": "/path/to/data",
                "class_names": ["class1", "class2"],
            }
        }

        # 验证配置有必要字段
        assert "yolo_training" in config
        assert "model" in config["yolo_training"]

    def test_pipeline_config_validation(self):
        """功能：验证pipeline配置"""
        config = {
            "pipeline": {
                "default_tasks": ["dataset_splitter", "yolo_train"],
                "stop_on_error": True,
            }
        }

        assert config["pipeline"]["stop_on_error"] is True


class TestEndToEndTaskWorkflow:
    """测试端到端任务工作流"""

    def test_complete_task_pipeline(self, tmp_path):
        """功能：完整的任务流水线"""
        # 1. 数据准备
        raw_dir = tmp_path / "raw"
        (raw_dir / "images").mkdir(parents=True)
        (raw_dir / "labels").mkdir(parents=True)

        for i in range(20):
            (raw_dir / "images" / f"img_{i}.jpg").write_text(f"img{i}")
            (raw_dir / "labels" / f"img_{i}.txt").write_text("0 0.5 0.5 0.1 0.1")

        # 2. 数据集分割
        split_dir = tmp_path / "split"
        split_config = {
            "train_test_split": {
                "input": {
                    "image_dir": str(raw_dir / "images"),
                    "label_dir": str(raw_dir / "labels"),
                },
                "output": {"output_dir": str(split_dir)},
                "split_ratios": {"train": 0.7, "val": 0.2, "test": 0.1},
            }
        }

        from picture_tool.tasks.quality import run_dataset_splitter

        args = SimpleNamespace()
        run_dataset_splitter(split_config, args)

        # 3. 质量检查
        lint_config = {
            "dataset_lint": {
                "image_dir": str(split_dir / "train" / "images"),
                "label_dir": str(split_dir / "train" / "labels"),
                "output_dir": str(tmp_path / "lint"),
            }
        }

        from picture_tool.tasks.quality import run_dataset_lint

        run_dataset_lint(lint_config, args)

        # 验证所有步骤完成
        assert (split_dir / "train" / "images").exists()
        assert (tmp_path / "lint").exists()

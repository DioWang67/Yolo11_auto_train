"""
功能测试：训练与评估流程
覆盖功能：YOLO训练、模型评估、权重管理
"""

import logging
import threading
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest


class TestYoloTrainingFunctionality:
    """测试YOLO训练功能"""

    def test_finetuning_profile_is_passed_to_ultralytics(self, tmp_path):
        from picture_tool.train.yolo_trainer import _parse_yolo_config

        params = _parse_yolo_config(
            {
                "yolo_training": {
                    "dataset_dir": str(tmp_path),
                    "epochs": 20,
                    "optimizer": "AdamW",
                    "lr0": 0.0005,
                    "lrf": 0.1,
                    "patience": 10,
                    "freeze": 10,
                    "mosaic": 0.0,
                    "fliplr": 0.0,
                    "close_mosaic": 0,
                    "cos_lr": True,
                    "warmup_epochs": 1.0,
                }
            }
        )

        assert params["optimizer"] == "AdamW"
        assert params["lr0"] == 0.0005
        assert params["freeze"] == 10
        assert params["mosaic"] == 0.0
        assert params["warmup_epochs"] == 1.0

    def test_training_workflow_executes(self, tmp_path, monkeypatch):
        """功能：执行完整的训练工作流"""
        from picture_tool.train.yolo_trainer import train_yolo

        # 准备数据集
        dataset_dir = tmp_path / "dataset"
        (dataset_dir / "train" / "images").mkdir(parents=True)
        (dataset_dir / "val" / "images").mkdir(parents=True)

        # 创建data.yaml
        data_yaml = dataset_dir / "data.yaml"
        data_yaml.write_text("""
train: train/images
val: val/images
names:
  0: red
  1: green
  2: blue
""")

        config = {
            "yolo_training": {
                "model": "yolov11n.pt",
                "dataset_dir": str(dataset_dir),
                "class_names": ["red", "green", "blue"],
                "epochs": 1,
                "imgsz": 640,
                "batch": 16,
                "device": "cpu",
                "project": str(tmp_path / "runs"),
                "name": "test_train",
            }
        }

        # Mock YOLO训练
        mock_yolo_class = MagicMock()
        mock_model = MagicMock()
        mock_yolo_class.return_value = mock_model

        # Mock训练结果
        mock_result = MagicMock()
        mock_result.save_dir = tmp_path / "runs" / "detect" / "test_train"
        mock_result.save_dir.mkdir(parents=True, exist_ok=True)
        (mock_result.save_dir / "weights").mkdir(exist_ok=True)
        (mock_result.save_dir / "weights" / "best.pt").write_text("weights")

        mock_model.train.return_value = mock_result

        monkeypatch.setattr("picture_tool.train.yolo_trainer.YOLO", mock_yolo_class)

        args = SimpleNamespace()
        logger = logging.getLogger("test")

        # 执行训练
        run_dir = train_yolo(config, logger=logger, args=args)

        # 验证训练执行
        assert mock_model.train.called
        assert run_dir is not None

    def test_training_continues_from_stripped_checkpoint_weights(
        self, tmp_path, monkeypatch
    ):
        from picture_tool.train.yolo_trainer import train_yolo

        dataset_dir = tmp_path / "dataset"
        (dataset_dir / "train" / "images").mkdir(parents=True)
        (dataset_dir / "val" / "images").mkdir(parents=True)
        checkpoint = tmp_path / "runs" / "train3" / "weights" / "last.pt"
        checkpoint.parent.mkdir(parents=True)
        checkpoint.write_bytes(b"checkpoint")
        (checkpoint.parent / "best.pt").write_bytes(b"best")

        mock_model = MagicMock()
        mock_model.train.return_value = SimpleNamespace(
            save_dir=checkpoint.parent.parent
        )
        model_factory = MagicMock(return_value=mock_model)
        args = SimpleNamespace(model_factory=model_factory)
        config = {
            "yolo_training": {
                "dataset_dir": str(dataset_dir),
                "class_names": ["item"],
                "epochs": 20,
                "project": str(tmp_path / "runs"),
                "name": "train",
                "resume_checkpoint": str(checkpoint),
                "resume_completed_epochs": 2,
                "resume_native": False,
            }
        }

        run_dir = train_yolo(config, args=args)

        model_factory.assert_called_once_with(str(checkpoint.resolve()))
        assert "resume" not in mock_model.train.call_args.kwargs
        assert mock_model.train.call_args.kwargs["epochs"] == 18
        assert run_dir == checkpoint.parent.parent.resolve()

    def test_training_uses_native_resume_checkpoint_when_preserved(
        self, tmp_path, monkeypatch
    ):
        from picture_tool.train.yolo_trainer import train_yolo

        dataset_dir = tmp_path / "dataset"
        (dataset_dir / "train" / "images").mkdir(parents=True)
        (dataset_dir / "val" / "images").mkdir(parents=True)
        checkpoint = tmp_path / "runs" / "train3" / "weights" / "last.resume.pt"
        checkpoint.parent.mkdir(parents=True)
        checkpoint.write_bytes(b"checkpoint")
        (checkpoint.parent / "best.pt").write_bytes(b"best")

        mock_model = MagicMock()
        mock_model.train.return_value = SimpleNamespace(
            save_dir=checkpoint.parent.parent
        )
        model_factory = MagicMock(return_value=mock_model)
        config = {
            "yolo_training": {
                "dataset_dir": str(dataset_dir),
                "class_names": ["item"],
                "epochs": 20,
                "project": str(tmp_path / "runs"),
                "name": "train",
                "resume_checkpoint": str(checkpoint),
                "resume_completed_epochs": 2,
                "resume_completed_before_run": 0,
                "resume_native": True,
            }
        }

        train_yolo(
            config,
            args=SimpleNamespace(model_factory=model_factory),
        )

        assert mock_model.train.call_args.kwargs["resume"] is True
        assert mock_model.train.call_args.kwargs["epochs"] == 20

    def test_safe_stop_raises_with_the_actual_resume_checkpoint(
        self, tmp_path, monkeypatch
    ):
        from picture_tool.train.yolo_trainer import TrainingInterrupted, train_yolo

        dataset_dir = tmp_path / "dataset"
        (dataset_dir / "train" / "images").mkdir(parents=True)
        (dataset_dir / "val" / "images").mkdir(parents=True)
        run_dir = tmp_path / "runs" / "train7"
        (run_dir / "weights").mkdir(parents=True)
        (run_dir / "weights" / "last.pt").write_bytes(b"checkpoint")

        mock_model = MagicMock()
        mock_model.train.return_value = SimpleNamespace(save_dir=run_dir)
        args = SimpleNamespace(
            model_factory=MagicMock(return_value=mock_model),
            stop_event=threading.Event(),
        )
        args.stop_event.set()
        config = {
            "yolo_training": {
                "dataset_dir": str(dataset_dir),
                "class_names": ["item"],
                "epochs": 20,
                "project": str(tmp_path / "runs"),
                "name": "train",
            }
        }

        with pytest.raises(TrainingInterrupted, match="last.pt"):
            train_yolo(config, args=args)

    def test_training_saves_weights(self, tmp_path, monkeypatch):
        """功能：训练后保存权重文件"""

        dataset_dir = tmp_path / "data"
        dataset_dir.mkdir()

        {
            "yolo_training": {
                "model": "yolov11n.pt",
                "dataset_dir": str(dataset_dir),
                "class_names": ["test"],
                "project": str(tmp_path / "runs"),
                "name": "train",
            }
        }

        # Mock训练并创建权重
        run_dir = tmp_path / "runs" / "detect" / "train"
        run_dir.mkdir(parents=True)
        weights_dir = run_dir / "weights"
        weights_dir.mkdir()
        (weights_dir / "best.pt").write_text("model_weights")
        (weights_dir / "last.pt").write_text("model_weights")

        # 验证权重文件
        assert (weights_dir / "best.pt").exists()
        assert (weights_dir / "last.pt").exists()

    def test_position_validation_prefers_auto_config_when_target_area_missing(
        self, tmp_path, monkeypatch
    ):
        """Should use run-local auto_position_config when default config lacks area."""
        import yaml

        from picture_tool.tasks import training

        run_dir = tmp_path / "runs" / "PCBA1" / "B" / "train2"
        weights_dir = run_dir / "weights"
        weights_dir.mkdir(parents=True)
        weights_path = weights_dir / "best.pt"
        weights_path.write_bytes(b"weights")
        auto_config = run_dir / "auto_position_config.yaml"
        auto_config.write_text(
            yaml.safe_dump(
                {"PCBA1": {"B": {"enabled": True, "expected_boxes": {"J2-1": {}}}}}
            ),
            encoding="utf-8",
        )

        default_config = tmp_path / "position_config.yaml"
        default_config.write_text(
            yaml.safe_dump(
                {"PCBA1": {"A": {"enabled": True, "expected_boxes": {"J5-1": {}}}}}
            ),
            encoding="utf-8",
        )

        config = {
            "yolo_training": {
                "project": str(run_dir.parent),
                "name": "train",
                "position_validation": {
                    "enabled": True,
                    "product": "PCBA1",
                    "area": "B",
                    "config": {},
                    "config_path": str(default_config),
                },
            }
        }
        captured = {}

        def fake_run_position_validation(updated_config, detected_run_dir, logger=None):
            captured["config"] = updated_config
            captured["run_dir"] = detected_run_dir
            return tmp_path / "position_validation.json"

        monkeypatch.setattr(
            training,
            "detect_existing_weights",
            lambda *_args, **_kwargs: (weights_path, run_dir),
        )
        monkeypatch.setattr(
            training, "run_position_validation", fake_run_position_validation
        )

        result = training.run_position_validation_task(config, MagicMock())

        assert result == tmp_path / "position_validation.json"
        pv_cfg = captured["config"]["yolo_training"]["position_validation"]
        assert pv_cfg["config"] == str(auto_config)
        assert captured["run_dir"] == run_dir

    def test_disabled_position_validation_is_an_explicit_noop(self, monkeypatch):
        from picture_tool.tasks import training

        run_validation = MagicMock()
        monkeypatch.setattr(training, "run_position_validation", run_validation)

        result = training.run_position_validation_task(
            {
                "yolo_training": {
                    "position_validation": {"enabled": False},
                }
            },
            MagicMock(),
        )

        assert result is None
        run_validation.assert_not_called()


class TestYoloEvaluationFunctionality:
    """测试YOLO评估功能"""

    def test_evaluation_loads_trained_weights(self, tmp_path, monkeypatch):
        """功能：加载训练好的权重进行评估"""
        from picture_tool.eval.yolo_evaluator import evaluate_yolo

        # 准备权重文件
        run_dir = tmp_path / "runs" / "detect" / "train"
        weights_dir = run_dir / "weights"
        weights_dir.mkdir(parents=True)
        weights_file = weights_dir / "best.pt"
        weights_file.write_text("weights")

        dataset_dir = tmp_path / "data"
        dataset_dir.mkdir()
        (dataset_dir / "data.yaml").write_text("dummy: data")

        config = {
            "yolo_training": {
                "dataset_dir": str(dataset_dir),
                "class_names": ["test"],
                "project": str(tmp_path / "runs"),
                "name": "train",
            },
            "yolo_evaluation": {
                "weights": str(weights_file),
                "imgsz": 640,
                "device": "cpu",
            },
        }

        # Mock评估
        mock_yolo = MagicMock()
        mock_model = MagicMock()
        mock_result = MagicMock(spec=["results_file", "results_dict"])
        mock_result.results_file = str(tmp_path / "results.csv")
        mock_result.results_dict = {}
        mock_yolo.return_value = mock_model
        mock_model.val.return_value = mock_result

        monkeypatch.setattr("picture_tool.eval.yolo_evaluator.YOLO", mock_yolo)

        # 执行评估
        evaluate_yolo(config)

        # 验证加载了正确的权重
        mock_yolo.assert_called_with(str(weights_file))
        assert mock_model.val.called

    def test_evaluation_generates_metrics(self, tmp_path, monkeypatch):
        """功能：生成评估指标"""
        from picture_tool.eval.yolo_evaluator import evaluate_yolo

        weights_file = tmp_path / "best.pt"
        weights_file.write_text("weights")
        (tmp_path / "data.yaml").write_text("dummy: data")

        config = {
            "yolo_training": {"dataset_dir": str(tmp_path), "class_names": ["test"]},
            "yolo_evaluation": {"weights": str(weights_file)},
        }

        # Mock评估结果
        mock_yolo = MagicMock()
        mock_model = MagicMock()
        mock_result = MagicMock(spec=["results_file", "results_dict"])
        mock_result.results_file = str(tmp_path / "results.csv")
        mock_result.results_dict = {
            "metrics/precision": 0.95,
            "metrics/recall": 0.92,
            "metrics/mAP50": 0.93,
        }
        mock_model.val.return_value = mock_result
        mock_yolo.return_value = mock_model

        monkeypatch.setattr("picture_tool.eval.yolo_evaluator.YOLO", mock_yolo)

        evaluate_yolo(config)

        # 验证生成了指标
        assert mock_model.val.called


class TestModelExportFunctionality:
    """测试模型导出功能"""

    def test_onnx_export_creates_file(self, tmp_path, monkeypatch):
        """功能：导出ONNX格式模型"""
        from picture_tool.utils.onnx_exporter import OnnxExporter

        run_dir = tmp_path / "run"
        run_dir.mkdir()
        weights_dir = run_dir / "weights"
        weights_dir.mkdir()
        (weights_dir / "best.pt").write_text("weights")

        config = {
            "yolo_training": {
                "export_onnx": {"enabled": True, "weights_name": "best.pt"}
            }
        }

        # Mock ONNX导出
        onnx_file = weights_dir / "best.onnx"
        onnx_file.write_text("onnx_model" * 100)

        monkeypatch.setattr("importlib.util.find_spec", lambda x: MagicMock())

        mock_yolo = MagicMock()
        mock_model = MagicMock()
        mock_model.export.return_value = str(onnx_file)
        mock_yolo.return_value = mock_model
        monkeypatch.setattr("picture_tool.utils.onnx_exporter.YOLO", mock_yolo)

        # Mock验证
        monkeypatch.setattr(
            "picture_tool.utils.onnx_validation.validate_onnx_structure", lambda x: None
        )
        monkeypatch.setattr(
            "picture_tool.utils.onnx_validation.validate_onnx_runtime",
            lambda x, **kwargs: None,
        )

        logger = logging.getLogger("test")
        result = OnnxExporter.export(config, run_dir, logger)

        # 验证导出成功
        assert result == onnx_file.resolve()


class TestTrainingWorkflowIntegration:
    """测试训练流程集成"""

    def test_complete_training_pipeline(self, tmp_path):
        """功能：完整的训练流水线"""
        # 1. 数据准备
        dataset_dir = tmp_path / "dataset"
        (dataset_dir / "train" / "images").mkdir(parents=True)
        (dataset_dir / "val" / "images").mkdir(parents=True)

        # 2. 训练配置
        config = {
            "yolo_training": {
                "model": "yolov11n.pt",
                "dataset_dir": str(dataset_dir),
                "class_names": ["obj1", "obj2"],
                "epochs": 10,
                "project": str(tmp_path / "runs"),
                "name": "exp",
            }
        }

        # 3. 验证配置正确
        assert config["yolo_training"]["model"] == "yolov11n.pt"
        assert config["yolo_training"]["epochs"] == 10

        # 4. 训练后产物
        run_dir = tmp_path / "runs" / "detect" / "exp"
        run_dir.mkdir(parents=True)
        (run_dir / "weights").mkdir()
        (run_dir / "weights" / "best.pt").write_text("trained_model")

        # 验证产物
        assert (run_dir / "weights" / "best.pt").exists()

from pathlib import Path
from types import SimpleNamespace
import zipfile

import pytest
import yaml

from picture_tool import anomalib_cli
from picture_tool.train import anomalib_trainer
from picture_tool.tasks.training import skip_anomalib_train


def _write_png(path: Path) -> None:
    path.write_bytes(
        b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01"
        b"\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde\x00"
        b"\x00\x00\x0cIDATx\x9cc\xf8\xff\xff?\x00\x05\xfe"
        b"\x02\xfeA\xe2!|\x00\x00\x00\x00IEND\xaeB`\x82"
    )


def test_parse_anomalib_training_config_accepts_normal_folder(tmp_path):
    normal_dir = tmp_path / "data" / "train" / "good"
    normal_dir.mkdir(parents=True)
    _write_png(normal_dir / "sample.png")

    cfg = anomalib_trainer.parse_anomalib_training_config(
        {
            "anomalib_training": {
                "root": str(tmp_path / "data"),
                "normal_dir": "train/good",
                "project": str(tmp_path / "runs"),
                "model": "padim",
                "image_size": 64,
            }
        }
    )

    assert cfg.normal_dir == normal_dir
    assert cfg.model == "padim"
    assert cfg.image_size == 64


def test_parse_anomalib_training_config_rejects_empty_normal_folder(tmp_path):
    normal_dir = tmp_path / "data" / "train" / "good"
    normal_dir.mkdir(parents=True)

    with pytest.raises(ValueError, match="No normal training images"):
        anomalib_trainer.parse_anomalib_training_config(
            {
                "anomalib_training": {
                    "root": str(tmp_path / "data"),
                    "normal_dir": "train/good",
                }
            }
        )


def test_parse_anomalib_training_config_requires_abnormal_when_enabled(tmp_path):
    normal_dir = tmp_path / "data" / "train" / "good"
    normal_dir.mkdir(parents=True)
    _write_png(normal_dir / "sample.png")

    with pytest.raises(ValueError, match="require_anomalous_validation"):
        anomalib_trainer.parse_anomalib_training_config(
            {
                "anomalib_training": {
                    "root": str(tmp_path / "data"),
                    "normal_dir": "train/good",
                    "require_anomalous_validation": True,
                }
            }
        )


def test_skip_anomalib_train_detects_existing_checkpoint(tmp_path):
    normal_dir = tmp_path / "data" / "train" / "good"
    normal_dir.mkdir(parents=True)
    _write_png(normal_dir / "sample.png")
    checkpoint_dir = (
        tmp_path
        / "runs"
        / "Padim"
        / "demo_anomaly"
        / "latest"
        / "weights"
        / "lightning"
    )
    checkpoint_dir.mkdir(parents=True)
    (checkpoint_dir / "model.ckpt").write_text("checkpoint", encoding="utf-8")

    reason = skip_anomalib_train(
        {
            "anomalib_training": {
                "root": str(tmp_path / "data"),
                "normal_dir": "train/good",
                "project": str(tmp_path / "runs"),
                "name": "demo_anomaly",
                "model": "padim",
            }
        },
        SimpleNamespace(force=False),
    )

    assert reason is not None
    assert "existing Anomalib checkpoint" in reason


def test_infer_anomalib_folder_layout_prefers_split_train_images(tmp_path):
    train_dir = tmp_path / "PCBA1" / "B" / "split" / "train" / "images"
    train_dir.mkdir(parents=True)
    _write_png(train_dir / "sample.png")

    layout = anomalib_trainer.infer_anomalib_folder_layout(tmp_path / "PCBA1" / "B")

    assert layout["root"] == tmp_path / "PCBA1" / "B" / "split"
    assert layout["normal_dir"] == Path("train/images")
    assert layout["abnormal_dir"] is None


def test_train_anomalib_folder_writes_baseline_report(tmp_path, monkeypatch):
    train_dir = tmp_path / "PCBA1" / "B" / "split" / "train" / "images"
    train_dir.mkdir(parents=True)
    _write_png(train_dir / "sample.png")

    def fake_train(config, logger=None):
        run_dir = tmp_path / "runs" / "Padim" / "PCBA1_B" / "latest"
        checkpoint_dir = run_dir / "weights" / "lightning"
        checkpoint_dir.mkdir(parents=True)
        (checkpoint_dir / "model.ckpt").write_text("checkpoint", encoding="utf-8")
        anomalib_trainer._write_training_metadata(
            run_dir,
            anomalib_trainer.parse_anomalib_training_config(config),
        )
        return run_dir

    monkeypatch.setattr(anomalib_trainer, "train_anomalib", fake_train)

    result = anomalib_trainer.train_anomalib_folder(
        tmp_path / "PCBA1" / "B",
        product="PCBA1",
        area="B",
        project=tmp_path / "runs",
    )

    assert result.baseline_only is True
    assert result.normal_image_count == 1
    assert result.abnormal_image_count == 0
    assert result.checkpoint_path is not None
    report = result.report_path.read_text(encoding="utf-8")
    assert '"baseline_only": true' in report


def test_lightweight_anomalib_cli_reports_result(tmp_path, monkeypatch, capsys):
    input_dir = tmp_path / "PCBA1" / "B"
    input_dir.mkdir(parents=True)
    report_path = tmp_path / "report.json"

    def fake_train_folder(**kwargs):
        assert kwargs["input_dir"] == input_dir
        assert kwargs["product"] == "PCBA1"
        assert kwargs["area"] == "B"
        return anomalib_trainer.AnomalibFolderTrainingResult(
            run_dir=tmp_path / "run",
            checkpoint_path=tmp_path / "run" / "weights" / "lightning" / "model.ckpt",
            report_path=report_path,
            baseline_only=True,
            normal_image_count=10,
            abnormal_image_count=0,
        )

    monkeypatch.setattr(anomalib_trainer, "train_anomalib_folder", fake_train_folder)
    monkeypatch.setattr(anomalib_cli, "train_anomalib_folder", fake_train_folder)

    exit_code = anomalib_cli.main(
        [
            "train-folder",
            "--input",
            str(input_dir),
            "--product",
            "PCBA1",
            "--area",
            "B",
        ],
    )

    output = capsys.readouterr().out
    assert exit_code == 0
    assert "baseline_only=true" in output
    assert "Normal images: 10" in output


def test_supported_models_cli_explains_tradeoffs(capsys):
    exit_code = anomalib_cli.main(["models"])

    output = capsys.readouterr().out
    assert exit_code == 0
    assert "padim" in output
    assert "patchcore" in output
    assert "efficientad" in output
    assert "trade_off" in output


def test_deploy_anomalib_run_writes_inference_layout(tmp_path):
    run_dir = tmp_path / "run"
    checkpoint_dir = run_dir / "weights" / "lightning"
    checkpoint_dir.mkdir(parents=True)
    (checkpoint_dir / "model.ckpt").write_text("checkpoint", encoding="utf-8")
    (run_dir / "training_report.json").write_text(
        """
{
  "baseline_only": true,
  "usable_for_deployment": false,
  "warnings": ["baseline only"],
  "config": {"model": "padim"}
}
""".strip(),
        encoding="utf-8",
    )

    result = anomalib_trainer.deploy_anomalib_run(
        run_dir,
        inference_root=tmp_path / "inference",
        product="PCBA1",
        area="B",
        threshold=0.42,
    )

    assert result.checkpoint_path.is_file()
    assert result.report_path is not None and result.report_path.is_file()
    cfg = yaml.safe_load(result.config_path.read_text(encoding="utf-8"))
    assert cfg["enable_anomalib"] is True
    assert cfg["enable_yolo"] is False
    acfg = cfg["anomalib_config"]
    assert acfg["baseline_only"] is True
    assert acfg["usable_for_deployment"] is False
    assert acfg["model"]["class_path"] == "anomalib.models.Padim"
    assert (
        acfg["models"]["PCBA1"]["B"]["ckpt_path"]
        == "models/PCBA1/B/anomalib/weights/model.ckpt"
    )
    assert acfg["models"]["PCBA1"]["B"]["threshold"] == 0.42


def test_lightweight_deploy_cli_reports_result(tmp_path, capsys):
    run_dir = tmp_path / "run"
    checkpoint_dir = run_dir / "weights" / "lightning"
    checkpoint_dir.mkdir(parents=True)
    (checkpoint_dir / "model.ckpt").write_text("checkpoint", encoding="utf-8")
    (run_dir / "training_report.json").write_text(
        '{"baseline_only": true, "usable_for_deployment": false, "config": {"model": "padim"}}',
        encoding="utf-8",
    )

    exit_code = anomalib_cli.main(
        [
            "deploy",
            "--run",
            str(run_dir),
            "--inference-root",
            str(tmp_path / "inference"),
            "--product",
            "PCBA1",
            "--area",
            "B",
        ]
    )

    output = capsys.readouterr().out
    assert exit_code == 0
    assert "Deploy directory:" in output
    assert "baseline_only: true" in output


def test_package_anomalib_run_creates_drop_in_zip(tmp_path):
    run_dir = tmp_path / "run"
    checkpoint_dir = run_dir / "weights" / "lightning"
    checkpoint_dir.mkdir(parents=True)
    (checkpoint_dir / "model.ckpt").write_text("checkpoint", encoding="utf-8")
    (run_dir / "training_report.json").write_text(
        """
{
  "baseline_only": true,
  "usable_for_deployment": false,
  "warnings": ["baseline only"],
  "config": {"model": "efficientad"}
}
""".strip(),
        encoding="utf-8",
    )

    result = anomalib_trainer.package_anomalib_run(
        run_dir,
        output_dir=tmp_path / "packages",
        product="PCBA1",
        area="B",
        threshold=0.37,
    )

    assert result.zip_path.is_file()
    assert result.checkpoint_path.is_file()
    with zipfile.ZipFile(result.zip_path) as package:
        names = set(package.namelist())
        cfg = yaml.safe_load(package.read("PCBA1/B/anomalib/config.yaml").decode("utf-8"))

    assert "PCBA1/B/anomalib/config.yaml" in names
    assert "PCBA1/B/anomalib/training_report.json" in names
    assert "PCBA1/B/anomalib/weights/model.ckpt" in names
    assert "PCBA1/B/anomalib/package_manifest.json" in names
    acfg = cfg["anomalib_config"]
    assert acfg["model"]["class_path"] == "anomalib.models.EfficientAd"
    assert acfg["models"]["PCBA1"]["B"]["ckpt_path"] == (
        "models/PCBA1/B/anomalib/weights/model.ckpt"
    )
    assert acfg["models"]["PCBA1"]["B"]["threshold"] == 0.37


def test_package_anomalib_run_force_replaces_existing_package(tmp_path):
    run_dir = tmp_path / "run"
    checkpoint_dir = run_dir / "weights" / "lightning"
    checkpoint_dir.mkdir(parents=True)
    (checkpoint_dir / "model.ckpt").write_text("new checkpoint", encoding="utf-8")
    (run_dir / "training_report.json").write_text(
        '{"baseline_only": true, "usable_for_deployment": false, "config": {"model": "padim"}}',
        encoding="utf-8",
    )
    package_dir = tmp_path / "packages" / "PCBA1_B_anomalib_padim_package"
    stale_file = package_dir / "PCBA1" / "B" / "anomalib" / "stale.txt"
    stale_file.parent.mkdir(parents=True)
    stale_file.write_text("stale", encoding="utf-8")
    (tmp_path / "packages" / "PCBA1_B_anomalib_padim_package.zip").write_text(
        "old zip",
        encoding="utf-8",
    )

    result = anomalib_trainer.package_anomalib_run(
        run_dir,
        output_dir=tmp_path / "packages",
        product="PCBA1",
        area="B",
        force=True,
    )

    assert result.zip_path.is_file()
    assert not stale_file.exists()
    assert result.checkpoint_path.read_text(encoding="utf-8") == "new checkpoint"


def test_lightweight_package_cli_reports_result(tmp_path, capsys):
    run_dir = tmp_path / "run"
    checkpoint_dir = run_dir / "weights" / "lightning"
    checkpoint_dir.mkdir(parents=True)
    (checkpoint_dir / "model.ckpt").write_text("checkpoint", encoding="utf-8")
    (run_dir / "training_report.json").write_text(
        '{"baseline_only": true, "usable_for_deployment": false, "config": {"model": "padim"}}',
        encoding="utf-8",
    )

    exit_code = anomalib_cli.main(
        [
            "package",
            "--run",
            str(run_dir),
            "--output-dir",
            str(tmp_path / "packages"),
            "--product",
            "PCBA1",
            "--area",
            "B",
        ]
    )

    output = capsys.readouterr().out
    assert exit_code == 0
    assert "Package zip:" in output
    assert "Extract under: yolo11_inference/models" in output


def test_efficientad_is_supported_model_option():
    options = {option.name: option for option in anomalib_trainer.supported_anomalib_models()}

    assert "efficientad" in options
    assert options["efficientad"].class_path == "anomalib.models.EfficientAd"


def test_train_anomalib_folder_forces_efficientad_batch_size_one(tmp_path, monkeypatch):
    train_dir = tmp_path / "PCBA1" / "B" / "split" / "train" / "images"
    train_dir.mkdir(parents=True)
    _write_png(train_dir / "sample.png")
    captured_config = {}

    def fake_train(config, logger=None):
        captured_config.update(config["anomalib_training"])
        run_dir = tmp_path / "runs" / "EfficientAd" / "PCBA1_B" / "latest"
        checkpoint_dir = run_dir / "weights" / "lightning"
        checkpoint_dir.mkdir(parents=True)
        (checkpoint_dir / "model.ckpt").write_text("checkpoint", encoding="utf-8")
        anomalib_trainer._write_training_metadata(
            run_dir,
            anomalib_trainer.parse_anomalib_training_config(config),
        )
        return run_dir

    monkeypatch.setattr(anomalib_trainer, "train_anomalib", fake_train)

    anomalib_trainer.train_anomalib_folder(
        tmp_path / "PCBA1" / "B",
        product="PCBA1",
        area="B",
        project=tmp_path / "runs",
        model="efficientad",
        batch_size=8,
        force=True,
    )

    assert captured_config["model"] == "efficientad"
    assert captured_config["train_batch_size"] == 1
    assert captured_config["eval_batch_size"] == 1

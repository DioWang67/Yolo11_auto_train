from picture_tool.gui.readiness import (
    build_project_readiness,
    count_images,
    format_readiness_preview,
)
from picture_tool.gui.main_window import (
    _operator_error_message,
    _operator_error_state,
)


def test_operator_error_message_keeps_line_leader_guidance_actionable():
    message = _operator_error_message(
        "Dataset readiness failed: split_underrepresented:test:1:required=2"
    )

    assert "資料已安全保存" in message
    assert "不會更新產線模型" in message


def test_operator_error_message_explains_quality_gate_rollback():
    message = _operator_error_message(
        "Deployment quality gate failed: recall=0.85 is below 0.90; "
        "map50 regressed from 0.95 to 0.89"
    )

    assert "現場仍使用原本模型" in message
    assert "recall=0.85" in message
    assert "map50 regressed" in message


def test_operator_error_message_explains_model_pair_and_feedback_preflight():
    message = _operator_error_message(
        "operator_training_preflight_failed: operator_feedback_not_actionable | "
        "deployed_training_pair_missing"
    )

    assert "原本就辨識正確" in message
    assert "同版本的 PT" in message
    assert "成對驗證與部署工具" in message
    assert "產線模型不會變更" in message


def test_position_preflight_shortage_waits_for_feedback_with_counts():
    raw = (
        "position_training_preflight_failed: position calibration/validation "
        "requires at least 10 eligible OK golden samples; found 0 OK and 0 NG."
    )

    message = _operator_error_message(raw)

    assert _operator_error_state(raw) == "waiting_feedback"
    assert "位置檢測補訓尚未開始" in message
    assert "OK 0 張、NG 0 張" in message
    assert "至少需要 OK 10 張" in message


def test_unexpected_pipeline_error_remains_failed():
    assert _operator_error_state("CUDA out of memory") == "failed"


def test_count_images_recurses_supported_extensions(tmp_path):
    image_dir = tmp_path / "images"
    (image_dir / "nested").mkdir(parents=True)
    (image_dir / "a.jpg").write_bytes(b"img")
    (image_dir / "nested" / "b.PNG").write_bytes(b"img")
    (image_dir / "ignore.txt").write_text("x", encoding="utf-8")

    assert count_images(image_dir) == 2


def test_build_project_readiness_reports_pcba_area_counts(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    data_root = tmp_path / "data" / "PCBA1" / "C"
    raw_images = data_root / "raw" / "images"
    raw_labels = data_root / "raw" / "labels"
    normal_dir = data_root / "train" / "good"
    abnormal_dir = data_root / "test" / "bad"
    for directory in (raw_images, raw_labels, normal_dir, abnormal_dir):
        directory.mkdir(parents=True)
    (raw_images / "a.jpg").write_bytes(b"img")
    (raw_labels / "a.txt").write_text("0 0.5 0.5 1 1", encoding="utf-8")
    (normal_dir / "good.png").write_bytes(b"img")
    (abnormal_dir / "bad.png").write_bytes(b"img")

    yolo_run = tmp_path / "runs" / "PCBA1" / "C" / "train"
    (yolo_run / "weights").mkdir(parents=True)
    (yolo_run / "weights" / "best.pt").write_bytes(b"pt")

    anomalib_run = (
        tmp_path / "runs" / "anomalib" / "PCBA1" / "C" / "EfficientAd" / "latest"
    )
    checkpoint = anomalib_run / "weights" / "lightning" / "model.ckpt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"ckpt")

    config = {
        "pipeline": {},
        "yolo_training": {"project": "runs/project", "name": "train"},
        "anomalib_training": {
            "root": "data/project/A",
            "normal_dir": "train/good",
            "abnormal_dir": "test/bad",
            "project": "runs/anomalib",
        },
        "anomalib_package": {"output_dir": "runs/anomalib_packages"},
    }

    readiness = build_project_readiness(config, "PCBA1,C")

    assert readiness.product == "PCBA1"
    assert readiness.area == "C"
    assert readiness.raw_images == 1
    assert readiness.raw_labels == 1
    assert readiness.anomalib_normal_images == 1
    assert readiness.anomalib_abnormal_images == 1
    assert readiness.latest_yolo_run == yolo_run.resolve()
    assert readiness.latest_anomalib_run == anomalib_run.resolve()
    assert readiness.is_ready_for_yolo is True
    assert readiness.is_ready_for_anomalib is True


def test_format_readiness_preview_surfaces_warnings(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    config = {"yolo_training": {}, "anomalib_training": {}, "anomalib_package": {}}

    readiness = build_project_readiness(config, "PCBA1,C")
    text = format_readiness_preview(readiness)

    assert "Target: PCBA1 / C" in text
    assert "Latest YOLO run: none" in text
    assert "Warnings:" in text

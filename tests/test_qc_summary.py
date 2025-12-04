import json

from picture_tool.report.qc_summary import generate_qc_summary


def test_qc_summary_aggregates(tmp_path):
    color_json = tmp_path / "color.json"
    color_json.write_text(json.dumps({"summary": {}, "records": [1, 2]}), encoding="utf-8")

    pos_dir = tmp_path / "pos"
    pos_dir.mkdir()
    pos_json = pos_dir / "position_validation.json"
    pos_json.write_text(
        json.dumps({"summary": {"status_counts": {"PASS": 1}, "samples": 1}}),
        encoding="utf-8",
    )

    infer_dir = tmp_path / "infer"
    infer_dir.mkdir()
    (infer_dir / "predictions.csv").write_text("image,conf\nimg,0.9\n", encoding="utf-8")

    config = {
        "color_verification": {"output_json": str(color_json)},
        "yolo_training": {"position_validation": {"output_dir": str(pos_dir)}},
        "batch_inference": {"output_dir": str(infer_dir)},
    }

    out = generate_qc_summary(config, output_path=tmp_path / "qc.json")
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["color_verification"]["exists"] is True
    assert data["color_verification"]["count"] == 2
    assert data["position_validation"]["status_counts"] == {"PASS": 1}
    assert data["detection"]["predictions"] == 1

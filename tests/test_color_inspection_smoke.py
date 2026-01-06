from types import SimpleNamespace

import pytest

from picture_tool.color import color_inspection as color_mod


@pytest.fixture()
def dummy_model():
    color_model = color_mod.EnhancedColorModel(
        avg_color_hist=[0.1] * 10,
        avg_rotation_hist=[0.1] * 10,
        hist_thr=0.5,
        rotation_hist_thr=0.5,
        mean_v_mu=0.5,
        mean_v_std=0.1,
        std_v_mu=0.1,
        std_v_std=0.02,
        uniformity_mu=0.8,
        uniformity_std=0.05,
        area_ratio_mu=0.2,
        area_ratio_std=0.01,
        hole_ratio_mu=0.05,
        hole_ratio_std=0.01,
        aspect_ratio_mu=1.0,
        aspect_ratio_std=0.1,
        compactness_mu=0.9,
        compactness_std=0.05,
        regularity_mu=0.95,
        regularity_std=0.04,
        texture_energy_mu=0.2,
        texture_energy_std=0.01,
        samples=10,
        avg_confidence=0.98,
        last_updated="2025-01-01T00:00:00Z",
    )
    model = color_mod.EnhancedReferenceModel(
        version=2,
        config={"sigma_multiplier": 2.0},
        colors={"RED": color_model},
        creation_time="2025-01-01T00:00:00Z",
        total_samples=10,
    )
    return model


def test_cmd_info(monkeypatch, tmp_path, capsys, dummy_model):
    model_path = tmp_path / "model.json"
    model_path.write_text("{}", encoding="utf-8")

    monkeypatch.setattr(
        color_mod.EnhancedReferenceModel,
        "from_json",
        staticmethod(lambda path: dummy_model),
    )

    color_mod.cmd_info(SimpleNamespace(model=str(model_path)))
    out = capsys.readouterr().out
    assert "total_samples" in out


def test_cmd_analyze_selects_directory(monkeypatch, tmp_path, dummy_model):
    called = {}

    model_path = tmp_path / "model.json"
    model_path.write_text("{}", encoding="utf-8")

    monkeypatch.setattr(
        color_mod.EnhancedReferenceModel,
        "from_json",
        staticmethod(lambda path: dummy_model),
    )

    def fake_analyze_directory(args, model):
        called["dir"] = args.dir

    monkeypatch.setattr(color_mod, "_analyze_directory", fake_analyze_directory)
    monkeypatch.setattr(color_mod, "_analyze_single", lambda args, model: None)

    args = SimpleNamespace(
        model=str(model_path),
        image=None,
        dir=str(tmp_path),
        out_dir=None,
        visualize=False,
        stability=False,
    )
    color_mod.cmd_analyze(args)
    assert called["dir"] == str(tmp_path)

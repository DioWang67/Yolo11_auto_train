import pytest
from pydantic import ValidationError
from picture_tool.tasks.quality_schemas import ColorInspectionConfig, ColorVerificationConfig

def test_color_inspection_config_valid():
    data = {
        "sam": {
            "checkpoint": "model.pt",
            "model_type": "vit_l"
        },
        "colors": ["Red"],
    }
    cfg = ColorInspectionConfig.model_validate(data)
    assert cfg.enabled is True
    assert cfg.sam.checkpoint == "model.pt"
    assert cfg.sam.model_type == "vit_l"
    assert cfg.colors == ["Red"]

def test_color_inspection_config_invalid():
    # Missing required 'sam' config
    data = {}
    with pytest.raises(ValidationError):
        ColorInspectionConfig.model_validate(data)

def test_color_verification_config_defaults():
    # Only input_dir is strictly needed as default fallback handles the rest
    data = {}
    cfg = ColorVerificationConfig.model_validate(data)
    assert cfg.enabled is True
    assert cfg.hsv_margin == (8.0, 35.0, 40.0)
    assert cfg.strip_sampling.segments == 10

def test_color_verification_config_overrides():
    data = {
        "segments": 20,
        "strip_sampling": {
            "enabled": True,
            "segments": 15 # overridden by root segments in quality.py Logic, but parsed correctly here
        }
    }
    cfg = ColorVerificationConfig.model_validate(data)
    assert cfg.segments == 20
    assert cfg.strip_sampling.segments == 15
    assert cfg.strip_sampling.enabled is True

"""Feature flag and configuration validation for autonomous training.

The first test in this file is the one that matters most: with the flag off,
every entry point must refuse before it touches anything.
"""

from __future__ import annotations

import pytest
import yaml

from picture_tool.autotrain import AutoTrainDisabledError
from picture_tool.autotrain.config import (
    DEFAULT_CONFIG_RELPATH,
    DEFAULT_MIN_GROUP_SAMPLES,
    PLANNED_SELECTORS,
    AutoTrainConfig,
    AutoTrainConfigError,
)


def _write(tmp_path, payload) -> str:
    path = tmp_path / "autonomous_training.yaml"
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return str(path)


def _enabled_payload(**overrides):
    section = {
        "enabled": True,
        "station": {"product": "Cable1", "area": "A"},
        "selectors": {"low_confidence": {"enabled": True, "conf_below": 0.55}},
    }
    section.update(overrides)
    return {"autonomous_training": section}


# --- the flag ---------------------------------------------------------------


def test_disabled_config_refuses_every_entry_point(tmp_path):
    config = AutoTrainConfig.load(_write(tmp_path, {"autonomous_training": {"enabled": False}}))

    assert config.enabled is False
    with pytest.raises(AutoTrainDisabledError):
        config.require_enabled()


def test_missing_file_is_disabled_not_an_error(tmp_path):
    config = AutoTrainConfig.load(tmp_path / "does-not-exist.yaml")

    assert config.enabled is False
    with pytest.raises(AutoTrainDisabledError):
        config.require_enabled()


def test_empty_file_is_disabled(tmp_path):
    path = tmp_path / "autonomous_training.yaml"
    path.write_text("", encoding="utf-8")

    assert AutoTrainConfig.load(path).enabled is False


def test_file_without_the_section_is_disabled(tmp_path):
    config = AutoTrainConfig.load(_write(tmp_path, {"something_else": {"enabled": True}}))

    assert config.enabled is False


def test_disabled_error_names_the_file_to_edit(tmp_path):
    path = _write(tmp_path, {"autonomous_training": {"enabled": False}})
    config = AutoTrainConfig.load(path)

    with pytest.raises(AutoTrainDisabledError, match="autonomous_training.yaml"):
        config.require_enabled()


def test_require_enabled_passes_when_switched_on(tmp_path):
    config = AutoTrainConfig.load(_write(tmp_path, _enabled_payload()))

    config.require_enabled()  # must not raise
    assert config.station.product == "Cable1"


# --- the shipped default ----------------------------------------------------


def test_shipped_default_config_is_disabled_and_valid():
    """The file checked into the repository must be inert as shipped."""
    from pathlib import Path

    import picture_tool

    project_root = Path(picture_tool.__file__).resolve().parents[2]
    shipped = project_root / DEFAULT_CONFIG_RELPATH
    assert shipped.is_file(), f"missing shipped config: {shipped}"

    config = AutoTrainConfig.load(shipped)
    assert config.enabled is False


# --- validation -------------------------------------------------------------


def test_enabled_requires_a_station(tmp_path):
    payload = _enabled_payload(station={"product": "", "area": ""})

    with pytest.raises(AutoTrainConfigError, match="station.product"):
        AutoTrainConfig.load(_write(tmp_path, payload))


def test_station_must_be_a_single_path_segment(tmp_path):
    payload = _enabled_payload(station={"product": "../escape", "area": "A"})

    with pytest.raises(AutoTrainConfigError, match="single path segment"):
        AutoTrainConfig.load(_write(tmp_path, payload))


def test_enabled_requires_at_least_one_selector(tmp_path):
    payload = _enabled_payload(selectors={"low_confidence": {"enabled": False}})

    with pytest.raises(AutoTrainConfigError, match="At least one selector"):
        AutoTrainConfig.load(_write(tmp_path, payload))


def test_unknown_selector_is_rejected_rather_than_ignored(tmp_path):
    payload = _enabled_payload(selectors={"lowconfidence": {"enabled": True}})

    with pytest.raises(AutoTrainConfigError, match="not a known selector"):
        AutoTrainConfig.load(_write(tmp_path, payload))


@pytest.mark.parametrize("name", PLANNED_SELECTORS)
def test_planned_selectors_may_be_declared_but_not_enabled(tmp_path, name):
    declared = _enabled_payload()
    declared["autonomous_training"]["selectors"][name] = {"enabled": False}
    assert AutoTrainConfig.load(_write(tmp_path, declared)).selector(name) is not None

    switched_on = _enabled_payload()
    switched_on["autonomous_training"]["selectors"][name] = {"enabled": True}
    with pytest.raises(AutoTrainConfigError, match="reserved for a later phase"):
        AutoTrainConfig.load(_write(tmp_path, switched_on))


def test_out_of_range_numbers_name_their_key(tmp_path):
    payload = _enabled_payload(collector={"lookback_days": 0})

    with pytest.raises(AutoTrainConfigError, match="collector.lookback_days"):
        AutoTrainConfig.load(_write(tmp_path, payload))


def test_booleans_must_be_booleans(tmp_path):
    payload = {"autonomous_training": {"enabled": "yes"}}

    with pytest.raises(AutoTrainConfigError, match="enabled must be true or false"):
        AutoTrainConfig.load(_write(tmp_path, payload))


def test_malformed_yaml_is_reported_with_the_path(tmp_path):
    path = tmp_path / "autonomous_training.yaml"
    path.write_text("autonomous_training: [unclosed\n", encoding="utf-8")

    with pytest.raises(AutoTrainConfigError, match="Unable to read"):
        AutoTrainConfig.load(path)


def test_non_mapping_top_level_is_rejected(tmp_path):
    path = tmp_path / "autonomous_training.yaml"
    path.write_text("- just\n- a list\n", encoding="utf-8")

    with pytest.raises(AutoTrainConfigError, match="YAML mapping"):
        AutoTrainConfig.load(path)


# --- accessors --------------------------------------------------------------


def test_selector_options_exclude_the_enabled_flag(tmp_path):
    config = AutoTrainConfig.load(_write(tmp_path, _enabled_payload()))
    selector = config.selector("low_confidence")

    assert selector is not None
    assert selector.enabled is True
    assert selector.options == {"conf_below": 0.55}


def test_enabled_selectors_skips_switched_off_ones(tmp_path):
    payload = _enabled_payload(
        selectors={
            "low_confidence": {"enabled": True},
            "random_sample": {"enabled": False},
        }
    )
    config = AutoTrainConfig.load(_write(tmp_path, payload))

    assert [s.name for s in config.enabled_selectors()] == ["low_confidence"]
    assert config.selector("random_sample") is not None
    assert config.selector("nope") is None


def test_golden_is_not_configured_when_path_is_blank(tmp_path):
    config = AutoTrainConfig.load(_write(tmp_path, _enabled_payload()))

    assert config.golden.is_configured is False


def test_golden_is_configured_when_a_path_is_given(tmp_path):
    payload = _enabled_payload(golden={"dataset_path": "D:/golden/cable1", "manifest_sha256": "ab"})
    config = AutoTrainConfig.load(_write(tmp_path, payload))

    assert config.golden.is_configured is True
    assert config.golden.manifest_sha256 == "ab"


def test_the_group_floor_has_a_default_and_is_overridable(tmp_path):
    """A group score over a handful of images reads like a measurement."""
    default = AutoTrainConfig.load(_write(tmp_path, _enabled_payload()))
    assert default.golden.min_group_samples == DEFAULT_MIN_GROUP_SAMPLES

    payload = _enabled_payload(golden={"min_group_samples": 25})
    config = AutoTrainConfig.load(_write(tmp_path, payload))

    assert config.golden.min_group_samples == 25


def test_a_group_floor_below_one_is_refused(tmp_path):
    payload = _enabled_payload(golden={"min_group_samples": 0})

    with pytest.raises(AutoTrainConfigError, match="min_group_samples"):
        AutoTrainConfig.load(_write(tmp_path, payload))


def test_promotion_defaults_match_the_existing_deployment_gate(tmp_path):
    """0.02 is the deployment gate's own max_regression; they must not drift."""
    config = AutoTrainConfig.load(_write(tmp_path, _enabled_payload()))

    assert config.promotion.max_overall_regression == pytest.approx(0.02)
    assert config.promotion.require_golden_pass is True
    assert config.promotion.max_false_negatives == 0


def test_critical_classes_accept_a_list(tmp_path):
    payload = _enabled_payload(promotion={"critical_classes": ["Red", " Orange "]})
    config = AutoTrainConfig.load(_write(tmp_path, payload))

    assert config.promotion.critical_classes == ("Red", "Orange")


def test_critical_classes_reject_a_bare_string(tmp_path):
    payload = _enabled_payload(promotion={"critical_classes": "Red"})

    with pytest.raises(AutoTrainConfigError, match="list of strings"):
        AutoTrainConfig.load(_write(tmp_path, payload))

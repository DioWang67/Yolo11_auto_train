"""The candidate registry, and reading the champion from production."""

from __future__ import annotations

import json

import pytest
import yaml

from picture_tool.autotrain.registry import (
    ARCHIVED,
    EVALUATING,
    FAILED,
    PROMOTION_CANDIDATE,
    REJECTED,
    TRAINED,
    TRAINING,
    CandidateRegistry,
    ModelRegistryError,
    read_champion,
)


@pytest.fixture()
def registry(tmp_path) -> CandidateRegistry:
    return CandidateRegistry(
        tmp_path / "models" / "candidates" / "Cable1" / "A",
        product="Cable1",
        area="A",
    )


# ---------------------------------------------------------------------------
# Registration


def test_a_candidate_is_registered_before_training_starts(registry):
    model = registry.register(
        "Cable1_A_v1.3.0-candidate",
        dataset_version="dataset_v018",
        parent_model="Cable1_A_v1.2.0",
        cycle_id="cycle_20260910_001",
        training_config={"epochs": 50},
    )

    assert model.status == TRAINING
    assert model.dataset_version == "dataset_v018"
    assert model.parent_model == "Cable1_A_v1.2.0"
    assert model.training_config == {"epochs": 50}
    assert registry.get(model.model_version).status == TRAINING


def test_a_crashed_run_still_leaves_a_record(registry):
    """Registering first is what makes a failed run explicable afterwards."""
    registry.register("c1", dataset_version="dataset_v001")

    registry.update_status("c1", FAILED, notes="ultralytics raised")

    assert registry.get("c1").status == FAILED
    assert registry.get("c1").notes == "ultralytics raised"


def test_registering_the_same_version_twice_is_refused(registry):
    registry.register("c1")

    with pytest.raises(ModelRegistryError, match="already registered"):
        registry.register("c1")


def test_an_unknown_model_is_an_error(registry):
    with pytest.raises(ModelRegistryError, match="Unknown candidate model"):
        registry.get("never-existed")


def test_the_manifest_records_everything_the_request_listed(registry, tmp_path):
    registry.register(
        "c1",
        dataset_version="dataset_v018",
        dataset_content_id="abc",
        parent_model="Cable1_A_v1.2.0",
        training_config={"epochs": 50},
    )
    registry.update_status("c1", TRAINED, training_metrics={"map50": 0.81})
    registry.update_status("c1", EVALUATING)
    registry.update_status(
        "c1",
        PROMOTION_CANDIDATE,
        evaluation_metrics={"map50": 0.81},
        promotion_decision={"decision": PROMOTION_CANDIDATE},
    )

    manifest = registry.directory_for("c1") / "manifest.json"
    payload = json.loads(manifest.read_text(encoding="utf-8"))

    for key in (
        "model_version",
        "parent_model",
        "dataset_version",
        "training_config",
        "training_metrics",
        "evaluation_metrics",
        "created_at",
        "updated_at",
        "status",
        "artifact_path",
    ):
        assert key in payload, key


# ---------------------------------------------------------------------------
# The state machine


def test_the_normal_lifecycle_is_allowed(registry):
    registry.register("c1")

    assert registry.update_status("c1", TRAINED).status == TRAINED
    assert registry.update_status("c1", EVALUATING).status == EVALUATING
    assert registry.update_status("c1", REJECTED).status == REJECTED
    assert registry.update_status("c1", ARCHIVED).status == ARCHIVED


def test_a_candidate_cannot_skip_evaluation(registry):
    """A model nobody evaluated must not be able to reach PROMOTION_CANDIDATE."""
    registry.register("c1")

    with pytest.raises(ModelRegistryError, match="Cannot move"):
        registry.update_status("c1", PROMOTION_CANDIDATE)

    registry.update_status("c1", TRAINED)
    with pytest.raises(ModelRegistryError, match="Cannot move"):
        registry.update_status("c1", PROMOTION_CANDIDATE)


def test_archived_is_terminal(registry):
    registry.register("c1")
    registry.update_status("c1", ARCHIVED)

    with pytest.raises(ModelRegistryError, match="Cannot move"):
        registry.update_status("c1", TRAINED)


def test_production_cannot_be_assigned_here(registry):
    """Only the inference project decides what production runs."""
    registry.register("c1")

    with pytest.raises(ModelRegistryError, match="not a candidate status"):
        registry.update_status("c1", "PRODUCTION")


def test_an_unknown_status_is_refused(registry):
    registry.register("c1")

    with pytest.raises(ModelRegistryError, match="not a candidate status"):
        registry.update_status("c1", "SHIPPED")


def test_re_setting_the_same_status_is_allowed(registry):
    """So a retried step is idempotent rather than an error."""
    registry.register("c1")
    registry.update_status("c1", TRAINED)

    assert registry.update_status("c1", TRAINED).status == TRAINED


# ---------------------------------------------------------------------------
# Listing


def test_models_are_listed_newest_first(registry):
    registry.register("c1")
    registry.register("c2")

    versions = [model.model_version for model in registry.list_models()]

    assert set(versions) == {"c1", "c2"}
    assert len(versions) == 2


def test_models_can_be_filtered_by_status(registry):
    registry.register("c1")
    registry.register("c2")
    registry.update_status("c2", TRAINED)

    assert [m.model_version for m in registry.by_status(TRAINED)] == ["c2"]
    assert [m.model_version for m in registry.by_status(TRAINING)] == ["c1"]


def test_an_empty_registry_lists_nothing(tmp_path):
    assert CandidateRegistry(tmp_path / "nothing").list_models() == ()


def test_an_unreadable_manifest_is_skipped_not_fatal(registry):
    registry.register("good")
    broken = registry.directory_for("broken")
    broken.mkdir(parents=True)
    (broken / "manifest.json").write_text("{oops", encoding="utf-8")

    assert [m.model_version for m in registry.list_models()] == ["good"]


# ---------------------------------------------------------------------------
# The champion, read from production


def test_the_champion_is_read_from_the_deployment_manifest(tmp_path):
    """The manifest is copied from one a real deploy wrote.

    ``deploy`` records ``deployed_file`` and ``training_weight_file`` as bare
    filenames and puts the files in the station's ``weights`` directory, so
    resolving them against the station directory alone names files that do
    not exist --- which is what a challenger would then try to train from.
    """
    model_dir = tmp_path / "models" / "Cable1" / "A" / "yolo"
    (model_dir / "weights").mkdir(parents=True)
    (model_dir / "deployment_manifest.yaml").write_text(
        yaml.safe_dump(
            {
                "deployed_version": "1.2.0",
                "deployed_file": "Cable1_A_v1.2.0_20260727.onnx",
                "weight_sha256": "abc123",
                "training_weight_file": "Cable1_A_v1.2.0_20260727.training.pt",
                "dataset_id": "dataset-abc",
                "evaluation_metrics": {"map50": 0.81, "recall": 0.9},
            }
        ),
        encoding="utf-8",
    )

    champion = read_champion(model_dir)

    assert champion.model_version == "1.2.0"
    assert champion.weight_sha256 == "abc123"
    assert champion.evaluation_metrics["map50"] == 0.81
    assert champion.status == "PRODUCTION"
    weights_dir = (model_dir / "weights").resolve()
    assert champion.weights_path == str(weights_dir / "Cable1_A_v1.2.0_20260727.onnx")
    assert champion.training_weight_path == str(
        weights_dir / "Cable1_A_v1.2.0_20260727.training.pt"
    )


def test_a_manifest_naming_a_subdirectory_is_taken_as_written(tmp_path):
    """A hand-built or older manifest may spell the directory out."""
    model_dir = tmp_path / "models" / "Cable1" / "A" / "yolo"
    model_dir.mkdir(parents=True)
    (model_dir / "deployment_manifest.yaml").write_text(
        yaml.safe_dump(
            {"deployed_version": "1.2.0", "deployed_file": "weights/best.onnx"}
        ),
        encoding="utf-8",
    )

    champion = read_champion(model_dir)

    assert champion.weights_path == str((model_dir / "weights" / "best.onnx").resolve())


def test_a_station_without_a_manifest_falls_back_to_its_config(tmp_path):
    """The station config's weights value is project-root relative.

    Unlike the manifest it spells out the whole ``models/...`` path, because
    the inference project resolves it from its own root. Joining that to the
    station directory would repeat the prefix.
    """
    model_dir = tmp_path / "models" / "Cable1" / "A" / "yolo"
    (model_dir / "weights").mkdir(parents=True)
    (model_dir / "config.yaml").write_text(
        yaml.safe_dump(
            {
                "weights": "models/Cable1/A/yolo/weights/best.pt",
                "model_version": "1.0.0",
            }
        ),
        encoding="utf-8",
    )

    champion = read_champion(model_dir)

    assert champion.model_version == "1.0.0"
    assert champion.weights_path == str(
        (model_dir / "weights" / "best.pt").resolve()
    )
    assert champion.weight_sha256 == ""


def test_a_station_with_nothing_deployed_reports_no_champion(tmp_path):
    model_dir = tmp_path / "empty"
    model_dir.mkdir()

    assert read_champion(model_dir) is None
    assert read_champion(tmp_path / "does-not-exist") is None


def test_a_corrupt_deployment_manifest_falls_back_rather_than_raising(tmp_path):
    model_dir = tmp_path / "models" / "Cable1" / "A" / "yolo"
    model_dir.mkdir(parents=True)
    (model_dir / "deployment_manifest.yaml").write_text("{[bad", encoding="utf-8")
    (model_dir / "config.yaml").write_text(
        yaml.safe_dump({"weights": "models/Cable1/A/yolo/weights/best.pt"}),
        encoding="utf-8",
    )

    champion = read_champion(model_dir)

    assert champion is not None
    assert champion.weights_path == str((model_dir / "weights" / "best.pt").resolve())


def test_reading_the_champion_never_writes_to_production(tmp_path):
    model_dir = tmp_path / "models" / "Cable1" / "A" / "yolo"
    model_dir.mkdir(parents=True)
    (model_dir / "deployment_manifest.yaml").write_text(
        yaml.safe_dump({"deployed_version": "1.2.0"}), encoding="utf-8"
    )
    before = sorted(p.name for p in model_dir.rglob("*"))

    read_champion(model_dir)

    assert sorted(p.name for p in model_dir.rglob("*")) == before

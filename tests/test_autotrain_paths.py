"""Path layout, and the guard that keeps writes out of the production project."""

from __future__ import annotations

import pytest

from picture_tool.autotrain.paths import AutoTrainPaths, ProductionWriteAttemptError
from picture_tool.workspace_paths import WorkspacePaths


@pytest.fixture()
def paths() -> AutoTrainPaths:
    # conftest routes discovery to an isolated temporary workspace.
    return AutoTrainPaths.from_workspace(WorkspacePaths.discover())


def test_everything_writable_lives_in_the_training_project(paths):
    training_project = paths.workspace.training_project

    for writable in (
        paths.root,
        paths.pool_root,
        paths.labeling_root,
        paths.datasets_root,
        paths.cycles_root,
        paths.candidates_root,
    ):
        assert writable.is_relative_to(training_project), writable


def test_candidate_models_never_land_in_the_production_models_tree(paths):
    candidate = paths.candidate_dir("Cable1", "A", "v1.3.0-candidate")

    assert not candidate.is_relative_to(paths.workspace.inference_models)
    assert not candidate.is_relative_to(paths.workspace.inference_project)
    paths.assert_not_production(candidate)  # must not raise


def test_station_scoped_paths_are_distinct(paths):
    assert paths.pool_dir("Cable1", "A") != paths.pool_dir("Cable1", "B")
    assert paths.dataset_dir("Cable1", "A", "dataset_v001") != paths.dataset_dir(
        "Cable1", "A", "dataset_v002"
    )
    assert paths.dataset_dir("Cable1", "A", "dataset_v001").is_relative_to(
        paths.dataset_station_root("Cable1", "A")
    )


def test_cycle_directories_are_named_by_cycle_id(paths):
    assert paths.cycle_dir("cycle_20260910_001").name == "cycle_20260910_001"


@pytest.mark.parametrize(
    "guarded_attribute",
    ["inference_project", "inference_models", "inference_results", "station_data"],
)
def test_guard_rejects_each_production_root(paths, guarded_attribute):
    target = getattr(paths.workspace, guarded_attribute)

    with pytest.raises(ProductionWriteAttemptError):
        paths.assert_not_production(target)
    with pytest.raises(ProductionWriteAttemptError):
        paths.assert_not_production(target / "Cable1" / "A" / "yolo")


def test_guard_rejects_a_traversal_back_into_production(paths):
    escape = (
        paths.candidates_root
        / ".."
        / ".."
        / ".."
        / "yolo11_inference"
        / "models"
        / "Cable1"
    )

    with pytest.raises(ProductionWriteAttemptError):
        paths.assert_not_production(escape)


def test_ensure_dir_creates_only_safe_directories(paths):
    created = paths.ensure_dir(paths.pool_dir("Cable1", "A"))

    assert created.is_dir()
    with pytest.raises(ProductionWriteAttemptError):
        paths.ensure_dir(paths.workspace.inference_models / "Cable1")
    assert not (paths.workspace.inference_models / "Cable1").exists()


def test_production_locations_point_at_the_inference_project(paths):
    model_dir = paths.production_model_dir("Cable1", "A")

    assert model_dir.is_relative_to(paths.workspace.inference_models)
    assert model_dir.name == "yolo"
    assert paths.production_database().name == "inspection_records.sqlite3"
    assert paths.production_results_root() == paths.workspace.inference_results

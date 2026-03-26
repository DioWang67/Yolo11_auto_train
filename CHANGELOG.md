# Changelog

All notable changes to this project will be documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

---

## [Unreleased]

### Fixed
- `yolo_train` skip logic now checks the **latest versioned run directory** (`train`, `train2`, …) instead of always checking the base `train/` directory. This prevented the skip message from correctly reflecting which run was current after force-runs.
- Changed `exist_ok=False` in YOLO trainer so each forced retrain creates a new versioned run directory instead of overwriting the previous one.

### Added
- `_find_latest_run_dir()` helper in `tasks/training.py` — finds the most recently modified run directory matching the Ultralytics version pattern (`^<name>\d*$`).
- 6 new tests in `tests/test_pipeline_skip.py` covering versioned directory detection and skip behaviour.

---

## [0.4.0] — 2026-03-10

### Added
- `deploy` task: copies training artefacts directly to yolo11_inference `models/` directory.
- `artifact_bundle` task: zips training artefacts for archival.
- `DetectionConfigExporter`: generates `detection_config.yaml` embeddable by yolo11_inference.
- Async pipeline support (`stop_event`) — GUI can cancel a running training gracefully.
- `--describe-task <name>` CLI flag to print task description and dependencies.

### Changed
- `position_validation` task dependency removed from `yolo_train` hard chain; now resolved at runtime via weight detection.

---

## [0.3.0] — 2026-02-11

### Added
- Experiment tracker integration (`get_tracker`) — logs params, metrics, artefacts to local YAML or MLflow.
- `write_experiment()` utility — writes structured experiment log after each training run.
- Hash-based skip logic (`compute_dir_hash`, `compute_config_hash`) — avoids redundant training when dataset and config are unchanged.
- `last_run_metadata.json` written to each run directory to persist hash state.
- `OnnxExporter` — auto-exports `best.onnx` after training when `export_onnx.enabled: true`.
- `PositionConfigGenerator` — auto-generates `auto_position_config.yaml` from training sample inferences.

### Changed
- `dataset_splitter` task now also generates `classes.txt` for class name auto-detection.

---

## [0.2.0] — 2026-01-15

### Added
- `color_inspection` and `color_verification` tasks for LED colour QC.
- `qc_summary` task — aggregates colour/position/inference results into a single JSON.
- `position_validation` task — offline position validation using trained weights and sample images.
- `batch_inference` task.
- `dataset_lint` and `aug_preview` tasks.
- GUI: log viewer, style manager, annotation tracker.
- `picture-tool-doctor` CLI for environment health checks.
- DVC integration via `data_sync` task.

### Changed
- Pipeline refactored to DAG-based executor (`pipeline/core.py`) with topological sort and `skip_fn` support.
- All task implementations moved to `tasks/` package.

---

## [0.1.0] — 2025-12-10

### Added
- Initial release.
- `format_conversion`, `yolo_augmentation`, `image_augmentation`, `dataset_splitter` tasks.
- `yolo_train`, `yolo_evaluation` tasks (Ultralytics YOLO11).
- `generate_report` task.
- PyQt5 GUI (`picture-tool-gui`).
- CLI entry point (`picture-tool-pipeline`).
- Pydantic config validation.

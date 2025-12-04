Picture Tool

Overview
- A structured toolkit for image pipelines: format conversion, augmentation (image/YOLO), anomaly masks, dataset splitting, and a Qt GUI orchestrator.

Package Layout
- picture_tool/
  - augment/: image and YOLO augmentation implementations
  - anomaly/: anomaly mask generation
  - format/: image format conversion
  - split/: train/val/test dataset splitter
  - pipeline/: pipeline helpers (task selection, execution)
  - utils/: IO and logging helpers
- GUI:
  - picture_tool/gui/app.py (Qt window)
  - gui/pipeline_controller.py (reusable pipeline mixin for custom frontends)
  - gui/task_thread.py (worker thread dispatch)
- Pipeline: main_pipeline.py (uses package APIs)
- Shims: top-level files import from picture_tool to keep compatibility

Usage (Python)
- from picture_tool.augment import ImageAugmentor, YoloDataAugmentor
- from picture_tool.anomaly import process_anomaly_detection
- from picture_tool.format import convert_format
- from picture_tool.split import split_dataset
- from picture_tool.pipeline import run_pipeline, load_config, setup_logging

CLI Examples
- python -m picture_tool.main_pipeline --config configs/default_pipeline.yaml --tasks image_augmentation
- picture-tool-pipeline --config configs/default_pipeline.yaml --tasks full
- python -m picture_tool.gui.app  # launch Qt GUI programmatically

Notes
- Implementations are centralized in picture_tool to improve maintainability.
- Top-level files are thin shims for backward compatibility.
- Albumentations + OpenCV operations use ThreadPool by default for Windows reliability.
- Pytest smoke suite available under `tests/` (`pytest -q`) to guard core pipeline behaviour.
- `PipelineControllerMixin` (picture_tool/gui/pipeline_controller.py) exposes the non-UI workflow logic so alternate frontends can reuse the same task orchestration.
- Detailed configuration reference: `docs/config_reference.md`.


## ?祇? API嚗icture_tool嚗?

撣貊??舐?亙? `picture_tool` ?臬嚗?雿輻摮芋蝯?

- 憓撥嚗ugment嚗?
  - `from picture_tool import ImageAugmentor, YoloDataAugmentor`
  - `from picture_tool.augment import ImageAugmentor, YoloDataAugmentor`
- ?啣虜?桃蔗嚗nomaly嚗?
  - `from picture_tool import process_anomaly_detection`
  - `from picture_tool.anomaly import process_anomaly_detection`
- ?澆?頧?嚗ormat嚗?
  - `from picture_tool import convert_format`
  - `from picture_tool.format import convert_format`
- 鞈???嚗plit嚗?
  - `from picture_tool import split_dataset`
  - `from picture_tool.split import split_dataset`
- 瘚偌蝺?Pipeline嚗?
  - `from picture_tool import run_pipeline, load_config, setup_logging`
  - `from picture_tool.pipeline import run_pipeline, load_config, setup_logging`

### Console scripts嚗? `pip install -e .`嚗?
- `picture-tool-gui`嚗???GUI
- `picture-tool-pipeline --config configs/default_pipeline.yaml --tasks image_augmentation`嚗銵?摰遙??


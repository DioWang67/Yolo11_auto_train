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
- GUI: gui.py (uses package APIs)
- Pipeline: main_pipeline.py (uses package APIs)
- Shims: top-level files import from picture_tool to keep compatibility

Usage (Python)
- from picture_tool.augment import ImageAugmentor, YoloDataAugmentor
- from picture_tool.anomaly import process_anomaly_detection
- from picture_tool.format import convert_format
- from picture_tool.split import split_dataset
- from picture_tool.pipeline import run_pipeline, load_config, setup_logging

CLI Examples
- python main_pipeline.py --config config.yaml --tasks image_augmentation
- python yolo_data_augmentor.py  # shim to package implementation
- python image_format_converter.py  # shim to package implementation

Notes
- Implementations are centralized in picture_tool to improve maintainability.
- Top-level files are thin shims for backward compatibility.
- Albumentations + OpenCV operations use ThreadPool by default for Windows reliability.


## 公開 API（picture_tool）

常用功能可直接從 `picture_tool` 匯入，或使用子模組：

- 增強（Augment）
  - `from picture_tool import ImageAugmentor, YoloDataAugmentor`
  - `from picture_tool.augment import ImageAugmentor, YoloDataAugmentor`
- 異常遮罩（Anomaly）
  - `from picture_tool import process_anomaly_detection`
  - `from picture_tool.anomaly import process_anomaly_detection`
- 格式轉換（Format）
  - `from picture_tool import convert_format`
  - `from picture_tool.format import convert_format`
- 資料切分（Split）
  - `from picture_tool import split_dataset`
  - `from picture_tool.split import split_dataset`
- 流水線（Pipeline）
  - `from picture_tool import run_pipeline, load_config, setup_logging`
  - `from picture_tool.pipeline import run_pipeline, load_config, setup_logging`

### Console scripts（需 `pip install -e .`）
- `picture-tool-gui`：啟動 GUI
- `picture-tool-pipeline --config config.yaml --tasks image_augmentation`：執行指定任務


# picture-tool

影像處理與 YOLO 自動化訓練/驗證工具，提供 CLI 與 PyQt GUI。涵蓋：格式轉換、資料增強、資料分割、資料檢查、YOLO 訓練/評估、批次推論、顏色檢測、位置驗證、報告產生等。

## 功能總覽（任務 key）
- `format_conversion`：影像格式轉換（含批次、品質設定）。
- `yolo_augmentation` / `image_augmentation`：YOLO/一般影像增強。
- `dataset_splitter`：切分 train/val/test。
- `dataset_lint` / `aug_preview`：資料品質檢查、增強預覽。
- `yolo_train`：YOLO 訓練（自動選 GPU/CPU）。
- `yolo_evaluation`：訓練後評估。
- `generate_report`：匯總報告。
- `batch_inference`：批次推論。
- `anomaly_detection`：簡易異常檢測流程。
- `color_inspection`：SAM 建模顏色範本（產生 color_stats）。
- `color_verification`：顏色批次檢測/比對，輸出 JSON/CSV。
- `position_validation`：離線位置驗證（使用已訓練權重 + sample images）。

## 任務步驟與資料需求（速查）
- `format_conversion`：輸入資料夾 + 原格式/目標格式；輸出轉檔影像（品質可調）。
- `yolo_augmentation`：需影像/標註資料夾（input.image_dir/label_dir），設定增強操作與輸出路徑；產生增強後影像/標註。
- `image_augmentation`：僅影像輸入目錄，設定增強操作與輸出路徑；產生增強影像。
- `dataset_splitter`：需原始影像/標註目錄與分割合數；輸出 train/val/test 影像與標註。
- `dataset_lint`：需影像/標註目錄；輸出 lint 報告（CSV/圖示）。
- `aug_preview`：需增強後影像/標註目錄；輸出增強預覽圖。
- `yolo_train`：需 split 資料集、class_names、初始權重；輸出 runs/detect/<name>/weights/{best,last}.pt 及 log/metrics。
- `yolo_evaluation`：需權重與驗證集；輸出評估結果與指標。
- `generate_report`：依訓練/評估結果產生報告；輸出 reports/*。
- `batch_inference`：需輸入影像資料夾與權重；輸出推論結果（影像/CSV）。
- `anomaly_detection`：需 anomaly 資料夾設定（reference/test 等）；輸出 anomaly 結果/報告。
- `color_inspection`：需 SAM checkpoint、輸入影像資料夾；輸出 color_stats.json（顏色範本）。
- `color_verification`：需 color_stats.json、輸入影像資料夾（可 expected_map）；輸出 JSON/CSV（可選 debug_plot 圖）。
- `position_validation`：需位置設定 expected_boxes、sample 圖片、訓練權重；輸出 position_validation.json。

## 常用操作範例（交接速用）
- 完整訓練→評估→報告（依預設任務）：
  ```bash
  picture-tool-pipeline --config configs/default_pipeline.yaml --tasks full
  ```
- 只訓練 YOLO（跳過增強/分割）：
  ```bash
  picture-tool-pipeline --config configs/default_pipeline.yaml --tasks yolo_train
  ```
- Training export for inference config (detection_config.yaml); includes weights/conf/iou and, if position_validation is enabled, embeds position_config.
  ```bash
  picture-tool-pipeline --config configs/default_pipeline.yaml --tasks yolo_train
  # output: runs/detect/<name>/detection_config.yaml
  ```
- 批次推論（指定輸入/輸出、權重）：
  ```bash
  picture-tool-pipeline --config configs/default_pipeline.yaml \
    --tasks batch_inference --weights runs/detect/train/weights/best.pt \
    --infer-input data/raw/images --infer-output reports/infer
  ```
- 顏色範本建立 + 批次顏色檢測：
  ```bash
  picture-tool-pipeline --config configs/default_pipeline.yaml --tasks color_inspection
  picture-tool-pipeline --config configs/default_pipeline.yaml --tasks color_verification
  ```
  或直接用 CLI：
  ```bash
  picture-tool-color-verify --input-dir data/led_qc/infer \
    --color-stats reports/led_qc/color_stats.json --output-json reports/led_qc/verify.json
  ```
- 位置驗證（需先訓練產出 weights、設定 product/area/position config）：
  ```bash
  picture-tool-pipeline --config configs/default_pipeline.yaml --tasks position_validation
  ```
  輸出：`runs/detect/<name>/position_validation/position_validation.json`（或自訂 output_dir）。

## 專案結構
```
Yolo11_auto_train/
├─ src/picture_tool/           # 主程式碼
│  ├─ gui/                     # PyQt GUI
│  ├─ pipeline/                # 任務管線與註冊
│  ├─ position/                # 位置驗證
│  ├─ color/, augment/, split/, train/, ... 等子模組
│  └─ resources/               # 內建範例設定
├─ configs/                    # 可覆蓋的設定 (default_pipeline.yaml, gui_presets.yaml)
├─ models/, data/, reports/, runs/ ...
├─ pyproject.toml, requirements-dev.txt, README.md
```

## 安裝
```bash
python -m pip install .
python -m pip install .[gui]      # 需要 PyQt5 GUI 時
```
開發環境：
```bash
python -m venv .venv
. .venv/Scripts/activate          # PowerShell
pip install -r requirements-dev.txt
pip install -e .[dev,gui]
```

## 快速開始
CLI：
```bash
picture-tool-pipeline --config configs/default_pipeline.yaml --tasks full
```
GUI：
```bash
picture-tool-gui --config configs/gui_presets.yaml
```
顏色檢測（LED）：
```bash
picture-tool-color-verify \
  --input-dir data/led_qc/infer \
  --color-stats reports/led_qc/color_stats.json \
  --expected-map reports/led_qc/expected.csv
```
## 新增功能
- 核心/CLI 拆分：任務執行集中在核心，CLI/GUI 只負責參數解析並自動處理依賴。
- 實驗紀錄：train/eval 後在 `reports/experiments/*.yaml|json` 留存 config、環境、git commit、產物路徑與 metrics。
- QC 彙總：`qc_summary` 任務（`picture-tool-pipeline --tasks qc_summary`）會把 color verification / position validation / batch inference 輸出整合成一份 JSON。
  支援 `/predict` 接口，接收圖片上傳，回傳 JSON 格式的偵測結果。

### 3. 容器化部署 (Docker Deployment)
支援標準化容器部署，適用於雲端或邊緣裝置。
- **建置映像檔**：
  ```bash
  docker build -t picture-tool .
  ```
- **啟動服務**：
  ```bash
  docker run -p 8000:8000 picture-tool
  ```
  服務啟動後，API 將在 `http://localhost:8000` 上線。

### 4. 數據版本控制 (DVC)
支援大規模數據集管理。
- **初始化**：
  ```bash
  dvc init
  ```
- **同步數據**：
  pipeline 已整合 `data_sync` 任務，執行時會自動檢查並拉取最新數據（需先配置 remote）。
  ```bash
  picture-tool-pipeline --tasks data_sync
  ```


- 健康檢查：`picture-tool-doctor --create-demo` 檢查 ffmpeg/torch/onnxruntime 等匯入並產生 `data/demo_doctor` 小型資料集，可配合 `configs/demo_doctor.yaml` 直接跑。
- 任務查詢：`picture-tool-pipeline --list-tasks` 或 `--describe-task <name>` 可查看可用任務、依賴與描述。

## 角色導向
- **我只想完整跑一輪訓練（含 split/訓練/評估/報告）**  
  `picture-tool-pipeline --config configs/default_pipeline.yaml --tasks full`  
  （若只訓練：`--tasks yolo_train`；GPU/epochs 可用 `--device 0 --epochs 50` 覆蓋）
- **我只想對現有推論圖做顏色 QC**  
  `picture-tool-color-verify --input-dir data/led_qc/infer --color-stats reports/led_qc/color_stats.json --output-json reports/led_qc/verify.json --output-csv reports/led_qc/verify.csv`  
  如需依檔名推 expected，加 `--expected-from-name`；有 mapping 就帶 `--expected-map <csv>`.
- **我只想做位置驗證**  
  確保已訓練好的權重在 `runs/detect/<name>/weights/best.pt`，config 設好 product/area：  
  position_validation:
    enabled: true
    product: Cable1          # 必填
    area: A                  # 必填
    config_path: ./models/yolo/position_config.yaml   # 或直接填 config: {...}
    sample_dir: ./data/split/val/images               # 選填，預設 dataset_dir/val/images
    weights: null            # 選填，預設 runs/detect/<name>/weights/best.pt
    output_dir: ./reports/position_validation         # 選填
    conf: 0.25               # 選填
    device: auto             # 選填
    tolerance_override: null # 選填，百分比
```
> Note: class names must match the model; with `auto_generate: true`, training will infer expected_boxes from sample images, save to `runs/detect/<name>/auto_position_config.yaml`, and detection_config export will include that position config.
2. 確認 `yolo_training.project/name` 指向已有訓練結果（預設 `runs/detect/train`）。
3. 執行：GUI 勾選「Position Validation」，或 CLI `--tasks position_validation`。
4. 輸出：`runs/detect/<name>/position_validation/position_validation.json`（或自訂 `output_dir`）。
> `enabled: false` 會直接跳過且不產出檔案。

## 顏色檢測流程
- `color_inspection`：啟動 SAM 範本建立，生成 `color_stats.json`。
- `color_verification`：使用 `color_stats.json` 對資料夾做批次顏色檢測，產出 JSON/CSV，可啟用 `debug_plot` 生成可視化。

## 測試與建置
```bash
ruff check src tests
pytest --cov=picture_tool
python -m build
```

## 授權
預設為專案內標示的 Proprietary License；若需開源，請同步更新 pyproject.toml 與 LICENSE。

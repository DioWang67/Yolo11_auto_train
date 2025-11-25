# picture-tool

picture-tool 提供一套可封裝至 PyPI 的影像處理 / YOLO 自動化工具鏈，涵蓋資料前處理、增強、資料切分、YOLO11 訓練與評估，以及 PyQt GUI 介面。所有模組共享同一個管線核心，CLI 與 GUI 可交互使用。

## 功能特色
- 任務導向的資料管線：支援資料轉檔、增強、異常遮罩、資料切分、訓練、推論與報告生成。
- YOLO11 訓練輔助：自動判斷 GPU/CPU、批次推論、評估與結果彙整。
- LED 品質檢驗模組：可自定義顏色映射與檢測容忍度，產出 CSV / JSON 報表；支援 `picture-tool-color-verify` 以既有 color_stats 自動判斷線材顏色。
- PyQt5 GUI：以 mixin 形式重複利用管線邏輯，適合客製化前端。
- src/ 佈局與型別標記 (py.typed)，利於發佈至 PyPI 並保持 IDE 體驗。

## 專案結構
`
Yolo11_auto_train/
├─ src/picture_tool/              # 發佈套件主體
│  ├─ anomaly/                    # 異常遮罩工具
│  ├─ augment/                    # 影像與 YOLO 增強
│  ├─ color/                      # LED 品質檢驗
│  ├─ format/                     # 影像格式轉換
│  ├─ gui/                        # PyQt GUI
│  ├─ pipeline/                   # 管線流程 helpers
│  ├─ split/, train/, utils/, …   # 其他模組
│  ├─ main_pipeline.py            # CLI 管線入口
│  └─ preset_config.yaml          # 預設 GUI 流程設定
├─ configs/                       # 範例設定檔 (不隨套件安裝)
├─ docs/                          # 詳細文件
├─ tests/                         # Pytest 測試
└─ pyproject.toml                 # 打包與中繼資料
`

## 安裝
`ash
python -m pip install .             # 安裝基礎套件
python -m pip install .[gui]        # 加上 PyQt5 GUI 依賴
`

開發環境建議：
`ash
python -m venv .venv
. .venv/Scripts/activate   # Windows PowerShell
pip install -r requirements-dev.txt
pip install -e .[dev,gui]
`

## 快速開始
`ash
# CLI：執行整合管線
picture-tool-pipeline --config configs/default_pipeline.yaml --tasks full

# GUI：啟動圖形介面
picture-tool-gui --config configs/gui_presets.yaml

# LED 顏色批次檢測
picture-tool-color-verify --input-dir data/led_qc/infer --color-stats reports/led_qc/color_stats.json --expected-map reports/led_qc/expected.csv
`

執行流程建議：先透過 `color_inspection.collect` (或 pipeline 的 `color_inspection` 任務) 建立 color_stats，之後在任意資料夾上執行 `picture-tool-color-verify`。工具會自動去除背景、取得主體平均色，並將結果寫入 JSON/CSV；若檔名含標準顏色或提供 `--expected-map` 對照表，即可直接標記出色差/誤判的樣本。
若需要除錯，可加上 `--debug-plot`（或在 pipeline 的 `color_verification.debug_plot` 設為 true）輸出每張圖的遮罩、色彩分佈與統計摘要。
若輸入圖已是 YOLO/ROI，可：
- 將 `mask_strategy` 設為 `full`（或 CLI `--mask-strategy full`）直接以整個 ROI 作為遮罩，避免自動遮罩蓋掉細線。
- 若輸入圖已是 YOLO/ROI，可開啟 `strip_sampling`（或 CLI `--strip-enabled`）沿線材寬度切成窄條逐一檢測 HSV/LAB 範圍，並可透過 `edge_margin`、`sat_threshold`、`val_threshold`、`min_sat_ratio`、`max_edge_ratio`、`center_sigma`、`top_k` 等參數排除邊緣雜訊、金屬高亮或低飽和區域，同時搭配 `--debug-plot` 產生 ROI/strip 視覺化圖，進一步降低誤判。
- `color_inspection.collect` 會為每次標註紀錄 HSV/LAB 的 10/90 百分位與遮罩覆蓋率，並寫入 `reports/led_qc/color_stats.json`。`picture-tool-color-verify` 會依據這些統計自動調整各顏色的信心門檻，若要檢查調整結果，可加上 `--threshold-report reports/led_qc/thresholds.json` 保存動態門檻。
- GUI 版 color inspection 會在下方顯示 `SAM: Idle / Running / Queued` 狀態，長時間推論時仍可持續操作，不必等候主執行緒回應。

也可以程式化呼叫核心 API：
`python
from picture_tool import ImageAugmentor, run_pipeline
from picture_tool.pipeline import load_config

config = load_config("configs/default_pipeline.yaml")
run_pipeline(config, tasks=["image_augmentation", "yolo_training"])
`

## 測試與品質
`ash
ruff check src tests
pytest --cov=picture_tool
python -m build
`

- 測試涵蓋關鍵模組（augment、split、pipeline 等）。
- pyproject.toml 提供 dev 與 gui extras，方便在 CI/CD 中安裝。
- picture_tool/__init__.py 對外釋出常用 API，__version__ 由套件版本自動帶入。

## 發佈提示
1. 更新 pyproject.toml 內的 ersion 與 CHANGELOG。
2. 執行 python -m build 產生 sdist / wheel。
3. 使用 	wine upload dist/* 上傳至 PyPI（或 TestPyPI）。
4. GUI 預設讀取 preset_config.yaml（隨套件一併打包）。範例設定則維持在 configs/ 供使用者修改。

## 授權
此專案預設為專有授權（License :: Other/Proprietary License），若需開源釋出請更新 pyproject.toml 與 LICENSE。

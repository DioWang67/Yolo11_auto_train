# picture-tool

picture-tool 提供一套可封裝至 PyPI 的影像處理 / YOLO 自動化工具鏈，涵蓋資料前處理、增強、資料切分、YOLO11 訓練與評估，以及 PyQt GUI 介面。所有模組共享同一個管線核心，CLI 與 GUI 可交互使用。

## 功能特色
- 任務導向的資料管線：支援資料轉檔、增強、異常遮罩、資料切分、訓練、推論與報告生成。
- YOLO11 訓練輔助：自動判斷 GPU/CPU、批次推論、評估與結果彙整。
- LED 品質檢驗模組：可自定義顏色映射與檢測容忍度，產出 CSV / JSON 報表。
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
`

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

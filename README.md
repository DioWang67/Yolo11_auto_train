# Yolo11 Auto Train & LED QC Toolkit

## 簡介

本工具整合 YOLO 資料處理/訓練流程與 LED QC 增強檢測模組，提供 CLI 與 PyQt5 GUI 兩種操作方式，協助快速完成資料前處理、模型訓練、批次推論以及 LED 組件的顏色/缺陷檢測與分析。

## 功能亮點

- **自動化流程編排**：透過 `picture_tool/Yolo11_Auto_Train_GUI.py` 整合格式轉換、資料增強、資料分割、YOLO 訓練/評估與報告產出。
- **LED QC 增強模組** (`picture_tool/picture_tool/color/led_qc_enhanced.py`)
  - 顏色色盤 (`ColorPalette`) 管理名稱、別名與色相範圍。
  - 自動正規化設定 (`normalize_color_config`) 確保 YAML/JSON 參數一致性。
  - 改良顏色判定 `_decide_color_robust`：結合色相覆蓋率與 Bhattacharyya 距離，支援白光專屬判定邏輯。
  - `enhanced_detect_one` 輸出顏色診斷、異常特徵、缺陷區域與建議。
- **CLI / GUI 雙介面**：
  - CLI 指令位於 `picture_tool/picture_tool/color/led_qc_enhanced.py` (`cmd_build`, `cmd_detect`, `cmd_detect_dir`, `cmd_analyze`)。
  - GUI 在 LED QC 面板提供建模、單張/批次檢測與分析按鈕，並支援即時日誌顯示。
- **診斷輸出**：JSON 與 CSV 會包含 `color_diagnostics`（距離/色相覆蓋率）以支援產線追蹤。

## 安裝需求

- Python 3.10+
- 主要套件：`opencv-python-headless`, `numpy`, `PyQt5`, `yaml`
- 建議使用虛擬環境：

```bash
python -m venv .venv
. .venv/Scripts/activate  # Windows PowerShell 可用: .venv\Scripts\Activate.ps1
pip install -r requirements.txt  # 若尚未建立，請依實際模組安裝
```

## 專案結構重點

```
Yolo11_auto_train/
├─ picture_tool/
│  ├─ Yolo11_Auto_Train_GUI.py        # GUI 主程式
│  ├─ gui/                            # GUI 相關邏輯模組
│  │  ├─ __init__.py
│  │  ├─ led_qc_manager.py            # LED QC 任務封裝
│  │  └─ task_thread.py               # 背景工作執行緒
│  └─ picture_tool/
│     └─ color/
│        └─ led_qc_enhanced.py        # LED QC 核心模組
├─ models/led_qc/                     # LED QC 模型與預設設定輸出
├─ reports/led_qc/                    # 檢測與分析輸出資料夾
└─ config.yaml                        # 全域工作流程設定檔
```

## 設定說明

### config.yaml - `led_qc_enhanced` 區段

```yaml
led_qc_enhanced:
  config_path: null                     # 指向自訂設定 JSON，可留空使用預設
  config_overrides: {}                  # 全域覆寫參數
  colors: [Red, Green, Blue, White]     # 色盤名稱
  color_aliases: {}                     # 別名對應
  color_conf_min_per_color: {}          # 個別顏色置信度門檻覆寫
  color_hue_range_margin: 8.0           # 色相範圍外擴餘裕 (度)
  color_hue_range_weight: 0.35          # 色相覆蓋率在顏色挑選時的權重
  color_conf_hue_balance: 0.4           # 顏色置信度中 hue coverage 比例
  build:
    ref_dir: ./data/led_qc/reference    # 參考樣本資料夾（根據檔名或父資料夾推斷顏色）
    model_out: ./models/led_qc/enhanced_model.json
    save_config: ./models/led_qc/default_config.json
  analyze:
    model: ./models/led_qc/enhanced_model.json
    image: ./data/led_qc/samples/sample.png
    out_dir: ./reports/led_qc/analyze
    visualize: true
    stability: false
  detect:
    model: ./models/led_qc/enhanced_model.json
    image: ./data/led_qc/samples/sample.png
    out_dir: ./reports/led_qc/single
    label: null
    sensitivity: 1.0
    save_annotated: true
    save_json: true
  detect_dir:
    model: ./models/led_qc/enhanced_model.json
    dir: ./data/led_qc/batch
    out_dir: ./reports/led_qc/batch
    sensitivity: 0.85
    save_annotated: true
    save_json: false
    csv_name: enhanced_summary.csv
```

> 可透過 `color_conf_min_per_color`、`color_hue_range_margin` 等參數細調顏色決策邏輯；若提供 `config_path` 則會先載入 JSON 後再套用覆寫。

### LED QC 預設值 (`models/led_qc/default_config.json`)

- `color_conf_min_per_color`, `color_hue_range_margin`, `color_hue_range_weight`, `color_conf_hue_balance` 等鍵與 `config.yaml` 對應。
- 可藉由建模流程自動輸出目前設定的快照，便於版本控管。

## 使用方式

### GUI

1. 執行 GUI：
   ```bash
   python picture_tool/Yolo11_Auto_Train_GUI.py
   ```
2. 左側面板載入 `config.yaml`，勾選欲執行的任務。
3. LED QC 區塊提供：
   - **建模**：讀取參考影像產生 `enhanced_model.json`。
   - **單張檢測**：輸入圖片與模型，輸出帶標註圖/JSON 診斷資料。
   - **批次檢測**：掃描資料夾並產出 CSV 與對應輸出。
   - **分析**：呼叫 `cmd_analyze` 生成可視化分析圖與統計。
4. 日誌視窗即時顯示狀態與錯誤，亦會擷取顏色診斷資訊（顏色名稱、信度、色相覆蓋率等）。

### CLI（位於 `picture_tool/picture_tool/color/led_qc_enhanced.py`）

```bash
# 建模
python picture_tool/picture_tool/color/led_qc_enhanced.py build --ref ./data/led_qc/reference --out ./models/led_qc/enhanced_model.json

# 單張檢測
python picture_tool/picture_tool/color/led_qc_enhanced.py detect --model ./models/led_qc/enhanced_model.json --image ./data/led_qc/samples/sample.png --out ./reports/led_qc/single

# 批次檢測
python picture_tool/picture_tool/color/led_qc_enhanced.py detect-dir --model ./models/led_qc/enhanced_model.json --dir ./data/led_qc/batch --out ./reports/led_qc/batch

# 圖像分析（含色相覆蓋統計與可視化）
python picture_tool/picture_tool/color/led_qc_enhanced.py analyze --model ./models/led_qc/enhanced_model.json --image ./data/led_qc/samples/sample.png --out ./reports/led_qc/analyze
```

> 指令均支援 `--config`/`--override` 等參數（詳見程式內的 `argparse` 定義）。

## 輸出結果

- 單張 JSON：
  - `result.color_diagnostics.distances`：各顏色與參考模型的 Bhattacharyya 距離。
  - `result.color_diagnostics.hue_coverage`：遮罩像素在顏色色相範圍內的比例。
- 批次 CSV：新增 `color_dist` 與 `color_hue_cov` 欄位，便於追蹤顏色漂移與遮罩品質。
- 標註圖：根據異常區域以綠/紅框標記。

## 建議流程

1. **準備參考資料**：依照顏色分資料夾或檔名，放入 `data/led_qc/reference/<Color>/...`。
2. **調整設定**：在 `config.yaml` 或 `default_config.json` 調整顏色相關參數，必要時設定 `color_conf_min_per_color`。
3. **執行建模**：透過 GUI/CLI 產出 `enhanced_model.json`。
4. **檢測與分析**：單張或批次檢測，查看 JSON/CSV/標註圖；使用 `analyze` 取得完整報告。
5. **迭代調整**：根據輸出的 `color_diagnostics` 調整色相範圍或置信度門檻，重新建模。

## 後續工作建議

- 撰寫單元測試針對 `_decide_color_robust` 與 `enhanced_detect_one`，確保顏色診斷邏輯在調參後仍維持穩定。
- 規劃資料蒐集與版本管控流程，將 `models/` 與 `reports/` 重要輸出納入版本記錄或備份。
- 若需要於 CI 中執行批次檢測，可將 CLI 指令封裝成腳本並搭配自動化測試資料。

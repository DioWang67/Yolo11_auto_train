# YOLO11 自動訓練與 LED QC 工具箱

這是一套整合 YOLO11 影像訓練流程與 LED 品質檢測的工具集，提供命令列與 PyQt5 圖形介面兩種操作方式，協助您完成資料前處理、模型訓練、批次推論與 LED 顏色檢測等工作。

## 核心特色

- **圖形化流程管理**：`picture_tool/Yolo11_Auto_Train_GUI.py` 支援來源轉檔、資料增強、資料切割、YOLO 訓練/評估、報表產生、批次推論等任務。左側任務面板可自訂流程，在 GUI 內直接儲存/刪除常見任務組合（流程資訊存於 `picture_tool/preset_config.yaml`）。
- **LED QC 增強模組**：位於 `picture_tool/picture_tool/color/led_qc_enhanced.py`，提供顏色定義、參數正規化、顏色決策與建模/檢測/分析等功能，可導出 JSON、CSV 與標註影像作追蹤。
- **批次推論輸出**：批次推論任務將結果寫入 `config.yaml` 中的 `batch_inference.output_dir`（預設 `./reports/infer`），便於後續檢視或自動化彙整。
- **命令列與 GUI 雙模式**：所有流程皆可透過 GUI 操作，也可直接呼叫 CLI 指令整合至 CI/CD 或自動化腳本。

## 環境需求

- Python 3.10+
- 主要套件：`opencv-python-headless`, `numpy`, `PyQt5`, `PyYAML`

```bash
python -m venv .venv
. .venv/Scripts/activate       # Windows PowerShell
pip install -r requirements.txt
```

## 專案結構速覽

```
Yolo11_auto_train/
├─ picture_tool/
│  ├─ Yolo11_Auto_Train_GUI.py        # GUI 主程式
│  ├─ preset_config.yaml              # GUI 自訂流程設定檔
│  ├─ gui/
│  │  ├─ task_thread.py               # 背景任務執行緒
│  │  └─ led_qc_manager.py            # LED 任務封裝
│  └─ picture_tool/color/led_qc_enhanced.py   # LED QC 核心模組
├─ models/led_qc/                    # LED 模型與預設參數
├─ reports/                          # 訓練、推論、LED QC 輸出
└─ config.yaml                       # Pipeline 主設定
```

## 主要設定

### Pipeline 與批次推論路徑

```yaml
batch_inference:
  input_dir: ./data/augmented/images
  output_dir: ./reports/infer   # 批次推論輸出位置
  weights: null
  imgsz: 640
  device: cpu
  conf: 0.25
```

### LED QC (`config.yaml` → `led_qc_enhanced` 節點)

- `colors` / `color_aliases`：定義可辨識顏色與別名。
- `color_conf_min_per_color`、`color_hue_range_margin` 等數值可微調顏色判定。
- `build` / `detect` / `detect_dir` / `analyze` 子節點配置建模、單張檢測、批次檢測與分析輸出的路徑。

### 自訂流程設定 (`picture_tool/preset_config.yaml`)

```yaml
presets:
  全流程示例:
    - 格式轉換
    - YOLO資料增強
    - 影像增強
    - 資料切割
    - YOLO訓練
    - YOLO評估
    - 報表生成
```

- 可直接手動編輯或透過 GUI 左側「儲存流程 / 刪除流程」按鈕操作。
- GUI 載入 `config.yaml` 後會同步讀取此檔案並更新流程下拉選單。

## 使用方式

### 啟動 GUI

```bash
python picture_tool/Yolo11_Auto_Train_GUI.py
```

1. **載入設定檔**：在左側「設定檔」區塊選擇或輸入 `config.yaml` 路徑。
2. **選擇任務**：勾選所需任務或從「流程」下拉選單套用已儲存流程。完成的流程會寫回 `preset_config.yaml` 供下次使用。
3. **LED QC 功能**：可於專屬區塊執行建模、單張檢測、批次檢測與分析，執行紀錄會顯示在右側日誌。
4. **執行 / 停止**：按「開始流程」啟動，執行過程中進度條與日誌會即時更新；可隨時按「停止流程」。

### 命令列範例（LED QC）

```bash
# 建立 LED 參考模型
python picture_tool/picture_tool/color/led_qc_enhanced.py build \
  --ref ./data/led_qc/reference \
  --out ./models/led_qc/enhanced_model.json

# 單張檢測
python picture_tool/picture_tool/color/led_qc_enhanced.py detect \
  --model ./models/led_qc/enhanced_model.json \
  --image ./data/led_qc/samples/sample.png \
  --out ./reports/led_qc/single

# 批次檢測（輸出 CSV 與標註影像）
python picture_tool/picture_tool/color/led_qc_enhanced.py detect-dir \
  --model ./models/led_qc/enhanced_model.json \
  --dir ./data/led_qc/batch \
  --out ./reports/led_qc/batch

# 統計與視覺化分析
python picture_tool/picture_tool/color/led_qc_enhanced.py analyze \
  --model ./models/led_qc/enhanced_model.json \
  --image ./data/led_qc/samples/sample.png \
  --out ./reports/led_qc/analyze
```

## 依賴套件與版本建議

| 類別 | 套件 | 建議版本 | 備註 |
| --- | --- | --- | --- |
| 核心 | Python | 3.10 以上 | 建議固定小版本，利於重現 |
| GUI | PyQt5 | 5.15.x | 與現有 UI 相容 |
| 影像處理 | opencv-python-headless | 4.8.x | 無 GUI 版，適合 Server |
| 數值運算 | numpy | 1.26.x | 與 PyTorch/Ultralytics 相容 |
| YOLO | ultralytics | 8.x | GUI 內部呼叫 YOLO11 模型 |
| 其他 | PyYAML | 最新穩定版 | 解析 config 與流程設定 |

> 如需 GPU 訓練，請另外安裝對應版本的 CUDA、cuDNN 與 PyTorch。

## 任務流程依賴示意

```
原始資料 → 格式轉換 → YOLO / 影像增強 → 資料切割 → YOLO 訓練 → YOLO 評估 → 報表產生 → 批次推論
                                         ↘———— LED QC 建模 → LED QC 檢測 / 分析
```

- **格式轉換** 會輸入 `data/raw`，輸出檔案供增強與切割。
- **YOLO 資料增強 / 影像增強** 依賴轉換後的影像，輸出 `data/augmented`。
- **資料切割** 需有影像與標註，產生 `data/split` 供訓練、評估、推論。
- **LED QC 模組** 可獨立執行，但若要與 YOLO 訓練成果比對，可同步使用報表輸出。

## 完整流程示範（建議交接時演練一次）

1. `data/raw` 內放入原始影像 (`images/`) 與 YOLO 標註 (`labels/`)。
2. 啟動 GUI，載入 `config.yaml`，左側勾選：
   - 格式轉換 → YOLO資料增強 → 影像增強 → 資料切割 → YOLO訓練 → YOLO評估 → 報表生成。
   - 若需要推論，可於訓練完成後再執行「批次推論」。
3. 執行流程，觀察右側日誌確認無錯；完成後檢查：
   - `runs/` 內的 YOLO 訓練成果與 `reports/` 內的報表。
   - 若有勾 LED QC，相對輸出會在 `reports/led_qc/*`。
4. 透過「儲存流程」輸入名稱（如「完整訓練」），確認 `picture_tool/preset_config.yaml` 出現對應條目。
5. 切換資料或部署到另一台主機時，只要補齊資料目錄、載入設定檔、選擇流程即可複製操作。

## 常見問題與排除

| 現象 | 可能原因 | 排除方式 |
| --- | --- | --- |
| GUI 顯示「請先載入設定檔」 | `config.yaml` 路徑錯誤或語法錯誤 | 重新指定路徑，並以 `python -m yaml` 驗證語法 |
| 開始流程立即失敗 | 任務需求的資料夾不存在 | 確認 `config.yaml` 中 `input_dir` / `output_dir` 皆已建立 |
| 指定 GPU 仍使用 CPU | GUI 覆寫欄位或 `config.yaml` 未設定 `device: 0` | 在 GUI 覆寫欄位輸入 `0`，或直接調整設定檔 |
| 儲存流程失敗 | `preset_config.yaml` 無寫入權限或路徑只讀 | 檢查權限，必要時刪除檔案讓 GUI 重新建立 |
| LED QC 建模報錯 | 參考樣本不足或顏色命名不一致 | 確保各顏色都有樣本，並與 `colors` 中名稱一致 |
| 批次推論無輸出 | `batch_inference.weights` 未指定或權重檔不存在 | 指定訓練產出的 `best.pt` 並確認路徑正確 |

## 快速驗證 (Smoke Test)

1. 啟動 GUI：`python picture_tool/Yolo11_Auto_Train_GUI.py`，確認可載入預設設定檔。
2. 勾選單一任務（如「格式轉換」）執行一次，是否在設定的輸出資料夾看到新檔案。
3. 於 GUI 儲存流程名稱「測試流程」，檢查 `picture_tool/preset_config.yaml` 是否新增條目。
4. 在命令列執行 `python picture_tool/picture_tool/color/led_qc_enhanced.py analyze --help`，確認 LED 模組依賴可正常載入。
5. （選用）以小型資料集跑 YOLO 訓練 1~2 epoch，確保 `runs/` 生成輸出。

若以上步驟皆成功，代表系統已完成基本部署，可交接給下一位同事使用。

## 輸出說明

- **YOLO 訓練**：結果與報表位於 `runs/`、`reports/` 等資料夾，依 Ultralytics 預設結構生成。
- **批次推論**：輸出至 `config.yaml` 中 `batch_inference.output_dir`（預設 `./reports/infer`）。
- **LED QC**：
  - 單張：產出標註影像、診斷 JSON。
  - 批次：產出標註影像與彙整 CSV，CSV 中包含顏色距離、遮罩覆蓋率等指標。

## 建議工作流程

1. **準備資料**：整理原始影像與標註，放置於 `data/raw` 或自訂路徑。
2. **設定調整**：編輯 `config.yaml` 與 `preset_config.yaml`，設定任務流程、輸出路徑與需求參數。
3. **資料前處理/增強**：透過 GUI 或 CLI 執行格式轉換、增強與資料切割。
4. **訓練與評估**：執行 YOLO 訓練、評估與報表生成，確認結果。
5. **批次推論**：使用訓練後的權重進行批次推論，輸出位置由 `batch_inference.output_dir` 控制。
6. **LED QC 檢測**：如需 LED 品質分析，先建模再進行單張或批次檢測，最後利用 `analyze` 觀察統計結果。
7. **迭代調整**：根據報表與 LED 輸出調整參數後再次執行。

## 後續延伸建議

- 為自訂流程與 LED 模組撰寫測試，確保調整參數時結果穩定。
- 將 `reports/`、`models/` 重要輸出納入版控或自動備份策略，以利回溯。
- 可將 CLI 指令包裝為 CI 工作流程，於資料更新時自動執行推論或 QC。

若需更多協助，歡迎檢查 GUI 右側的「流程設定」頁籤或參考程式中的註解說明。

# Integration Guide: Yolo11_auto_train → yolo11_inference

本指南說明如何將 `Yolo11_auto_train` 的訓練產物部署到 `yolo11_inference` 進行生產推論。

---

## 整體流程

```
Yolo11_auto_train                          yolo11_inference
─────────────────────────────────          ─────────────────────────────────
資料準備 → 增強 → 分割
        ↓
   yolo_train
        ↓
runs/detect/<name>/
  ├── weights/
  │   ├── best.pt          ──────────────→  models/<product>/<area>/yolo/best.pt
  │   └── last.pt
  ├── detection_config.yaml ─────────────→  models/<product>/<area>/yolo/config.yaml
  └── auto_position_config.yaml  ────────→  models/<product>/<area>/yolo/position_config.yaml
                                             (若啟用 position_validation)
```

---

## Step 1：訓練完成後確認產物

```bash
picture-tool-pipeline --config configs/Cable1.yaml --tasks yolo_train
```

完成後確認以下檔案存在：

```
runs/detect/Cable1/
├── weights/
│   ├── best.pt                    # 最佳權重
│   └── last.pt
├── detection_config.yaml          # 推論所需設定
├── auto_position_config.yaml      # 位置驗證設定（需啟用 position_validation）
└── last_run_metadata.json         # 訓練 hash（供 skip 邏輯使用）
```

> 若執行多次訓練，目錄名稱會自動遞增（`Cable1`、`Cable12`、`Cable13`...）。
> 請使用最新版本目錄的產物。

---

## Step 2：複製模型到 yolo11_inference

### 方法 A：手動複製

```bash
# 建立目標目錄
mkdir -p ../yolo11_inference/models/Cable1/A/yolo

# 複製權重
cp runs/detect/Cable1/weights/best.pt \
   ../yolo11_inference/models/Cable1/A/yolo/best.pt

# 複製設定（重新命名為 config.yaml）
cp runs/detect/Cable1/detection_config.yaml \
   ../yolo11_inference/models/Cable1/A/yolo/config.yaml

# （選填）位置驗證設定
cp runs/detect/Cable1/auto_position_config.yaml \
   ../yolo11_inference/models/Cable1/A/yolo/position_config.yaml
```

### 方法 B：使用 `deploy` 任務（自動部署）

在 `configs/Cable1.yaml` 設定部署目標：

```yaml
deploy:
  enabled: true
  target_dir: ../yolo11_inference/models/Cable1/A/yolo
  copy_position_config: true   # 同時複製 position_config.yaml
```

然後執行：

```bash
picture-tool-pipeline --config configs/Cable1.yaml --tasks deploy
```

---

## Step 3：確認 yolo11_inference config.yaml

`detection_config.yaml` 會自動生成，內容範例如下：

```yaml
# models/Cable1/A/yolo/config.yaml
imgsz: 640
conf_thres: 0.25
iou_thres: 0.45
device: auto
class_names:
  - Red
  - Green
  - Orange

# 若有位置驗證設定會嵌入此處
position_check:
  enabled: true
  config_path: ./position_config.yaml
```

若需調整推論門檻值（如提高 `conf_thres` 降低誤報），直接編輯此檔，**不需重新訓練**。

---

## Step 4：在 yolo11_inference 執行推論

```bash
cd ../yolo11_inference

# CLI 單次推論
python main.py --product Cable1 --area A --type yolo --image path/to/image.jpg

# CLI 互動模式
python main.py

# GUI 模式
python GUI.py
```

---

## 設定欄位對應關係

| Yolo11_auto_train (`yolo_training`) | yolo11_inference (`models/.../yolo/config.yaml`) |
|-------------------------------------|--------------------------------------------------|
| `class_names` | `class_names` |
| `imgsz` | `imgsz` |
| `name` (run name) | 目錄名稱 `models/<product>/<area>/yolo/` |
| `position_validation.product` | 目錄層級 `models/<product>/` |
| `position_validation.area` | 目錄層級 `models/<product>/<area>/` |
| `position_validation.tolerance` | `position_check.tolerance_px` |

---

## 多產品部署範例

同一個 yolo11_inference 可服務多個產品，各自放置在獨立目錄：

```
yolo11_inference/models/
├── Cable1/
│   └── A/
│       └── yolo/
│           ├── best.pt
│           ├── config.yaml
│           └── position_config.yaml
├── LED/
│   └── A/
│       └── yolo/
│           ├── best.pt
│           └── config.yaml
└── Connector/
    └── B/
        └── yolo/
            ├── best.pt
            └── config.yaml
```

---

## 常見問題

**Q: 訓練多次後，哪個 `best.pt` 應該使用？**
用 runs/detect 目錄中 mtime 最新的版本。可直接用 deploy 任務避免手動判斷。

**Q: 部署後推論結果不符預期，需要調整 conf 門檻怎麼辦？**
直接編輯 `models/<product>/<area>/yolo/config.yaml` 中的 `conf_thres`，不需重新訓練。

**Q: `auto_position_config.yaml` 為什麼不存在？**
需在訓練 config 中啟用 `position_validation.enabled: true` 且 `auto_generate: true`，訓練時會對 val 集跑推論並記錄 bbox 位置。

**Q: 可以同時部署 YOLO + Anomalib 模型嗎？**
可以。在 yolo11_inference 的 `models/<product>/<area>/` 下同時建立 `yolo/` 和 `anomalib/` 子目錄，各自放置對應設定與模型。Anomalib 模型由 `anomalib_inference_model.py` 另外訓練，不屬於本工具的產出。

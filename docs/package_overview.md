# Picture Tool：套件概覽

本頁是 `picture_tool` 的目錄、CLI 與公開 API 速查表。
模組之間怎麼互動、資料怎麼流、哪些路徑會寫進產線，請看 [ARCHITECTURE.md](ARCHITECTURE.md)。

## 目錄

```text
src/picture_tool/
├── gui/                     # PyQt5：操作員補訓與工程工作台（app.py、main_window.py、operator_handoff.py …）
├── autotrain/               # 旁掛式自動訓練：挑樣本 → 人工標註 → 候選訓練 → 評估 → 報告（永不部署）
├── bootstrap/               # 冷啟動／自動標註實驗，只有 scripts/ 會呼叫
├── pipeline/                # Task、Pipeline（DAG 排序、跳過快取、preflight）
├── tasks/                   # 20 個 pipeline 任務；deploy.py 是唯一寫入產線的任務
├── split/                   # 依來源分組切分 train/val/test，寫出 training_provenance.json
├── augment/                 # Albumentations：YOLO 標註感知增強、純影像增強
├── quality/                 # dataset lint、訓練前 readiness gate、重複／衝突分析
├── train/                   # YOLO 與 Anomalib 訓練、fine-tune recipe
├── eval/                    # YOLO 評估與評估 Gate
├── infer/                   # 批次推論
├── report/                  # 訓練報告、QC 彙總
├── position/                # 位置校正、設定產生、離線驗證、位置 Gate
├── color/                   # SAM 顏色統計、離線顏色驗證、門檻校正
├── anomaly/                 # 傳統統計式異常遮罩（與 Anomalib 無關）
├── format/                  # 影像格式轉換
├── tracking/                # MLflow／Null tracker
├── utils/                   # ONNX 匯出與驗證、detection_config、雜湊、實驗紀錄
├── libs/labelImg/           # 內嵌的上游 LabelImg 1.8.6（排除在 lint／型別檢查外）
├── resources/               # 套件內建的預設 pipeline 設定
├── main_pipeline.py         # run_pipeline：CLI、GUI、autotrain 共用的執行路徑
├── cli.py                   # Typer：run、list-tasks、describe、anomalib-train-folder
├── config_loader.py         # 載入 YAML；檔案不存在時改用 resources/default_pipeline.yaml
├── config_validation.py     # Pydantic 寬鬆驗證（只記警告）
├── path_resolver.py         # 依 --product P,A 改寫為 data/P/A/…、runs/P/A/…
├── workspace_paths.py       # 解析 ../workspace.yaml，找出推論 repo 與共用資料夾
├── runtime_pair_deployment.py   # ONNX＋PT 成對等價證明與發布
├── portable_training_package.py # 匯入推論站匯出的補訓包
├── pending_annotations.py   # 待補標工作項與全有或全無的晉升
├── operator_job.py          # 工作狀態 status.json、站點訓練鎖、心跳
├── operator_acceptance.py   # 讀取部署後的驗收摘要（唯讀）
├── dataset_manifest_lock.py # 跨行程鎖
├── dataset_inventory.py     # 唯讀盤點 data/
├── doctor.py                # 環境檢查
├── anomalib_cli.py          # Anomalib 訓練、打包、部署 CLI
├── serve.py                 # FastAPI 推論服務（只給 deploy/Dockerfile 使用；依賴在 `serve` extra）
└── exceptions.py            # PictureToolError 階層（大多未被使用，見 ARCHITECTURE §13）
```

## CLI 入口

安裝 `pip install -e .` 之後可用。現場的 conda 環境可能沒有安裝 `typer`，此時兩個 Typer 指令無法執行，請改用 bat 啟動器（見 ARCHITECTURE §2）。

| 指令 | 說明 |
|---|---|
| `picture-tool-pipeline` | Pipeline CLI（Typer）：`run`、`list-tasks`、`describe`、`anomalib-train-folder` |
| `picture-tool-gui` | 啟動 PyQt5 GUI |
| `picture-tool-autotrain` | autotrain CLI（Typer）：`run`、`cycle`、`labeling …`、`golden …`、`status` 等 |
| `picture-tool-runtime-pair` | 驗證（預設）或以 `--deploy` 發布 ONNX＋PT 成對模型 |
| `picture-tool-color-verify` | 離線 LED 顏色驗證 |
| `picture-tool-color-calibrate` | 依人工回饋產生顏色門檻建議；核准後才寫入 |
| `picture-tool-anomalib` | Anomalib：`models`、`train-folder`、`deploy`、`package` |
| `picture-tool-dataset-inventory` | 盤點 `data/` 的重複與 split 洩漏 |
| `picture-tool-doctor` | 環境健康檢查 |

```bash
# 完整流程（任務群組用 --task-groups；--tasks full 會被當成任務名稱）
picture-tool-pipeline run --config configs/default_pipeline.yaml --task-groups full

# 只跑指定任務；多個任務要重複寫旗標
picture-tool-pipeline run --config configs/default_pipeline.yaml --tasks yolo_train
picture-tool-pipeline run --config configs/default_pipeline.yaml --tasks dataset_splitter --tasks yolo_train

# 資料或設定沒變也強制重訓
picture-tool-pipeline run --config configs/default_pipeline.yaml --tasks yolo_train --force

# 指定產品／站別（路徑改寫為 data/P/A、runs/P/A）
picture-tool-pipeline run --config configs/default_pipeline.yaml --tasks yolo_train --product Cable1,A

# 列出與說明任務
picture-tool-pipeline list-tasks --config configs/default_pipeline.yaml
picture-tool-pipeline describe yolo_train
```

## 公開 API

```python
from picture_tool.augment import ImageAugmentor, YoloDataAugmentor
from picture_tool.anomaly import process_anomaly_detection
from picture_tool.format import convert_format
from picture_tool.split import split_dataset
from picture_tool.pipeline.core import Pipeline, Task
from picture_tool.main_pipeline import run_pipeline, build_task_registry
from picture_tool.config_validation import validate_config_schema
from picture_tool.train import train_yolo, train_anomalib   # 延遲載入
from picture_tool.eval.yolo_evaluator import evaluate_yolo
from picture_tool.position import run_position_validation, evaluate_position_gate
from picture_tool.workspace_paths import WorkspacePaths
```

## 備註

- 任務只在 `tasks/` 裡註冊。依賴關係只看程式碼裡的 `Task.dependencies`，YAML 裡的 `dependencies` 不會被讀取。
- `yolo_train` 每次訓練都會建立新的 run 目錄（`train`、`train2`…）。資料或設定沒變時會跳過，用 `--force` 可強制重訓。
- 設定鍵說明見 [config_reference.md](config_reference.md)；訓練到部署的完整流程見 [ARCHITECTURE.md §7](ARCHITECTURE.md)。

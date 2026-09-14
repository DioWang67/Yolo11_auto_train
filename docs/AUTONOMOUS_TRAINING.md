# 旁掛式自動訓練（Autonomous Training, Phase 1）

這是一條**與現有流程並行**的自動訓練路徑。它不取代
[`SEAMLESS_WORKFLOW.md`](SEAMLESS_WORKFLOW.md) 描述的操作員補訓閉環，也不會部署任何模型。

它做的事：讀取產線推論紀錄 → 自動挑出值得學的樣本 → 交由人工標註 →
建立不可變資料集版本 → 訓練 challenger → 與線上 champion 比較 → 依照
**確定性規則**給出建議 → 產出報告。**採用與否由人決定。**

---

## 1. 為什麼它碰不到產線

`yolo11_inference` **沒有任何一行程式 import 這個 package**。整套系統住在
`Yolo11_auto_train`，對產線只有唯讀存取。

Collector 不掛 hook、不註冊 pipeline step、不進推論執行緒。它讀的是產線**本來就會寫**
的每次檢測完整紀錄：

```
Result/<日期>/<產品>/<站別>/<狀態>/metadata/<detector>/<檔名>_config_snapshot.json
```

這份檔案已經包含 `inspection_id`、`timestamp`、`model_info`（權重、版本、
`conf_thres`、類別清單）、`detections`（類別／信心／bbox）、影像路徑、
`equipment`，以及整份 merged config（因此 `exposure_time`／`gain`／
`light_brightness` 也在內）。亮度／飽和度／模糊分數不在裡面，由本系統**離線**
從已存影像計算。

所以「collector 失敗不影響推論」不是靠 try/except 保證的 —— 是產線根本不呼叫它。

**代價（要知道）**：資料新鮮度等於 cycle 執行頻率，不是即時。且產線的保存政策
（`inspection_retention_cleanup_enabled`，PASS 影像預設 30 天）會刪圖，因此
collector 在挑中樣本的當下就**立刻把影像複製進自己的池子**。排程頻率請高於保存期。

---

## 2. 啟用與停用

設定檔：`configs/autonomous_training.yaml`

```yaml
autonomous_training:
  enabled: false        # 預設關閉
  station: {product: Cable1, area: A}
```

- `enabled: false`（預設）：所有指令在**碰任何檔案之前**直接拒絕，離開碼 `2`。
- 停用不只是旗標。產線沒有 import 這個 package，所以停用等於它在結構上不存在。
- 回退：刪掉 `data/.autotrain/` 與 `models/candidates/` 即可。產線從未依賴它們。

---

## 3. 一次完整流程

```powershell
# 0) 確認狀態（隨時可用，關閉時也可用）
picture-tool-autotrain status

# 1) 跑一輪。資料不足時會停在該步並說明原因，不會失敗
picture-tool-autotrain cycle --product Cable1 --area A
```

第一次跑通常會停在「沒有任何人工確認標註」。這是**正常且健康的狀態**，不是錯誤 ——
自動挑樣不等於自動標註（見第 5 節）。接著：

```powershell
# 2) 匯出待標註佇列（類別順序必須與線上模型一致）
picture-tool-autotrain labeling export --classes "Black,Green,Orange,Red,Yellow"

# 3) 用既有標註工具標完 <request>/images/，把 .txt 放進 <request>/labels/

# 4) 匯入。任一張不合法則整批不套用，修正後重跑即可
picture-tool-autotrain labeling import <request 路徑>

# 5) 再跑一次 cycle，這次會真的訓練
picture-tool-autotrain cycle
```

每一步也可以單獨執行、單獨重試：

```powershell
picture-tool-autotrain collect
picture-tool-autotrain select  --cycle-id <cycle_id>
picture-tool-autotrain dataset
picture-tool-autotrain cycle   --cycle-id <cycle_id>   # 從中斷處續跑
```

查看結果：

```powershell
picture-tool-autotrain registry            # champion + 所有 candidate
picture-tool-autotrain history             # 歷次 cycle 與決策
picture-tool-autotrain report <cycle_id>   # 單次報告
```

---

## 4. 報告長什麼樣

```
Training Cycle: cycle_20260910T081500Z

Champion:    Cable1_A_v1.2.0
Challenger:  Cable1_A_cycle_20260910T081500Z_candidate
Dataset:     dataset_v018

Evaluation:
  Overall:
    Precision         +0.8%  (0.851 -> 0.859)
    Recall            +1.1%  (0.902 -> 0.913)
    mAP50             +1.2%  (0.801 -> 0.813)
    mAP50-95          +0.4%  (0.551 -> 0.555)
  Per-class recall:
    Orange Recall     +3.7%  (0.880 -> 0.917)
    Red Recall        +0.1%  (0.900 -> 0.901)
    Yellow Recall     -0.2%  (0.860 -> 0.858)

Golden Dataset:  NOT_CONFIGURED
  No golden evaluation dataset is configured, so a challenger cannot be
  recommended for promotion. Register one and set golden.dataset_path.

Decision:        REJECTED
  - golden dataset status is NOT_CONFIGURED, not a pass: No golden evaluation
    dataset is configured, ...

This is a recommendation only. Nothing has been deployed: adopting a
challenger remains a named human action in the existing inspection release
flow.
```

注意上面這個例子：**每一項指標都進步了，決策仍然是 `REJECTED`** ——
因為沒有 golden dataset 就無法確認這個進步在獨立資料上成立。這是刻意的，
不是設定錯誤。

同一份內容也寫成 `report.json`（含 `"deployed": false`），供工具或未來的 agent 讀取。
兩份都在 `data/.autotrain/cycles/<cycle_id>/`。

---

## 5. 為什麼標註不能自動化

低信心與隨機抽樣的樣本**沒有真值**，模型自己的預測**不是證據**。用預測當標註送訓，
等於教模型鞏固它原本就犯的錯 —— 這正是既有操作員流程明訂
「誤報不能直接把原推理框當正確答案」的原因。

所以候選池只產生**待標註佇列**，每筆進池時一律是 `NEEDS_LABEL`。
匯入的標註使用與操作員流程**同一支** `validate_yolo_label_text` 驗證。
只有 `VERIFIED` 的樣本會進入 dataset 版本。

空標註檔是合法的「已確認負樣本」；**檔案不存在**則不是 —— 兩者必須能區分。

---

## 6. Golden dataset（預設未設定，且刻意如此）

`golden.dataset_path` 空白時狀態是 `NOT_CONFIGURED`，而
`promotion.require_golden_pass: true` 會因此**拒絕**給出 `PROMOTION_CANDIDATE`。
這是 fail-closed：「沒辦法檢查」絕不能讀成「通過了」。

本系統**不會自己挑資料當 golden set**。哪些影像可以當一個站別的評測真值，
是有長期後果的人為判斷；由腳本隨手湊出來的 golden set 比沒有更糟，因為它看起來像證據。

由人準備好目錄後：

```powershell
picture-tool-autotrain golden register D:\golden\Cable1_A --registered-by <姓名>
# 把輸出的 manifest_sha256 填進設定檔的 golden.manifest_sha256
picture-tool-autotrain golden check
```

註冊後內容被鎖定：manifest 或任何一張影像變動都會判為 `MISMATCH`；
與訓練資料重疊（以影像 SHA-256 比對，改檔名藏不住）判為 `CONTAMINATED`。
要讓 golden set 能算 detection 指標，目錄裡需要一份 `data.yaml` 與標註；
沒有時會跳過 golden 比較並記錄原因，而不是假裝跑過。

### 6.1 representative / hard_case 分組

一個整體分數答不了這個問題：challenger 在日常影像上變好、在難例上變差，
整體數字仍可能上升。所以 golden set 可以帶一份分組，評測時**分開量**。

分組來自候選報告（`scripts/golden_candidates.py` 產出的 `candidates.csv`），
註冊時一起帶進去：

```powershell
picture-tool-autotrain golden register D:\golden\Cable1_A `
  --registered-by <姓名> `
  --groups runs\golden_candidates\v1
```

對應鍵是**影像內容的 SHA-256**（報告裡的 `image_sha256` 欄），不是 `sample_id`
（那是檔名 stem）。所以從候選池挑出來複製到別處、改了檔名，分組仍然對得上。
報告涵蓋的影像通常遠多於你實際留下的，對不到的會自動捨棄；
**一張都對不到則拒絕註冊**——那代表帶錯檔案或接錯鍵，
若默默註冊成「無分組」，它看起來會像一個有分組的集合。

輸出會同時報 `groups`（各組張數）與 `ungrouped`（沒有分組標籤的張數）。
沒帶 `--groups` 就是不分組，golden 比較照舊只有整體一組。

評測端對每一組**各跑一次 champion 與 challenger**（所以一組多兩次 val）。
子集是一份影像路徑清單，golden 目錄不會被複製、搬移或寫入。
下列情況回報原因而不是給數字：

| 狀態 | 何時 |
| --- | --- |
| `INSUFFICIENT` | 該組張數低於 `golden.min_group_samples`（預設 10） |
| `NO_LABELS` | 該組沒有任何標註可定位（ultralytics 以 `images` → `labels` 路徑對應） |
| `FAILED` | 沒有 `data.yaml`、描述檔無法解析、影像不在磁碟上，或該組驗證拋錯 |

`NO_LABELS` 特別重要：版面不對時若照算，每個物件都會被算成漏檢，
然後把那個當成 recall 崩盤發佈出去。

分組數字**目前只進報告，不進 promotion 閘門**——在第一次真跑量到數字之前，
任何門檻都只是猜的。

> **golden 目錄會多出一個檔案。** 評測時 ultralytics 會自己在標註目錄旁寫下
> `labels/<split>.cache`（對 8.3.156 實測確認，不是推測）。那是衍生資料，
> 不動任何已註冊影像，`verify_content` 只雜湊影像，所以狀態仍是 `OK`。
> 但「golden 目錄逐位元不變」在真實路徑下**不成立**，用整棵樹快照比對的人會看到它。
> golden set 放在唯讀儲存上是安全的，只是 ultralytics 每次都會警告寫不進去。

要重跑這段對真實 ultralytics 的驗證（不需要正式 golden set，會自建拋棄式 fixture）：

```powershell
python scripts\autotrain_group_smoke.py --fresh > runs\group_smoke.log 2>&1
```

> 註：本站既有的 `station_data/yolo11_inference/acceptance/` 驗收集是天然的候選 ——
> 它有人工真值、不可變快照，且文件明訂不可進訓練集。但它是「整體檢測組合」層級的
> OK/NG 真值，不是 YOLO 框標註，要轉用需要另做指標對應。這是人的決定，不在本階段。

---

## 7. Promotion 規則（確定性）

`promotion.py` 是純函式，輸入是量測到的指標、golden 狀態與設定門檻。
沒有任何模型或 agent 對「新模型是不是比較好」有發言權。

要成為 `PROMOTION_CANDIDATE`，必須全部通過：

| 規則 | 設定鍵 | 預設 |
| --- | --- | --- |
| mAP50 進步不低於 | `min_map50_delta` | `0.0` |
| 任一整體指標退步不超過 | `max_overall_regression` | `0.02` |
| 關鍵類別 recall 掉落不超過 | `max_critical_class_recall_drop` | `0.005` |
| 漏檢（FN）相對 champion 增加不超過 | `max_false_negatives` | `0` |
| Golden dataset 通過 | `require_golden_pass` | `true` |

全部 fail-closed：**量不到的指標會擋下升級**，不會被略過 —— 「沒量到退步」和
「沒有退步」從這裡看是一樣的。漏檢是與 champion 相比，不是與 0 相比：
在 champion 本來就有漏檢的站別上，絕對門檻 0 會否決掉每一個可能的改善。

即使判為 `PROMOTION_CANDIDATE`，**也不會部署**。採用仍須走既有的
「模型組合驗收 → 建立檢測發布版本 → 具名啟用」流程。

---

## 8. 不可能部署的三道保證

1. 任務清單裡沒有 `deploy`。`deploy`／`artifact_bundle`／`anomalib_package`
   在送進 pipeline 前會被**硬性拒絕**（`trainer.assert_no_forbidden_tasks`），
   不是靠設定關掉 —— 這個保證不應該取決於某個 YAML 檔維持正確。
2. 產生的訓練 config 另外把 `deploy`／`export_onnx`／`artifact_bundle` 全部設為
   `enabled: false`。
3. 所有寫入路徑都經過 `AutoTrainPaths.assert_not_production()`，
   解析後只要落在 `yolo11_inference/` 之下就丟例外（含 `..` 逃逸）。

Registry 也不接受 `PRODUCTION` 狀態：線上跑什麼模型是讀產線自己的
`deployment_manifest.yaml`，本系統沒有第二份真相。

---

## 9. 目錄

```
Yolo11_auto_train/
  data/.autotrain/
    pool/<產品>/<站別>/          # 候選池（含已複製影像與已驗證標註）
    labeling/<request_id>/        # 待標註佇列
    datasets/<產品>/<站別>/dataset_vNNN/   # 不可變，含 lineage.json
    cycles/<cycle_id>/            # cycle 狀態、work/、report.md、report.json
  models/candidates/<產品>/<站別>/<版本>/  # challenger 權重與 manifest
```

Champion 永遠是 `yolo11_inference/models/<產品>/<站別>/yolo/` 指到的權重，唯讀。

---

## 10. 目前仍然需要人

- 決定哪份資料成為 golden dataset。
- 所有標註與複核。
- **所有部署決定。**
- 決定何時跑 cycle（本階段由人排程；見 `src/picture_tool/autotrain/agent/README.md`
  對下一階段的邊界說明）。

## 11. 已知限制

- Cycle 佔用 GPU，可能拖慢同機推論。用 `training.device` 指定其他裝置，或錯開排程。
- Golden 比較需要 golden 目錄內有 `data.yaml` 與標註；只有影像時會跳過該比較。
- 分組評測的代價是每組多兩次 val（一組 champion、一組 challenger）。兩組就是
  golden 段從 2 次變 6 次。要省，就不要帶 `--groups`。
- 分組指標只出現在報告裡，promotion 閘門還沒用它。
- 四個保留 selector（`model_disagreement`、`class_imbalance`、`embedding_novelty`、
  `distribution_drift`）已在設定與註冊表中預留，但**啟用會被拒絕**，各自缺什麼寫在
  `selectors/_planned.py`。

# 自動訓練 Phase 1 — 接手紀錄（2026-09-10）

這份是**接手用**的工作紀錄，不是操作手冊。要知道這套東西怎麼用，看
[`AUTONOMOUS_TRAINING.md`](AUTONOMOUS_TRAINING.md)；要知道**做到哪、下一步做什麼、
有哪些坑**，看這裡。

---

## 0. 先確認你在正確的狀態

```powershell
cd D:\Git\robotlearning\yolo11_workspace

git -C Yolo11_auto_train branch --show-current
#   應為 feature/autonomous-training-phase1

git -C yolo11_inference branch --show-current
#   應為 fix/restore-color-verification-and-duplicate-suppression
```

### 這個工作樹裡有**兩批互不相干**的工作

`git add -A` 會把它們黏成一坨，**不要這樣做**。

| 位置 | 內容 | 狀態 |
| --- | --- | --- |
| `Yolo11_auto_train` | `autotrain/` 全套 + 2 個既有檔案的小改動 | **已提交**（2026-09-11，見 §3(1)），未 push |
| `yolo11_inference` | golden sample 開線檢查（8 個未追蹤檔 + 5 個修改） | **仍未提交**，先前既有的工作，本次完全沒碰 |

`yolo11_inference` 的 `git status` 在本次工作開始前與結束後**逐字相同**，
其 2247 個測試也全數通過。那批工作請獨立處理。

---

## 1. 做完了什麼

在 `Yolo11_auto_train` 新增 `src/picture_tool/autotrain/`，一條與產線並行、
可完全停用、不會部署的自動訓練路徑。

| 模組 | 狀態 | 備註 |
| --- | --- | --- |
| `config.py` / `paths.py` | 完成 | feature flag、路徑守衛（寫入落進 `yolo11_inference/` 會丟例外） |
| `collector.py` / `image_quality.py` | 完成 | 唯讀掃 `*_config_snapshot.json` + 唯讀開 sqlite |
| `candidate_pool.py` / `selectors/` | 完成 | 3 個實作 selector，4 個保留 selector 只註冊不實作 |
| `labeling.py` | 完成 | 待標註佇列 export/import，all-or-nothing |
| `dataset_versions.py` | 完成 | `dataset_vNNN` 不可變 + lineage + `verify()` |
| `golden.py` | 完成 | 註冊／鎖定／防污染。**尚未挑資料**（刻意） |
| `trainer.py` | 完成 | 呼叫既有 `run_pipeline`，任務清單無 `deploy` |
| `evaluator.py` / `metrics.py` | 完成 | champion vs challenger 同 split，含 per-class |
| `registry.py` / `promotion.py` | 完成 | 候選狀態機；確定性決策 |
| `orchestrator.py` / `reports.py` | 完成 | 可單步、可續跑、可失敗 |
| `service.py` / `cli.py` | 完成 | 11 個 agent-facing 函式；typer CLI |
| `agent/` | 骨架 | 只有 README 寫下邊界，無實作（刻意） |

### 只改了兩個既有檔案

1. `pyproject.toml` — 加 `picture-tool-autotrain` console script。
2. `dataset_manifest_lock.py` — 加 `autotrain_store_lock()`。既有三個 wrapper 都把鎖檔
   放進 `.operator_handoff/`，自動訓練不該在操作員流程的地盤建檔案。

**`yolo11_inference` 零修改**，這是刻意的設計結果，不是還沒做。

---

## 2. 驗證到哪（2026-09-10）

提交前於 2026-09-11 重跑確認，training 端數字與下表逐項相同
（1268 passed / 6 skipped、ruff 與 mypy 全綠、coverage 82.08%）。

| 項目 | 結果 |
| --- | --- |
| training 全套 | 1268 passed / 6 skipped |
| training ruff / mypy | 全綠（mypy 是 blocking gate） |
| training coverage | 82.08%（gate 80） |
| inference 全套 | 2247 passed / 6 skipped，工作樹逐檔未變 |
| workspace | 31 passed、`validate_workspace.py` 通過 |
| 顏色 conformance | 19 agreed + 1 recorded divergence |

重跑指令：

```powershell
cd Yolo11_auto_train
python -m pytest -q
python -m ruff check src tests
python -m mypy src/picture_tool

cd ..\yolo11_inference
python -m pytest -q --basetemp=D:\tmp\pytest-autotrain

cd ..
python -m pytest -q tests
python scripts\validate_workspace.py --root .
python scripts\generate_color_conformance.py --check
```

### 沒有驗證的事

- ~~從來沒有跑過一次真的 YOLO 訓練。~~ **2026-09-11 跑過了**，見 §3(2)。
  改用 `python scripts/autotrain_smoke.py --fresh` 重跑。
- **CLI 本機一次都沒執行過。** `typer` 在本機三個 python 環境都沒安裝
  （系統 3.11、`anomalib_env`、`yolo_anomalib`），所以 `test_autotrain_cli.py`
  本機一律 skip。CI 的 `requirements-dev.txt` 有 `typer==0.21.0`，在那裡才會跑。

---

## 3. 下一步，依價值排序

### ~~(1) 先把工作提交~~ —— 已完成（2026-09-11）

已提交，未 push（沿用本 workspace 政策）。

| commit | 內容 |
| --- | --- |
| `5f0eed7` | `feat(lock): add an autotrain store lock` |
| `0d6fda5` | `feat(autotrain): add the parallel autonomous training path` |
| `c683afc` | `chore(autotrain): register the console script` |

workspace 端 `bd81783` 只推進 `Yolo11_auto_train` 一個 pointer（不是慣例的
「advance both」——因為本次對 `yolo11_inference` 零修改，它沒有東西可推進），
另有 `8e01450` 把 workspace 根目錄的 `.coverage` 加進 `.gitignore`。

**順序不可對調。** `autotrain/` 會 `import autotrain_store_lock`，所以鎖必須排在
package 之前，否則中間那個 commit 自己是壞的；console script 指向
`picture_tool.autotrain.cli:app`，必須排在 package 之後。這三個 commit 各自都能單獨
跑測試（已確認 `test_autotrain_cli.py` 是直接 import typer 的 `app` 物件、
不經過安裝的 console script，也沒有任何測試斷言 entry point 清單）。

`yolo11_inference` 的 golden sample 那批**仍未提交**，本次同樣沒碰。

### ~~(2) 跑一次真的訓練~~ —— 已完成（2026-09-11），挖出三個缺陷

用 `scripts/autotrain_smoke.py` 跑通了：真實 operator handoff job 的 46 張已標註影像
→ 增生 → 切分 → CPU 1 epoch，從真實 champion 續訓，產出 challenger。

**假 runner 測試抓不到、只有真跑才會出現的三個缺陷**（都已修）：

| commit | 缺陷 |
| --- | --- |
| `3a77dab` | 版本庫的唯讀位元經 `copy2` 傳染給工作副本，**續跑必定 `PermissionError`**，還會蓋掉原始失敗原因 |
| `7ea5985` | `read_champion()` 三處路徑／欄位名與 `deploy.py` 實際寫出的不符，**真實站別上從來無法開始訓練** |
| `6e57de5` | `build_candidate_config()` 留下失效的 `position_validation.sample_dir`，每個任務都吐 schema 警告 |

`3a109ed` 另外補上 `TrainingResult.trained_this_run`。

**原先列的三個疑點，結論與預期不同：**

- `validate_config_schema` **不是**通過，而是 `run_pipeline` 用 `strict=False` 呼叫，
  失敗降級成警告繼續跑。清掉 `sample_dir` 後只剩 `dataset_dir` 那項，屬順序性
  （split 目錄要到 `dataset_splitter` 才建），跑到 `yolo_train` 時已乾淨，不需處理。
- `minimum_source_groups` 沒有卡住（46 張原圖足夠）。
- `_resolve_run_dir()` 的「重跑會變成 `candidate2`」**至今仍未被觸發**：重跑時
  `skip_yolo_train` 會先把整個 run 跳掉，根本不會產生新目錄。要真的觸發它，
  得讓 dataset 或 config 改變。

### (2b) 還沒做：一次完整 cycle

上面只驗證了 `train_candidate` 這一段。`run_training_cycle` 的七個步驟串起來跑一次
還沒做過，而 `collect` / `select` 依賴真實產線紀錄，`evaluate` 依賴 golden（見 (3)）。

### (3) 決定 golden dataset

**在這件事完成前，每次 cycle 都會是 `REJECTED`**，即使每項指標都進步 —— 這是
fail-closed，不是 bug。

天然候選是 `station_data/yolo11_inference/acceptance/<產品>/<站別>/`：有人工真值、
不可變快照、文件明訂不可進訓練集。但它是**整體檢測組合層級的 OK/NG 真值，不是 YOLO
框標註**，直接拿來算 detection 指標需要另做對應。這是要你決定的事，我沒有替你挑。

短期替代：另外準備一份帶 `data.yaml` 與標註的 YOLO 評測集，
`picture-tool-autotrain golden register <路徑> --registered-by <姓名>`。

### (4) 走完一次完整的人在迴圈流程

`collect → select → labeling export → 真的標一批 → labeling import → cycle`。
這會驗證 `labeling.py` 的檔名慣例對實際標註工具好不好用（目前用
`<selector>-<sample_id>.jpg`，匯入時兩種檔名都接受）。

### (5) 之後才談的

- 四個保留 selector（`_planned.py` 寫了各自缺什麼）。
- 觸發政策（何時該跑 cycle）—— 應該是確定性程式碼，不是讓 agent 每次評估。
- `agent/` 的實作。邊界寫在 `agent/README.md`，動手前先讀。

---

## 4. 坑（都是查過才知道的）

- **重跑一次不代表重訓一次。** `skip_yolo_train` 比對 `dataset_hash` 與 `config_hash`，
  相符就把整條 run 跳掉（增生／lint／切分／訓練全跳），而且只寫在 log 裡。
  看 `TrainingResult.trained_this_run`，不要看有沒有產出權重。
- **`expected_items` 不是類別表，而且它「看起來可以用」。**
  Cable1/A 的 `expected_items` 是 `['Red','Green','Orange','Yellow','Black','Black']`，
  真正的契約是 `['Black','Green','Orange','Red','Yellow']`——順序全不同（每一個
  class id 都會錯），而且 `Black` 出現兩次，因為站別實體上預期兩條黑線。
  真跑的 per-class 數字證實了這點：Black 的實例數正好是其他類別的兩倍。
  `core/detector.py` 用 `set(expected_items) - detected` 對待它，語意是「預期看到的
  物件多重集合」。**2026-09-11 起這條路徑不再可能誤用它**：class schema 拒絕重複
  名稱，所以 `expected_items` 是被結構性擋掉，不是靠記得別用（見 §6）。
- **`tests/conftest.py` 的隔離 workspace 是 session 級的。** dataset 版本號、pool、
  registry 會跨測試累積，斷言變順序相依。`test_autotrain_orchestrator.py` 與
  `test_autotrain_service.py` 因此**每個測試自建一份 `WorkspacePaths`** 指向 `tmp_path`。
  新測試照抄那個 fixture，不要用 `WorkspacePaths.discover()`。
- **mypy 是 blocking gate。** selector 的 `options` 來自 YAML，必須標
  `Mapping[str, Any]`；標 `Mapping[str, object]` 會讓 `float(options.get(...))` 直接報錯。
- **命名衝突：** `yolo11_inference` 的「golden sample」是**顏色開線基準**
  （`core/services/golden_sample.py`，4×4 Lab 網格），和這裡的 golden **評測資料集**
  完全是兩回事。
- **C: 槽滿了**，跑 inference 測試要 `--basetemp=D:\tmp\...`（且不能在 workspace 內）。
- IDE 顯示的 E501（79 字元）**不是**本 repo 的規則。實際閘門是 ruff 預設 88 與
  flake8 160，`python -m ruff check src tests` 才是準的。

---

## 5. 設計上不要「順手改掉」的三件事

這三點是刻意的，看起來像未完成，其實是約束：

1. **任務清單裡沒有 `deploy`**，而且 `assert_no_forbidden_tasks` 會硬性拒絕。
   不要改成「用設定關掉」—— 這個保證不應該取決於某個 YAML 維持正確。
2. **候選一律以 `NEEDS_LABEL` 進池，預測永遠不當真值。** repo 自己就明訂
   「誤報不能直接把原推理框當正確答案」。
3. **golden 未設定 → 拒絕升級。** 「沒辦法檢查」不能讀成「通過了」。

---

## 6. Class schema（2026-09-11 新增）

`class_id -> class_name` 是這條路徑上唯一「錯了不會當掉、只會靜靜產出語意錯亂模型」
的東西，所以它被做成一個有身分的型別：`autotrain/class_schema.py`。

### Cable1/A 的真正契約

```
0=Black  1=Green  2=Orange  3=Red  4=Yellow
hash 05f915927011ba63db6d16d535e714c014b53f3f6602391688536aa5b3119df9
```

三個獨立來源完全一致：champion checkpoint 的 `model.names`、handoff manifest 的
`class_names_json` / `class_schema_hash`、以及 job 自己的 `data.yaml`。
掃過 29 個真實 manifest、9576 個 id→name 對照，**零不一致**。

雜湊直接沿用 operator handoff 既有的 `_class_schema_hash`（不是另寫一份），
所以這條路徑寫出的 schema 與 operator 流程寫出的可以直接比對。

### Source of truth 優先序

1. dataset version 記錄的 schema
2. champion checkpoint 的 `model.names`
3. 站別 `config.yaml` **明確宣告**的 `class_names` / `names`
4. 產線紀錄的 `model_info.class_names`
5. 以上皆無 → **拒絕**（fail closed）

優先序只決定「回報時標示哪個來源」，**不決定誰贏**：所有存在的來源必須一致，
不一致就停。**同名不同序算不一致**——這正是整件事存在的理由，用 set 比較會誤判相等。

### 誰在什麼時候檢查

| 位置 | 檢查什麼 | 失敗行為 |
| --- | --- | --- |
| `build_dataset` | 解析契約 | 無來源 → **BLOCK**（不是 fail，等待是合理狀態） |
| `DatasetVersionStore.create` | 標註 class_id 在範圍內 | raise（版本不可變，錯了就永久了） |
| `train`（orchestrator） | dataset + champion + 紀錄三方一致 | fail，訊息含兩邊 mapping、dataset 版本、champion 版本 |
| `train_candidate` | 版本 schema 與傳入 schema 一致；工作副本標註範圍 | raise，**runner 完全不會被呼叫** |

### 兩個之後會用到的缺口

- **deploy manifest 完全沒有 class 欄位**（48 個 key 一個都沒有）。所以 champion 的
  契約只能開 checkpoint 才讀得到。要讓它變便宜，得請 `deploy.py` 寫進去。
- **本機沒有真實產線紀錄**（`Result/` 是空的），所以「產線紀錄帶不帶 `class_names`」
  是從產生端程式碼確認的（`yolo11_inference/core/yolo_inference_model.py:441`，
  註解明寫 ordered names 是訓練資料契約的一部分），不是從真實紀錄檔。

---

## 7. Golden dataset v1 候選（2026-09-11）

### 先更正一件事

先前 §6 寫「本機沒有真實產線紀錄（`Result/` 空）」是**錯的**——查錯目錄了。
真實產線輸出在 **workspace 根目錄**的 `Result/`：43 個日期目錄、2095 張 Cable1/A 原圖、
2196 份 config snapshot。`collect` 對它實跑過，2096 筆紀錄、0 筆無法讀取。

順帶解答 §6 留下的問題：2196 份 snapshot 中 **1392 份有 `model_info.class_names`、
804 份沒有**（較舊、`model_version` 為 None 那批）。第 4 順位來源覆蓋率 63%。

### 最重要的發現：production 紀錄不能直接當真值

production 存的是「模型**找到**的框 + 人工對每個框的判定」。漏掉的物件不會留下任何痕跡。

```
每列偵測框數（共 2607 列）：
   0 框  935 列      ← 完全漏檢
 1-5 框  214 列      ← 部分漏檢
   6 框 1247 列      ← 預期（2 Black + 各 1）
  7+ 框  211 列      ← 多檢

低於 6 框：1149 列（44.1%）
```

拿它當 golden 真值 = **把每一個 false negative 都算成正確**。所以候選分兩種 status：
`ready_to_review`（71 張，有人工框）與 `needs_annotation`（production，需先標註）。

### 工具

| 檔案 | 用途 |
| --- | --- |
| `src/picture_tool/autotrain/golden_candidates.py` | 選候選、去重、統計、報告 |
| `scripts/golden_candidates.py` | CLI 進入點 |

```powershell
python scripts/golden_candidates.py --out runs/golden_candidates/v1
```

唯讀。production 影像只被引用，不複製、不移動、不修改。

### 實跑結果（Cable1/A）

1730 檢視 → 582 去重 → **1148 候選**（71 ready_to_review + 1077 needs_annotation；
871 hard_case + 277 representative）。五種 hard case 全部有樣本，無 coverage gap。

`ready_to_review` 恰為 71，與獨立量到的唯一已標註影像數相符——這個數字是修掉一個缺陷
後才對的：handoff 會把 production 影像複製進 job，同一張圖同時以「已標註」與
「production 證據」出現，原本的排序會丟掉有真值的那份。現在先**合併證據**再去重。

### 去重是兩層的

sha256 抓重複複製的檔案（本專案 510 個檔案只有 71 張唯一影像），dHash 抓「人眼看來相同
但位元不同」的。**精確重複只留 1 份**（兩份同一個檔案不是兩個樣本），近重複可留多份
（那真的是不同照片）。

### ~~尚未做~~ —— 已完成（2026-09-14），見 §8

---

## 8. 分組評測（2026-09-14）

### 先更正一件事

§7 說「evaluator 雙組指標尚未實作，golden manifest 已經有 `groups` 欄位可承載」。
前半對，後半誤導：`groups` 欄位存在，但**整條鏈斷在三個地方**，所以那個欄位在
實務上永遠是空的——`golden register` 從來沒有把分組傳進去。

| 環節 | 當時狀態 |
| --- | --- |
| `golden_candidates.py` 算出每張的 `group` | 有 |
| `golden register` 寫進 manifest | **無**（`cli.py` 沒傳 `groups=`） |
| `GoldenDataset.groups` | 有欄位，恆為空 |
| evaluator 分組量測 | **無** |
| report 顯示 | **無** |

順帶查到：`golden_comparison`（整體 golden 比較）**算出來但沒有任何人讀**，
只進了 `to_dict()`。花了兩次 val 的錢，報告與決策都沒用到。現在報告會印了。

### 做了什麼

| 檔案 | 內容 |
| --- | --- |
| `golden_candidates.py` | `read_group_assignments()`：從 `candidates.csv` 讀回分組 |
| `golden.py` | `register` 將分組與實際影像取交集；`group_counts` / `ungrouped_sample_ids` / `sample_ids_in_group` |
| `cli.py` | `golden register --groups <報告目錄或 csv>` |
| `evaluator.py` | `GroupEvaluation` + `_evaluate_golden_groups`，每組各跑一次 champion/challenger |
| `config.py` | `golden.min_group_samples`（預設 10） |
| `reports.py` | 印出 golden 整體與各組 |
| `tests/test_autotrain_reports.py` | **新檔**——報告渲染先前完全沒有測試 |

### 最容易踩的一個點：join key

候選報告的 `sample_id` 是**檔名 stem**（`golden_candidates.py` 裡是 `image.stem`），
golden 的 `sample_ids` 是**影像 bytes 的 sha256**。用 `sample_id` 去 join
**一張都對不到，而且是靜靜地對不到**。所以 `read_group_assignments()` 只讀
`image_sha256` 欄，永遠不讀 `sample_id`。

衍生的設計：一張都對不到時**拒絕註冊**。對不到多數是正常的（報告涵蓋 1148 張，
留下的遠少於此），但全部對不到代表帶錯檔案或接錯鍵——若默默註冊成無分組，
它看起來會像一個有分組的集合。

### 子集怎麼做的，以及為什麼

ultralytics `val()` 只給整體聚合，拿不到 per-image 拆解，所以分組只能**各跑一次**。
子集是一份**影像絕對路徑清單 txt**，另寫一份 `data.yaml`
把 `val` 指過去、`path` 設為 golden root、移除 `train`/`test`，
其餘（特別是 `names`）從 golden 的 `data.yaml` **原樣繼承**——
在這裡重算 names 等於多開一個讓 class 契約出錯的地方。

golden 目錄完全不被複製、搬移或寫入（有測試守這件事）。

**代價**：golden 段的 val 次數從 2 變 6（整體 2 + 每組 2 × 2）。
替代方案是從整體那一次推估分組分數——那是算術，不是量測。

### 四個「回報原因而不是給數字」的狀態

| 狀態 | 何時 | 為什麼不給數字 |
| --- | --- | --- |
| `INSUFFICIENT` | 張數 < `golden.min_group_samples` | 少量樣本的 rate 以整張為單位跳動，看起來卻像量測 |
| `NO_LABELS` | 該組沒有任何標註可定位 | 版面不對時照算，會把每個物件算成漏檢並發佈成 recall 崩盤 |
| `FAILED` | 無 `data.yaml`／描述檔無法解析／影像不在磁碟／驗證拋錯 | — |
| （無分組） | manifest 沒有 groups | 回傳空 tuple，不是回傳「兩組都 0」 |

`NO_LABELS` 的判斷是複製 ultralytics 的 `images` → `labels` 路徑規則，
**刻意複製而不是 import**：這段要在任何 ultralytics import 之前就能跑
（pytest 下禁止載入）。判準是「**至少一張**能定位到標註」——
要求每張都有會誤殺合法的背景圖，一張都不要求則擋不住版面錯誤。

### 沒有進 promotion 閘門（使用者決定）

分組指標目前只進報告。理由是現在**還沒有任何 golden set 被註冊**，
門檻只能用猜的，而這個 repo 的閘門一向是從量測定出來的確定性數值。
等第一次真跑有數字再定。`PromotionConfig` 沒有新增欄位。

### 驗證

`1384 passed / 6 skipped`（前次 1381）、ruff 全綠、mypy 全綠、coverage 82.36%（gate 80）。
`yolo11_inference` 本次完全沒碰。

**`golden register --groups` 的 CLI 路徑本機仍一次都沒執行過**——`typer` 在本機
三個 python 環境仍然都沒有，`test_autotrain_cli.py` 一律 skip（新增的三個也是）。
CLI 以外的邏輯都有測試涵蓋，包含「從候選報告讀回分組 → 註冊 → manifest
帶著分組」的端到端那一段，以及下面的真跑。

### 對真實 ultralytics 跑過了（2026-09-14），又挖到一個缺陷

`scripts/autotrain_group_smoke.py`（新增）。它**不建立正式 golden set**：
在 `runs/autotrain_group_smoke/` 下自建一份標記為 `SMOKE/TEST-ONLY` 的拋棄式
fixture，champion 與 challenger 都是**複製**出來的副本，production 權重全程沒被開啟。

素材：最新 handoff job 的 46 張已標註影像——26 張進 fixture
（hard_case 12／representative 12／`tiny_smoke` 2，最後一組刻意低於門檻），
其餘 20 張當主 split。ultralytics 8.3.156、torch 2.4.1+cpu、imgsz 320。

**第一個缺陷（我寫的，真跑才會出現）**：`_write_group_descriptor` 原本
`payload.pop("train")`，本意是「不讓分組被拿去對訓練影像評分」。但 ultralytics 的
`check_det_dataset` **要求 `train` 與 `val` 兩個 key 都在**，少一個直接
`SyntaxError`。也就是說**每一組在真實環境下都會是 `FAILED`**，而假 validator
從不解析那份 yaml，所以 1384 個測試全綠。

修法不是把 `train` 放回原值（那才真的會指向訓練影像），而是**指向同一份子集清單**：
格式要求滿足了，「這份描述檔不可能指名真訓練集」的性質也保住。
回歸測試：`test_the_subset_keeps_a_train_key_pointing_at_the_subset`。

**第二個發現（不是缺陷，是文件不實）**：ultralytics 會在標註目錄旁寫
`labels/<split>.cache`。`golden.py` 原本明寫「Never written --- nothing here
modifies the golden directory」，在真實路徑下不成立。
影響評估：衍生資料、不動任何已註冊影像、`verify_content` 只雜湊影像，
所以狀態仍是 `OK`。**這一點早於本次改動就存在**——既有的整體 golden 比較
同樣會觸發它，只是從來沒人真跑過所以沒發現。已改成據實描述。

**結果**：

| 項目 | 結果 |
| --- | --- |
| `evaluation_status` | `COMPLETED` |
| 子集 `data.yaml` 被 ultralytics 接受 | 是（修掉 `train` 之後） |
| `hard_case` | `MEASURED`，12 張，champion mAP50 0.9950 |
| `representative` | `MEASURED`，12 張，champion mAP50 0.9938 |
| `tiny_smoke` | `INSUFFICIENT`，2 張，**完全沒有呼叫 val** |
| per-class | 五類全數回收（Black/Green/Orange/Red/Yellow） |
| 標註真的有解析到 | 每組 72 instances，與離線數標註檔算出的 72 相符 |
| production 權重 | sha256 前後相同 |
| golden 目錄 | 多出 `labels/val.cache`，無既有檔案被改 |

**標註張數那一列是這次最有價值的檢查**：子集若定位不到標註，ultralytics 仍會跑完
並回傳一組漂亮的 0，看起來與「模型全漏檢」無法區分。所以腳本先離線數一次
標註框數，再跟 ultralytics 自己報的 instances 對。72 = 12 張 × 6 框，
且 Black 是 24（其他各 12）——正好是 §4 記的「站別實體上預期兩條黑線」。

**不構成缺陷但要知道**：充當 challenger 的 `runs/Cable1/train/weights/best.pt`
在 conf 0.4 下什麼都測不到，三組 mAP50 全是 0.0。所以本次真跑驗證的是**管路**，
不是任何模型表現；delta 數字沒有意義。

### 下一步

§3 的排序不變（(2b) 完整 cycle、(3) 決定 golden dataset、(4) 人在迴圈）。
分組這件事真正剩下的只有一件：**第一次有真實 golden set 之後**，
看 hard_case 與 representative 的實際差距，再決定要不要把它變成閘門規則。

---

## 9. Golden Review Pack v1（2026-09-14）

分組評測到此視為驗證完成，不再擴功能。這一節是接手 review pack 的部分。

### 做了什麼

| 檔案 | 內容 |
| --- | --- |
| `src/picture_tool/autotrain/review_pack.py` | **新** —— 排除、source 收斂、分組、多樣性挑選、配額、輸出 |
| `scripts/golden_review_pack.py` | **新** —— CLI 進入點，負責所有影像 I/O |
| `tests/test_autotrain_review_pack.py` | **新**，37 個測試 |
| `golden_candidates.py` | `thin_by_group(per_group=None)`：保留近重複叢集，讓 pack 自己挑 |

輸出：`runs/review_pack/v1/`（`review_pack.csv` + `summary.json` + `REVIEW.md` + `images/` 250 張）。
**不註冊 golden、不自動標註、不接 LLM、不碰 production inference／model／trainer。**

### 兩個只有真資料才會暴露的缺陷

**(1) 關鍵組被淹沒。** 第一版 `red_orange_critical` 是 721/1135（63%）。原因：
數量規則只看「Red 或 Orange 的驗證數量 ≠ 期望」，而這個站別有 **935 列 production
是零框**（模型什麼都沒找到），每一列的 Red 數量都是 0。那是漏檢，不是顏色混淆。
現在數量訊號**只在總框數等於期望值時**成立——也就是模型抓到六個、但 Red/Orange
分配錯了，那才是替換。關鍵組降到 311。修正組合成 `hard_case`。

**(2) 訓練影像從兩道檢查中間漏過去。** 第一版產出的 250 張裡，
**有 5 張與 handoff 訓練影像感知上完全相同**，sha 與 source-id 兩道檢查都沒攔到。

原因值得記：handoff 副本是**重新編碼且改名**的。

```
handoff    : 17b1fad3-yolo_Cable1_A_142254.jpg
production : yolo_Cable1_A_142252_433429_4f905ddf33c3.jpg
```

兩種命名結構沒有共同部分，所以 `source_image_id` 的 lineage 比對在這兩者之間
**不是比不到，是結構上不可能比到**。我一度把「source 重疊 = 0」當成獨立性的證據，
那個零其實是命名格式的產物。

加了第三道：dHash 比對，用 repo 既有的 `difference_hash`（不另寫一套，否則會與
既有去重不一致）。實跑攔下 8 列 production 候選。

**dHash 是較粗的訊號**——同一個治具的兩張不同照片可能同雜湊，這個站別確實會發生
（250 張裡有 20 組內部近重複）。所以它 fail-closed 施用並單獨記一個排除理由：
少收幾張可用候選不痛不癢，放一張模型記過的影像進尺，尺就沒意義了。

### 三道排除的分工（缺一不可）

| 檢查 | 抓什麼 | 為什麼不夠 |
| --- | --- | --- |
| sha256 | 被複製的同一個檔案 | 重新編碼就失效 |
| source lineage | 增生衍生檔（`_aug_<n>`） | 命名體系不同就失效 |
| dHash | 重新編碼＋改名的同一張照片 | 較粗，會誤傷；故 fail-closed |

### 近重複用分散而不是刪光

每個感知叢集最多貢獻 `--per-cluster`（預設 3）張，用 farthest-point 在既有量測
（時間、亮度、飽和度、模糊、信心、框數）上挑彼此最不像的。第一張挑「最有料」的
那張，所以叢集即使只剩一個名額也交出最值得看的。

實跑：**只稀釋掉 28 列**。因為這批 production 資料本來就少有近重複
（250 張裡 196 張完全沒有近鄰）。機制在，但這批資料沒怎麼用到它。

### 尚未做／已知限制

- **champion v1.0.6 自己的訓練 provenance 本機沒有任何紀錄**（deploy manifest 48 個
  key 沒有 dataset 欄位，`ChampionModel.dataset_id` 是空的）。所以「與訓練無交集」
  的強度上限就是「與 12 個 handoff job 資料集 + smoke run provenance 無交集」。
  要更強，得請 `deploy.py` 把訓練 dataset id 寫進 manifest。
- pack 內部仍有 20 組近重複（`--per-cluster 3` 允許的），人工複核時可再收斂。
- 本節工作**尚未提交**。

---

## 10. 人工標註匯入與 Golden v1 流程（2026-09-14）

Review pack selection 自此凍結為 v1，除非發現 correctness bug 不再調整。
真正的 blocker 是 ground truth，不是選樣演算法。

### 新增

| 檔案 | 內容 |
| --- | --- |
| `src/picture_tool/autotrain/label_review.py` | **新** —— 五狀態、驗證、核可、coverage、staging |
| `tests/test_autotrain_label_review.py` | **新**，36 個測試 |
| `cli.py` | `golden validate-labels` / `approve-labels` / `reject-labels` / `coverage` / `register-from-review` |

### 這個模組存在的理由只有一句

**一個 label 檔不是一次核可。** 一整個目錄的 `.txt` 只證明有人畫了框。
所以驗證從內容推出狀態（`NEEDS_LABEL` / `LABELED` / `NEEDS_REVIEW`），
人做出決定（`APPROVED` / `REJECTED`），**兩者都到齊**才是 golden eligible。

### 三個不明顯但重要的設計

**核可綁定內容雜湊。** 每個決定記下當時的影像與標註 sha256。核可後改標註，
核可不會跟著走——退回 `NEEDS_REVIEW` 並標 `approval_stale`。
沒有這個，approve-then-edit 是敞開的。這是 golden manifest 上鎖那套邏輯往前挪一步。

**身分用影像 sha256 與 review pack 對接，不是檔名。** 人一定會改名、改副檔名、
改大小寫。反過來這也讓「不在 pack 裡的影像」被抓出來——pack 正是做污染檢查的地方，
繞過它塞進來的影像等於沒檢查過。

**`register-from-review` 不採信任何 cache。** 全部從磁碟重跑：影像/標註雜湊、
source lineage、class schema、訓練污染、group、核可狀態。
昨天的核可說的是「有人看過那些位元」，不是「那些位元還在、還乾淨」。

### 框數不符不自動修

回報 `label_incomplete` 並轉 `NEEDS_REVIEW`，訊息指名缺哪一類（`Yellow 0/1`）。
語法驗證沿用與 operator handoff **同一支** `validate_yolo_label_text`
（涵蓋欄位數、數值、class id 範圍、NaN/inf、座標邊界、寬高 > 0），
兩條路不可能對「什麼叫合法標註」有不同意見。

### 真檔案上驗過

用 review pack 裡 3 張真影像跑完整條：標 2 張 → `LABELED` 2 →
核可 → `APPROVED` 2、eligible 2 → **竄改其中一份標註 → 該張退回 `NEEDS_REVIEW`
並標 `approval_stale`** → 還原重核 → stage 2 張 → `golden.register` 回報 `OK`，
群組 `{red_orange_critical: 2}`、ungrouped 0，coverage 誠實回報三組全 `INSUFFICIENT`。
（該 scratch golden 已刪除，避免被誤認為正式註冊。）

## 11. Forward-only provenance（2026-09-14）

### 先更正 §8 的一句話

§8 說「deploy manifest 完全沒有 class 欄位」——**class 那部分對，dataset 那部分錯**。
manifest 早就有 `dataset_id`、`dataset_image_count`、`training_job_id`、
`training_provenance`、`provenance_confidence`、`dataset_hash`、`training_config_hash`。
問題是 **v1.0.6 這些值全是 `None`**——欄位在，值沒填。差別很重要：
不是要加欄位，是那次 deploy 沒有東西可填。

### 改了什麼（都是 forward-only）

| 位置 | 內容 |
| --- | --- |
| `tasks/deploy.py` | manifest 新增 `class_names` 與 `class_schema_hash` |
| `autotrain/registry.py` | `CandidateModel` 新增 `base_model` 與 `training_provenance` |
| `autotrain/orchestrator.py` | 訓練後把 provenance 檔的路徑＋sha256＋張數/來源數寫進候選紀錄 |

`class_names` 取自**已驗證的 ONNX/PT pair**——那是唯一對兩個部署產物都交叉檢查過的
來源。沒有 pair verification 就留 `None`，不從站別 config 猜（那是「預期」不是「契約」）。
雜湊沿用 `pending_annotations._class_schema_hash`（`autotrain/class_schema.py` 早就這樣做），
所以 manifest、handoff、autotrain dataset version 三者可以直接比對而不是「大概一樣」。

`training_provenance` 記**雜湊**而不只是路徑：路徑在檔案被改之後仍然解析得到，
而「這個模型看過哪些影像」正是今天對 champion 無法回答的問題。

**沒有做的事**：不改現有 production model、不重新部署、不重寫舊 manifest、
不改 production inference 行為。舊 champion 的 provenance 不硬追。

---

## 12. Historical replay（2026-09-14）

驗證 AutoTrain 能否接管既有人工 dataset 走完
Dataset → Train → Evaluate → Registry → Report。**不是**驗證模型好壞。

`scripts/autotrain_historical_replay.py` + `tests/test_autotrain_historical_replay.py`。

### 歷史 dataset 是追出來的，不是猜的

champion manifest 的 `dataset_hash 4fec918397fe` → `runs/Cable1/A/train10/last_run_metadata.json`
（同一個 hash）→ `dataset_dir` =
`data/.operator_handoff/jobs/20260727T060103Z-e6d896ebb9/dataset/Cable1/A/split`，
`trained_at 2026-07-27T16:03:28`。用日期猜大概也會猜到同一個，但「大概」不是 provenance。

### Dataset 實況

882 physical / 882 unique sha（**零** exact duplicate）/ **46 distinct source** /
836 augmented（約 19×）/ 314 unique dHash。train 585、val 98、test 199。

**source 層級乾淨**：46 個 source 沒有任何一個跨 split。`_aug_<n>` 慣例有效。

**dHash 層級有洩漏**：9 組跨 split 近重複，涉及 141 個檔案、8 個 source。
這些是**不同的原始拍攝**在 8×8 灰階下無法區分。注意這是冗餘的證據，
不是同一張照片的證明——這個站別確實會有不同照片同雜湊。

### 真跑

從 champion `.training.pt` 續訓 1 epoch、imgsz 320、CPU，`trained_this_run: True`，
產出 `runs/historical_replay/registry/Cable1_A_historical_replay/weights/best.pt`。
原始 dataset 前後 1769 個檔案雜湊**逐檔相同**（0 added / 0 removed / 0 changed）。

### 一個只有真跑才會撞到的坑

第一次失敗在 `yolo_augmentation`：albumentations `std_range=(0, 6)` 超出新版要求的 0–1。
原因不是 albumentations，是**我用了 `training_project/config.yaml` 當 base config，
而訓練專案根目錄根本沒有這個檔案**——`load_config` 於是落到打包預設值，那份是舊格式。
既有 cycle 走 `_load_base_pipeline_config`（挑 `configs/default_pipeline.yaml`），改用同一支即通。
**教訓**：replay 要沿用 cycle 自己的每一個載入路徑，換一個就不是同一條路。

### 指標，以及為什麼不能拿來歸因

全部標記 `NON_INDEPENDENT`，decision 在任何量測之前就寫死為
`NOT_PROMOTABLE_NON_INDEPENDENT`。

| val set | champion mAP50-95 | challenger mAP50-95 |
| --- | --- | --- |
| 原始 historical val（98） | 0.8160 | 0.8160 |
| source-safe val（179） | 0.7399 | 0.7481 |

precision/recall 在兩邊對兩個模型**全部是 1.0000**。

**一個看似合理但錯誤的歸因，必須寫下來免得被重複**：
「原始 val 因近重複洩漏而虛高」聽起來能解釋 0.076 的差距，但查證後方向不對——
source-safe val 的 179 個檔案裡**有 58 個是 champion 當初訓練用過的**
（另有 100 個來自原始 test）。它對 champion 反而**更**污染，分數卻更低。
所以這 0.076 不能歸因於洩漏；兩個 val set 在大小（98 vs 179）與組成上都不同，
這個設計**無法分離洩漏效應**。要分離，得在 champion 訓練前就把整個 family 留出來，
而 champion 已經固定，回頭做不到。

**這次真正立得住的 dataset quality 結論**：原始 val **完全無法區分兩個不同的模型**
——precision、recall、mAP50、mAP50-95 四項的 delta 全是 0.0000，
而同樣兩個模型在另一個切分上是有差距的。一個讓兩個不同模型拿到相同分數的 val set
沒有在量測任何東西。加上 44% 的檔案與近重複相連，這份 val 不適合當升級判準。

### 沒有做的事

不改 production model／inference／trainer、不部署、不註冊 golden、
不因為數值好而產生 `PROMOTION_CANDIDATE`。

---

## 13. Known risk：Vision gateway 是內網明文 HTTP（2026-09-14）

`bootstrap/vision_client.py` 呼叫的公司 gateway 位於 `http://<company-gateway>:12808`，
**沒有 TLS**。已實測確認，過程中未關閉、未繞過任何 TLS 設定（驗證全程維持預設）：

| 探測 | 結果 |
| --- | --- |
| `https://<company-gateway>:12808` | `SSL: WRONG_VERSION_NUMBER` —— 該埠說明文 HTTP，不是 TLS |
| TLS handshake on 12808 | 同上，握手失敗 |
| `https://<company-gateway>:443` | 連線被拒，沒有 HTTPS listener |

**風險**：API key 以 `x-api-key` header 明文送出，production 影像以 base64 明文送出。
在內網同網段上可被被動側錄。這是**基礎設施層級**的問題，不是這段程式碼能修的——
程式端已經做到不 hard-code key、不寫進 log/provenance/commit，但傳輸本身無法自保。

**不要做的事**：不要為了讓 HTTPS「看起來能通」而設 `verify=False`、
自訂 `ssl.SSLContext(check_hostname=False)` 或塞自簽憑證繞過。
那會把「已知的明文」換成「假裝加密」，更糟。

**償還路徑**：請 IT 在 gateway 前面加 TLS termination，然後把 `ANTHROPIC_BASE_URL`
改成 `https://`。程式端零修改——endpoint 本來就是設定。

在那之前，這條路徑只適合內網、只適合非機密影像。Cable1/A 的產線影像是否算機密，
是需要你們判斷的事，我沒有替你們判斷。

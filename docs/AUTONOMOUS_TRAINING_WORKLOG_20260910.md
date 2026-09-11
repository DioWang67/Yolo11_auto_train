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

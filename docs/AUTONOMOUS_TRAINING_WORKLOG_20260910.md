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

### 兩件**沒有**驗證的事

- **從來沒有跑過一次真的 YOLO 訓練。** 所有測試都注入假的 runner／validator。
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

### (2) 跑一次真的訓練（最大的未知）

目前 `trainer.py` 與既有 `run_pipeline` 的介面只在假 runner 下驗證過。真跑一次會遇到
的疑點：

- `build_candidate_config()` 產出的 config 能否通過 `validate_config_schema`。
- `dataset_splitter` 的 `minimum_source_groups`（val ≥5、test ≥10 組獨立原圖）——
  一開始樣本少會直接卡住，這是既有的安全下限，不要為了跑通把它調低。
- `_resolve_run_dir()` 對 ultralytics `exist_ok=False` 自動遞增目錄的假設。

建議先用一小批已標註資料、`epochs: 1`、`device: cpu` 跑通，再談品質。

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

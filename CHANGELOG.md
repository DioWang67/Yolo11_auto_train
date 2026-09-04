# Changelog

All notable changes to this project will be documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

---

## [Unreleased]

### Added
- 顏色關卡與推論端執行期之間新增跨實作一致性測試。兩者是各自獨立的程式碼，
  卻在把關同一個產品，任何一邊的判定規則移動而另一邊沒有，就會讓模型通過關卡
  卻在產線表現不同，而兩個 repo 各自都看不到。兩邊共用一份逐位元相同的
  `tests/data/color_conformance.json`（顏色模型直接內嵌，所以各自的測試不需要
  對方的 checkout 就能跑），各自把自己釘在案例上；workspace CI 比對兩份副本
  是否相同。已知的合理差異記錄在 `known_divergences` 並附理由，而不是被抹掉。
- Operator review handoff for confirmed OK, overkill, missed detections,
  annotation repair and color-only calibration routes with stable SHA-256
  sample identities.
- Explicit per-job position retraining, calibration, Golden Set validation and
  Position Gate; activation remains a separate non-persistent operator choice.
- Versioned deployment transaction that pairs runtime ONNX with its training
  PT, preserves station settings and publishes `config.yaml` last.
- Dataset readiness and family-aware train/validation/test separation,
  including reviewed-sample routing and leakage checks.
- Runtime profile diagnostics for the supported Python 3.10/3.11 dependency
  sets used by Picture Tool.

### Changed
- `picture_tool.color` 的中心裁切統一為 `strategies.base.center_crop()`。
  原本評分路徑用逐軸 15%、規則路徑用 `min(h, w)`、黃色快速判定又是另一種，
  同一張圖的三個判定看的不是同一批像素。
- 色相環形平均、色相區間比對、權重正規化與安全比值抽到
  `strategies/base.py`，與推論端 `core/stats_color_checker.py` 的對應實作
  對齊。兩者仍是各自獨立的程式碼，但把關的是同一個產品——判定規則不一致
  會讓模型通過訓練關卡卻在產線表現不同。
- Operator training opens from the inference application's PIN-protected
  engineering settings and reports the existing terminal job instead of
  silently starting a duplicate.
- Training continues only from a manifest-verified PT matching the deployed
  runtime model; missing lineage now fails closed.
- Position calibration uses reviewed label evidence and a disjoint holdout
  instead of calibrating from candidate predictions.

### Fixed
- The color gate is measured over the whole detection box now, restricted to
  its largest connected match, mirroring the inference runtime's
  `stats-robust-v6` contract. Every strategy's `match_ratio` used to receive a
  pre-flattened, pre-cropped pixel list from a fixed geometric center-crop; it
  now receives the whole 2D box and does its own saturation-gating and
  connected-component selection (`measure_color_region`/
  `largest_matching_blob` in `strategies/base.py`), because a wire's position
  and curve vary board to board and a fixed crop assumed a fixed position.
  Black drops `coverage_mean` -- a figure that recorded how much of a
  *differently framed* crop matched during calibration -- and scores its
  matched region's own share of the whole box instead; a region already
  isolated by connectivity does not need a geometry-coupled number to
  normalize away contamination it no longer contains. `lab_ratio` shares
  `hsv_ratio`'s denominator (the whole candidate pool) rather than the
  matched region's own size, which would have let an internally-consistent
  minority region outscore a legitimate majority whose exact LAB rendering
  missed its own recorded envelope by a hair -- caught by the shared
  conformance fixture, not assumed. When nothing forms a connected region at
  all, LAB and hue-mean terms fall back to the whole saturation-gated pool
  rather than a hard zero, reproducing the previous behavior for a genuinely
  weak (not contaminated) hue signal such as real desaturation.

  Verified against the shared conformance fixture: 19 of 21 cases agree with
  the runtime outright, and the one recorded divergence (`desaturated red`)
  still diverges for its already-documented reason, not a new one.

- The training color gate now measures what the runtime measures. Three
  differences had it reaching different verdicts on the same part, and the
  shared conformance fixture could not see any of them because every one of its
  cases was square:
  - `strategies/base.py` `center_crop` derived one margin from `min(h, w)`
    while the runtime crops each axis independently. On a square patch the two
    are the same crop, which is why the suite stayed green; on an elongated
    wire ROI they are not. A 384x96 red strip with an orange end came back Red
    from the runtime and Orange from the gate.
  - The `fast_detect` loop returned on the first color it recognized, reporting
    that color with its own confidence and every other color zeroed -- so a 40%
    yellow band beat a 60% green one. The indicators are still recorded in
    debug info, and no longer adjust any score. The dead
    `_extract_center_pixels`, a third copy of the old crop, is gone.
  - `GreenStrategy.post_correction` flipped a Red verdict to Green whenever
    bare hue pixels in [70, 100] exceeded 0.3 of the crop -- no saturation or
    value filter, so unlit pixels counted -- and replaced the confidence with
    that raw ratio. It manufactured confidence from a disambiguation step,
    which the orange/red tie-break was already fixed not to do, and the runtime
    has no counterpart. Green now competes on its score.
- `color_decision_tuning` survives a detector deployment. v5 binds a baseline
  to all of its resolved tuning keys, so reverting a station's tuning
  invalidates its approved baseline and fails the color check closed under
  strict enforcement -- the same reason `color_roi_policy` was already
  preserved.
- `tests/test_color_integration.py` no longer leaves a `MagicMock` standing in
  for the color verifier. It replaced the module in `sys.modules` at import
  time and never restored it, so whichever suite ran afterwards tested the mock
  instead of the real gate -- silently, as passes. The conformance guard was
  among them, which means it could be switched off by test ordering.
- Detector deployment and portable bundles no longer publish training
  `quality/color/stats.json` as a runtime `stats` baseline. Training records SAM
  mask coverage, while the runtime records coverage in the station ROI; treating
  them as interchangeable silently shifts Black scores. Deployments now preserve
  the existing station baseline or fail when none exists, bundles omit the
  incompatible file and require strict station calibration, and detector deploys
  preserve the station's provenance-enforcement setting.
- Black 改為與推論端相同的 learned S/V + LAB 聯合匹配，並用基準的
  `coverage_mean` 正規化；手寫的 `s < 50 & v < 80` shortcut 移除。缺少或無效的
  `coverage_mean` 會 fail closed，而非對著不存在的參考值計分。這個 shortcut
  原本兩頭不到岸 —— 既不是學出來的，也不是乾淨的規則 —— 而它的統計基準在自己的
  holdout 上只有 3.6% 準確率。
- 部署會保留已簽核的 `color_roi_policy`，不讓訓練輸出默默覆蓋站點的取樣幾何。
  顏色基準是在特定 ROI 幾何下校正的，換掉幾何等於換掉基準的意義。
- 顏色決勝不再憑空製造信心。Orange/Red tie-break 會把勝方分數乘上 1.3（或
  1.1），於是一個「只負責區分橘或紅」的步驟可以讓勝方超車一個本來分數更高
  的無關顏色——這正是黑色被報成橘色的成因。現在改為把該配對原本的最佳分數
  轉移給勝方，不創造任何新分數，與推論端 `_separate_orange_red` 一致。
- Black 不再無條件白拿 0.2 分。`BlackStrategy` 把 hue 相似度硬寫成 1.0 再乘
  0.2 權重，註解說是「忽略」但實際是「給滿分」，任何區域（包含完全不黑的
  區域）都能拿到這 0.2。現在該項目直接不參與計分，其餘權重重新正規化。
- 色相改用環形平均。OpenCV 色相在 0/179 環繞，但評分端與基準產出端都用線性
  平均，紅色像素 3 與 178 會平均成 ~90（綠色）。這同時修在三處：策略評分的
  `mean_hue`、`compute_hsv_lab_stats()` 的逐張 `hsv_mean`、以及
  `RunningStats` 跨張聚合（改為累積單位向量，否則逐張修好仍會在聚合層重現）。
- 統計缺失不再變成加分。缺少 `hsv_mean` / `lab_mean` 的顏色，其相似度項目
  原本預設為完美的 1.0 且仍計入完整權重，導致統計不完整的顏色贏過統計完整
  的顏色。現在缺失的項目直接排除、其餘權重重新正規化（統計齊全時為無變化）。
- 空區域不再讓流程崩潰。`BlackStrategy.fast_detect` 與
  `YellowStrategy.fast_detect` 以遮罩大小為分母卻未防零，空裁切會拋
  `ZeroDivisionError`。
- 有效像素過少時不再捏造 Black 判定。`_evaluate_image_improved()` 原本直接
  給 `Black = 0.7`（高於 Black 自己的 0.45 門檻），使曝光不足或洗白的影像
  「通過」顏色關卡。現在回報 `insufficient_pixels` 並讓所有分數為 0，
  fail closed；後校正在此情況下一併跳過，否則所有 ratio 皆為 0 會讓橘/紅
  決勝的「差距 < margin」條件恆成立而憑空生出預測。
- 未特別處理的顏色，其色相區間比對現在能跨 0/179 環繞，因此 margin 把區間
  推出邊界、或未來出現校正於色相 0 附近的顏色（粉紅、洋紅），不再被判成
  「永不匹配」。
- Black 快速判定回報的信心值現在對應實際觸發的規則。原本一律回報 coverage，
  但判定可由 mean 或 median 規則觸發，此時 coverage 與被比較的門檻無因果
  關係，會出現「這是黑色，而黑色不合格」的自相矛盾結果。
- Promotion and deployment now publish artifacts atomically, verify portable
  package receipts, and bind position-gate evidence to the canonical target
  configuration and recomputed metrics.
- Python 3.10 exception notes and cross-platform lock handling now preserve the
  original failure context without breaking Linux or Windows type checks.
- Pytest now runs in an isolated workspace and rejects dynamic attempts to
  redirect tests into live station data.
- `yolo_train` skip logic now checks the **latest versioned run directory** (`train`, `train2`, …) instead of always checking the base `train/` directory. This prevented the skip message from correctly reflecting which run was current after force-runs.
- Changed `exist_ok=False` in YOLO trainer so each forced retrain creates a new versioned run directory instead of overwriting the previous one.

### Added
- `_find_latest_run_dir()` helper in `tasks/training.py` — finds the most recently modified run directory matching the Ultralytics version pattern (`^<name>\d*$`).
- 6 new tests in `tests/test_pipeline_skip.py` covering versioned directory detection and skip behaviour.

---

## [0.4.0] — 2026-03-10

### Added
- `deploy` task: copies training artefacts directly to yolo11_inference `models/` directory.
- `artifact_bundle` task: zips training artefacts for archival.
- `DetectionConfigExporter`: generates `detection_config.yaml` embeddable by yolo11_inference.
- Async pipeline support (`stop_event`) — GUI can cancel a running training gracefully.
- `--describe-task <name>` CLI flag to print task description and dependencies.

### Changed
- `position_validation` task dependency removed from `yolo_train` hard chain; now resolved at runtime via weight detection.

---

## [0.3.0] — 2026-02-11

### Added
- Experiment tracker integration (`get_tracker`) — logs params, metrics, artefacts to local YAML or MLflow.
- `write_experiment()` utility — writes structured experiment log after each training run.
- Hash-based skip logic (`compute_dir_hash`, `compute_config_hash`) — avoids redundant training when dataset and config are unchanged.
- `last_run_metadata.json` written to each run directory to persist hash state.
- `OnnxExporter` — auto-exports `best.onnx` after training when `export_onnx.enabled: true`.
- `PositionConfigGenerator` — auto-generates `auto_position_config.yaml` from training sample inferences.

### Changed
- `dataset_splitter` task now also generates `classes.txt` for class name auto-detection.

---

## [0.2.0] — 2026-01-15

### Added
- `color_inspection` and `color_verification` tasks for LED colour QC.
- `qc_summary` task — aggregates colour/position/inference results into a single JSON.
- `position_validation` task — offline position validation using trained weights and sample images.
- `batch_inference` task.
- `dataset_lint` and `aug_preview` tasks.
- GUI: log viewer, style manager, annotation tracker.
- `picture-tool-doctor` CLI for environment health checks.
- DVC integration via `data_sync` task.

### Changed
- Pipeline refactored to DAG-based executor (`pipeline/core.py`) with topological sort and `skip_fn` support.
- All task implementations moved to `tasks/` package.

---

## [0.1.0] — 2025-12-10

### Added
- Initial release.
- `format_conversion`, `yolo_augmentation`, `image_augmentation`, `dataset_splitter` tasks.
- `yolo_train`, `yolo_evaluation` tasks (Ultralytics YOLO11).
- `generate_report` task.
- PyQt5 GUI (`picture-tool-gui`).
- CLI entry point (`picture-tool-pipeline`).
- Pydantic config validation.

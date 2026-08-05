# 位置檢測補訓與部署

## 目標

位置設定與 YOLO 權重共用同一個部署交易，但使用不同的品質關卡：

1. YOLO Gate 驗證辨識 Precision、Recall、mAP 與基準模型退化。
2. Position Gate 驗證正常品誤殺率、異常品召回率與基準設定退化。
3. 兩個 Gate 都通過後才發布；`config.yaml` 最後寫入，作為原子啟用點。

位置校正只使用人工確認的 YOLO 標註，不使用候選模型自己的預測結果，
避免模型以自己的誤差校正自己。

## 資料分工

| 資料 | 用途 | 可否重疊 |
|---|---|---|
| `calibration_image_dir` + `calibration_label_dir` | 計算每一類的中心、尺寸與容許偏移 | 不可與 Golden Set 重疊 |
| `golden_ok_dir` | 計算正常品誤殺率 | 不可與校正集重疊 |
| `golden_ng_dir` | 計算位置異常召回率 | 不可與校正集重疊 |

校正集預設排除檔名含 `_aug_` 的增強圖，且每張圖必須包含設定要求的所有
class。缺標註、重複 class、越界座標或類別契約不一致時，流程 Fail-closed。
當 `require_disjoint_calibration: true` 時，校正 manifest 是必要部署證據；
缺少 manifest 或與 Golden Set 的影像 SHA-256 重疊，Position Gate 都會拒絕。
若要求 baseline，候選與 baseline 也必須能以影像 SHA-256 證明使用相同
Golden Set，避免拿不同資料集的指標做退化比較。

Operator 補訓預設使用 `golden_manifest_path` 搭配 test holdout 目錄，不會把
整個 YOLO test split 當成位置 OK。manifest 只收錄兩種位置專用證據：
`position_false_reject` 視為 PASS；只含 `POSITION_SHIFT` 的
`confirmed_ng` 視為 FAIL。缺件、錯框、錯類與顏色異常不納入 Position Gate。

## 設定範例

```yaml
yolo_training:
  position_validation:
    enabled: true
    auto_generate: true
    product: Cable1
    area: A
    calibration_source: labels
    calibration_image_dir: ./data/split/train/images
    calibration_label_dir: ./data/split/train/labels
    calibration_min_samples: 3
    calibration_require_all_classes: true
    calibration_exclude_augmented: true
    golden_manifest_path: ./data/metadata/review_dataset_manifest.csv
    golden_manifest_image_dir: ./data/split/test/images
    tolerance_unit: pixel
    gate:
      enabled: true
      min_ok_samples: 10
      max_ok_false_reject_rate: 0.005
      min_ng_samples: 0
      min_ng_recall: 0.0
      require_baseline: false
      max_ok_false_reject_regression: 0.0
      max_ng_recall_regression: 0.0
      require_disjoint_calibration: true

  deploy:
    enabled: true
    preserve_station_settings: true
    position_activation: preserve
```

有正式位置異常 Golden Set 時，應設定 `golden_ng_dir`、提高
`min_ng_samples`，並依風險設定 `min_ng_recall`。尚無 NG 樣本時可以先以正常品
誤殺率 Gate 部署校正值，但不得把這解讀為已驗證異常召回率。

## 執行與輸出

主推理 GUI 的送出畫面提供兩個逐筆、不記憶的選項：

- `啟用位置檢測補訓`：產生位置基準並執行位置驗證與 Gate；
- `位置驗證通過後啟用現場位置檢測`：將
  `position_activation`設為 `enable_after_gate`。未勾選時為
  `preserve`。

若第一個選項未勾選，本次只執行 YOLO 補訓。現場位置檢測已啟用時，
YOLO-only 工作會在訓練前被阻擋，避免新權重沿用不相容的舊位置基準。

啟用位置補訓時會自動執行：

```text
人工覆核
  -> YOLO 標註資料
  -> YOLO 訓練與 YOLO Gate
  -> 人工標註位置校正
  -> Golden Set 位置驗證
  -> Position Gate
  -> ONNX/設定成對驗證
  -> 版本化部署
```

每次成功部署會在站點 `versions/` 目錄保留：

- 版本化 runtime weight；
- `*.position_gate.json`；
- `*.position_validation.json`；
- `deployment_manifest.yaml` 中的 SHA-256、位置指標與實際 runtime
  enabled 狀態。

任何 Gate、報告、checksum 或候選設定缺失，都會中止部署。

GUI 預設要求至少 10 張合格 Golden OK。若 manifest 在 test holdout 中
沒有 `position_false_reject`，也沒有只有 `POSITION_SHIFT` 的
`confirmed_ng`，會明確失敗：

```text
Position golden manifest has no eligible samples in the holdout image directory.
```

此錯誤代表資料不具備位置驗證資格，不是按「續訓」即可修復。應先回到推論
複核流程收集位置誤殺 OK 或純位置偏移 NG。

## 啟用與回滾

`position_activation: preserve` 是預設值：

- 現場原本 `enabled: false`，補訓後仍為 `false`；
- 現場原本 `enabled: true`，Gate 通過後保持 `true`；
- 新站點沒有 incumbent 設定時，預設不啟用。

只有工程師完成 Golden Set 驗收後，才使用
`position_activation: enable` 明確啟用。推論端重新載入時會先完成候選引擎
初始化，再原子替換舊引擎；候選初始化失敗時舊引擎保持可用。

回滾時必須以同一版本的 weight、config 與 deployment manifest 成對回復，
不可只替換單一檔案。

角色操作與現場處置另見：

- [`yolo11_inference/docs/manuals/OPERATOR_MANUAL.md`](../../yolo11_inference/docs/manuals/OPERATOR_MANUAL.md)
- [`yolo11_inference/docs/manuals/ENGINEERING_MANUAL.md`](../../yolo11_inference/docs/manuals/ENGINEERING_MANUAL.md)

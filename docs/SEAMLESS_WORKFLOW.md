# YOLO11 推論、回訓與部署閉環

`yolo11_inference` 與 `Yolo11_auto_train` 使用檔案式交接。作業人員不需要輸入指令、路徑或產品名稱。

## 文件入口

- 日常檢測、紀錄、Excel 與班組長操作：
  [`yolo11_inference/docs/manuals/OPERATOR_MANUAL.md`](../../yolo11_inference/docs/manuals/OPERATOR_MANUAL.md)
- 設定、Gate、資料庫、部署與回滾：
  [`yolo11_inference/docs/manuals/ENGINEERING_MANUAL.md`](../../yolo11_inference/docs/manuals/ENGINEERING_MANUAL.md)

## 產線組長快速操作

組長只需要做三件事：在複核畫面判定「有目標但沒框」、於自動開啟的標註工具拖曳正確框並選類別、按 `Ctrl+S` 儲存後關閉工具。資料切分、補訓、新舊模型比較與部署由系統處理；任何安全檢查失敗都保留目前產線模型。

## 作業流程

1. 在推理系統停止檢測，選擇產品與站別，再進入
   「工程設定 → 模型補訓 → 開啟補訓資料與送出」。工程設定需要 PIN，
   離開頁面後會重新鎖定。也可雙擊推理專案根目錄的
   `一鍵蒐集訓練資料.bat`開啟備援入口。
2. 預設顯示最近 7 日資料；選擇「全部」、「本日」或「最近 30 日」會立即更新，自訂時間才需要按「套用」。系統另納入每 100 張 PASS 影像中的 1 張，供漏檢抽查。
3. 逐張檢查推論結果。介面先判產品結果，再處理標註：AI 判 NG 時可選「確認 NG（AI 判定正確）」、「實際 OK（AI 過殺）」、「確認 NG，但標註需修正」或「圖片無法判定（不採用）」；選擇標註修正後，再指定框的位置／數量或類別錯誤。若失敗原因是顏色，畫面改顯示「顏色確實 NG（顏色判定正確）」、「顏色其實 OK（門檻過嚴）」及「顏色覆核＋框需修正」，並直接列出預測顏色、`diff` 與門檻。PASS 抽查則可選「確認 OK（AI 判定正確）」或「實際 NG（AI 漏檢）」。無框時可選實際無目標或實際 NG。無法判定的圖片不會送訓。
   - 現場發現不在清單中的漏檢時，選擇「從已保存結果回報漏檢」。只能選擇目前推理系統已保存、且包含完整類別契約的結果；外部圖片須先用目前產線模型檢測一次，避免把錯誤產品、站別或類別送訓。
4. 複核期間可開啟「查看補訓清單」檢查目前候選縮圖。所選範圍全部確認後，「下一步：建立訓練資料」才會啟用，並開啟全螢幕候選總覽；可依「直接訓練／需要補標／顏色校正」篩選、雙擊放大或取消勾選誤入資料。顏色單一路由按「送出顏色校正資料」，不會啟動 YOLO 補訓；同時有框錯誤才會進入補標。排除狀態會保存，確認清單後才進入後續流程。
5. 若有待標註資料，LabelImg 會自動開啟並載入正確類別。漏檢案件依序操作「拖曳框選 → 選擇類別 → `Ctrl+S` → 關閉工具」，系統會自動驗證並回流 `raw`；未完成時才顯示「開始標註」供繼續處理。誤檢與錯類案件即使已有系統產生的草稿框，也必須實際修正並儲存；直接關閉 LabelImg 不會把未確認草稿送入訓練。
6. 待標註案件清空後，送出畫面要求逐筆確認補訓參數與位置選項：
   - 未勾選「啟用位置檢測補訓」時，本次只執行 YOLO 補訓；
   - 勾選後才會產生位置基準、執行位置 Golden Set 驗證與 Position Gate；
   - 只有再勾選「位置驗證通過後啟用現場位置檢測」，且全部 Gate 通過，
     才會把現場位置檢測切為啟用；
   - 兩個位置選擇均不保存到下一筆工作。
   確認後系統自動執行適用的增強、資料檢查、資料切分、訓練、品質驗證、
   報告及部署，不需另選 task。

完成時會顯示「訓練與部署完成」。推理系統在下一次檢測時會偵測
`config.yaml`更新並安全載入新模型。資料不足時會進入「累積改善案例」；
失敗、取消或等待補標的既有工作可從主 GUI
「工程設定 → 模型補訓 → 補訓進度」選取後按「繼續這筆補訓」。

也可以直接雙擊 `yolo11_inference/一鍵蒐集訓練資料.bat` 開啟相同複核畫面。

## 自動分類結果

```text
Yolo11_auto_train/data/<product>/<area>/
  raw/images/                         # OP 確認框正確的乾淨影像
  raw/labels/                         # 由已確認的框轉成 YOLO 標註
  metadata/review_dataset_manifest.csv
  review_pending/images/              # 誤檢、漏檢、類別錯誤或無法判定
  review_pending/manifest.csv          # 待人工標註原因
  color_review/images/                 # 顏色覆核證據，不進 YOLO raw dataset
  color_review/feedback.csv            # item-level 顏色真值、diff 與執行時門檻
```

系統使用影像 SHA-256 產生穩定 sample ID 並保存最新複核狀態。同一影像重複提交時不會重複加入；已送訓影像若改判為誤檢、漏檢或錯類，舊 `raw` 影像與 label 會自動撤銷。舊版紀錄若沒有影像寬高，系統會從已保存的前處理影像取得；證據或座標不完整時，一律轉入待人工標註區。

漏檢案例可能完全沒有推論框。此類案例不會預先建立空白 YOLO label，也不會直接進入訓練。標註人員必須補上至少一個正確框與類別；漏檢案件即使儲存空白 label 也會被拒絕。只有在推理複核畫面明確選擇「沒有目標」的負樣本，才會以 `verified_empty` 進入訓練。

顏色案例以 `action_route=color` 分流，永遠不會建立 YOLO label。相同影像與顏色項目再次覆核時以最新真值覆蓋，不重複計數。若 `diff` 已通過門檻但其他顏色規則失敗，資料標記為 `failure_kind=rule`，只供規則分析而不參與門檻最佳化。舊版快照若缺少 item-level `diff` 與門檻，系統拒絕匯出，避免用猜測值校正。

## 顏色門檻校正

顏色門檻採兩階段發布。`picture-tool-color-calibrate recommend data --output color-threshold-report.json` 只產生 shadow report，不修改模型。預設每個產品／站別／模型／顏色至少 30 筆，且實際 OK、NG 各至少 5 筆；候選不得增加 NG 誤放率。工程師抽驗後，使用 `picture-tool-color-calibrate apply color-threshold-report.json --models-root <inference-models> --approver <姓名或工號>` 具名批准。

套用時會檢查報告 checksum、設定漂移與目標路徑，備份原設定後才原子替換 `config.yaml`，並寫入 `color_threshold_history.json`。`stats` checker 會自動將公開的 diff 門檻轉換回設定使用的相似度門檻；禁止人工心算後直接改設定。

## 訓練與部署保護

- readiness gate 會阻擋缺標註、非法 bbox、class ID 越界及 train/val/test 洩漏。
- handoff 會核對推理與訓練的類別名稱、ID 與順序；不一致時禁止訓練。
- 組長補訓模式要求每類至少 5 個有效 instance，其中 train 至少 3 個、test 至少 1 個；train/val/test 圖片數至少為 3/2/2。產品設定只能提高，不會降低這些安全下限。
- splitter 依原圖 family 分組，避免 `_aug_N` 版本跨 split。
- 本次送出的補標 sample ID 強制進入 train；歷史資料負責形成互相隔離的 validation/test，不會讓新漏檢案例誤落到 test 而沒有參與學習。
- 作業流程固定執行：YOLO 標註感知增強 → 資料 lint → 依原圖 family
  切分 → YOLO 訓練 → YOLO 評估 → 訓練報告 → 測試集批次推論 →
  QC 彙總 → 部署；只有本筆工作明確啟用位置補訓時，才插入位置校正、
  位置驗證與 Position Gate。預設每張已標註正樣本產生 20 張增強圖並
  保留原圖；已確認的空白負樣本只保留原圖。補訓會關閉色相、翻轉、
  透視及 YOLO 內建線上增強，避免線材顏色與方向語意被破壞。
- 回訓只會從 manifest 記錄且 SHA-256 驗證通過的同版本 `.pt` 延續，保留既有類別能力。產線若使用 ONNX 卻缺少配對 PT，系統會停止補訓，不會退回舊 `best.pt` 或基礎模型冒充現行版本。
- 權重先寫暫存檔並驗證 SHA-256，最後才原子發布 `config.yaml`。
- 部署會產生新版本且保留站別的曝光、增益、燈光、校正、輸出設定與既有 `color_stats.json`；只有訓練輸出提供新的顏色模型時才會替換。部署 manifest 會記錄顏色檔來源及 SHA-256。
- 未成功完成時不會顯示部署完成，介面會顯示錯誤訊息。
- 品質 gate 使用目前產線的信心門檻，在同一份 test split 上分別執行 challenger 與 incumbent；precision、recall、mAP50、mAP50-95 低於門檻、任一核心指標退步超過 0.02，或找不到 incumbent 時，一律保留舊模型。

資料不足時會顯示「補標資料已安全保存」，組長只需繼續累積漏檢案例後再次送出。品質比較未通過時會顯示「現場仍使用原本模型」，不需要手動回復權重。

位置補訓至少需要 10 張合格的正常位置 Golden 樣本。Operator handoff
只接受 `position_false_reject` 作為 Golden OK，或只有
`POSITION_SHIFT` 的 `confirmed_ng` 作為 Golden NG。缺件、錯框、錯類、
顏色與混合原因不納入 Position Gate。出現
`Position golden manifest has no eligible samples`時，必須先收集上述
位置專用證據，不可把整個 test split 當成位置正常。

### 遠端訓練模型的成對部署

`scripts/run_remote_train_cable1.bat` 會下載 `best.onnx`、來源 `best.pt`，以及同時記錄兩者 SHA-256 的 `runtime_export_manifest.json`。部署前再以同一個固定輸入比較 ONNX 與 PT 的原始輸出；檔案缺失、下載後遭修改、輸出 shape 不同或數值不等價都會阻擋部署。

```bash
python scripts/validate_and_deploy_cable1.py \
  --weights best_cable1_remote.onnx \
  --training-weights best_cable1_remote.pt \
  --contract cable1_remote_runtime_export_manifest.json \
  --deploy
```

舊模型沒有 export contract 時仍可使用同一工具，但必須通過數值等價驗證；成功後會產生版號化 ONNX、同版 `.training.pt`、config snapshot 與 `deployment_manifest.yaml`。未加 `--deploy` 時只驗證，不更動產線。

## 工程師注意事項

- 「誤報」不能直接把原推理框當正確答案，否則會強化錯誤，因此必須補標。
- 自動驗證可確認資料與模型流程可執行，但不能取代 golden-image regression 與現場 pilot；正式大量上線前仍應抽驗。
- 各產品可在 `yolo_evaluation.gate` 調整絕對門檻與允許退步幅度；調整必須有驗證資料依據。

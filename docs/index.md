# Picture Tool Documentation

Picture Tool 是一個工業視覺訓練工具包，自動化從資料準備到模型部署的完整流程。訓練完成的模型可直接部署到 [yolo11_inference](../yolo11_inference) 進行生產推論。

## 文件導覽

| 文件 | 說明 |
|------|------|
| [快速開始](guides/quickstart.md) | 安裝、設定、第一次訓練 |
| [設定參考](config_reference.md) | 所有 YAML 設定鍵值完整說明 |
| [整合指南](INTEGRATION_GUIDE.md) | 訓練 → 部署到 yolo11_inference 的完整流程 |
| [架構說明](ARCHITECTURE.md) | 系統架構、模組職責、資料流 |
| [Package 概覽](package_overview.md) | 公開 API 與 CLI 入口點 |
| [貢獻指南](CONTRIBUTING.md) | 開發環境、測試、PR 流程 |
| [程式碼品質](CODE_QUALITY_CHECKS.md) | Ruff、Mypy、測試覆蓋率說明 |

## API 文件

| 模組 | 說明 |
|------|------|
| [Pipeline](api/pipeline.md) | `Task`、`Pipeline`、`skip_fn` 合約 |
| [Tasks](api/tasks.md) | 所有任務的 `run` / `skip_fn` 函式 |
| [Augmentation](api/augmentation.md) | `ImageAugmentor`、`DataAugmentor` |
| [Config Validation](api/config.md) | Pydantic schema、`validate_config_schema` |

## 功能總覽

- **格式轉換** (`format_conversion`)：批次轉換影像格式
- **資料增強** (`yolo_augmentation` / `image_augmentation`)：Albumentations 增強
- **資料分割** (`dataset_splitter`)：切分 train / val / test
- **資料品質** (`dataset_lint`, `aug_preview`)：標註檢查、增強預覽
- **YOLO 訓練** (`yolo_train`)：自動版本化、hash-based skip、ONNX 匯出
- **評估** (`yolo_evaluation`)：驗證集指標評估
- **位置驗證** (`position_validation`)：離線位置校驗
- **顏色檢測** (`color_inspection` / `color_verification`)：LED 顏色統計驗證
- **報告** (`generate_report`, `qc_summary`)：訓練報告、QC 彙總

## 快速指令

```bash
# 完整訓練流程
picture-tool-pipeline --config configs/my_project.yaml --tasks full

# 僅重新訓練（強制，即使資料未變動）
picture-tool-pipeline --config configs/my_project.yaml --tasks yolo_train --force

# 列出所有任務
picture-tool-pipeline --list-tasks

# 啟動 GUI
picture-tool-gui --config configs/my_project.yaml
```

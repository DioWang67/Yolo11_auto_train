# 代碼質量檢查指南

## 🎯 目標

確保每次代碼修改都能通過 lint 檢查，避免 CI 失敗。

## 方案選擇

### 方案 1: Pre-commit Hooks（推薦）⭐

**優點**: 
- 自動執行，不會忘記
- 在提交前攔截問題
- 團隊成員統一標準

**設置步驟**:

```bash
# 1. 安裝 pre-commit
pip install pre-commit

# 2. 安裝 hooks
pre-commit install

# 3. 測試（可選）
pre-commit run --all-files
```

**使用**:
```bash
# 正常提交，hooks 會自動執行
git add .
git commit -m "feat: 新功能"
# 👆 會自動執行 ruff、mypy 等檢查

# 如果檢查失敗，修正後再次提交即可
```

---

### 方案 2: 手動檢查腳本（備用）

**優點**:
- 更靈活，可隨時執行
- 不依賴 git hooks

**使用**:

#### Windows (PowerShell):
```powershell
.\scripts\check_code.ps1
```

#### Linux/Mac:
```bash
python scripts/check_code.py
```

**何時使用**:
- 修改文件後，提交前
- CI 失敗後，本地重現問題

---

### 方案 3: IDE 整合（最佳體驗）

#### VS Code 設置

在專案根目錄的 `.vscode/settings.json` 中添加（如果資料夾或檔案不存在，請自行建立）：

```json
{
  "[python]": {
    "editor.formatOnSave": true,
    "editor.codeActionsOnSave": {
      "source.fixAll": true,
      "source.organizeImports": true
    },
    "editor.defaultFormatter": "charliermarsh.ruff"
  },
  "ruff.lint.enable": true,
  "ruff.format.enable": true,
  "python.linting.enabled": true,
  "python.linting.mypyEnabled": true
}
```

**安裝 VS Code 擴展**:
- Ruff (charliermarsh.ruff)
- Pylance (ms-python.vscode-pylance)
- MyPy Type Checker (matangover.mypy)

#### PyCharm 設置

1. Settings → Tools → External Tools
2. 添加 `ruff check --fix`
3. 設置快捷鍵（如 Ctrl+Alt+L）

---

## 📋 最佳實踐

### 開發工作流程

```bash
# 1. 修改代碼
vim src/picture_tool/some_module.py

# 2. 本地檢查（方案 2）
.\scripts\check_code.ps1

# 3. 運行測試
pytest tests/test_some_module.py -v

# 4. 提交（方案 1 會自動檢查）
git add .
git commit -m "fix: 修復某問題"

# 5. 推送
git push
```

### CI 失敗處理

```bash
# 1. 查看 CI 錯誤日誌
# 2. 本地重現
.\scripts\check_code.ps1

# 3. 修復
# 編輯代碼...

# 4. 驗證
ruff check src tests --fix
pytest

# 5. 提交修復
git add .
git commit -m "fix: 修復 lint 錯誤"
git push
```

---

## 🔧 常見問題

### Q: Pre-commit 太慢怎麼辦？

A: 可以跳過某次檢查：
```bash
git commit --no-verify -m "緊急修復"
```

### Q: 如何只檢查修改的文件？

A: Pre-commit 默認只檢查 staged 文件。手動運行：
```bash
ruff check $(git diff --name-only --cached | grep '\.py$')
```

### Q: Ruff 和 Black 衝突怎麼辦？

A: 使用 `ruff format` 替代 Black，完全相容且更快。

---

## 📊 檢查項目說明

| 工具 | 檢查內容 | 自動修復 |
|------|---------|---------|
| ruff check | 代碼風格、未使用導入等 | ✅ (--fix) |
| ruff format | 代碼格式化 | ✅ |
| mypy | 類型註解 | ❌ |
| pytest | 單元測試 | ❌ |

---

## 🎓 推薦順序

1. **立即設置**: Pre-commit hooks（方案 1）
2. **長期使用**: IDE 整合（方案 3）
3. **緊急備用**: 手動腳本（方案 2）

現在開始設置，徹底告別 lint 錯誤！🚀

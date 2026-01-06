# PowerShell 快速檢查腳本
# 使用方法: .\scripts\check_code.ps1

Write-Host "🚀 開始代碼檢查..." -ForegroundColor Cyan

# 1. Ruff Lint 檢查並自動修復
Write-Host "`n============================================================" -ForegroundColor Yellow
Write-Host "🔍 Ruff Lint 檢查並自動修復" -ForegroundColor Yellow
Write-Host "============================================================" -ForegroundColor Yellow
ruff check src tests --fix
if ($LASTEXITCODE -ne 0) { $failed = $true }

# 2. Ruff 代碼格式化
Write-Host "`n============================================================" -ForegroundColor Yellow
Write-Host "🔍 Ruff 代碼格式化" -ForegroundColor Yellow
Write-Host "============================================================" -ForegroundColor Yellow
ruff format src tests
if ($LASTEXITCODE -ne 0) { $failed = $true }

# 3. MyPy 類型檢查（僅檢查新模塊）
Write-Host "`n============================================================" -ForegroundColor Yellow
Write-Host "🔍 MyPy 類型檢查" -ForegroundColor Yellow
Write-Host "============================================================" -ForegroundColor Yellow
mypy src/picture_tool/exceptions.py src/picture_tool/validation.py --strict
if ($LASTEXITCODE -ne 0) { $failed = $true }

# 結果
Write-Host "`n============================================================" -ForegroundColor Cyan
if (-not $failed) {
    Write-Host "✅ 所有檢查通過！可以安心提交。" -ForegroundColor Green
    exit 0
} else {
    Write-Host "⚠️  部分檢查失敗，請檢查上方錯誤訊息。" -ForegroundColor Red
    exit 1
}

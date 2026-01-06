#!/usr/bin/env python3
"""
快速代碼檢查腳本
在提交前運行此腳本，自動修復常見問題
"""
import subprocess
import sys
from pathlib import Path

def run_command(cmd, description):
    """執行命令並顯示結果"""
    print(f"\n{'='*60}")
    print(f"🔍 {description}")
    print(f"{'='*60}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print(result.stderr)
    return result.returncode == 0

def main():
    """主函數"""
    print("🚀 開始代碼檢查...")
    
    checks = [
        ("ruff check src tests --fix", "Ruff Lint 檢查並自動修復"),
        ("ruff format src tests", "Ruff 代碼格式化"),
        ("mypy src/picture_tool/exceptions.py src/picture_tool/validation.py --strict", "MyPy 類型檢查"),
    ]
    
    all_passed = True
    for cmd, desc in checks:
        if not run_command(cmd, desc):
            all_passed = False
    
    print(f"\n{'='*60}")
    if all_passed:
        print("✅ 所有檢查通過！可以安心提交。")
        return 0
    else:
        print("⚠️  部分檢查失敗，請檢查上方錯誤訊息。")
        return 1

if __name__ == "__main__":
    sys.exit(main())

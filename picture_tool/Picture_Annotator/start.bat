@echo off
REM 指定 Miniconda 路徑
set CONDA_PATH=D:\miniconda

REM 設定環境變數 PATH，確保可以找到 conda
set PATH=%CONDA_PATH%;%CONDA_PATH%\Scripts;%CONDA_PATH%\Library\bin;%PATH%

REM 啟動 Conda 並激活環境
call %CONDA_PATH%\Scripts\activate.bat
call conda activate D:\Git\robotlearning\yolo_train\env

REM 切換到指定目錄
cd /d D:\Git\robotlearning\picture_tool\Picture_Annotator

REM 確認 Python 環境和 torch 模組
python --version
where python
python -c "import torch; print(torch.__version__)"

REM 執行 YOLOAutoAnnotator.py 並保持視窗開啟
python Image_AutoAnnotator.py
cmd /k

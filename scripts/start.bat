@echo off
setlocal EnableExtensions
title Picture Tool
for %%I in ("%~dp0..") do set "PROJECT_ROOT=%%~fI"
set "CHECK_MODE=0"
if /I "%~1"=="--check" set "CHECK_MODE=1"
cd /d "%PROJECT_ROOT%" || (
    echo ERROR: Training project directory was not found: %PROJECT_ROOT%
    if "%CHECK_MODE%"=="0" pause
    exit /b 1
)

set "PYTHON_EXE="
if defined PICTURE_TOOL_PYTHON if exist "%PICTURE_TOOL_PYTHON%" set "PYTHON_EXE=%PICTURE_TOOL_PYTHON%"
if not defined PYTHON_EXE if exist "%PROJECT_ROOT%\.venv\Scripts\python.exe" set "PYTHON_EXE=%PROJECT_ROOT%\.venv\Scripts\python.exe"
if not defined PYTHON_EXE if exist "D:\miniconda\envs\anomalib_env\python.exe" set "PYTHON_EXE=D:\miniconda\envs\anomalib_env\python.exe"
if not defined PYTHON_EXE if exist "D:\miniconda\envs\yolo_anomalib\python.exe" set "PYTHON_EXE=D:\miniconda\envs\yolo_anomalib\python.exe"
if not defined PYTHON_EXE (
    where python >nul 2>nul
    if not errorlevel 1 set "PYTHON_EXE=python"
)
if not defined PYTHON_EXE (
    echo ERROR: Python was not found.
    if "%CHECK_MODE%"=="0" pause
    exit /b 1
)

set "PYTHONPATH=%PROJECT_ROOT%\src;%PYTHONPATH%"
"%PYTHON_EXE%" -c "import picture_tool" >nul 2>nul
if errorlevel 1 (
    echo ERROR: Python cannot import picture_tool.
    echo Python: %PYTHON_EXE%
    echo Source: %PROJECT_ROOT%\src
    if "%CHECK_MODE%"=="0" pause
    exit /b 1
)
if "%CHECK_MODE%"=="1" (
    echo Picture Tool launcher OK
    echo Python: %PYTHON_EXE%
    echo Source: %PROJECT_ROOT%\src
    exit /b 0
)

"%PYTHON_EXE%" -m picture_tool.gui.app
if errorlevel 1 (
    echo ERROR: Picture Tool failed to start.
    pause
    exit /b 1
)
exit /b 0

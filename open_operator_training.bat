@echo off
setlocal EnableExtensions
title YOLO Operator Training Center
cd /d "%~dp0"

if /I "%~1"=="--check" (
    echo Operator training launcher OK
    exit /b 0
)

if "%~1"=="" (
    echo ERROR: Missing operator handoff path.
    pause
    exit /b 1
)

set "PYTHON_EXE="
if defined PICTURE_TOOL_PYTHON if exist "%PICTURE_TOOL_PYTHON%" set "PYTHON_EXE=%PICTURE_TOOL_PYTHON%"
if not defined PYTHON_EXE if exist ".venv\Scripts\python.exe" set "PYTHON_EXE=.venv\Scripts\python.exe"
if not defined PYTHON_EXE if exist "D:\miniconda\envs\anomalib_env\python.exe" set "PYTHON_EXE=D:\miniconda\envs\anomalib_env\python.exe"
if not defined PYTHON_EXE if exist "D:\miniconda\envs\yolo_anomalib\python.exe" set "PYTHON_EXE=D:\miniconda\envs\yolo_anomalib\python.exe"
if not defined PYTHON_EXE (
    where python >nul 2>nul
    if not errorlevel 1 set "PYTHON_EXE=python"
)
if not defined PYTHON_EXE (
    echo ERROR: Python was not found.
    pause
    exit /b 1
)

set "PYTHONPATH=%CD%\src;%PYTHONPATH%"
"%PYTHON_EXE%" -m picture_tool.gui.app --handoff "%~1"
if errorlevel 1 (
    echo ERROR: Training center failed to start.
    pause
    exit /b 1
)
exit /b 0

@echo off
setlocal EnableExtensions
title YOLO Operator Training Center
set "PROJECT_ROOT=%~dp0"
set "CHECK_MODE=0"
set "BACKGROUND_ARG="
if /I "%~1"=="--check" set "CHECK_MODE=1"
if /I "%~2"=="--background" set "BACKGROUND_ARG=--background"
cd /d "%PROJECT_ROOT%" || (
    echo ERROR: Training project directory was not found: %PROJECT_ROOT%
    if "%CHECK_MODE%"=="0" pause
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
    if "%CHECK_MODE%"=="0" if not defined BACKGROUND_ARG pause
    exit /b 1
)

set "PYTHONPATH=%PROJECT_ROOT%src;%PYTHONPATH%"
"%PYTHON_EXE%" -c "import picture_tool" >nul 2>nul
if errorlevel 1 (
    echo ERROR: Python cannot import picture_tool.
    echo Python: %PYTHON_EXE%
    echo Source: %PROJECT_ROOT%src
    if "%CHECK_MODE%"=="0" if not defined BACKGROUND_ARG pause
    exit /b 1
)
if "%CHECK_MODE%"=="1" (
    echo Operator training launcher OK
    echo Python: %PYTHON_EXE%
    echo Source: %PROJECT_ROOT%src
    exit /b 0
)

if /I "%~1"=="--import" (
    if "%~2"=="" (
        echo ERROR: Missing portable training ZIP path.
        pause
        exit /b 1
    )
    "%PYTHON_EXE%" -m picture_tool.gui.app --import-package "%~2"
) else if "%~1"=="" (
    "%PYTHON_EXE%" -m picture_tool.gui.app --resume-latest
) else (
    "%PYTHON_EXE%" -m picture_tool.gui.app --handoff "%~1" %BACKGROUND_ARG%
)
if errorlevel 1 (
    echo ERROR: Training center failed to start.
    if not defined BACKGROUND_ARG pause
    exit /b 1
)
exit /b 0

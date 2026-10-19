@echo off
REM Autonomous training, unattended: import every labelled folder waiting in
REM the inbox, train and evaluate a challenger, write the report. Never deploys.
REM
REM   run_autotrain.bat                 one run with configs\autonomous_training.yaml
REM   run_autotrain.bat --config X.yaml one run with another settings file
REM   run_autotrain.bat --check         only check that the runtime can start
REM
REM Safe to put in Task Scheduler as is: with nothing new it does nothing, and
REM it refuses to start while another run is still training. It never pauses,
REM so a scheduled run cannot hang on a prompt. Exit code 0 = all fine,
REM 1 = a batch was refused or held, or a cycle ended without a result,
REM 2 = autonomous training is disabled in the settings file.
setlocal EnableExtensions
set "PROJECT_ROOT=%~dp0"
set "YOLO_CONFIG_DIR=%PROJECT_ROOT%data\.ultralytics"
rem Ultralytics must not pip-install into the locked env (stations are offline).
set "YOLO_AUTOINSTALL=False"
if not exist "%YOLO_CONFIG_DIR%" mkdir "%YOLO_CONFIG_DIR%"
cd /d "%PROJECT_ROOT%" || (
    echo ERROR: Training project directory was not found: %PROJECT_ROOT%
    exit /b 1
)

REM The same interpreter search as open_operator_training.bat, so a run trains
REM in exactly the environment operator retraining uses.
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
    exit /b 1
)

REM Source on the path directly: the runner needs nothing installed, not even
REM typer, and the cycle it launches inherits this environment.
set "PYTHONPATH=%PROJECT_ROOT%src;%PYTHONPATH%"
"%PYTHON_EXE%" -c "import picture_tool" >nul 2>nul
if errorlevel 1 (
    echo ERROR: Python cannot import picture_tool.
    echo Python: %PYTHON_EXE%
    echo Source: %PROJECT_ROOT%src
    exit /b 1
)
"%PYTHON_EXE%" -m picture_tool.doctor --runtime-only
if errorlevel 1 (
    echo ERROR: Training runtime environment check failed.
    echo Python: %PYTHON_EXE%
    exit /b 1
)
if /I "%~1"=="--check" (
    echo Autonomous training launcher OK
    echo Python: %PYTHON_EXE%
    echo Source: %PROJECT_ROOT%src
    exit /b 0
)

"%PYTHON_EXE%" -m picture_tool.autotrain.runner %*
exit /b %ERRORLEVEL%

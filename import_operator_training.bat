@echo off
setlocal EnableExtensions
title Import Portable YOLO Training Package
if "%~1"=="" (
    echo Usage: import_operator_training.bat ^<portable-training-package.zip^>
    pause
    exit /b 1
)
call "%~dp0open_operator_training.bat" --import "%~1"
exit /b %errorlevel%

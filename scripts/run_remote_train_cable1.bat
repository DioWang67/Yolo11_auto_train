@echo off
setlocal
REM ===========================================================================
REM  Remote train the Cable1 color detector on the GPU box (mujoco-style).
REM  Uploads script + dataset + base model, trains in an ISOLATED venv (so it
REM  never disturbs the mujoco anaconda env), then pulls the verified ONNX/PT
REM  pair and its checksum contract back.
REM
REM  Usage:   scripts\run_remote_train_cable1.bat
REM  Override: set REMOTE_HOST / REMOTE_DIR / REMOTE_BASE_PY before running.
REM ===========================================================================

if "%REMOTE_HOST%"==""  set "REMOTE_HOST=root@10.6.243.55"
if "%REMOTE_DIR%"==""   set "REMOTE_DIR=/root/anaconda3/yolo11_cable1_train"
REM yolo_train_env already has ultralytics 8.3.56 + torch CUDA on the GPU box.
if "%REMOTE_PY%"==""    set "REMOTE_PY=/root/anaconda3/envs/yolo_train_env/bin/python"

REM --- repo root = parent of this script ---
pushd "%~dp0.."
set "LOCAL_ROOT=%CD%"
popd

echo [1/5] create remote dirs on %REMOTE_HOST%
ssh %REMOTE_HOST% "mkdir -p %REMOTE_DIR%/scripts %REMOTE_DIR%/models %REMOTE_DIR%/data/Cable1"
if errorlevel 1 goto fail

echo [2/5] upload training script + base model
scp "%LOCAL_ROOT%\scripts\train_cable1_color.py" %REMOTE_HOST%:%REMOTE_DIR%/scripts/
if errorlevel 1 goto fail
scp "%LOCAL_ROOT%\models\yolo11n.pt" %REMOTE_HOST%:%REMOTE_DIR%/models/
if errorlevel 1 goto fail

echo [3/5] upload dataset (this is the slow step, a few hundred MB)
scp -r "%LOCAL_ROOT%\data\Cable1\split" %REMOTE_HOST%:%REMOTE_DIR%/data/Cable1/
if errorlevel 1 goto fail

echo [3b/5] remove stale label .cache (built by a different numpy; breaks loading)
ssh %REMOTE_HOST% "find %REMOTE_DIR%/data/Cable1/split -name *.cache -delete"

echo [4/5] train on GPU using yolo_train_env (no install needed)
ssh -t %REMOTE_HOST% "bash -lc 'set -o pipefail; cd %REMOTE_DIR% && %REMOTE_PY% -u scripts/train_cable1_color.py --device 0 2>&1 | tee train_cable1.log'"
if errorlevel 1 goto fail

echo [5/5] pull paired artifacts and their export contract back
set "RW=%REMOTE_DIR%/runs/Cable1/train_color_fixed/weights"
set "RR=%REMOTE_DIR%/runs/Cable1/train_color_fixed"
set "LPT=%LOCAL_ROOT%\best_cable1_remote.pt.download"
set "LONNX=%LOCAL_ROOT%\best_cable1_remote.onnx.download"
set "LMANIFEST=%LOCAL_ROOT%\cable1_remote_runtime_export_manifest.json.download"
if exist "%LPT%" del /q "%LPT%"
if exist "%LONNX%" del /q "%LONNX%"
if exist "%LMANIFEST%" del /q "%LMANIFEST%"
scp %REMOTE_HOST%:%RW%/best.pt "%LPT%"
if errorlevel 1 goto fail
scp %REMOTE_HOST%:%RW%/best.onnx "%LONNX%"
if errorlevel 1 goto fail
scp %REMOTE_HOST%:%RR%/runtime_export_manifest.json "%LMANIFEST%"
if errorlevel 1 goto fail
move /y "%LPT%" "%LOCAL_ROOT%\best_cable1_remote.pt" >nul
if errorlevel 1 goto fail
move /y "%LONNX%" "%LOCAL_ROOT%\best_cable1_remote.onnx" >nul
if errorlevel 1 goto fail
move /y "%LMANIFEST%" "%LOCAL_ROOT%\cable1_remote_runtime_export_manifest.json" >nul
if errorlevel 1 goto fail

echo.
echo DONE. Downloaded a checksum-bound ONNX/PT pair.
echo Next (on the inference machine):
echo   python scripts\validate_and_deploy_cable1.py --weights "%LOCAL_ROOT%\best_cable1_remote.onnx" --training-weights "%LOCAL_ROOT%\best_cable1_remote.pt" --contract "%LOCAL_ROOT%\cable1_remote_runtime_export_manifest.json" --deploy
endlocal
exit /b 0

:fail
echo.
echo [ERROR] remote train step failed (exit %errorlevel%). Check SSH access / remote env.
if defined LPT if exist "%LPT%" del /q "%LPT%"
if defined LONNX if exist "%LONNX%" del /q "%LONNX%"
if defined LMANIFEST if exist "%LMANIFEST%" del /q "%LMANIFEST%"
endlocal
exit /b 1

@echo off
REM ============================================================
REM  Whispering - Windows one-click launcher
REM
REM  What it does:
REM    1. Checks for Python 3.11+
REM    2. Creates a virtual environment (.venv)
REM    3. Installs the app and dependencies (first run only)
REM    4. (GPU mode) Installs CUDA libraries for faster-whisper
REM    5. Downloads the model to %USERPROFILE%\whispering-models
REM    6. Launches Whispering
REM
REM  Usage:
REM    run.bat            GPU (CUDA) mode - default, for GTX 1660
REM    run.bat --cpu      CPU-only mode (skips CUDA libraries)
REM    set MODEL=large-v3 && run.bat     use a different model
REM ============================================================
setlocal enableextensions
cd /d "%~dp0"

set "VENV_DIR=.venv"
if "%MODELS_DIR%"=="" set "MODELS_DIR=%USERPROFILE%\whispering-models"
set "WHISPERING_MODELS_DIR=%MODELS_DIR%"
if "%MODEL%"=="" set "MODEL=medium"
set "MODEL_REPO=Systran/faster-whisper-%MODEL%"
set "MODEL_PATH=%MODELS_DIR%\faster-whisper-%MODEL%"
REM Tell the app exactly which local model directory to use (no UI selection).
set "WHISPERING_MODEL_PATH=%MODEL_PATH%"

set "USE_GPU=1"
if /i "%~1"=="--cpu" set "USE_GPU=0"

REM --- 1) Python check ---
where python >nul 2>&1
if errorlevel 1 (
    echo [run] Python not found on PATH.
    echo [run] Install Python 3.11+ from https://www.python.org/downloads/
    echo [run] and be sure to check "Add python.exe to PATH" during install.
    pause
    exit /b 1
)

REM --- 2) Virtual environment (recreate if not a valid Windows venv) ---
if not exist "%VENV_DIR%\Scripts\python.exe" (
    echo [run] Creating virtual environment...
    if exist "%VENV_DIR%" rmdir /s /q "%VENV_DIR%"
    python -m venv "%VENV_DIR%"
    if errorlevel 1 ( echo [run] Failed to create venv & pause & exit /b 1 )
)
call "%VENV_DIR%\Scripts\activate.bat"

REM --- 3) Install app + dependencies (first run only) ---
REM Editable install so the latest source is always used (avoids pip skipping
REM reinstall when the version number is unchanged, which can leave stale code).
python -c "import whispering, faster_whisper" >nul 2>&1
if errorlevel 1 (
    echo [run] Installing app and dependencies...
    python -m pip install --upgrade pip
    python -m pip install -e .
    python -m pip install huggingface_hub
    if errorlevel 1 ( echo [run] Install failed & pause & exit /b 1 )
) else (
    python -m pip install -e . --no-deps --quiet
)

REM --- 4) GPU libraries (CUDA 12 / cuDNN 9) ---
if "%USE_GPU%"=="1" (
    python -c "import nvidia.cublas, nvidia.cudnn" >nul 2>&1
    if errorlevel 1 (
        echo [run] Installing CUDA libraries ^(cuBLAS, cuDNN 9^)...
        python -m pip install nvidia-cuda-runtime-cu12 nvidia-cublas-cu12 "nvidia-cudnn-cu12==9.*"
    )
) else (
    echo [run] CPU-only mode.
)

REM --- 5) Download model (first run only) ---
if not exist "%MODEL_PATH%\model.bin" (
    echo [run] Downloading model %MODEL_REPO% to "%MODEL_PATH%" ...
    if not exist "%MODELS_DIR%" mkdir "%MODELS_DIR%"
    hf download %MODEL_REPO% --local-dir "%MODEL_PATH%"
    if errorlevel 1 ( echo [run] Model download failed & pause & exit /b 1 )
) else (
    echo [run] Model already present: %MODEL_PATH%
)

REM --- 6) Launch ---
echo [run] Starting Whispering...
whispering
endlocal

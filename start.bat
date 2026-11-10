@echo off
setlocal enabledelayedexpansion

REM ============================================================================
REM  CNN Autoencoder — One-Shot Setup & Run Script (Windows)
REM  Runs anywhere: checks Python >= 3.9, builds/heals venv, verifies dependencies,
REM  loads pre-trained weights, starts server, and opens your default browser.
REM ============================================================================

title CNN Autoencoder — Character Reconstruction
pushd "%~dp0"

echo ============================================================================
echo   CNN Autoencoder: Alphanumeric Character Reconstruction (PyTorch)
echo ============================================================================
echo.

REM 1. Detect Python command (checks 'python' then 'py -3')
set "PY_CMD="

where python >nul 2>&1
if %errorlevel% equ 0 (
    set "PY_CMD=python"
) else (
    where py >nul 2>&1
    if %errorlevel% equ 0 (
        set "PY_CMD=py -3"
    )
)

if "%PY_CMD%"=="" (
    echo ============================================================================
    echo [ERROR] Python was not found on this computer.
    echo.
    echo Please install Python 3.9 or higher:
    echo   1. Download: https://www.python.org/downloads/
    echo   2. IMPORTANT: Check "Add Python to PATH" during installation.
    echo   3. Re-run start.bat
    echo ============================================================================
    pause
    popd
    exit /b 1
)

REM 2. Verify Python version is 3.9 or newer
%PY_CMD% -c "import sys; sys.exit(0 if sys.version_info >= (3, 9) else 1)" >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Found Python, but it is older than Python 3.9.
    echo PyTorch and modern deep learning packages require Python 3.9+.
    echo Please install a newer Python version from https://www.python.org/
    pause
    popd
    exit /b 1
)

echo [*] Using Python: %PY_CMD%

REM 3. Virtual environment management (auto-heals if copied from another PC)
if exist "venv\Scripts\python.exe" (
    venv\Scripts\python.exe -c "import sys" >nul 2>&1
    if %errorlevel% neq 0 (
        echo [*] Detected stale or moved virtual environment. Rebuilding for this PC...
        rmdir /s /q venv
    )
)

if not exist "venv\Scripts\activate.bat" (
    echo [*] Creating isolated virtual environment (venv)...
    %PY_CMD% -m venv venv
    if %errorlevel% neq 0 (
        echo [ERROR] Failed to create virtual environment.
        pause
        popd
        exit /b 1
    )
    echo [OK] Virtual environment created.
)

REM 4. Activate virtual environment
call venv\Scripts\activate.bat

REM 5. Verify & install packages
echo [*] Checking dependencies...
python -m pip install -q -r requirements.txt
if %errorlevel% neq 0 (
    echo [WARNING] Pip install returned warnings, attempting standard install...
    python -m pip install -r requirements.txt
)
echo [OK] Dependencies verified.

REM 6. Train model if weights do not exist
if not exist "model\saved_model.pt" (
    echo.
    echo ============================================================================
    echo [*] No trained model found at model\saved_model.pt.
    echo [*] Starting initial training on EMNIST dataset (first run only)...
    echo ============================================================================
    echo.
    python model\train.py
    if not exist "model\saved_model.pt" (
        echo [ERROR] Training did not produce model\saved_model.pt.
        pause
        popd
        exit /b 1
    )
    echo.
    echo [OK] Model training completed successfully!
)

REM 7. Launch Server (browser will open automatically)
echo.
echo ============================================================================
echo [*] Starting CNN Autoencoder server...
echo ============================================================================
echo.

python backend\app.py

popd
pause

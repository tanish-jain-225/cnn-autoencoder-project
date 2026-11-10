@echo off
setlocal

title CNN Autoencoder — Character Reconstruction
pushd "%~dp0"

echo ============================================================================
echo   CNN Autoencoder: Alphanumeric Character Reconstruction [PyTorch]
echo ============================================================================
echo.

REM 1. Detect Python command (checks PATH, py launcher, and common install folders)
set "PY_CMD="

REM Check py launcher (official Windows Python launcher)
where py >nul 2>&1
if not errorlevel 1 (
    py -3 -c "import sys" >nul 2>&1
    if not errorlevel 1 set "PY_CMD=py -3"
)

REM Check python in PATH
if "%PY_CMD%"=="" (
    where python >nul 2>&1
    if not errorlevel 1 (
        python -c "import sys" >nul 2>&1
        if not errorlevel 1 set "PY_CMD=python"
    )
)

REM Check common default Windows installation folders if PATH was not set
if "%PY_CMD%"=="" (
    for %%V in (313 312 311 310 39) do (
        if exist "%LOCALAPPDATA%\Programs\Python\Python%%V\python.exe" (
            set "PY_CMD=%LOCALAPPDATA%\Programs\Python\Python%%V\python.exe"
            goto :FOUND_PYTHON
        )
        if exist "%ProgramFiles%\Python%%V\python.exe" (
            set "PY_CMD=%ProgramFiles%\Python%%V\python.exe"
            goto :FOUND_PYTHON
        )
        if exist "C:\Python%%V\python.exe" (
            set "PY_CMD=C:\Python%%V\python.exe"
            goto :FOUND_PYTHON
        )
    )
)

:FOUND_PYTHON
if "%PY_CMD%"=="" goto :NO_PYTHON

REM 2. Verify Python version is 3.9 or newer
%PY_CMD% -c "import sys; sys.exit(0 if sys.version_info >= (3, 9) else 1)" >nul 2>&1
if errorlevel 1 goto :OLD_PYTHON

echo [*] Using Python: %PY_CMD%

REM 3. Virtual environment management (auto-heals if copied from another PC)
if exist "venv\Scripts\python.exe" (
    venv\Scripts\python.exe -c "import sys" >nul 2>&1
    if errorlevel 1 (
        echo [*] Detected stale or moved virtual environment from another PC.
        echo [*] Automatically rebuilding environment for this computer...
        rmdir /s /q venv
    )
)

if not exist "venv\Scripts\activate.bat" (
    echo [*] Setting up isolated virtual environment [venv]...
    %PY_CMD% -m venv venv
    if errorlevel 1 goto :VENV_FAILED
    echo [OK] Virtual environment created.
)

REM 4. Activate virtual environment
call venv\Scripts\activate.bat

REM 5. Verify & install dependencies
python -c "import torch, torchvision, flask, numpy, PIL, tqdm" >nul 2>&1
if not errorlevel 1 (
    echo [OK] All required packages are ready.
) else (
    echo [*] Checking and installing dependencies from requirements.txt...
    python -m pip install --quiet -r requirements.txt
    if errorlevel 1 (
        echo [WARNING] Quiet pip install had warnings. Running standard install...
        python -m pip install -r requirements.txt
        if errorlevel 1 goto :PIP_FAILED
    )
    echo [OK] Dependencies verified.
)

REM 6. Check trained model weights
if not exist "model\saved_model.pt" goto :TRAIN_MODEL
for %%F in ("model\saved_model.pt") do if %%~zF equ 0 goto :TRAIN_MODEL
echo [OK] Loaded pre-trained model weights [model\saved_model.pt].
goto :RUN_SERVER

:TRAIN_MODEL
echo.
echo ============================================================================
echo [*] No trained model found at model\saved_model.pt.
echo [*] Starting initial training on EMNIST dataset [happens once]...
echo ============================================================================
echo.
python model\train.py
if not exist "model\saved_model.pt" goto :TRAIN_FAILED
echo [OK] Model training completed successfully!

:RUN_SERVER
if "%1"=="--dry-run" (
    echo [OK] start.bat dry-run verification passed successfully!
    popd
    exit /b 0
)
echo.
echo ============================================================================
echo [*] Starting CNN Autoencoder server...
echo ============================================================================
echo.

python backend\app.py

popd
pause
exit /b 0

:NO_PYTHON
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

:OLD_PYTHON
echo ============================================================================
echo [ERROR] Found Python, but it is older than Python 3.9.
echo PyTorch and modern deep learning packages require Python 3.9+.
echo Please install a newer Python version from https://www.python.org/
echo ============================================================================
pause
popd
exit /b 1

:VENV_FAILED
echo ============================================================================
echo [ERROR] Failed to create virtual environment.
echo Please ensure you have write permissions in this folder.
echo ============================================================================
pause
popd
exit /b 1

:PIP_FAILED
echo ============================================================================
echo [ERROR] Failed to install dependencies via pip.
echo Please verify your internet connection on first run and try again.
echo ============================================================================
pause
popd
exit /b 1

:TRAIN_FAILED
echo ============================================================================
echo [ERROR] Training did not produce model\saved_model.pt.
echo ============================================================================
pause
popd
exit /b 1

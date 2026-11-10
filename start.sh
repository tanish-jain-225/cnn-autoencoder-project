#!/usr/bin/env bash
# ============================================================================
#  CNN Autoencoder — One-Shot Setup & Run Script (Mac / Linux / Unix / WSL)
#  Runs anywhere: detects Python 3.9+, auto-heals virtual environments,
#  verifies dependencies, loads pre-trained weights, and launches server.
# ============================================================================

set -e
cd "$(dirname "$0")"

echo "============================================================================"
echo "  CNN Autoencoder: Handwritten Digit Reconstruction (0-9) [PyTorch]"
echo "============================================================================"
echo ""

# 1. Detect Python 3 (checks PATH, Homebrew, and system binary locations)
PYTHON_BIN=""

for candidate in python3 python /opt/homebrew/bin/python3 /usr/local/bin/python3 /usr/bin/python3; do
    if command -v "$candidate" >/dev/null 2>&1; then
        if "$candidate" -c "import sys; sys.exit(0 if sys.version_info >= (3, 9) else 1)" >/dev/null 2>&1; then
            PYTHON_BIN="$candidate"
            break
        fi
    fi
done

if [ -z "$PYTHON_BIN" ]; then
    echo "============================================================================"
    echo "[ERROR] Python 3.9 or higher was not found on this computer."
    echo ""
    echo "Please install Python 3.9 or newer:"
    echo "  - Mac:   brew install python3"
    echo "  - Linux: sudo apt update && sudo apt install -y python3 python3-venv python3-pip"
    echo "============================================================================"
    exit 1
fi

echo "[*] Using Python: $PYTHON_BIN"

# 2. Virtual environment management (auto-heals if copied from another computer)
VENV_DIR="venv"
VENV_PY="venv/bin/python"

if [ -f "$VENV_PY" ]; then
    if ! "$VENV_PY" -c "import sys" >/dev/null 2>&1; then
        echo "[*] Detected stale or moved virtual environment from another computer."
        echo "[*] Automatically rebuilding environment for this computer..."
        rm -rf "$VENV_DIR"
    fi
fi

if [ ! -f "$VENV_PY" ]; then
    echo "[*] Setting up isolated virtual environment [venv]..."
    if ! "$PYTHON_BIN" -m venv "$VENV_DIR" >/dev/null 2>&1; then
        if ! "$PYTHON_BIN" -m virtualenv "$VENV_DIR" >/dev/null 2>&1; then
            echo "============================================================================"
            echo "[ERROR] Failed to create virtual environment."
            echo "On Ubuntu/Debian, install the venv package:"
            echo "  sudo apt install -y python3-venv"
            echo "============================================================================"
            exit 1
        fi
    fi
    echo "[OK] Virtual environment created."
fi

# 3. Activate virtual environment if activate script exists
if [ -f "venv/bin/activate" ]; then
    . "venv/bin/activate" 2>/dev/null || source "venv/bin/activate" 2>/dev/null || true
fi

# 4. Verify dependencies (skips pip if already installed)
if "$VENV_PY" -c "import torch, torchvision, flask, numpy, PIL, tqdm" >/dev/null 2>&1; then
    echo "[OK] All required packages are ready."
else
    echo "[*] Checking and installing dependencies from requirements.txt..."
    "$VENV_PY" -m pip install --quiet -r requirements.txt 2>/dev/null || \
    "$VENV_PY" -m pip install -r requirements.txt || {
        echo "[WARNING] Retrying after pip upgrade..."
        "$VENV_PY" -m pip install --upgrade pip >/dev/null 2>&1 || true
        "$VENV_PY" -m pip install -r requirements.txt
    }
    echo "[OK] Dependencies verified."
fi

# 5. Check model weights
if [ ! -s "model/saved_model.pt" ]; then
    echo ""
    echo "============================================================================"
    echo "[*] No trained model found at model/saved_model.pt."
    echo "[*] Starting initial training on EMNIST Digits dataset [happens once]..."
    echo "============================================================================"
    echo ""
    "$VENV_PY" "model/train.py"
    if [ ! -s "model/saved_model.pt" ]; then
        echo "[ERROR] Training did not produce model/saved_model.pt."
        exit 1
    fi
    echo "[OK] Model training completed successfully!"
else
    echo "[OK] Loaded pre-trained model weights [model/saved_model.pt]."
fi

# 6. Launch Server (browser opens automatically)
if [ "$1" = "--dry-run" ]; then
    echo "[OK] start.sh dry-run verification passed successfully!"
    exit 0
fi

echo ""
echo "============================================================================"
echo "[*] Starting CNN Autoencoder server..."
echo "============================================================================"
echo ""

"$VENV_PY" "backend/app.py"

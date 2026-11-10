#!/usr/bin/env bash
# ============================================================================
#  CNN Autoencoder — One-Shot Setup & Run Script (Mac / Linux)
#  Runs anywhere: detects Python 3.9+, auto-heals virtual environments,
#  verifies dependencies, loads pre-trained weights, and launches server.
# ============================================================================

set -e
cd "$(dirname "$0")"

echo "============================================================================"
echo "  CNN Autoencoder: Alphanumeric Character Reconstruction [PyTorch]"
echo "============================================================================"
echo ""

# 1. Detect Python 3
PYTHON_BIN=""

for candidate in python3 python /opt/homebrew/bin/python3 /usr/local/bin/python3 /usr/bin/python3; do
    if command -v "$candidate" &> /dev/null; then
        if "$candidate" -c "import sys; sys.exit(0 if sys.version_info >= (3, 9) else 1)" 2>/dev/null; then
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
if [ -f "venv/bin/python" ]; then
    if ! venv/bin/python -c "import sys" 2>/dev/null; then
        echo "[*] Detected stale or moved virtual environment from another computer."
        echo "[*] Automatically rebuilding environment for this computer..."
        rm -rf venv
    fi
fi

if [ ! -f "venv/bin/activate" ]; then
    echo "[*] Setting up isolated virtual environment [venv]..."
    if ! "$PYTHON_BIN" -m venv venv 2>/dev/null; then
        echo "============================================================================"
        echo "[ERROR] Failed to create virtual environment."
        echo "On Ubuntu/Debian, install the venv package:"
        echo "  sudo apt install -y python3-venv"
        echo "============================================================================"
        exit 1
    fi
    echo "[OK] Virtual environment created."
fi

# 3. Activate virtual environment
if [ -f "venv/bin/activate" ]; then
    . "venv/bin/activate" 2>/dev/null || source "venv/bin/activate"
fi

# 4. Verify dependencies (skips pip if already installed)
if python3 -c "import torch, torchvision, flask, numpy, PIL, tqdm" 2>/dev/null || python -c "import torch, torchvision, flask, numpy, PIL, tqdm" 2>/dev/null; then
    echo "[OK] All required packages are ready."
else
    echo "[*] Checking and installing dependencies from requirements.txt..."
    python3 -m pip install --quiet -r requirements.txt 2>/dev/null || python -m pip install --quiet -r requirements.txt || python -m pip install -r requirements.txt
    echo "[OK] Dependencies verified."
fi

# 5. Check model weights
if [ ! -s "model/saved_model.pt" ]; then
    echo ""
    echo "============================================================================"
    echo "[*] No trained model found at model/saved_model.pt."
    echo "[*] Starting initial training on EMNIST dataset [happens once]..."
    echo "============================================================================"
    echo ""
    python3 "model/train.py" 2>/dev/null || python "model/train.py"
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

python3 "backend/app.py" 2>/dev/null || python "backend/app.py"

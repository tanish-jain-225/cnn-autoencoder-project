#!/usr/bin/env bash
# ============================================================================
#  CNN Autoencoder — One-Shot Setup & Run Script (Mac / Linux)
# ============================================================================

set -e
cd "$(dirname "$0")"

echo "============================================================================"
echo "  CNN Autoencoder: Alphanumeric Character Reconstruction (PyTorch)"
echo "============================================================================"
echo ""

# 1. Detect Python 3
PYTHON_BIN="python3"
if ! command -v "$PYTHON_BIN" &> /dev/null; then
    PYTHON_BIN="python"
fi

if ! command -v "$PYTHON_BIN" &> /dev/null; then
    echo "[ERROR] Python 3 was not found. Please install Python 3.9 or newer."
    exit 1
fi

# 2. Check Python version >= 3.9
if ! "$PYTHON_BIN" -c "import sys; sys.exit(0 if sys.version_info >= (3, 9) else 1)"; then
    echo "[ERROR] Found Python, but it is older than 3.9."
    echo "Please upgrade Python to 3.9 or newer."
    exit 1
fi

echo "[*] Using Python: $PYTHON_BIN"

# 3. Virtual environment management (auto-heals if copied from another machine)
if [ -f "venv/bin/python" ]; then
    if ! venv/bin/python -c "import sys" &> /dev/null; then
        echo "[*] Detected stale or moved virtual environment. Rebuilding for this machine..."
        rm -rf venv
    fi
fi

if [ ! -f "venv/bin/activate" ]; then
    echo "[*] Creating isolated virtual environment (venv)..."
    "$PYTHON_BIN" -m venv venv
    echo "[OK] Virtual environment created."
fi

# 4. Activate
source venv/bin/activate

# 5. Dependencies
echo "[*] Checking dependencies..."
python -m pip install -q -r requirements.txt
echo "[OK] Dependencies verified."

# 6. Model check
if [ ! -f "model/saved_model.pt" ]; then
    echo ""
    echo "[*] No trained model found at model/saved_model.pt."
    echo "[*] Starting initial training on EMNIST dataset (first run only)..."
    echo ""
    python model/train.py
    if [ ! -f "model/saved_model.pt" ]; then
        echo "[ERROR] Training did not produce model/saved_model.pt."
        exit 1
    fi
    echo "[OK] Model training completed successfully!"
fi

# 7. Start server (browser opens automatically)
echo ""
echo "============================================================================"
echo "[*] Starting CNN Autoencoder server..."
echo "============================================================================"
echo ""

python backend/app.py

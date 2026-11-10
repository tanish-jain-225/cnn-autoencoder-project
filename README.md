# CNN Autoencoder — Handwritten Digit Reconstruction (0–9)

An end-to-end Generative AI deep learning application: draw any digit (0–9) on an interactive canvas, and a Convolutional Autoencoder (trained on the EMNIST Digits dataset) compresses it into a compact **64-dimensional latent vector** and reconstructs it locally with PyTorch.

---

## ✨ Features

- **End-to-End Local Execution:** No cloud APIs, no external keys — runs 100% locally on CPU or CUDA GPU.
- **Deep CNN Encoder-Decoder:**
  - **Encoder:** Convolutions downsampling $28\times28 \to 14\times14 \to 7\times7 \to 1\times1 \to 64\text{D}$ latent code ($12.25\times$ compression).
  - **Decoder:** Transposed convolutions upsampling $64\text{D} \to 1\times1 \to 7\times7 \to 14\times14 \to 28\times28$.
- **Interactive Web Interface:**
  - **Dual Drawing Tools:** Pen and Eraser with Thin (10px), Medium (18px), and Thick (26px) brush sizes.
  - **Click-to-Dot & Touch Support:** Optimized for desktop mouse and mobile/tablet touch screens.
  - **Quick Digit Presets:** Instant test numbers (`0`, `1`, `2`, `3`, `4`, `5`, `6`, `7`, `8`, `9`).
  - **Upload Support:** Upload any digit image file directly from your computer.
- **Side-by-Side Visual Pipeline:**
  1. *Your Drawing* (280×280 interactive canvas)
  2. *What the Model Saw* (28×28 centered, padded, and normalized input)
  3. *Reconstructed Output* (decoded output from latent vector)
- **Quantitative Metrics:** Real-time MSE Loss, Reconstruction Fidelity percentage, and inference latency in milliseconds.
- **Latent Space Exploration:**
  - **64-D Latent Heatmap:** Interactive visual grid displaying exact values and polarity of each compressed latent dimension on hover.

---

## 🚀 Quick Start (One Command)

### Windows
Double-click `start.bat`, or run in PowerShell / Command Prompt:
```bat
start.bat
```

### Mac / Linux
Open a terminal in the project directory and run:
```bash
./start.sh
```
*(Or `bash start.sh`)*

**What the launcher scripts handle automatically:**
1. **Python Detection:** Scans PATH, `py` launcher, and system install paths for Python 3.9+.
2. **Environment Isolation & Auto-Healing:** Builds a local `venv/` automatically. If the folder was copied from another computer, it detects path invalidation and auto-rebuilds.
3. **Instant Launch:** If dependencies are already installed, pip is skipped so the server starts in under 1 second.
4. **Pre-trained Weights Included:** The repo ships with pre-trained weights (`model/saved_model.pt`). If missing, the script downloads EMNIST and trains automatically.
5. **Port Collision Prevention:** If port 5000 is occupied, it automatically falls back to an available open port and launches your default browser.

---

## 🧪 Verification & Testing

To run the complete automated test and verification suite on any computer:
```bash
# Windows
.\venv\Scripts\python.exe verify_all.py

# Mac / Linux
python3 verify_all.py
```
This tests:
- Model forward pass, shapes, bounds $[0, 1]$, and numerical stability
- Model metadata integrity
- Preprocessing robustness (contrast auto-inversion, transparency compositing, bounding box normalization, blank canvas guard)
- Flask API endpoints (`/`, `/status`, `/predict`, `/interpolate`, `/sample`, and error handling)
- Frontend template and static asset DOM IDs
- Jupyter Notebook format
- Unit test suite (`test_e2e.py`)

---

## 🛠️ Manual Setup

If you prefer to run the steps manually:

```bash
# 1. Create and activate virtual environment
python -m venv venv
venv\Scripts\activate       # On Windows
source venv/bin/activate    # On Mac/Linux

# 2. Install dependencies
pip install -r requirements.txt

# 3. Train model (creates model/saved_model.pt)
python model/train.py

# 4. Start Flask web server
python backend/app.py
```
Then navigate to **http://127.0.0.1:5000** in your browser.

---

## 📂 Project Architecture

```
cnn-autoencoder-project/
├── backend/
│   ├── app.py              # Flask REST API (/predict, /interpolate, /sample, /status)
│   └── preprocess.py       # Canvas/image -> 28x28 normalized, centered tensor
├── frontend/
│   ├── static/
│   │   ├── canvas.js       # Drawing tools, touch handlers, API calls, latent grid
│   │   └── style.css       # Dark-mode styling, animations, responsive layout
│   └── templates/
│       └── index.html      # Responsive dashboard interface
├── model/
│   ├── autoencoder.py      # ConvAutoencoder PyTorch module (Conv2d + ConvTranspose2d)
│   ├── train.py            # EMNIST training script with progress bar and metrics
│   ├── saved_model.pt      # Pre-trained weights (included in repo)
│   └── model_meta.json     # Saved training loss history and hyperparameter metadata
├── notebooks/
│   └── explore_autoencoder.ipynb  # Interactive Jupyter notebook for analysis
├── data/                   # EMNIST dataset cache (auto-downloaded if retrained)
├── start.bat               # Windows one-click launcher (auto-setup + launch)
├── start.sh                # Mac/Linux one-click launcher (auto-setup + launch)
├── test_e2e.py             # End-to-end unit test suite
├── verify_all.py           # Comprehensive verification script
├── requirements.txt        # Python dependency manifest
└── .gitattributes          # Cross-platform line endings specification
```

---

## 🔬 How the Pipeline Works

1. **Drawing / Ingestion:** User draws on a 280×280 HTML5 canvas (or uploads an image).
2. **Preprocessing (`backend/preprocess.py`):**
   - Transparencies are composited on white background.
   - Automatically detects dark-on-light vs light-on-dark contrast and inverts stroke to white-on-black.
   - Low-noise thresholding and bounding box extraction.
   - Aspect-ratio preserving square padding + margin centering (matching EMNIST's 20×20 character scale inside a 28×28 frame).
   - Lanczos downsampling to 28×28 float32 array normalized to $[0.0, 1.0]$.
3. **Encoding (`model/autoencoder.py`):**
   - Passed through 3 convolutional layers with ReLU activations.
   - Dense linear projection down to 64 scalar latent values.
4. **Decoding:**
   - Linear projection + unflattening to $(64, 1, 1)$.
   - 3 transposed convolutional layers with ReLU and Sigmoid output activations yielding $1 \times 28 \times 28$.
5. **Display & Metrics:**
   - Preprocessed input and reconstructed output rendered side by side at 280×280 using pixelated interpolation.
   - Computes MSE loss and similarity score.
   - Visualizes all 64 latent variables on an interactive color-coded heatmap.

---

## 📡 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | `GET` | Main interactive web application |
| `/status` | `GET` | Health check, device info, model loaded status, and metadata |
| `/predict` | `POST` | Accepts base64 image; returns reconstruction, input preview, 64D latent vector, and MSE metrics |
| `/interpolate` | `POST` | Linearly interpolates between two images in latent space across $N$ steps |
| `/sample` | `POST` | Decodes custom/perturbed latent vectors or samples from Gaussian latent prior |

---

## 💡 Troubleshooting

- **Server starts before model training completes:** Run `python model/train.py` once to generate `model/saved_model.pt`.
- **First-time training download:** EMNIST balanced dataset is ~535 MB. The training script displays a live progress bar. Once downloaded, it is cached in `data/` and never downloaded again.
- **Off-center handwriting:** The preprocessing pipeline automatically crops to stroke bounding boxes and adds proportional padding. For best results, draw characters with continuous, confident strokes.

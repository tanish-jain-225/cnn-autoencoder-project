"""
Flask server: serves the canvas web page and API endpoints:
  - GET  /             : web interface
  - GET  /status       : model status and environment info
  - POST /predict      : runs autoencoder on drawn image (returns recon, input preview, latent vector, metrics)
  - POST /interpolate  : linearly interpolates between two images in latent space
  - POST /sample       : decodes a custom or perturbed latent vector

Run: python backend/app.py
Then open: http://127.0.0.1:5000
"""

import os
import sys
import io
import time
import base64
import json
import socket
import threading
import webbrowser

import torch
import numpy as np
from flask import Flask, request, jsonify, render_template
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(HERE, "..", "model"))
sys.path.append(HERE)

from autoencoder import ConvAutoencoder   # noqa: E402
from preprocess import preprocess_canvas_image  # noqa: E402

MODEL_PATH = os.path.join(HERE, "..", "model", "saved_model.pt")
META_PATH = os.path.join(HERE, "..", "model", "model_meta.json")
LATENT_DIM = 64

app = Flask(
    __name__,
    template_folder=os.path.join(HERE, "..", "frontend", "templates"),
    static_folder=os.path.join(HERE, "..", "frontend", "static"),
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = None
model_load_error = None


def load_model_if_available():
    global model, model_load_error
    if os.path.exists(MODEL_PATH):
        try:
            m = ConvAutoencoder(latent_dim=LATENT_DIM).to(DEVICE)
            ckpt = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)
            if isinstance(ckpt, dict) and "state_dict" in ckpt:
                m.load_state_dict(ckpt["state_dict"])
            else:
                m.load_state_dict(ckpt)
            m.eval()
            model = m
            model_load_error = None
            print(f"[OK] Loaded trained model from {MODEL_PATH} on {DEVICE}")
            return True
        except Exception as exc:
            model = None
            model_load_error = str(exc)
            print(f"[Warning] Failed to load model weights: {exc}")
            return False
    else:
        model = None
        model_load_error = f"Model weights not found at {MODEL_PATH}. Run 'python model/train.py' first."
        return False


# Attempt initial load
load_model_if_available()


def array_to_data_url(arr_28x28_float: np.ndarray, display_size: int = 280) -> str:
    """Converts a 28x28 float array in [0, 1] to a display PNG data URL."""
    uint8_img = np.clip(arr_28x28_float * 255.0, 0, 255).astype(np.uint8)
    img = Image.fromarray(uint8_img, mode="L").resize((display_size, display_size), Image.Resampling.NEAREST)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{b64}"


def find_available_port(preferred_port: int = 5000, max_attempts: int = 15) -> int:
    """Finds an available TCP port starting from preferred_port."""
    for p in range(preferred_port, preferred_port + max_attempts):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("0.0.0.0", p))
                return p
            except OSError:
                continue
    return preferred_port


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/status", methods=["GET"])
def status():
    if model is None:
        load_model_if_available()

    meta = {}
    if os.path.exists(META_PATH):
        try:
            with open(META_PATH, "r") as f:
                meta = json.load(f)
        except Exception:
            pass

    return jsonify({
        "ready": model is not None,
        "device": str(DEVICE),
        "latent_dim": LATENT_DIM,
        "error": model_load_error,
        "model_path": MODEL_PATH,
        "metadata": meta
    })


@app.route("/predict", methods=["POST"])
def predict():
    global model
    if model is None and not load_model_if_available():
        return jsonify({
            "error": "Trained model not found. Please train the model using 'python model/train.py' first."
        }), 503

    payload = request.get_json(silent=True) or {}
    data_url = payload.get("image")
    if not data_url:
        return jsonify({"error": "No image data provided in request body."}), 400

    try:
        t0 = time.time()
        arr_28x28 = preprocess_canvas_image(data_url)
    except Exception as exc:
        return jsonify({"error": f"Image preprocessing failed: {exc}"}), 400

    tensor = torch.from_numpy(arr_28x28).unsqueeze(0).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        recon, latent = model(tensor)

    inference_ms = round((time.time() - t0) * 1000, 2)

    recon_arr = recon.squeeze().cpu().numpy()
    latent_vec = latent.squeeze().cpu().tolist()

    mse = float(np.mean((arr_28x28 - recon_arr) ** 2))
    fidelity = max(0.0, min(100.0, round((1.0 - (mse * 3.5)) * 100.0, 1)))

    return jsonify({
        "reconstructed": array_to_data_url(recon_arr),
        "preprocessed_input": array_to_data_url(arr_28x28),
        "latent_dim": LATENT_DIM,
        "latent_vector": [round(v, 4) for v in latent_vec],
        "metrics": {
            "mse_loss": round(mse, 5),
            "fidelity_pct": fidelity,
            "inference_ms": inference_ms
        }
    })


@app.route("/interpolate", methods=["POST"])
def interpolate():
    """Interpolates linearly between two images in latent space."""
    global model
    if model is None and not load_model_if_available():
        return jsonify({"error": "Model not available"}), 503

    payload = request.get_json(silent=True) or {}
    img1_url = payload.get("image1")
    img2_url = payload.get("image2")
    steps = max(3, min(20, int(payload.get("steps", 7))))

    if not img1_url or not img2_url:
        return jsonify({"error": "Both 'image1' and 'image2' must be provided."}), 400

    try:
        arr1 = preprocess_canvas_image(img1_url)
        arr2 = preprocess_canvas_image(img2_url)
    except Exception as exc:
        return jsonify({"error": f"Preprocessing failed: {exc}"}), 400

    t1 = torch.from_numpy(arr1).unsqueeze(0).unsqueeze(0).to(DEVICE)
    t2 = torch.from_numpy(arr2).unsqueeze(0).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        z1 = model.encode(t1)
        z2 = model.encode(t2)

        frames = []
        for i in range(steps):
            alpha = i / float(steps - 1)
            z_interp = (1.0 - alpha) * z1 + alpha * z2
            recon = model.decode(z_interp).squeeze().cpu().numpy()
            frames.append({
                "step": i,
                "alpha": round(alpha, 2),
                "image": array_to_data_url(recon, display_size=140)
            })

    return jsonify({"steps": steps, "frames": frames})


@app.route("/sample", methods=["POST"])
def sample():
    """Decodes a modified or perturbed latent vector."""
    global model
    if model is None and not load_model_if_available():
        return jsonify({"error": "Model not available"}), 503

    payload = request.get_json(silent=True) or {}
    vector = payload.get("vector")

    if vector and len(vector) == LATENT_DIM:
        z = torch.tensor(vector, dtype=torch.float32).unsqueeze(0).to(DEVICE)
    else:
        z = torch.randn(1, LATENT_DIM, device=DEVICE)

    with torch.no_grad():
        recon = model.decode(z).squeeze().cpu().numpy()

    return jsonify({
        "image": array_to_data_url(recon),
        "latent_vector": [round(float(v), 4) for v in z.squeeze().cpu().tolist()]
    })


def open_browser_later(url: str, delay_sec: float = 1.0):
    """Opens the user's default browser after the Flask server starts listening."""
    def _open():
        time.sleep(delay_sec)
        try:
            webbrowser.open(url)
        except Exception:
            pass
    threading.Thread(target=_open, daemon=True).start()


if __name__ == "__main__":
    requested_port = int(os.environ.get("PORT", 5000))
    port = find_available_port(requested_port)
    if port != requested_port:
        print(f"[*] Port {requested_port} is busy. Switched to available port {port}.")

    url = f"http://127.0.0.1:{port}"
    print("=" * 60)
    print(f" CNN Autoencoder Server Running at: {url}")
    print(" Press Ctrl+C in this terminal to stop.")
    print("=" * 60)

    # Launch browser automatically
    open_browser_later(url, delay_sec=1.0)

    app.run(host="0.0.0.0", port=port, debug=False)

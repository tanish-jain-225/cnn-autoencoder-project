"""
Trains the CNN autoencoder on EMNIST Digits (0-9) and saves weights to model/saved_model.pt.

Run directly:  python model/train.py
The EMNIST dataset is cached locally in data/.
"""

import os
import sys
import time
import json

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from PIL import Image
from tqdm import tqdm

sys.path.append(os.path.dirname(__file__))
from autoencoder import ConvAutoencoder

# ---------------- Config ----------------
SPLIT = "digits"       # EMNIST split: 'digits' (10 classes: 0-9, 240k train images)
EPOCHS = 5
BATCH_SIZE = 256
LATENT_DIM = 64
LEARNING_RATE = 1e-3
NUM_WORKERS = 0        # 0 is most reliable across Windows and virtual environments
# -----------------------------------------

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "..", "data")
MODEL_PATH = os.path.join(HERE, "saved_model.pt")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def fix_orientation(img):
    """EMNIST images are stored transposed (column-major MATLAB order).
    PIL Image.TRANSPOSE swaps x and y cleanly without rotation artifacts.
    """
    return img.transpose(Image.TRANSPOSE)


transform = transforms.Compose([
    transforms.Lambda(fix_orientation),
    transforms.ToTensor(),
])


def train():
    print("=" * 60)
    print(" CNN Autoencoder Training — EMNIST Digits (0-9)")
    print("=" * 60)
    print(f"Device: {DEVICE}")
    if DEVICE.type == "cpu":
        num_threads = min(8, max(2, (os.cpu_count() or 4) - 1))
        torch.set_num_threads(num_threads)
        print(f"CPU threads allocated: {num_threads}")
    print(f"Split: '{SPLIT}'  |  Epochs: {EPOCHS}  |  Batch size: {BATCH_SIZE}  |  Latent dim: {LATENT_DIM}")
    print(f"Data directory: {DATA_DIR}")
    print(f"Target model save path: {MODEL_PATH}")
    print("-" * 60)

    os.makedirs(DATA_DIR, exist_ok=True)

    print("Checking / downloading EMNIST dataset (first run downloads ~535MB)...")
    try:
        train_dataset = datasets.EMNIST(
            root=DATA_DIR, split=SPLIT, train=True, download=True, transform=transform
        )
        test_dataset = datasets.EMNIST(
            root=DATA_DIR, split=SPLIT, train=False, download=True, transform=transform
        )
    except Exception as exc:
        print(f"\n[Error] Failed to load or download EMNIST: {exc}")
        print("Tip: If downloading from NIST is slow or interrupted, try running the script again.")
        raise exc

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=(DEVICE.type == "cuda")
    )
    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=(DEVICE.type == "cuda")
    )

    print(f"Dataset ready: {len(train_dataset):,} train samples, {len(test_dataset):,} test samples.\n")

    model = ConvAutoencoder(latent_dim=LATENT_DIM).to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    history = {"train_loss": [], "val_loss": []}
    best_val_loss = float("inf")

    total_start = time.time()
    for epoch in range(1, EPOCHS + 1):
        epoch_start = time.time()
        model.train()
        running_loss = 0.0
        seen_samples = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS} [Train]", unit="batch", leave=False)
        for imgs, _ in pbar:
            imgs = imgs.to(DEVICE)
            optimizer.zero_grad()
            recon, _ = model(imgs)
            loss = criterion(recon, imgs)
            loss.backward()
            optimizer.step()

            batch_sz = imgs.size(0)
            running_loss += loss.item() * batch_sz
            seen_samples += batch_sz
            pbar.set_postfix({"loss": f"{running_loss / seen_samples:.4f}"})

        train_loss = running_loss / len(train_dataset)

        # Validation
        model.eval()
        val_running_loss = 0.0
        with torch.no_grad():
            for imgs, _ in test_loader:
                imgs = imgs.to(DEVICE)
                recon, _ = model(imgs)
                val_running_loss += criterion(recon, imgs).item() * imgs.size(0)
        val_loss = val_running_loss / len(test_dataset)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        epoch_time = time.time() - epoch_start
        print(f"Epoch {epoch:2d}/{EPOCHS} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Time: {epoch_time:.1f}s")

        if val_loss < best_val_loss:
            best_val_loss = val_loss

    total_time = time.time() - total_start
    print("-" * 60)
    print(f"Training completed in {total_time:.1f}s. Best Val Loss: {best_val_loss:.4f}")

    # Save state dict
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"Model state saved successfully to: {MODEL_PATH}")

    # Save training metadata
    meta_path = os.path.join(HERE, "model_meta.json")
    try:
        with open(meta_path, "w") as f:
            json.dump({
                "latent_dim": LATENT_DIM,
                "split": SPLIT,
                "epochs": EPOCHS,
                "best_val_loss": round(best_val_loss, 5),
                "train_samples": len(train_dataset),
                "test_samples": len(test_dataset),
                "history": history
            }, f, indent=2)
        print(f"Training metadata saved to: {meta_path}")
    except Exception:
        pass


if __name__ == "__main__":
    train()

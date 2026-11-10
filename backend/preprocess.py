"""
Preprocessing utilities for converting canvas or uploaded images into 28x28
tensors matching EMNIST character conventions:
  - Grayscale 28x28 normalized to [0.0, 1.0]
  - White stroke on dark background (EMNIST convention)
  - Centered by bounding box, aspect ratio preserved, with realistic padding
  - Robust to transparent, white, or dark backgrounds
"""

import base64
import io
import numpy as np
from PIL import Image, ImageOps


def preprocess_canvas_image(data_url: str) -> np.ndarray:
    """Processes a base64 image data URL into a 28x28 float32 numpy array.

    Args:
        data_url: Base64 data URL string (e.g. 'data:image/png;base64,...') or raw base64.

    Returns:
        np.ndarray of shape (28, 28) with float32 values in [0.0, 1.0].
    """
    if "," in data_url:
        _, encoded = data_url.split(",", 1)
    else:
        encoded = data_url

    img_bytes = base64.b64decode(encoded)
    raw_img = Image.open(io.BytesIO(img_bytes))

    # Handle transparency by compositing onto a white background
    if raw_img.mode in ("RGBA", "LA") or ("transparency" in raw_img.info):
        bg = Image.new("RGBA", raw_img.size, (255, 255, 255, 255))
        bg.paste(raw_img, mask=raw_img.split()[-1])
        img = bg.convert("L")
    else:
        img = raw_img.convert("L")

    # Determine if stroke is dark-on-light (standard canvas) or light-on-dark (e.g. dark mode / MNIST sample)
    mean_val = np.mean(np.array(img))
    if mean_val >= 127.5:
        # Dark stroke on white canvas -> invert so stroke is white (255) on black (0)
        img = ImageOps.invert(img)

    # Threshold low-level noise before finding the bounding box
    thresh = img.point(lambda p: 255 if p > 25 else 0)
    bbox = thresh.getbbox()

    if not bbox:
        # Empty drawing
        return np.zeros((28, 28), dtype=np.float32)

    # Crop to stroke bounding box
    img = img.crop(bbox)
    w, h = img.size

    # Fit into a square canvas, preserving aspect ratio
    max_side = max(w, h)
    square = Image.new("L", (max_side, max_side), color=0)
    square.paste(img, ((max_side - w) // 2, (max_side - h) // 2))

    # Add margin: in EMNIST, standard strokes occupy roughly 20x20 in a 28x28 frame (~20% margin per side)
    margin = max(int(max_side * 0.20), 4)
    final_size = max_side + 2 * margin
    final_img = Image.new("L", (final_size, final_size), color=0)
    final_img.paste(square, (margin, margin))

    # Resize to model input size (28x28) using high-quality Lanczos resampling
    final_img = final_img.resize((28, 28), Image.Resampling.LANCZOS)

    arr = np.array(final_img, dtype=np.float32) / 255.0
    arr = np.clip(arr, 0.0, 1.0)
    return arr

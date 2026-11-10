"""
End-to-End Test Suite for CNN Autoencoder Project
Validates preprocessing, model architecture, model loading, and Flask API endpoints.
"""

import os
import sys
import io
import base64
import unittest
import numpy as np
import torch
from PIL import Image, ImageDraw

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(HERE, "backend"))
sys.path.append(os.path.join(HERE, "model"))

from autoencoder import ConvAutoencoder
from preprocess import preprocess_canvas_image


def create_dummy_character_data_url(char: str = "A") -> str:
    """Creates a base64 PNG data URL mimicking the HTML canvas drawing."""
    img = Image.new("RGB", (280, 280), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    # Draw simple cross-lines
    draw.line([(50, 230), (140, 50)], fill=(0, 0, 0), width=18)
    draw.line([(140, 50), (230, 230)], fill=(0, 0, 0), width=18)
    draw.line([(80, 160), (200, 160)], fill=(0, 0, 0), width=18)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{b64}"


class TestPreprocessing(unittest.TestCase):
    def test_preprocess_character(self):
        url = create_dummy_character_data_url("A")
        arr = preprocess_canvas_image(url)
        self.assertEqual(arr.shape, (28, 28))
        self.assertEqual(arr.dtype, np.float32)
        self.assertGreaterEqual(arr.min(), 0.0)
        self.assertLessEqual(arr.max(), 1.0)
        self.assertGreater(arr.max(), 0.1, "Drawn stroke should not be all black")

    def test_preprocess_blank(self):
        img = Image.new("RGB", (280, 280), color=(255, 255, 255))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        url = f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode('utf-8')}"
        arr = preprocess_canvas_image(url)
        self.assertEqual(arr.shape, (28, 28))
        self.assertEqual(np.count_nonzero(arr), 0, "Blank canvas should return all zeros")

    def test_preprocess_transparent_rgba(self):
        img = Image.new("RGBA", (280, 280), color=(255, 255, 255, 0))
        draw = ImageDraw.Draw(img)
        draw.ellipse([(90, 90), (190, 190)], outline=(0, 0, 0, 255), width=16)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        url = f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode('utf-8')}"
        arr = preprocess_canvas_image(url)
        self.assertEqual(arr.shape, (28, 28))
        self.assertGreater(arr.max(), 0.1)


class TestModelArchitecture(unittest.TestCase):
    def test_forward_pass(self):
        model = ConvAutoencoder(latent_dim=64)
        x = torch.rand(2, 1, 28, 28)
        recon, latent = model(x)
        self.assertEqual(latent.shape, (2, 64))
        self.assertEqual(recon.shape, (2, 1, 28, 28))
        self.assertGreaterEqual(recon.min().item(), 0.0)
        self.assertLessEqual(recon.max().item(), 1.0)


class TestFlaskEndpoints(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import app as flask_backend
        flask_backend.app.config["TESTING"] = True
        cls.client = flask_backend.app.test_client()

    def test_index_route(self):
        res = self.client.get("/")
        self.assertEqual(res.status_code, 200)
        self.assertIn(b"CNN Autoencoder", res.data)

    def test_status_route(self):
        res = self.client.get("/status")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertIn("ready", data)
        self.assertIn("latent_dim", data)
        self.assertEqual(data["latent_dim"], 64)

    def test_predict_route(self):
        url = create_dummy_character_data_url("A")
        res = self.client.post("/predict", json={"image": url})
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertIn("reconstructed", data)
        self.assertIn("preprocessed_input", data)
        self.assertIn("latent_vector", data)
        self.assertEqual(len(data["latent_vector"]), 64)
        self.assertIn("metrics", data)
        self.assertIn("mse_loss", data["metrics"])

    def test_interpolate_route(self):
        url1 = create_dummy_character_data_url("A")
        url2 = create_dummy_character_data_url("B")
        res = self.client.post("/interpolate", json={"image1": url1, "image2": url2, "steps": 5})
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertEqual(data["steps"], 5)
        self.assertEqual(len(data["frames"]), 5)

    def test_sample_route(self):
        res = self.client.post("/sample", json={})
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertIn("image", data)
        self.assertEqual(len(data["latent_vector"]), 64)


if __name__ == "__main__":
    unittest.main()


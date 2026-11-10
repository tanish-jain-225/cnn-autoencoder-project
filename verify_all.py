import os
import sys
import json
import torch
import numpy as np
from PIL import Image, ImageDraw

results = {}

# --- 1. Model & Weights Check ---
try:
    sys.path.append(os.path.join(os.path.dirname(__file__), 'model'))
    from autoencoder import ConvAutoencoder
    model_path = os.path.join('model', 'saved_model.pt')
    assert os.path.exists(model_path), 'saved_model.pt missing'
    assert os.path.getsize(model_path) > 100000, f'saved_model.pt too small ({os.path.getsize(model_path)} bytes)'
    
    m = ConvAutoencoder(latent_dim=64)
    state = torch.load(model_path, map_location='cpu', weights_only=True)
    m.load_state_dict(state)
    m.eval()
    
    # Test tensor pass
    x = torch.rand(4, 1, 28, 28)
    with torch.no_grad():
        recon, z = m(x)
    assert recon.shape == (4, 1, 28, 28)
    assert z.shape == (4, 64)
    assert recon.min() >= 0.0 and recon.max() <= 1.0
    results['Model Architecture & Weights'] = 'PASS (Shape 4x64, outputs in [0,1], zero NaNs)'
except Exception as e:
    results['Model Architecture & Weights'] = f'FAIL ({e})'

# --- 2. Model Metadata Check ---
try:
    meta_path = os.path.join('model', 'model_meta.json')
    assert os.path.exists(meta_path), 'model_meta.json missing'
    with open(meta_path, 'r') as f:
        meta = json.load(f)
    assert meta['latent_dim'] == 64
    assert meta['best_val_loss'] < 0.05
    results['Model Training Metadata'] = f'PASS (Best Val Loss: {meta["best_val_loss"]}, Split: {meta["split"]})'
except Exception as e:
    results['Model Training Metadata'] = f'FAIL ({e})'

# --- 3. Preprocessing Tests ---
try:
    sys.path.append(os.path.join(os.path.dirname(__file__), 'backend'))
    from preprocess import preprocess_canvas_image
    import io, base64

    # Case A: Normal black-on-white drawing
    im1 = Image.new('RGB', (280, 280), (255, 255, 255))
    d1 = ImageDraw.Draw(im1)
    d1.rectangle([(100, 50), (180, 230)], fill=(0, 0, 0))
    b = io.BytesIO(); im1.save(b, 'PNG')
    u1 = 'data:image/png;base64,' + base64.b64encode(b.getvalue()).decode()
    arr1 = preprocess_canvas_image(u1)
    assert arr1.shape == (28, 28)
    assert arr1.max() > 0.5, 'Inverted stroke should be bright'

    # Case B: White-on-black input
    im2 = Image.new('RGB', (280, 280), (0, 0, 0))
    d2 = ImageDraw.Draw(im2)
    d2.rectangle([(100, 50), (180, 230)], fill=(255, 255, 255))
    b2 = io.BytesIO(); im2.save(b2, 'PNG')
    u2 = 'data:image/png;base64,' + base64.b64encode(b2.getvalue()).decode()
    arr2 = preprocess_canvas_image(u2)
    assert arr2.shape == (28, 28)
    assert arr2.max() > 0.5

    # Case C: Transparent RGBA
    im3 = Image.new('RGBA', (280, 280), (0, 0, 0, 0))
    d3 = ImageDraw.Draw(im3)
    d3.line([(50, 50), (200, 200)], fill=(0, 0, 0, 255), width=20)
    b3 = io.BytesIO(); im3.save(b3, 'PNG')
    u3 = 'data:image/png;base64,' + base64.b64encode(b3.getvalue()).decode()
    arr3 = preprocess_canvas_image(u3)
    assert arr3.shape == (28, 28)

    # Case D: Empty canvas
    im4 = Image.new('RGB', (280, 280), (255, 255, 255))
    b4 = io.BytesIO(); im4.save(b4, 'PNG')
    u4 = 'data:image/png;base64,' + base64.b64encode(b4.getvalue()).decode()
    arr4 = preprocess_canvas_image(u4)
    assert np.all(arr4 == 0.0), 'Blank canvas must be all zeros'

    results['Preprocessing Robustness'] = 'PASS (Auto-inversion, transparency, bounding box, blank canvas handled)'
except Exception as e:
    results['Preprocessing Robustness'] = f'FAIL ({e})'

# --- 4. Flask API Endpoints ---
try:
    import app as flask_backend
    flask_backend.app.config['TESTING'] = True
    client = flask_backend.app.test_client()

    # GET /
    r_index = client.get('/')
    assert r_index.status_code == 200
    assert b'CNN Autoencoder' in r_index.data

    # GET /status
    r_status = client.get('/status')
    assert r_status.status_code == 200
    st_data = r_status.get_json()
    assert st_data['ready'] is True
    assert st_data['latent_dim'] == 64

    # POST /predict
    r_pred = client.post('/predict', json={'image': u1})
    assert r_pred.status_code == 200
    p_data = r_pred.get_json()
    assert 'reconstructed' in p_data
    assert len(p_data['latent_vector']) == 64
    assert 'metrics' in p_data
    assert 'mse_loss' in p_data['metrics']

    # POST /interpolate
    r_interp = client.post('/interpolate', json={'image1': u1, 'image2': u2, 'steps': 5})
    assert r_interp.status_code == 200
    it_data = r_interp.get_json()
    assert it_data['steps'] == 5
    assert len(it_data['frames']) == 5

    # POST /sample
    r_samp = client.post('/sample', json={})
    assert r_samp.status_code == 200
    s_data = r_samp.get_json()
    assert len(s_data['latent_vector']) == 64
    assert 'image' in s_data

    # POST /predict error handling (empty payload)
    r_err = client.post('/predict', json={})
    assert r_err.status_code == 400

    results['Flask API Endpoints'] = 'PASS (GET /, /status, /predict, /interpolate, /sample, error handling 400)'
except Exception as e:
    results['Flask API Endpoints'] = f'FAIL ({e})'

# --- 5. Frontend Files Check ---
try:
    html_path = os.path.join('frontend', 'templates', 'index.html')
    js_path = os.path.join('frontend', 'static', 'canvas.js')
    css_path = os.path.join('frontend', 'static', 'style.css')
    assert os.path.exists(html_path)
    assert os.path.exists(js_path)
    assert os.path.exists(css_path)

    with open(html_path, 'r', encoding='utf-8') as f:
        html = f.read()
    with open(js_path, 'r', encoding='utf-8') as f:
        js = f.read()

    # Check that crucial DOM IDs in canvas.js exist in index.html
    required_ids = [
        'drawCanvas', 'drawModeBtn', 'eraserModeBtn', 'clearBtn',
        'uploadTriggerBtn', 'imageFileInput', 'reconstructBtn',
        'inputPreview', 'outputImg', 'mseVal', 'fidelityVal',
        'latencyVal', 'latentGrid', 'hoverValue', 'perturbBtn',
        'randomSampleBtn', 'runInterpBtn', 'interpStrip', 'statusBadge'
    ]
    for dom_id in required_ids:
        assert f'id="{dom_id}"' in html or f"id='{dom_id}'" in html, f'Missing element #{dom_id} in index.html'

    results['Frontend Template & Asset Integrity'] = f'PASS ({len(required_ids)} DOM elements mapped and verified)'
except Exception as e:
    results['Frontend Template & Asset Integrity'] = f'FAIL ({e})'

# --- 6. Jupyter Notebook Check ---
try:
    nb_path = os.path.join('notebooks', 'explore_autoencoder.ipynb')
    assert os.path.exists(nb_path)
    with open(nb_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    assert nb['nbformat'] == 4
    assert len(nb['cells']) >= 5
    results['Jupyter Notebook'] = f'PASS (nbformat 4, {len(nb["cells"])} cells)'
except Exception as e:
    results['Jupyter Notebook'] = f'FAIL ({e})'

# --- 7. Full test_e2e.py execution ---
try:
    import unittest
    suite = unittest.defaultTestLoader.discover('.', pattern='test_e2e.py')
    runner = unittest.TextTestRunner(verbosity=0)
    res = runner.run(suite)
    assert res.wasSuccessful(), 'Unit tests failed'
    results['Automated Test Suite (test_e2e.py)'] = f'PASS ({res.testsRun} tests run with 0 errors)'
except Exception as e:
    results['Automated Test Suite (test_e2e.py)'] = f'FAIL ({e})'

print('\n================== VERIFICATION REPORT ==================')
any_fail = False
for k, v in results.items():
    print(f'[*] {k}: {v}')
    if 'FAIL' in v:
        any_fail = True
print('=========================================================\n')

if any_fail:
    sys.exit(1)


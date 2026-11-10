// CNN Autoencoder — Frontend Interactive Script

document.addEventListener('DOMContentLoaded', () => {
  const canvas = document.getElementById('drawCanvas');
  const ctx = canvas.getContext('2d');

  let isDrawing = false;
  let lastX = 0;
  let lastY = 0;
  let currentBrushSize = 18;
  let isEraser = false;
  let currentLatentVector = null;

  // Initialize Canvas
  function resetCanvas() {
    ctx.fillStyle = '#ffffff';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.lineWidth = currentBrushSize;
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';
    ctx.strokeStyle = isEraser ? '#ffffff' : '#000000';
  }
  resetCanvas();

  // Coordinate mapping handling Retina and scaling
  function getPos(e) {
    const rect = canvas.getBoundingClientRect();
    const clientX = e.touches && e.touches.length > 0 ? e.touches[0].clientX : e.clientX;
    const clientY = e.touches && e.touches.length > 0 ? e.touches[0].clientY : e.clientY;
    return {
      x: (clientX - rect.left) * (canvas.width / rect.width),
      y: (clientY - rect.top) * (canvas.height / rect.height),
    };
  }

  function startDraw(e) {
    isDrawing = true;
    const pos = getPos(e);
    lastX = pos.x;
    lastY = pos.y;

    // Draw single dot on click/tap
    ctx.beginPath();
    ctx.arc(pos.x, pos.y, ctx.lineWidth / 2, 0, Math.PI * 2);
    ctx.fillStyle = ctx.strokeStyle;
    ctx.fill();

    if (e.cancelable && e.type.startsWith('touch')) {
      e.preventDefault();
    }
  }

  function draw(e) {
    if (!isDrawing) return;
    const pos = getPos(e);
    ctx.beginPath();
    ctx.moveTo(lastX, lastY);
    ctx.lineTo(pos.x, pos.y);
    ctx.stroke();
    lastX = pos.x;
    lastY = pos.y;

    if (e.cancelable && e.type.startsWith('touch')) {
      e.preventDefault();
    }
  }

  function endDraw() {
    isDrawing = false;
  }

  // Canvas Mouse Listeners
  canvas.addEventListener('mousedown', startDraw);
  canvas.addEventListener('mousemove', draw);
  canvas.addEventListener('mouseup', endDraw);
  canvas.addEventListener('mouseleave', endDraw);

  // Canvas Touch Listeners
  canvas.addEventListener('touchstart', startDraw, { passive: false });
  canvas.addEventListener('touchmove', draw, { passive: false });
  canvas.addEventListener('touchend', endDraw);
  canvas.addEventListener('touchcancel', endDraw);

  // Tools: Pen vs Eraser
  const drawModeBtn = document.getElementById('drawModeBtn');
  const eraserModeBtn = document.getElementById('eraserModeBtn');

  drawModeBtn.addEventListener('click', () => {
    isEraser = false;
    ctx.strokeStyle = '#000000';
    drawModeBtn.classList.add('active');
    eraserModeBtn.classList.remove('active');
  });

  eraserModeBtn.addEventListener('click', () => {
    isEraser = true;
    ctx.strokeStyle = '#ffffff';
    eraserModeBtn.classList.add('active');
    drawModeBtn.classList.remove('active');
  });

  // Brush Size Buttons
  const sizeBtns = document.querySelectorAll('.size-btn');
  sizeBtns.forEach(btn => {
    btn.addEventListener('click', () => {
      sizeBtns.forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      currentBrushSize = parseInt(btn.dataset.size, 10);
      ctx.lineWidth = currentBrushSize;
    });
  });

  // Clear Canvas
  const clearBtn = document.getElementById('clearBtn');
  clearBtn.addEventListener('click', () => {
    resetCanvas();
    clearOutputs();
    showStatus('');
  });

  function clearOutputs() {
    document.getElementById('inputPreview').src = "data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' width='280' height='280'><rect width='280' height='280' fill='%230b0f19'/><text x='50%25' y='50%25' fill='%23475569' font-size='14' font-family='sans-serif' dominant-baseline='middle' text-anchor='middle'>Awaiting drawing...</text></svg>";
    document.getElementById('outputImg').src = "data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' width='280' height='280'><rect width='280' height='280' fill='%230b0f19'/><text x='50%25' y='50%25' fill='%23475569' font-size='14' font-family='sans-serif' dominant-baseline='middle' text-anchor='middle'>Awaiting reconstruction...</text></svg>";
    document.getElementById('mseVal').textContent = '—';
    document.getElementById('fidelityVal').textContent = '—';
    document.getElementById('latencyVal').textContent = '—';
    currentLatentVector = null;
    renderLatentGrid([]);
  }

  // Upload Image Support
  const uploadTriggerBtn = document.getElementById('uploadTriggerBtn');
  const imageFileInput = document.getElementById('imageFileInput');

  uploadTriggerBtn.addEventListener('click', () => {
    imageFileInput.click();
  });

  imageFileInput.addEventListener('change', (e) => {
    const file = e.target.files[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onload = (event) => {
      const img = new Image();
      img.onload = () => {
        resetCanvas();
        // Scale and center image inside canvas
        const scale = Math.min((canvas.width * 0.8) / img.width, (canvas.height * 0.8) / img.height);
        const w = img.width * scale;
        const h = img.height * scale;
        const x = (canvas.width - w) / 2;
        const y = (canvas.height - h) / 2;
        ctx.drawImage(img, x, y, w, h);
        reconstruct();
      };
      img.src = event.target.result;
    };
    reader.readAsDataURL(file);
    imageFileInput.value = '';
  });

  // Quick Preset Characters
  function drawPresetChar(char) {
    resetCanvas();
    ctx.fillStyle = '#000000';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.font = 'bold 180px "Inter", "Segoe UI", sans-serif';
    ctx.fillText(char, canvas.width / 2, canvas.height / 2 + 10);
    reconstruct();
  }

  document.querySelectorAll('.preset-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      const char = btn.dataset.char;
      drawPresetChar(char);
    });
  });

  // Reconstruct Trigger
  const reconstructBtn = document.getElementById('reconstructBtn');
  reconstructBtn.addEventListener('click', reconstruct);

  async function reconstruct() {
    const spinner = reconstructBtn.querySelector('.spinner');
    const btnText = reconstructBtn.querySelector('.btn-text');
    reconstructBtn.disabled = true;
    spinner.style.display = 'inline-block';
    btnText.textContent = 'Processing...';

    showStatus('Encoding to 64D latent vector & decoding 28x28 reconstruction...', 'info');

    const dataUrl = canvas.toDataURL('image/png');

    try {
      const res = await fetch('/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ image: dataUrl }),
      });

      const data = await res.json();

      if (!res.ok) {
        showStatus('Server Error: ' + (data.error || res.statusText), 'error');
        return;
      }

      document.getElementById('inputPreview').src = data.preprocessed_input;
      document.getElementById('outputImg').src = data.reconstructed;

      if (data.metrics) {
        document.getElementById('mseVal').textContent = data.metrics.mse_loss;
        document.getElementById('fidelityVal').textContent = `${data.metrics.fidelity_pct}%`;
        document.getElementById('latencyVal').textContent = `${data.metrics.inference_ms} ms`;
      }

      currentLatentVector = data.latent_vector || [];
      renderLatentGrid(currentLatentVector);

      showStatus(`Reconstructed in ${data.metrics?.inference_ms || 0} ms with ${data.latent_dim}D compression.`, 'success');
    } catch (err) {
      showStatus('Connection failed: ' + err.message, 'error');
    } finally {
      reconstructBtn.disabled = false;
      spinner.style.display = 'none';
      btnText.textContent = 'Reconstruct →';
    }
  }

  // Render 64-Cell Latent Grid
  const latentGrid = document.getElementById('latentGrid');
  const hoverValue = document.getElementById('hoverValue');

  function renderLatentGrid(vector) {
    latentGrid.innerHTML = '';
    if (!vector || vector.length === 0) {
      for (let i = 0; i < 64; i++) {
        const cell = document.createElement('div');
        cell.className = 'latent-cell empty';
        latentGrid.appendChild(cell);
      }
      return;
    }

    const maxAbs = Math.max(0.1, ...vector.map(v => Math.abs(v)));

    vector.forEach((val, idx) => {
      const cell = document.createElement('div');
      cell.className = 'latent-cell';
      const norm = val / maxAbs; // in [-1, 1]

      if (norm >= 0) {
        // Positive: Warm Indigo to Violet
        cell.style.backgroundColor = `rgba(99, 102, 241, ${Math.min(1.0, 0.2 + norm * 0.8)})`;
      } else {
        // Negative: Cool Cyan to Teal
        cell.style.backgroundColor = `rgba(14, 165, 233, ${Math.min(1.0, 0.2 + Math.abs(norm) * 0.8)})`;
      }

      cell.addEventListener('mouseenter', () => {
        hoverValue.textContent = `z[${idx}] = ${val > 0 ? '+' : ''}${val.toFixed(4)}`;
        cell.classList.add('hovered');
      });
      cell.addEventListener('mouseleave', () => {
        cell.classList.remove('hovered');
      });

      latentGrid.appendChild(cell);
    });
  }
  renderLatentGrid([]);

  // Server Status Polling / Notification
  const statusBadge = document.getElementById('statusBadge');
  const statusBadgeText = document.getElementById('statusBadgeText');
  const deviceStat = document.getElementById('deviceStat');

  async function checkServerStatus() {
    try {
      const res = await fetch('/status');
      const data = await res.json();

      if (data.ready) {
        statusBadge.className = 'badge badge-ready';
        statusBadgeText.textContent = `Model Ready (${data.device.toUpperCase()})`;
        deviceStat.textContent = data.device.toUpperCase();
      } else {
        statusBadge.className = 'badge badge-warning';
        statusBadgeText.textContent = 'Model Weights Pending';
        deviceStat.textContent = 'Pending';
        showStatus('Model is currently training or not yet saved. Please run "python model/train.py".', 'info');
      }
    } catch (err) {
      statusBadge.className = 'badge badge-error';
      statusBadgeText.textContent = 'Server Offline';
    }
  }

  function showStatus(msg, type = 'info') {
    const el = document.getElementById('statusMessage');
    el.textContent = msg;
    el.className = `status-message ${type}`;
    if (!msg) {
      el.style.display = 'none';
    } else {
      el.style.display = 'block';
    }
  }

  // Initial check
  checkServerStatus();
});

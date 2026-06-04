const API = '';

function $(id) {
  return document.getElementById(id);
}

function setLog(id, text, isError = false) {
  const el = $(id);
  el.textContent = text;
  el.classList.toggle('error', isError);
}

async function api(path, options = {}) {
  const res = await fetch(`${API}${path}`, {
    headers: { 'Content-Type': 'application/json', ...options.headers },
    ...options,
  });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) {
    const msg = data.detail || data.message || res.statusText;
    throw new Error(typeof msg === 'string' ? msg : JSON.stringify(msg));
  }
  return data;
}

function datasetRoot() {
  const v = $('dataset-root').value.trim();
  if (!v) throw new Error('Enter dataset root folder path');
  return v;
}

// Tabs
document.querySelectorAll('.tab').forEach((btn) => {
  btn.addEventListener('click', () => {
    document.querySelectorAll('.tab').forEach((b) => b.classList.remove('active'));
    document.querySelectorAll('.panel').forEach((p) => p.classList.remove('active'));
    btn.classList.add('active');
    $(`panel-${btn.dataset.tab}`).classList.add('active');
  });
});

$('btn-gpu').addEventListener('click', async () => {
  setLog('log-gpu', 'Checking…');
  try {
    const r = await api('/api/environment');
    setLog('log-gpu', r.output, !r.ok);
  } catch (e) {
    setLog('log-gpu', String(e), true);
  }
});

$('btn-setup').addEventListener('click', async () => {
  setLog('log-dataset', 'Creating folders…');
  try {
    const r = await api('/api/dataset/setup', {
      method: 'POST',
      body: JSON.stringify({ root: datasetRoot() }),
    });
    setLog('log-dataset', `${r.message}\n\n${r.paths.join('\n')}`);
  } catch (e) {
    setLog('log-dataset', String(e), true);
  }
});

$('btn-save-yaml').addEventListener('click', async () => {
  setLog('log-dataset', 'Saving data.yaml…');
  try {
    const r = await api('/api/dataset/config', {
      method: 'POST',
      body: JSON.stringify({ root: datasetRoot() }),
    });
    setLog('log-dataset', `Saved: ${r.path}\n\n${r.yaml}`);
  } catch (e) {
    setLog('log-dataset', String(e), true);
  }
});

$('btn-validate').addEventListener('click', async () => {
  setLog('log-dataset', 'Validating…');
  try {
    const r = await api('/api/dataset/validate', { method: 'POST' });
    setLog('log-dataset', r.output, !r.ok);
  } catch (e) {
    setLog('log-dataset', String(e), true);
  }
});

let pollTimer = null;

function stopPoll() {
  if (pollTimer) {
    clearInterval(pollTimer);
    pollTimer = null;
  }
  $('train-badge').classList.add('hidden');
  $('btn-train').disabled = false;
}

async function pollTrainLog() {
  try {
    const r = await api('/api/train/status');
    setLog('log-train', r.lines.join('\n') || '(waiting for output…)');
    const el = $('log-train');
    el.scrollTop = el.scrollHeight;
    if (!r.running) {
      stopPoll();
      if (r.returncode !== 0) el.classList.add('error');
    }
  } catch (e) {
    stopPoll();
    setLog('log-train', String(e), true);
  }
}

$('btn-train').addEventListener('click', async () => {
  setLog('log-train', 'Starting training…');
  $('btn-train').disabled = true;
  $('train-badge').classList.remove('hidden');
  stopPoll();
  try {
    await api('/api/train/start', {
      method: 'POST',
      body: JSON.stringify({
        epochs: Number($('epochs').value),
        batch: Number($('batch').value),
        imgsz: Number($('imgsz').value),
        device: $('device').value,
        patience: Number($('patience').value),
      }),
    });
    pollTimer = setInterval(pollTrainLog, 800);
    pollTrainLog();
  } catch (e) {
    stopPoll();
    setLog('log-train', String(e), true);
  }
});

$('btn-status').addEventListener('click', refreshModelStatus);

async function refreshModelStatus() {
  try {
    const r = await api('/api/model/status');
    const lines = [
      `best.pt:   ${r.best_pt_exists ? 'OK' : 'missing'}`,
      `           ${r.best_pt}`,
      `best.onnx: ${r.best_onnx_exists ? 'OK' : 'missing'}`,
      `           ${r.best_onnx}`,
    ];
    setLog('log-model', lines.join('\n'), !r.best_onnx_exists);
  } catch (e) {
    setLog('log-model', String(e), true);
  }
}

$('btn-verify').addEventListener('click', async () => {
  setLog('log-model', 'Verifying ONNX…');
  try {
    const r = await api('/api/model/verify', { method: 'POST' });
    setLog('log-model', r.output, !r.ok);
  } catch (e) {
    setLog('log-model', String(e), true);
  }
});

refreshModelStatus();

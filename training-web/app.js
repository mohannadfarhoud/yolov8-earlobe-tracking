(function () {
  const API = '';

  function $(id) {
    return document.getElementById(id);
  }

  function setLog(id, text, isError) {
    const el = $(id);
    if (!el) {
      console.error('Missing element:', id);
      return;
    }
    el.textContent = text;
    el.classList.toggle('error', !!isError);
  }

  async function api(path, options) {
    const opts = options || {};
    const res = await fetch(API + path, {
      headers: { 'Content-Type': 'application/json', ...(opts.headers || {}) },
      ...opts,
    });
    let data = {};
    try {
      data = await res.json();
    } catch (_) {
      data = {};
    }
    if (!res.ok) {
      const msg = data.detail || data.message || res.statusText || 'Request failed';
      throw new Error(typeof msg === 'string' ? msg : JSON.stringify(msg));
    }
    return data;
  }

  function datasetRoot() {
    const v = ($('dataset-root') || {}).value;
    if (!v || !String(v).trim()) throw new Error('Enter dataset root folder path');
    return String(v).trim();
  }

  function initTabs() {
    document.querySelectorAll('.tab').forEach(function (btn) {
      btn.addEventListener('click', function () {
        document.querySelectorAll('.tab').forEach(function (b) {
          b.classList.remove('active');
        });
        document.querySelectorAll('.panel').forEach(function (p) {
          p.classList.remove('active');
        });
        btn.classList.add('active');
        const panel = $('panel-' + btn.dataset.tab);
        if (panel) panel.classList.add('active');
      });
    });
  }

  async function checkHealth() {
    const status = $('api-status');
    try {
      const r = await api('/api/health');
      if (status) status.textContent = 'API connected';
      setLog('log-gpu', 'Ready. Click "Check GPU" (first check may take 30–60 seconds).', false);
    } catch (e) {
      if (status) {
        status.textContent = 'API not reachable — is training_server.py running?';
        status.classList.add('error');
      }
      setLog(
        'log-gpu',
        'Cannot reach server at ' +
          window.location.origin +
          '\n\nStart in PowerShell:\n  python python/training_server.py\n\nThen open http://127.0.0.1:8000\n\n' +
          String(e),
        true
      );
    }
  }

  function initGpu() {
    const btn = $('btn-gpu');
    if (!btn) return;
    btn.addEventListener('click', async function () {
      btn.disabled = true;
      setLog('log-gpu', 'Checking GPU… (first time can take up to 60 seconds, please wait)', false);
      try {
        const r = await api('/api/environment');
        setLog('log-gpu', r.output || '(empty response)', !r.ok);
      } catch (e) {
        setLog('log-gpu', String(e), true);
      } finally {
        btn.disabled = false;
      }
    });
  }

  function initDataset() {
    $('btn-setup').addEventListener('click', async function () {
      setLog('log-dataset', 'Creating folders…');
      try {
        const r = await api('/api/dataset/setup', {
          method: 'POST',
          body: JSON.stringify({ root: datasetRoot() }),
        });
        setLog('log-dataset', r.message + '\n\n' + r.paths.join('\n'));
      } catch (e) {
        setLog('log-dataset', String(e), true);
      }
    });

    $('btn-save-yaml').addEventListener('click', async function () {
      setLog('log-dataset', 'Saving data.yaml…');
      try {
        const r = await api('/api/dataset/config', {
          method: 'POST',
          body: JSON.stringify({ root: datasetRoot() }),
        });
        setLog('log-dataset', 'Saved: ' + r.path + '\n\n' + r.yaml);
      } catch (e) {
        setLog('log-dataset', String(e), true);
      }
    });

    $('btn-validate').addEventListener('click', async function () {
      setLog('log-dataset', 'Validating…');
      try {
        const r = await api('/api/dataset/validate', { method: 'POST' });
        setLog('log-dataset', r.output, !r.ok);
      } catch (e) {
        setLog('log-dataset', String(e), true);
      }
    });
  }

  var pollTimer = null;

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

  function initTrain() {
    $('btn-train').addEventListener('click', async function () {
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
  }

  async function refreshModelStatus() {
    try {
      const r = await api('/api/model/status');
      setLog(
        'log-model',
        [
          'best.pt:   ' + (r.best_pt_exists ? 'OK' : 'missing'),
          '           ' + r.best_pt,
          'best.onnx: ' + (r.best_onnx_exists ? 'OK' : 'missing'),
          '           ' + r.best_onnx,
        ].join('\n'),
        !r.best_onnx_exists
      );
    } catch (e) {
      setLog('log-model', String(e), true);
    }
  }

  function initModel() {
    $('btn-status').addEventListener('click', refreshModelStatus);
    $('btn-verify').addEventListener('click', async function () {
      setLog('log-model', 'Verifying ONNX…');
      try {
        const r = await api('/api/model/verify', { method: 'POST' });
        setLog('log-model', r.output, !r.ok);
      } catch (e) {
        setLog('log-model', String(e), true);
      }
    });
    refreshModelStatus();
  }

  function init() {
    initTabs();
    initGpu();
    initDataset();
    initTrain();
    initModel();
    checkHealth();
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();

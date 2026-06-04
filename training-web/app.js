(function () {
  const API = '';
  var annotateState = {
    images: [],
    index: 0,
    clickX: 0,
    clickY: 0,
    naturalW: 0,
    naturalH: 0,
    hasClick: false,
  };

  function $(id) {
    return document.getElementById(id);
  }

  function setLog(id, text, isError) {
    var el = $(id);
    if (!el) return;
    el.textContent = text;
    el.classList.toggle('error', !!isError);
  }

  function datasetRoot() {
    var v = ($('dataset-root') || {}).value;
    if (!v || !String(v).trim()) throw new Error('Enter dataset folder path');
    return String(v).trim();
  }

  function rootQuery() {
    return '?root=' + encodeURIComponent(datasetRoot());
  }

  async function api(path, options) {
    var opts = options || {};
    var res = await fetch(API + path, {
      headers: { 'Content-Type': 'application/json', ...(opts.headers || {}) },
      ...opts,
    });
    var data = {};
    try {
      data = await res.json();
    } catch (_) {}
    if (!res.ok) {
      var msg = data.detail || data.message || res.statusText || 'Request failed';
      throw new Error(typeof msg === 'string' ? msg : JSON.stringify(msg));
    }
    return data;
  }

  function updateStats(total, annotated) {
    var el = $('annotate-stats');
    if (el) el.textContent = total + ' images · ' + annotated + ' annotated';
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
        var panel = $('panel-' + btn.dataset.tab);
        if (panel) panel.classList.add('active');
      });
    });
  }

  async function checkHealth() {
    var status = $('api-status');
    try {
      await api('/api/health');
      if (status) status.textContent = 'API connected — start on tab 1. Annotate';
      setLog('log-gpu', 'Optional: check GPU on GPU tab before training.', false);
    } catch (e) {
      if (status) {
        status.textContent = 'Server not running';
        status.classList.add('error');
      }
      setLog('log-annotate', 'Start server: python python/training_server.py\n' + String(e), true);
    }
  }

  function renderImageList() {
    var list = $('image-list');
    if (!list) return;
    list.innerHTML = '';
    annotateState.images.forEach(function (img, i) {
      var li = document.createElement('li');
      li.textContent = (img.annotated ? '✓ ' : '○ ') + img.filename;
      li.className = i === annotateState.index ? 'active' : '';
      if (!img.annotated) li.classList.add('pending');
      li.addEventListener('click', function () {
        annotateState.index = i;
        loadCurrentImage();
        renderImageList();
      });
      list.appendChild(li);
    });
  }

  async function refreshImageList() {
    var r = await api('/api/annotate/images' + rootQuery());
    annotateState.images = r.images || [];
    updateStats(r.images.length, r.images.filter(function (x) {
      return x.annotated;
    }).length);
    renderImageList();
    return r;
  }

  function findNextUnannotated(from) {
    for (var i = from; i < annotateState.images.length; i++) {
      if (!annotateState.images[i].annotated) return i;
    }
    for (var j = 0; j < from; j++) {
      if (!annotateState.images[j].annotated) return j;
    }
    return -1;
  }

  async function loadCurrentImage() {
    var imgs = annotateState.images;
    if (!imgs.length) {
      $('annotate-img').removeAttribute('src');
      $('btn-save-anno').disabled = true;
      return;
    }
    if (annotateState.index < 0 || annotateState.index >= imgs.length) {
      annotateState.index = findNextUnannotated(0);
      if (annotateState.index < 0) annotateState.index = 0;
    }
    var cur = imgs[annotateState.index];
    var url =
      '/api/annotate/image/' +
      cur.split +
      '/' +
      encodeURIComponent(cur.filename) +
      rootQuery();
    var imgEl = $('annotate-img');
    annotateState.hasClick = false;
    $('btn-save-anno').disabled = true;
    $('annotate-marker').classList.add('hidden');

    imgEl.onload = function () {
      annotateState.naturalW = imgEl.naturalWidth;
      annotateState.naturalH = imgEl.naturalHeight;
      try {
        api('/api/annotate/label/' + cur.split + '/' + cur.stem + rootQuery()).then(function (r) {
          if (r.label) {
            placeMarker(r.label.kx * annotateState.naturalW, r.label.ky * annotateState.naturalH);
          }
        });
      } catch (_) {}
    };
    imgEl.src = url;
    $('click-info').textContent = cur.filename + ' (' + (annotateState.index + 1) + '/' + imgs.length + ')';
  }

  function placeMarker(px, py) {
    var imgEl = $('annotate-img');
    var wrap = $('canvas-wrap');
    if (!imgEl.clientWidth) return;
    var scaleX = imgEl.clientWidth / annotateState.naturalW;
    var scaleY = imgEl.clientHeight / annotateState.naturalH;
    annotateState.clickX = px;
    annotateState.clickY = py;
    annotateState.hasClick = true;
    $('btn-save-anno').disabled = false;
    var marker = $('annotate-marker');
    marker.classList.remove('hidden');
    marker.style.left = px * scaleX - 6 + 'px';
    marker.style.top = py * scaleY - 6 + 'px';
    $('click-info').textContent =
      'Earlobe: ' +
      Math.round(px) +
      ', ' +
      Math.round(py) +
      ' (of ' +
      annotateState.naturalW +
      '×' +
      annotateState.naturalH +
      ')';
  }

  function initAnnotate() {
    var wrap = $('canvas-wrap');
    wrap.addEventListener('click', function (e) {
      var imgEl = $('annotate-img');
      if (!imgEl.src || !annotateState.naturalW) return;
      var rect = imgEl.getBoundingClientRect();
      var x = ((e.clientX - rect.left) / rect.width) * annotateState.naturalW;
      var y = ((e.clientY - rect.top) / rect.height) * annotateState.naturalH;
      placeMarker(x, y);
    });

    $('btn-init').addEventListener('click', async function () {
      setLog('log-annotate', 'Setting up…');
      try {
        var r = await api('/api/annotate/init', {
          method: 'POST',
          body: JSON.stringify({ root: datasetRoot() }),
        });
        updateStats(r.total, r.annotated);
        await refreshImageList();
        await loadCurrentImage();
        setLog('log-annotate', 'Ready. Upload images, then click each earlobe.');
      } catch (e) {
        setLog('log-annotate', String(e), true);
      }
    });

    $('file-upload').addEventListener('change', async function (ev) {
      var files = ev.target.files;
      if (!files || !files.length) return;
      setLog('log-annotate', 'Uploading ' + files.length + ' file(s)…');
      var fd = new FormData();
      for (var i = 0; i < files.length; i++) fd.append('files', files[i]);
      try {
        var res = await fetch('/api/annotate/upload' + rootQuery(), { method: 'POST', body: fd });
        var data = await res.json();
        if (!res.ok) throw new Error(data.detail || 'Upload failed');
        updateStats(data.total, data.annotated);
        await refreshImageList();
        annotateState.index = findNextUnannotated(0);
        if (annotateState.index < 0) annotateState.index = 0;
        await loadCurrentImage();
        setLog('log-annotate', 'Uploaded: ' + (data.saved || []).join(', '));
        ev.target.value = '';
      } catch (e) {
        setLog('log-annotate', String(e), true);
      }
    });

    $('btn-save-anno').addEventListener('click', async function () {
      if (!annotateState.hasClick) return;
      var cur = annotateState.images[annotateState.index];
      try {
        var r = await api('/api/annotate/save', {
          method: 'POST',
          body: JSON.stringify({
            root: datasetRoot(),
            split: cur.split,
            filename: cur.filename,
            x: annotateState.clickX,
            y: annotateState.clickY,
            image_width: annotateState.naturalW,
            image_height: annotateState.naturalH,
          }),
        });
        updateStats(r.total, r.annotated);
        await refreshImageList();
        var next = findNextUnannotated(annotateState.index + 1);
        if (next >= 0) {
          annotateState.index = next;
          await loadCurrentImage();
          setLog('log-annotate', 'Saved. Next image.');
        } else {
          setLog('log-annotate', 'All images annotated! Go to tab 2. Train.');
        }
      } catch (e) {
        setLog('log-annotate', String(e), true);
      }
    });

    $('btn-skip-anno').addEventListener('click', async function () {
      var next = annotateState.index + 1;
      if (next < annotateState.images.length) {
        annotateState.index = next;
        await loadCurrentImage();
        renderImageList();
      }
    });

    $('btn-split-val').addEventListener('click', async function () {
      try {
        var r = await api('/api/annotate/split-val', {
          method: 'POST',
          body: JSON.stringify({ root: datasetRoot() }),
        });
        await refreshImageList();
        setLog('log-annotate', r.message);
      } catch (e) {
        setLog('log-annotate', String(e), true);
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
    var r = await api('/api/train/status');
    setLog('log-train', r.lines.join('\n') || '…');
    var el = $('log-train');
    el.scrollTop = el.scrollHeight;
    if (!r.running) {
      stopPoll();
      if (r.returncode !== 0) el.classList.add('error');
    }
  }

  function initTrain() {
    $('btn-prepare').addEventListener('click', async function () {
      setLog('log-prepare', 'Preparing…');
      try {
        var r = await api('/api/annotate/prepare-train', {
          method: 'POST',
          body: JSON.stringify({ root: datasetRoot() }),
        });
        setLog('log-prepare', r.output, !r.ok);
      } catch (e) {
        setLog('log-prepare', String(e), true);
      }
    });

    $('btn-train').addEventListener('click', async function () {
      setLog('log-train', 'Starting…');
      $('btn-train').disabled = true;
      $('train-badge').classList.remove('hidden');
      stopPoll();
      try {
        await api('/api/annotate/prepare-train', {
          method: 'POST',
          body: JSON.stringify({ root: datasetRoot() }),
        });
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

  function initExport() {
    $('btn-export-lib').addEventListener('click', async function () {
      setLog('log-export', 'Exporting…');
      try {
        var r = await api('/api/model/export-library', { method: 'POST' });
        setLog(
          'log-export',
          r.message +
            '\n\nFolder: ' +
            r.export_dir +
            '\nZIP: ' +
            r.zip +
            '\n\n' +
            (r.verify || '')
        );
        $('link-download-zip').classList.remove('hidden');
      } catch (e) {
        setLog('log-export', String(e), true);
      }
    });
  }

  function initGpu() {
    $('btn-gpu').addEventListener('click', async function () {
      $('btn-gpu').disabled = true;
      setLog('log-gpu', 'Checking… (may take 60s)', false);
      try {
        var r = await api('/api/environment');
        setLog('log-gpu', r.output, !r.ok);
      } catch (e) {
        setLog('log-gpu', String(e), true);
      } finally {
        $('btn-gpu').disabled = false;
      }
    });
  }

  function init() {
    initTabs();
    initAnnotate();
    initTrain();
    initExport();
    initGpu();
    checkHealth();
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();

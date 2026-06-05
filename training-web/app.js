(function () {
  const API = '';
  var annotateState = {
    images: [],
    index: 0,
    naturalW: 0,
    naturalH: 0,
    activeSide: 'left',
    left: null,
    right: null,
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
    var el = $('dataset-root');
    if (!el) throw new Error('Dataset path field missing — refresh page (Ctrl+F5)');
    var v = (el.value || '').trim();
    if (!v) {
      throw new Error(
        'Enter dataset folder path in the box above (e.g. C:\\earlobe-tracking\\earlobe), then click "1. Setup folder".'
      );
    }
    return v;
  }

  function rootQuery() {
    return '?root=' + encodeURIComponent(datasetRoot());
  }

  function savePathToStorage() {
    try {
      var v = ($('dataset-root').value || '').trim();
      if (v) localStorage.setItem('earlobe_dataset_root', v);
    } catch (_) {}
  }

  async function ensureDatasetReady() {
    var root = datasetRoot();
    savePathToStorage();
    return api('/api/annotate/init', {
      method: 'POST',
      body: JSON.stringify({ root: root }),
    });
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
      var tags = '';
      if (img.has_left) tags += ' L';
      if (img.has_right) tags += ' R';
      li.textContent = (img.annotated ? '✓' : '○') + tags + ' ' + img.filename;
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

  function nextImageIndex(from) {
    var n = annotateState.images.length;
    if (n === 0) return 0;
    return (from + 1) % n;
  }

  async function loadLabelsForImage(cur) {
    try {
      var r = await api('/api/annotate/label/' + cur.split + '/' + cur.stem + rootQuery());
      annotateState.left = null;
      annotateState.right = null;
      if (r.ears) {
        if (r.ears.left) {
          annotateState.left = {
            x: r.ears.left.kx * annotateState.naturalW,
            y: r.ears.left.ky * annotateState.naturalH,
          };
        }
        if (r.ears.right) {
          annotateState.right = {
            x: r.ears.right.kx * annotateState.naturalW,
            y: r.ears.right.ky * annotateState.naturalH,
          };
        }
      }
      updateMarkers();
      updateSaveButton();
      updateClickInfo();
      requestAnimationFrame(updateMarkers);
    } catch (e) {
      setLog('log-annotate', 'Could not load labels: ' + String(e), true);
    }
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
      rootQuery() +
      '&_t=' +
      Date.now();
    var imgEl = $('annotate-img');
    annotateState.left = null;
    annotateState.right = null;
    updateMarkers();
    updateSaveButton();

    imgEl.onload = function () {
      annotateState.naturalW = imgEl.naturalWidth;
      annotateState.naturalH = imgEl.naturalHeight;
      loadLabelsForImage(cur);
    };
    imgEl.src = url;
    $('click-info').textContent =
      cur.filename + ' (' + (annotateState.index + 1) + '/' + imgs.length + ')' + (cur.annotated ? ' — edit' : '');
  }

  function setActiveSide(side) {
    annotateState.activeSide = side;
    document.querySelectorAll('.mode-btn').forEach(function (b) {
      b.classList.toggle('active', b.dataset.side === side);
    });
    updateClickInfo();
  }

  function updateSaveButton() {
    $('btn-save-anno').disabled = !(annotateState.left || annotateState.right);
  }

  function updateClickInfo() {
    var parts = [];
    if (annotateState.left) parts.push('L:' + Math.round(annotateState.left.x) + ',' + Math.round(annotateState.left.y));
    if (annotateState.right) parts.push('R:' + Math.round(annotateState.right.x) + ',' + Math.round(annotateState.right.y));
    var mode = annotateState.activeSide === 'left' ? 'Left ear' : 'Right ear';
    $('click-info').textContent =
      mode + ' — click earlobe' + (parts.length ? ' | ' + parts.join(' | ') : ' | (none marked yet)');
  }

  function updateMarkers() {
    var imgEl = $('annotate-img');
    if (!imgEl.clientWidth || !annotateState.naturalW) return;
    var scaleX = imgEl.clientWidth / annotateState.naturalW;
    var scaleY = imgEl.clientHeight / annotateState.naturalH;
    function show(side, point, elId) {
      var el = $(elId);
      if (!point) {
        el.classList.add('hidden');
        return;
      }
      el.classList.remove('hidden');
      el.style.left = point.x * scaleX - 6 + 'px';
      el.style.top = point.y * scaleY - 6 + 'px';
    }
    show('left', annotateState.left, 'marker-left');
    show('right', annotateState.right, 'marker-right');
  }

  function placeMarkerForSide(side, px, py) {
    var pt = { x: px, y: py };
    if (side === 'left') annotateState.left = pt;
    else annotateState.right = pt;
    updateMarkers();
    updateSaveButton();
    updateClickInfo();
  }

  function initAnnotate() {
    var wrap = $('canvas-wrap');
    wrap.addEventListener('click', function (e) {
      var imgEl = $('annotate-img');
      if (!imgEl.src || !annotateState.naturalW) return;
      var rect = imgEl.getBoundingClientRect();
      var x = ((e.clientX - rect.left) / rect.width) * annotateState.naturalW;
      var y = ((e.clientY - rect.top) / rect.height) * annotateState.naturalH;
      placeMarkerForSide(annotateState.activeSide, x, y);
    });

    $('btn-mode-left').addEventListener('click', function () {
      setActiveSide('left');
    });
    $('btn-mode-right').addEventListener('click', function () {
      setActiveSide('right');
    });
    $('btn-clear-left').addEventListener('click', function () {
      annotateState.left = null;
      updateMarkers();
      updateSaveButton();
      updateClickInfo();
    });
    $('btn-clear-right').addEventListener('click', function () {
      annotateState.right = null;
      updateMarkers();
      updateSaveButton();
      updateClickInfo();
    });

    var pathInput = $('dataset-root');
    if (pathInput) {
      try {
        var saved = localStorage.getItem('earlobe_dataset_root');
        if (saved && !pathInput.value) pathInput.value = saved;
      } catch (_) {}
      pathInput.addEventListener('input', function () {
        savePathToStorage();
        setLog('log-annotate', '', false);
      });
    }

    $('btn-init').addEventListener('click', async function () {
      setLog('log-annotate', 'Setting up…');
      try {
        var r = await ensureDatasetReady();
        updateStats(r.total, r.annotated);
        await refreshImageList();
        await loadCurrentImage();
        setLog('log-annotate', 'Setup OK at: ' + datasetRoot() + '\nNow click "2. Upload images".');
      } catch (e) {
        setLog('log-annotate', String(e), true);
      }
    });

    $('btn-upload').addEventListener('click', function () {
      try {
        datasetRoot();
        $('file-upload').click();
      } catch (e) {
        setLog('log-annotate', String(e), true);
      }
    });

    $('file-upload').addEventListener('change', async function (ev) {
      var files = ev.target.files;
      if (!files || !files.length) return;
      setLog('log-annotate', 'Uploading ' + files.length + ' file(s)…');
      try {
        await ensureDatasetReady();
        var fd = new FormData();
        for (var i = 0; i < files.length; i++) fd.append('files', files[i]);
        var res = await fetch('/api/annotate/upload' + rootQuery(), { method: 'POST', body: fd });
        var data = await res.json();
        if (!res.ok) {
          var errMsg = data.detail || data.message || 'Upload failed';
          throw new Error(typeof errMsg === 'string' ? errMsg : JSON.stringify(errMsg));
        }
        updateStats(data.total, data.annotated);
        await refreshImageList();
        annotateState.index = findNextUnannotated(0);
        if (annotateState.index < 0) annotateState.index = 0;
        await loadCurrentImage();
        setLog('log-annotate', 'Uploaded: ' + (data.saved || []).join(', ') + '\nClick the earlobe on the image.');
        ev.target.value = '';
      } catch (e) {
        setLog('log-annotate', String(e), true);
      }
    });

    async function saveAnnotation(advance) {
      if (!annotateState.left && !annotateState.right) {
        setLog('log-annotate', 'Mark at least one ear (left or right) before saving.', true);
        return;
      }
      if (!annotateState.naturalW || !annotateState.naturalH) {
        setLog('log-annotate', 'Wait for the image to finish loading.', true);
        return;
      }
      var cur = annotateState.images[annotateState.index];
      $('btn-save-anno').disabled = true;
      $('btn-save-stay').disabled = true;
      try {
        var r = await api('/api/annotate/save', {
          method: 'POST',
          body: JSON.stringify({
            root: datasetRoot(),
            split: cur.split,
            filename: cur.filename,
            image_width: annotateState.naturalW,
            image_height: annotateState.naturalH,
            left: annotateState.left,
            right: annotateState.right,
          }),
        });
        updateStats(r.total, r.annotated);
        await refreshImageList();
        if (advance) {
          var pending = findNextUnannotated(0);
          annotateState.index = pending >= 0 ? pending : nextImageIndex(annotateState.index);
          await loadCurrentImage();
          setLog('log-annotate', pending >= 0 ? 'Saved. Next unannotated.' : 'Saved. Next image.');
        } else {
          await loadLabelsForImage(cur);
          renderImageList();
          setLog('log-annotate', 'Saved changes to ' + cur.filename);
        }
      } catch (e) {
        setLog('log-annotate', 'Save failed: ' + String(e), true);
      } finally {
        updateSaveButton();
        $('btn-save-stay').disabled = false;
      }
    }

    $('btn-save-anno').addEventListener('click', function () {
      saveAnnotation(true);
    });

    $('btn-save-stay').addEventListener('click', function () {
      saveAnnotation(false);
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
        await ensureDatasetReady();
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
    setLog('log-annotate', 'Enter path → "1. Setup folder" → "2. Upload images".', false);
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();

(function () {
  const API = '/api/side-ear';
  const KPT_LABELS = ['Earlobe', 'Earring 1', 'Earring 2', 'Earring 3'];
  var annotateState = {
    images: [],
    index: 0,
    naturalW: 0,
    naturalH: 0,
    activeKpt: 0,
    points: [null, null, null, null],
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
    if (!el) throw new Error('Dataset path field missing');
    var v = (el.value || '').trim();
    if (!v) throw new Error('Enter side-ear dataset folder path, then click Setup folder.');
    return v;
  }

  function rootQuery() {
    return '?root=' + encodeURIComponent(datasetRoot());
  }

  function savePathToStorage() {
    try {
      var v = ($('dataset-root').value || '').trim();
      if (v) localStorage.setItem('side_ear_dataset_root', v);
    } catch (_) {}
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
    try {
      await fetch('/api/health');
      $('api-status').textContent = 'API connected — side-ear training ready';
    } catch (e) {
      $('api-status').textContent = 'Server not running';
      $('api-status').classList.add('error');
      setLog('log-annotate', 'Start: python python/training_server.py', true);
    }
  }

  function renderImageList() {
    var list = $('image-list');
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
    var r = await api('/annotate/images' + rootQuery());
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

  function clearPoints() {
    annotateState.points = [null, null, null, null];
    updateMarkers();
    updateSaveButton();
    updateClickInfo();
  }

  async function loadLabelsForImage(cur) {
    clearPoints();
    try {
      var r = await api('/annotate/label/' + cur.split + '/' + cur.stem + rootQuery());
      if (r.points && r.points.length === 4) {
        for (var i = 0; i < 4; i++) {
          annotateState.points[i] = {
            x: r.points[i].kx * annotateState.naturalW,
            y: r.points[i].ky * annotateState.naturalH,
          };
        }
      }
      updateMarkers();
      updateSaveButton();
      updateClickInfo();
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
      '/api/side-ear/annotate/image/' +
      cur.split +
      '/' +
      encodeURIComponent(cur.filename) +
      rootQuery() +
      '&_t=' +
      Date.now();
    var imgEl = $('annotate-img');
    clearPoints();
    imgEl.onload = function () {
      annotateState.naturalW = imgEl.naturalWidth;
      annotateState.naturalH = imgEl.naturalHeight;
      loadLabelsForImage(cur);
    };
    imgEl.src = url;
    $('click-info').textContent =
      cur.filename + ' (' + (annotateState.index + 1) + '/' + imgs.length + ')' + (cur.annotated ? ' — edit' : '');
  }

  function setActiveKpt(idx) {
    annotateState.activeKpt = idx;
    document.querySelectorAll('.kpt-btn').forEach(function (b) {
      b.classList.toggle('active', Number(b.dataset.kpt) === idx);
    });
    updateClickInfo();
  }

  function allPointsSet() {
    return annotateState.points.every(function (p) {
      return p !== null;
    });
  }

  function updateSaveButton() {
    $('btn-save-anno').disabled = !allPointsSet();
  }

  function updateClickInfo() {
    var parts = [];
    for (var i = 0; i < 4; i++) {
      var p = annotateState.points[i];
      parts.push(i + ':' + (p ? Math.round(p.x) + ',' + Math.round(p.y) : '—'));
    }
    $('click-info').textContent =
      'Active: ' + KPT_LABELS[annotateState.activeKpt] + ' | ' + parts.join(' | ');
  }

  function updateMarkers() {
    var imgEl = $('annotate-img');
    if (!imgEl.clientWidth || !annotateState.naturalW) return;
    var scaleX = imgEl.clientWidth / annotateState.naturalW;
    var scaleY = imgEl.clientHeight / annotateState.naturalH;
    for (var i = 0; i < 4; i++) {
      var el = $('marker-' + i);
      var pt = annotateState.points[i];
      if (!pt) {
        el.classList.add('hidden');
        continue;
      }
      el.classList.remove('hidden');
      el.style.left = pt.x * scaleX - 6 + 'px';
      el.style.top = pt.y * scaleY - 6 + 'px';
    }
  }

  function initAnnotate() {
    var pathInput = $('dataset-root');
    if (pathInput) {
      try {
        var saved = localStorage.getItem('side_ear_dataset_root');
        if (saved) pathInput.value = saved;
      } catch (_) {}
      pathInput.addEventListener('input', savePathToStorage);
    }

    document.querySelectorAll('.kpt-btn').forEach(function (btn) {
      btn.addEventListener('click', function () {
        setActiveKpt(Number(btn.dataset.kpt));
      });
    });

    $('canvas-wrap').addEventListener('click', function (e) {
      var imgEl = $('annotate-img');
      if (!imgEl.src || !annotateState.naturalW) return;
      var rect = imgEl.getBoundingClientRect();
      var x = ((e.clientX - rect.left) / rect.width) * annotateState.naturalW;
      var y = ((e.clientY - rect.top) / rect.height) * annotateState.naturalH;
      annotateState.points[annotateState.activeKpt] = { x: x, y: y };
      var next = annotateState.activeKpt;
      while (next < 3 && annotateState.points[next + 1]) next++;
      if (next < 3 && annotateState.points[next + 1] === null && annotateState.activeKpt === next) {
        setActiveKpt(next + 1);
      }
      updateMarkers();
      updateSaveButton();
      updateClickInfo();
    });

    $('btn-clear-all').addEventListener('click', clearPoints);

    $('btn-init').addEventListener('click', async function () {
      try {
        savePathToStorage();
        var r = await api('/annotate/init', {
          method: 'POST',
          body: JSON.stringify({ root: datasetRoot() }),
        });
        updateStats(r.total, r.annotated);
        await refreshImageList();
        await loadCurrentImage();
        setLog('log-annotate', 'Setup OK. Upload side-ear images.');
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
      try {
        var fd = new FormData();
        for (var i = 0; i < files.length; i++) fd.append('files', files[i]);
        var res = await fetch(API + '/annotate/upload' + rootQuery(), { method: 'POST', body: fd });
        var data = await res.json();
        if (!res.ok) throw new Error(data.detail || 'Upload failed');
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

    async function saveAnnotation(advance) {
      if (!allPointsSet()) {
        setLog('log-annotate', 'Mark all 4 points before saving.', true);
        return;
      }
      var cur = annotateState.images[annotateState.index];
      $('btn-save-anno').disabled = true;
      try {
        var r = await api('/annotate/save', {
          method: 'POST',
          body: JSON.stringify({
            root: datasetRoot(),
            split: cur.split,
            filename: cur.filename,
            image_width: annotateState.naturalW,
            image_height: annotateState.naturalH,
            points: annotateState.points,
          }),
        });
        updateStats(r.total, r.annotated);
        await refreshImageList();
        if (advance) {
          var pending = findNextUnannotated(0);
          annotateState.index = pending >= 0 ? pending : (annotateState.index + 1) % annotateState.images.length;
          await loadCurrentImage();
          setLog('log-annotate', 'Saved.');
        } else {
          await loadLabelsForImage(cur);
          renderImageList();
          setLog('log-annotate', 'Saved ' + cur.filename);
        }
      } catch (e) {
        setLog('log-annotate', String(e), true);
      } finally {
        updateSaveButton();
      }
    }

    $('btn-save-anno').addEventListener('click', function () {
      saveAnnotation(true);
    });
    $('btn-save-stay').addEventListener('click', function () {
      saveAnnotation(false);
    });
    $('btn-skip-anno').addEventListener('click', async function () {
      if (annotateState.index + 1 < annotateState.images.length) {
        annotateState.index++;
        await loadCurrentImage();
        renderImageList();
      }
    });

    $('btn-split-val').addEventListener('click', async function () {
      try {
        var r = await api('/annotate/split-val', {
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
    if (pollTimer) clearInterval(pollTimer);
    pollTimer = null;
    $('train-badge').classList.add('hidden');
    $('btn-train').disabled = false;
  }

  function initTrain() {
    $('btn-prepare').addEventListener('click', async function () {
      try {
        var r = await api('/annotate/prepare-train', {
          method: 'POST',
          body: JSON.stringify({ root: datasetRoot() }),
        });
        setLog('log-prepare', r.output, !r.ok);
      } catch (e) {
        setLog('log-prepare', String(e), true);
      }
    });

    $('btn-train').addEventListener('click', async function () {
      $('btn-train').disabled = true;
      $('train-badge').classList.remove('hidden');
      stopPoll();
      try {
        await api('/annotate/prepare-train', {
          method: 'POST',
          body: JSON.stringify({ root: datasetRoot() }),
        });
        await api('/train/start', {
          method: 'POST',
          body: JSON.stringify({
            epochs: Number($('epochs').value),
            batch: Number($('batch').value),
            imgsz: Number($('imgsz').value),
            device: $('device').value,
            patience: Number($('patience').value),
          }),
        });
        pollTimer = setInterval(async function () {
          var r = await api('/train/status');
          setLog('log-train', r.lines.join('\n') || '…');
          if (!r.running) stopPoll();
        }, 800);
      } catch (e) {
        stopPoll();
        setLog('log-train', String(e), true);
      }
    });
  }

  function initExport() {
    $('btn-export-lib').addEventListener('click', async function () {
      try {
        var r = await api('/model/export-library', { method: 'POST' });
        setLog('log-export', r.message + '\n\nFolder: ' + r.export_dir + '\nZIP: ' + r.zip);
        $('link-download-zip').classList.remove('hidden');
      } catch (e) {
        setLog('log-export', String(e), true);
      }
    });
  }

  function initGpu() {
    $('btn-gpu').addEventListener('click', async function () {
      try {
        var r = await fetch('/api/environment').then(function (x) {
          return x.json();
        });
        setLog('log-gpu', r.output, !r.ok);
      } catch (e) {
        setLog('log-gpu', String(e), true);
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
    setActiveKpt(0);
    setLog('log-annotate', 'Separate folder from front-face data → Setup → Upload side-ear images.');
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();

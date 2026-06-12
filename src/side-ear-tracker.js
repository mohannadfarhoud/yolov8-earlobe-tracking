/**
 * Side-ear detection library — 4 keypoints: earlobe + 3 earring anchors.
 */

import { getVideoFps, isMobileDevice } from './device.js';
import { decodeSideEar } from './decoder.js';
import { letterboxFromCanvas, letterboxToTensor, mapToScreen, mapToSource } from './letterbox.js';
import { createSession, runInference } from './onnx-engine.js';
import { EarPointSmoother } from './smoothing.js';
import { DEFAULT_SIDE_EAR_CFG } from './side-ear-defaults.js';

const TENSOR_SIZE = 1 * 3 * 640 * 640;
const POINT_KEYS = ['earlobe', 'earPos1', 'earPos2', 'earPos3'];
const EMPTY = { earlobe: null, earPos1: null, earPos2: null, earPos3: null };

function resolveScanTiming(options, mobile) {
  if (options.fps != null && options.fps > 0) {
    return { fps: options.fps, scanIntervalMs: 1000 / options.fps };
  }
  if (options.minIntervalMs != null && options.minIntervalMs > 0) {
    return { fps: 1000 / options.minIntervalMs, scanIntervalMs: options.minIntervalMs };
  }
  const fps = mobile ? DEFAULT_SIDE_EAR_CFG.mobileDefaultFps : DEFAULT_SIDE_EAR_CFG.defaultFps;
  return { fps, scanIntervalMs: 1000 / fps };
}

function mapPoint(pt, lb, mirrorX) {
  if (!pt) return null;
  const src = mapToSource(pt.x, pt.y, lb);
  const display = mirrorX ? mapToScreen(src.x, src.y, lb) : { x: src.x, y: src.y };
  return {
    x: display.x,
    y: display.y,
    confidence: pt.conf,
    xRaw: src.x,
    yRaw: src.y,
  };
}

function mapSideEarResult(raw, lb, mirrorX) {
  const out = { ...EMPTY };
  for (const key of POINT_KEYS) {
    out[key] = raw[key] ? mapPoint(raw[key], lb, mirrorX) : null;
  }
  return out;
}

class SideEarWorkerClient {
  constructor(workerUrl) {
    const url = workerUrl || new URL('./side-ear-worker.js', import.meta.url);
    this.worker = new Worker(url, { type: 'module' });
    this._seq = 0;
    this._pending = new Map();
    this._ready = new Promise((resolve, reject) => {
      this._resolveReady = resolve;
      this._rejectReady = reject;
    });
    this.worker.onmessage = (ev) => this._onMessage(ev);
    this.worker.onerror = (err) => {
      if (this._rejectReady) {
        this._rejectReady(err);
        this._rejectReady = null;
      }
      for (const [, p] of this._pending) p.reject(err);
      this._pending.clear();
    };
  }

  _onMessage(ev) {
    const msg = ev.data;
    if (msg.type === 'ready') {
      if (this._resolveReady) {
        this._resolveReady();
        this._resolveReady = null;
      }
      return;
    }
    if (msg.type === 'error') {
      const p = this._pending.get(msg.id);
      if (p) {
        this._pending.delete(msg.id);
        p.reject(new Error(msg.message || 'worker error'));
      }
      return;
    }
    if (msg.type === 'result') {
      const p = this._pending.get(msg.id);
      if (p) {
        this._pending.delete(msg.id);
        p.resolve(msg.points);
      }
    }
  }

  async init(opts) {
    this.worker.postMessage({
      type: 'init',
      modelUrl: opts.modelUrl,
      cfg: opts.cfg,
      mobile: !!opts.mobile,
      useGpu: !!opts.useGpu,
    });
    await this._ready;
  }

  infer(tensor) {
    const id = ++this._seq;
    const copy = new Float32Array(tensor);
    return new Promise((resolve, reject) => {
      this._pending.set(id, { resolve, reject });
      this.worker.postMessage({ type: 'infer', id, tensor: copy }, [copy.buffer]);
    });
  }

  dispose() {
    try {
      this.worker.postMessage({ type: 'dispose' });
    } catch (_) {}
    this.worker.terminate();
  }
}

function makeSmoothers(options, mobile) {
  const hyst = options.smoothHysteresis ?? DEFAULT_SIDE_EAR_CFG.smoothHysteresis;
  const tauMs =
    options.smoothTauMs ??
    (mobile ? DEFAULT_SIDE_EAR_CFG.mobileSmoothTauMs : DEFAULT_SIDE_EAR_CFG.smoothTauMs);
  const base = { tauMs, hystOn: hyst.on, hystOff: hyst.off };
  return Object.fromEntries(POINT_KEYS.map((k) => [k, new EarPointSmoother(base)]));
}

function smoothResult(raw, smoothers, now) {
  const out = { ...EMPTY };
  for (const key of POINT_KEYS) {
    const pt = raw[key];
    const smoothed = smoothers[key].update(
      pt ? { x: pt.x, y: pt.y, confidence: pt.confidence, xRaw: pt.xRaw, yRaw: pt.yRaw } : null,
      now
    );
    out[key] = smoothed
      ? { x: smoothed.x, y: smoothed.y, confidence: smoothed.confidence, xRaw: smoothed.xRaw, yRaw: smoothed.yRaw }
      : null;
  }
  return out;
}

export async function createSideEarTracker(options = {}) {
  const mobile = options.mobile ?? isMobileDevice();
  const cfg = {
    numClasses: options.numClasses ?? DEFAULT_SIDE_EAR_CFG.numClasses,
    numKpts: options.numKpts ?? DEFAULT_SIDE_EAR_CFG.numKpts,
    boxConfThreshold: options.boxConfThreshold ?? DEFAULT_SIDE_EAR_CFG.boxConfThreshold,
    kptConfThreshold: options.kptConfThreshold ?? DEFAULT_SIDE_EAR_CFG.kptConfThreshold,
    nmsIou: options.nmsIou ?? DEFAULT_SIDE_EAR_CFG.nmsIou,
    selectionRule: options.selectionRule ?? DEFAULT_SIDE_EAR_CFG.selectionRule,
  };
  const mirrorX = options.mirrorX !== false;
  const useWorker = options.useWorker !== false && typeof Worker !== 'undefined';
  const useGpu = mobile ? false : options.useGpu === true;

  let { fps, scanIntervalMs } = resolveScanTiming(options, mobile);
  const maxCaptureSide =
    options.maxCaptureSide ??
    (mobile ? DEFAULT_SIDE_EAR_CFG.mobileMaxCaptureSide : DEFAULT_SIDE_EAR_CFG.desktopMaxCaptureSide);

  let workerClient = null;
  let session = null;
  let inputName = '';
  let tensorReuse = {};
  let usingWorker = false;

  if (useWorker) {
    try {
      workerClient = new SideEarWorkerClient(options.workerUrl);
      await workerClient.init({ modelUrl: options.modelUrl, cfg, mobile, useGpu });
      usingWorker = true;
      console.info(`Side-ear tracker: Web Worker, ${fps} FPS (${scanIntervalMs.toFixed(1)}ms/cycle)`);
    } catch (e) {
      console.warn('Side-ear worker failed, main thread fallback:', e);
      workerClient?.dispose();
      workerClient = null;
    }
  }

  if (!usingWorker) {
    const created = await createSession(options.modelUrl, { useGpu, mobile });
    session = created.session;
    inputName = created.inputName;
    console.info(`Side-ear tracker: main thread, ${fps} FPS (${scanIntervalMs.toFixed(1)}ms/cycle)`);
  }

  const offscreen = document.createElement('canvas');
  const captureCanvas = document.createElement('canvas');
  let tensorBuf = new Float32Array(TENSOR_SIZE);

  let inferBusy = false;
  let lastInfer = 0;
  let lastResult = { ...EMPTY };
  let loopInferTimer = 0;
  let loopRaf = 0;
  let videoFrameHandle = 0;
  let loopRunning = false;
  let disposed = false;
  let onVisibilityChange = null;
  let smoothers = makeSmoothers(options, mobile);
  let boundFrameSource = null;

  function mapWorkerPoints(points, lb) {
    const raw = { ...EMPTY };
    for (const key of POINT_KEYS) {
      const p = points[key];
      raw[key] = p ? { x: p.x, y: p.y, conf: p.conf } : null;
    }
    return mapSideEarResult(raw, lb, mirrorX);
  }

  async function runInferenceBackend(lb) {
    if (usingWorker && workerClient) {
      const points = await workerClient.infer(lb.tensor);
      return mapWorkerPoints(points, lb);
    }
    const { data, dims } = await runInference(session, inputName, lb.tensor, tensorReuse);
    const decoded = decodeSideEar(data, dims, cfg);
    return mapSideEarResult(decoded, lb, mirrorX);
  }

  async function runScanCycle(frameSource) {
    if (disposed || inferBusy) return lastResult;

    let lb;
    if (frameSource instanceof HTMLVideoElement) {
      if (!frameSource.videoWidth) return lastResult;
      lb = letterboxToTensor(frameSource, offscreen, captureCanvas, tensorBuf, { maxCaptureSide });
    } else if (frameSource instanceof HTMLCanvasElement) {
      if (!frameSource.width) return lastResult;
      lb = letterboxFromCanvas(frameSource, offscreen, tensorBuf);
    } else {
      throw new Error('scan requires HTMLVideoElement or HTMLCanvasElement');
    }

    inferBusy = true;
    try {
      lastResult = await runInferenceBackend(lb);
      lastInfer = performance.now();
      if (!tensorBuf.byteLength) tensorBuf = new Float32Array(TENSOR_SIZE);
      return lastResult;
    } catch (e) {
      console.error('Side-ear scan failed:', e);
      return lastResult;
    } finally {
      inferBusy = false;
    }
  }

  function syncFpsFromVideo(video) {
    const live = getVideoFps(video);
    if (live && live > 0) {
      fps = live;
      scanIntervalMs = 1000 / fps;
      console.info(`Side-ear tracker: synced to camera ${fps} FPS`);
    }
    return fps;
  }

  function bindVisibilityStop() {
    if (onVisibilityChange) document.removeEventListener('visibilitychange', onVisibilityChange);
    onVisibilityChange = () => {
      if (document.hidden) stopLoop();
    };
    document.addEventListener('visibilitychange', onVisibilityChange);
  }

  function startFpsScanLoop(frameSource) {
    const tryScan = (now) => {
      if (!inferBusy && now - lastInfer >= scanIntervalMs) {
        runScanCycle(frameSource).catch((e) => console.error(e));
      }
    };

    if (frameSource instanceof HTMLVideoElement && frameSource.requestVideoFrameCallback) {
      const onVideoFrame = (now) => {
        if (!loopRunning || disposed) return;
        tryScan(now);
        videoFrameHandle = frameSource.requestVideoFrameCallback(onVideoFrame);
      };
      videoFrameHandle = frameSource.requestVideoFrameCallback(onVideoFrame);
      return;
    }

    const run = async () => {
      if (!loopRunning || disposed) return;
      const t0 = performance.now();
      tryScan(t0);
      if (!loopRunning || disposed) return;
      const wait = Math.max(0, scanIntervalMs - (performance.now() - t0));
      loopInferTimer = window.setTimeout(run, wait);
    };
    run();
  }

  function startLoop(frameSource, onResult, loopOpts = {}) {
    stopLoop();
    boundFrameSource = frameSource;
    smoothers = makeSmoothers(options, mobile);

    if (loopOpts.syncFps !== false && frameSource instanceof HTMLVideoElement) {
      syncFpsFromVideo(frameSource);
    }

    loopRunning = true;
    bindVisibilityStop();

    const displayTick = (now) => {
      if (!loopRunning || disposed) return;
      onResult(smoothResult(lastResult, smoothers, now));
      loopRaf = requestAnimationFrame(displayTick);
    };
    loopRaf = requestAnimationFrame(displayTick);
    startFpsScanLoop(frameSource);
    runScanCycle(frameSource).catch((e) => console.error(e));
    return stopLoop;
  }

  function stopLoop() {
    loopRunning = false;
    const src = boundFrameSource;
    boundFrameSource = null;
    if (loopInferTimer) {
      clearTimeout(loopInferTimer);
      loopInferTimer = 0;
    }
    if (loopRaf) {
      cancelAnimationFrame(loopRaf);
      loopRaf = 0;
    }
    if (videoFrameHandle && src?.cancelVideoFrameCallback) {
      try {
        src.cancelVideoFrameCallback(videoFrameHandle);
      } catch (_) {}
      videoFrameHandle = 0;
    }
  }

  async function dispose() {
    if (disposed) return;
    disposed = true;
    stopLoop();
    if (onVisibilityChange) {
      document.removeEventListener('visibilitychange', onVisibilityChange);
      onVisibilityChange = null;
    }
    if (usingWorker && workerClient) {
      workerClient.dispose();
      workerClient = null;
    } else if (session) {
      try {
        await session.release();
      } catch (e) {
        console.warn('session.release', e);
      }
      session = null;
    }
  }

  return {
    startLoop,
    stopLoop,
    dispose,
    syncFpsFromVideo,
    getLastResult: () => lastResult,
    isMobile: mobile,
    usingWorker,
    fps,
    scanIntervalMs,
    getStats: () => ({ fps, scanIntervalMs, lastInfer, inferBusy, usingWorker }),
  };
}

export { isMobileDevice, getMobileCameraConstraints, getVideoFps } from './device.js';
export default createSideEarTracker;

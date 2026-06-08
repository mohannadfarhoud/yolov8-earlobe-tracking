/**
 * Abstract earlobe detection library.
 * FPS-based scan cycles — pass `fps` at init; no MediaPipe or UI events required.
 */

import { getVideoFps, isMobileDevice } from './device.js';
import { decodeBothEars } from './decoder.js';
import { letterboxFromCanvas, letterboxToTensor, mapToScreen, mapToSource } from './letterbox.js';
import { createSession, runInference } from './onnx-engine.js';
import { EarPointSmoother, smoothBothEars } from './smoothing.js';
import { DEFAULT_TRACKING_CFG } from './tracking-defaults.js';
import { EarlobeWorkerClient } from './worker-client.js';

const EMPTY = { left: null, right: null };
const TENSOR_SIZE = 1 * 3 * 640 * 640;

function resolveScanTiming(options, mobile) {
  if (options.fps != null && options.fps > 0) {
    const fps = options.fps;
    return { fps, scanIntervalMs: 1000 / fps };
  }
  if (options.minIntervalMs != null && options.minIntervalMs > 0) {
    const scanIntervalMs = options.minIntervalMs;
    return { fps: 1000 / scanIntervalMs, scanIntervalMs };
  }
  const fps = mobile ? DEFAULT_TRACKING_CFG.mobileDefaultFps : DEFAULT_TRACKING_CFG.defaultFps;
  return { fps, scanIntervalMs: 1000 / fps };
}

function mapEar(ear, lb, mirrorX) {
  if (!ear) return null;
  const src = mapToSource(ear.x, ear.y, lb);
  const display = mirrorX ? mapToScreen(src.x, src.y, lb) : { x: src.x, y: src.y };
  return {
    x: display.x,
    y: display.y,
    confidence: ear.conf,
    xRaw: src.x,
    yRaw: src.y,
  };
}

function mapEarsFromWorker(ears, lb, mirrorX) {
  return {
    left: ears.left ? mapEar({ x: ears.left.x, y: ears.left.y, conf: ears.left.conf }, lb, mirrorX) : null,
    right: ears.right ? mapEar({ x: ears.right.x, y: ears.right.y, conf: ears.right.conf }, lb, mirrorX) : null,
  };
}

function makeSmoothers(options, mobile) {
  const hyst = options.smoothHysteresis ?? DEFAULT_TRACKING_CFG.smoothHysteresis;
  const tauMs =
    options.smoothTauMs ??
    (mobile ? DEFAULT_TRACKING_CFG.mobileSmoothTauMs : DEFAULT_TRACKING_CFG.smoothTauMs);
  const base = { tauMs, hystOn: hyst.on, hystOff: hyst.off };
  return {
    left: new EarPointSmoother(base),
    right: new EarPointSmoother(base),
  };
}

export async function createEarlobeTracker(options = {}) {
  const mobile = options.mobile ?? isMobileDevice();
  const cfg = {
    boxConfThreshold: options.boxConfThreshold ?? DEFAULT_TRACKING_CFG.boxConfThreshold,
    kptConfThreshold: options.kptConfThreshold ?? DEFAULT_TRACKING_CFG.kptConfThreshold,
    earlobeKptIndex: options.earlobeKptIndex ?? DEFAULT_TRACKING_CFG.earlobeKptIndex,
    leftClassId: options.leftClassId ?? DEFAULT_TRACKING_CFG.leftClassId,
    rightClassId: options.rightClassId ?? DEFAULT_TRACKING_CFG.rightClassId,
    numClasses: options.numClasses ?? DEFAULT_TRACKING_CFG.numClasses,
    nmsIou: options.nmsIou ?? DEFAULT_TRACKING_CFG.nmsIou,
  };
  const mirrorX = options.mirrorX !== false;
  const useWorker = options.useWorker !== false && typeof Worker !== 'undefined';
  const useGpu = mobile ? false : options.useGpu === true;

  let { fps, scanIntervalMs } = resolveScanTiming(options, mobile);
  const maxCaptureSide =
    options.maxCaptureSide ??
    (mobile ? DEFAULT_TRACKING_CFG.mobileMaxCaptureSide : DEFAULT_TRACKING_CFG.desktopMaxCaptureSide);

  let workerClient = null;
  let session = null;
  let inputName = '';
  let tensorReuse = {};
  let usingWorker = false;

  if (useWorker) {
    try {
      workerClient = new EarlobeWorkerClient(options.workerUrl);
      await workerClient.init({
        modelUrl: options.modelUrl,
        cfg,
        mobile,
        useGpu,
      });
      usingWorker = true;
      console.info(`Earlobe tracker: Web Worker, ${fps} FPS (${scanIntervalMs.toFixed(1)}ms/cycle)`);
    } catch (e) {
      console.warn('Earlobe worker failed, falling back to main thread:', e);
      workerClient?.dispose();
      workerClient = null;
    }
  }

  if (!usingWorker) {
    const created = await createSession(options.modelUrl, { useGpu, mobile });
    session = created.session;
    inputName = created.inputName;
    tensorReuse = {};
    console.info(`Earlobe tracker: main thread, ${fps} FPS (${scanIntervalMs.toFixed(1)}ms/cycle)`);
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

  async function runInferenceBackend(lb) {
    if (usingWorker && workerClient) {
      return workerClient.infer(lb.tensor);
    }
    const { data, dims } = await runInference(session, inputName, lb.tensor, tensorReuse);
    return decodeBothEars(data, dims, cfg);
  }

  function applyEars(ears, lb) {
    if (usingWorker) {
      return mapEarsFromWorker(ears, lb, mirrorX);
    }
    return {
      left: mapEar(ears.left, lb, mirrorX),
      right: mapEar(ears.right, lb, mirrorX),
    };
  }

  /** One clean scan cycle — no overlap, no double throttle. */
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
      const ears = await runInferenceBackend(lb);
      lastResult = applyEars(ears, lb);
      lastInfer = performance.now();
      if (!tensorBuf.byteLength) {
        tensorBuf = new Float32Array(TENSOR_SIZE);
      }
      return lastResult;
    } catch (e) {
      console.error('Earlobe scan failed:', e);
      return lastResult;
    } finally {
      inferBusy = false;
    }
  }

  async function detect(frameSource) {
    if (disposed) return lastResult;
    const now = performance.now();
    if (inferBusy || now - lastInfer < scanIntervalMs) return lastResult;
    return runScanCycle(frameSource);
  }

  function requestDetect(frameSource) {
    const src = frameSource ?? boundFrameSource;
    if (!src) return Promise.resolve(lastResult);
    return detect(src);
  }

  async function detectSide(frameSource, side) {
    const r = await detect(frameSource);
    return side === 'left' ? r.left : r.right;
  }

  function getSmoothedResult(now = performance.now()) {
    return smoothBothEars(lastResult, smoothers, now);
  }

  /** Re-read camera FPS from video track and update scan interval. */
  function syncFpsFromVideo(video) {
    const live = getVideoFps(video);
    if (live && live > 0) {
      fps = live;
      scanIntervalMs = 1000 / fps;
      console.info(`Earlobe tracker: synced to camera ${fps} FPS`);
    }
    return fps;
  }

  function bindVisibilityStop() {
    if (onVisibilityChange) {
      document.removeEventListener('visibilitychange', onVisibilityChange);
    }
    onVisibilityChange = () => {
      if (document.hidden) stopLoop();
    };
    document.addEventListener('visibilitychange', onVisibilityChange);
  }

  /**
   * FPS scan loop — one attempt per video frame (requestVideoFrameCallback).
   * Skips frame if previous scan still running (clean cycles, no pile-up).
   */
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
      const elapsed = performance.now() - t0;
      const wait = Math.max(0, scanIntervalMs - elapsed);
      loopInferTimer = window.setTimeout(run, wait);
    };
    run();
  }

  function startDisplayLoop(frameSource, onResult) {
    boundFrameSource = frameSource;
    smoothers = makeSmoothers(options, mobile);
    loopRunning = true;
    bindVisibilityStop();

    const displayTick = (now) => {
      if (!loopRunning || disposed) return;
      onResult(getSmoothedResult(now));
      loopRaf = requestAnimationFrame(displayTick);
    };
    loopRaf = requestAnimationFrame(displayTick);
  }

  /**
   * Start FPS-based scanning. Pass video after camera is playing.
   * @param {HTMLVideoElement|HTMLCanvasElement} frameSource
   * @param {(result: { left, right }) => void} onResult
   * @param {{ syncFps?: boolean }} [loopOpts] syncFps: read FPS from video track on start
   */
  function startLoop(frameSource, onResult, loopOpts = {}) {
    stopLoop();
    boundFrameSource = frameSource;

    if (loopOpts.syncFps !== false && frameSource instanceof HTMLVideoElement) {
      syncFpsFromVideo(frameSource);
    }

    startDisplayLoop(frameSource, onResult);
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
      disposeTensor(tensorReuse.inputTensor);
      tensorReuse = {};
      try {
        await session.release();
      } catch (e) {
        console.warn('session.release', e);
      }
      session = null;
    }
  }

  return {
    detect,
    requestDetect,
    detectSide,
    startLoop,
    stopLoop,
    dispose,
    syncFpsFromVideo,
    getLastResult: () => lastResult,
    getSmoothedResult,
    isMobile: mobile,
    usingWorker,
    fps,
    scanIntervalMs,
    get minIntervalMs() {
      return scanIntervalMs;
    },
    getStats: () => ({
      fps,
      scanIntervalMs,
      lastInfer,
      inferBusy,
      usingWorker,
    }),
  };
}

function disposeTensor(t) {
  if (t && typeof t.dispose === 'function') {
    try {
      t.dispose();
    } catch (_) {}
  }
}

export { isMobileDevice, getMobileCameraConstraints, getVideoFps } from './device.js';
export { EarlobeWorkerClient } from './worker-client.js';
export default createEarlobeTracker;

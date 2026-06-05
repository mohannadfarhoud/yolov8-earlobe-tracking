/**
 * Browser library: detect left and/or right earlobe from a trained 2-class pose model.
 * Auto-detects phones and uses lighter settings to avoid tab crashes.
 */

import { isMobileDevice } from './device.js';
import { decodeBothEars } from './decoder.js';
import { letterboxFromCanvas, letterboxToTensor, mapToScreen, mapToSource } from './letterbox.js';
import { createSession, runInference } from './onnx-engine.js';
import { DEFAULT_TRACKING_CFG } from './tracking-defaults.js';

const EMPTY = { left: null, right: null };

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
  const minIntervalMs =
    options.minIntervalMs ??
    (mobile ? DEFAULT_TRACKING_CFG.mobileInferenceIntervalMs : DEFAULT_TRACKING_CFG.inferenceIntervalMs);
  const maxCaptureSide =
    options.maxCaptureSide ??
    (mobile ? DEFAULT_TRACKING_CFG.mobileMaxCaptureSide : DEFAULT_TRACKING_CFG.desktopMaxCaptureSide);

  const { session, inputName } = await createSession(options.modelUrl, {
    useGpu: mobile ? false : options.useGpu === true,
    mobile,
  });
  const offscreen = document.createElement('canvas');
  const captureCanvas = document.createElement('canvas');
  const tensorBuf = new Float32Array(1 * 3 * 640 * 640);
  const tensorReuse = {};

  let inferBusy = false;
  let lastInfer = 0;
  let lastResult = { ...EMPTY };
  let loopTimer = 0;
  let loopRaf = 0;
  let loopRunning = false;
  let disposed = false;
  let onVisibilityChange = null;

  async function detect(frameSource) {
    if (disposed) return lastResult;
    if (inferBusy) return lastResult;

    const now = performance.now();
    if (now - lastInfer < minIntervalMs) return lastResult;

    let lb;
    if (frameSource instanceof HTMLVideoElement) {
      if (!frameSource.videoWidth) return lastResult;
      lb = letterboxToTensor(frameSource, offscreen, captureCanvas, tensorBuf, { maxCaptureSide });
    } else if (frameSource instanceof HTMLCanvasElement) {
      if (!frameSource.width) return lastResult;
      lb = letterboxFromCanvas(frameSource, offscreen, tensorBuf);
    } else {
      throw new Error('detect() requires HTMLVideoElement or HTMLCanvasElement');
    }

    inferBusy = true;
    try {
      const { data, dims } = await runInference(session, inputName, lb.tensor, tensorReuse);
      const ears = decodeBothEars(data, dims, cfg);
      lastResult = {
        left: mapEar(ears.left, lb, mirrorX),
        right: mapEar(ears.right, lb, mirrorX),
      };
      lastInfer = performance.now();
      return lastResult;
    } catch (e) {
      console.error('Earlobe detect failed:', e);
      return lastResult;
    } finally {
      inferBusy = false;
    }
  }

  async function detectSide(frameSource, side) {
    const r = await detect(frameSource);
    return side === 'left' ? r.left : r.right;
  }

  /**
   * Safe loop — one inference at a time. On mobile uses setTimeout (less CPU than rAF).
   */
  function startLoop(frameSource, onResult, loopOpts = {}) {
    stopLoop();
    const interval = loopOpts.intervalMs ?? minIntervalMs;
    const useTimer = loopOpts.useTimer ?? mobile;
    loopRunning = true;

    if (onVisibilityChange) {
      document.removeEventListener('visibilitychange', onVisibilityChange);
    }
    onVisibilityChange = () => {
      if (document.hidden) {
        stopLoop();
      }
    };
    document.addEventListener('visibilitychange', onVisibilityChange);

    if (useTimer) {
      const step = async () => {
        if (!loopRunning || disposed) return;
        if (!inferBusy) {
          await detect(frameSource);
        }
        onResult(lastResult);
        loopTimer = window.setTimeout(step, interval);
      };
      step();
    } else {
      let lastTickInfer = 0;
      const tick = (t) => {
        if (!loopRunning || disposed) return;
        if (!inferBusy && t - lastTickInfer >= interval) {
          lastTickInfer = t;
          detect(frameSource).then((r) => onResult(r)).catch((e) => console.error(e));
        } else {
          onResult(lastResult);
        }
        loopRaf = requestAnimationFrame(tick);
      };
      loopRaf = requestAnimationFrame(tick);
    }
    return stopLoop;
  }

  function stopLoop() {
    loopRunning = false;
    if (loopTimer) {
      clearTimeout(loopTimer);
      loopTimer = 0;
    }
    if (loopRaf) {
      cancelAnimationFrame(loopRaf);
      loopRaf = 0;
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
    disposeTensor(tensorReuse.inputTensor);
    tensorReuse.inputTensor = null;
    try {
      await session.release();
    } catch (e) {
      console.warn('session.release', e);
    }
  }

  return {
    detect,
    detectSide,
    startLoop,
    stopLoop,
    dispose,
    getLastResult: () => lastResult,
    isMobile: mobile,
  };
}

function disposeTensor(t) {
  if (t && typeof t.dispose === 'function') {
    try {
      t.dispose();
    } catch (_) {}
  }
}

export { isMobileDevice, getMobileCameraConstraints } from './device.js';
export default createEarlobeTracker;

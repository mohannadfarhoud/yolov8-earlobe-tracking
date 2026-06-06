/**
 * Browser library: detect left and/or right earlobe from a trained 2-class pose model.
 * Use startMotionDriven() with MediaPipe triggers, or startLoop() for interval mode.
 */

import { isMobileDevice } from './device.js';
import { decodeBothEars } from './decoder.js';
import { letterboxFromCanvas, letterboxToTensor, mapToScreen, mapToSource } from './letterbox.js';
import { MotionTrigger } from './motion-trigger.js';
import { createSession, runInference } from './onnx-engine.js';
import { EarPointSmoother, smoothBothEars } from './smoothing.js';
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
  let loopInferTimer = 0;
  let loopRaf = 0;
  let loopRunning = false;
  let disposed = false;
  let onVisibilityChange = null;
  let smoothers = makeSmoothers(options, mobile);
  let boundFrameSource = null;

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

  /** Run ONNX when an external trigger (e.g. MediaPipe) requests it. Respects cooldown. */
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
   * Smooth 60 FPS display only — no timed inference.
   */
  function startDisplayLoop(frameSource, onResult, loopOpts = {}) {
    stopLoop();
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

    if (loopOpts.initialDetect !== false) {
      requestDetect(frameSource).catch((e) => console.error(e));
    }

    return stopLoop;
  }

  /**
   * MediaPipe / motion-driven mode: ONNX runs only when landmarks move (plus cooldown).
   * @returns {{ stop, onLandmarks, requestDetect, motionTrigger }}
   */
  function startMotionDriven(frameSource, onResult, loopOpts = {}) {
    const motionTrigger = loopOpts.motionTrigger ?? new MotionTrigger(loopOpts.motionTriggerOptions);
    const shouldDetect = loopOpts.shouldDetect;

    startDisplayLoop(frameSource, onResult, loopOpts);

    function onLandmarks(landmarks) {
      if (!loopRunning || disposed) return;
      const fire = shouldDetect ? shouldDetect(landmarks) : motionTrigger.check(landmarks);
      if (fire) {
        requestDetect(frameSource).catch((e) => console.error(e));
      }
    }

    return {
      stop: stopLoop,
      onLandmarks,
      requestDetect: () => requestDetect(frameSource),
      motionTrigger,
    };
  }

  /**
   * Interval mode (legacy): timed inference + smooth display.
   */
  function startLoop(frameSource, onResult, loopOpts = {}) {
    stopLoop();
    const inferInterval = loopOpts.intervalMs ?? minIntervalMs;
    boundFrameSource = frameSource;
    smoothers = makeSmoothers(options, mobile);
    loopRunning = true;
    bindVisibilityStop();

    const scheduleInfer = () => {
      if (!loopRunning || disposed) return;
      if (!inferBusy) requestDetect(frameSource).catch((e) => console.error(e));
      loopInferTimer = window.setTimeout(scheduleInfer, inferInterval);
    };
    scheduleInfer();

    const displayTick = (now) => {
      if (!loopRunning || disposed) return;
      onResult(getSmoothedResult(now));
      loopRaf = requestAnimationFrame(displayTick);
    };
    loopRaf = requestAnimationFrame(displayTick);

    return stopLoop;
  }

  function stopLoop() {
    loopRunning = false;
    boundFrameSource = null;
    if (loopInferTimer) {
      clearTimeout(loopInferTimer);
      loopInferTimer = 0;
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
    requestDetect,
    detectSide,
    startLoop,
    startDisplayLoop,
    startMotionDriven,
    stopLoop,
    dispose,
    getLastResult: () => lastResult,
    getSmoothedResult,
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
export { MotionTrigger, MEDIAPIPE_LANDMARKS } from './motion-trigger.js';
export default createEarlobeTracker;

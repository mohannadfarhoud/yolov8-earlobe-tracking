/**
 * Browser library: load trained best.onnx and return earlobe (x, y) from a video frame.
 *
 * @example
 * const tracker = await createEarlobeTracker({ modelUrl: '/models/best.onnx' });
 * const point = await tracker.detect(videoElement);
 * if (point) console.log(point.x, point.y, point.confidence);
 */

import defaultCfg from '../config/tracking.json';
import { decodePoseOutput } from './decoder.js';
import { letterboxToTensor, mapToScreen, mapToSource } from './letterbox.js';
import { createSession, runInference } from './onnx-engine.js';

const IMGSZ = 640;

/**
 * @typedef {Object} EarlobePoint
 * @property {number} x - Pixel X (mirrored when mirrorX is true)
 * @property {number} y - Pixel Y
 * @property {number} confidence
 * @property {number} xRaw - Video pixel X (not mirrored)
 * @property {number} yRaw - Video pixel Y
 * @property {{ x1: number, y1: number, x2: number, y2: number }} [box]
 */

/**
 * @param {import('./letterbox.js').LetterboxResult} lb
 * @param {number} x640
 * @param {number} y640
 * @param {boolean} mirrorX
 */
function toDisplayCoords(lb, x640, y640, mirrorX) {
  const src = mapToSource(x640, y640, lb);
  return mirrorX ? mapToScreen(src.x, src.y, lb) : { x: src.x, y: src.y, xRaw: src.x, yRaw: src.y };
}

function letterboxFromCanvas(canvas, offscreen, tensorBuf) {
  const srcW = canvas.width;
  const srcH = canvas.height;
  const ctx = offscreen.getContext('2d', { willReadFrequently: true });
  offscreen.width = IMGSZ;
  offscreen.height = IMGSZ;
  const gain = Math.min(IMGSZ / srcW, IMGSZ / srcH);
  const newW = Math.round(srcW * gain);
  const newH = Math.round(srcH * gain);
  const padX = (IMGSZ - newW) / 2;
  const padY = (IMGSZ - newH) / 2;
  ctx.fillStyle = '#114';
  ctx.fillRect(0, 0, IMGSZ, IMGSZ);
  ctx.drawImage(canvas, 0, 0, srcW, srcH, padX, padY, newW, newH);
  const { data } = ctx.getImageData(0, 0, IMGSZ, IMGSZ);
  const size = 1 * 3 * IMGSZ * IMGSZ;
  const out = tensorBuf.length === size ? tensorBuf : new Float32Array(size);
  let p = 0;
  const plane = IMGSZ * IMGSZ;
  for (let y = 0; y < IMGSZ; y++) {
    for (let x = 0; x < IMGSZ; x++) {
      const i = (y * IMGSZ + x) * 4;
      out[p] = data[i] / 255;
      out[p + plane] = data[i + 1] / 255;
      out[p + 2 * plane] = data[i + 2] / 255;
      p++;
    }
  }
  return { tensor: out, gain, padX, padY, srcW, srcH };
}

/**
 * @param {Object} options
 * @param {string} options.modelUrl
 * @param {boolean} [options.mirrorX=true]
 */
export async function createEarlobeTracker(options) {
  const cfg = {
    boxConfThreshold: options.boxConfThreshold ?? defaultCfg.boxConfThreshold,
    kptConfThreshold: options.kptConfThreshold ?? defaultCfg.kptConfThreshold,
    earlobeKptIndex: options.earlobeKptIndex ?? defaultCfg.earlobeKptIndex,
    selectionRule: options.selectionRule ?? defaultCfg.selectionRule,
  };
  const mirrorX = options.mirrorX !== false;

  const { session, inputName } = await createSession(options.modelUrl);
  const offscreen = document.createElement('canvas');
  const tensorBuf = new Float32Array(1 * 3 * 640 * 640);

  /**
   * @param {HTMLVideoElement | HTMLCanvasElement} frameSource
   * @returns {Promise<EarlobePoint | null>}
   */
  async function detect(frameSource) {
    let lb;
    if (frameSource instanceof HTMLVideoElement) {
      if (!frameSource.videoWidth) return null;
      lb = letterboxToTensor(frameSource, offscreen, tensorBuf);
    } else if (frameSource instanceof HTMLCanvasElement) {
      if (!frameSource.width) return null;
      lb = letterboxFromCanvas(frameSource, offscreen, tensorBuf);
    } else {
      throw new Error('detect() requires HTMLVideoElement or HTMLCanvasElement');
    }

    const { data, dims } = await runInference(session, inputName, lb.tensor);
    const { detection, earlobe } = decodePoseOutput(data, dims, cfg);
    if (!earlobe) return null;

    const src = mapToSource(earlobe.x, earlobe.y, lb);
    const display = mirrorX ? mapToScreen(src.x, src.y, lb) : { x: src.x, y: src.y };

    /** @type {EarlobePoint} */
    const point = {
      x: display.x,
      y: display.y,
      confidence: earlobe.conf,
      xRaw: src.x,
      yRaw: src.y,
    };

    if (detection) {
      const tlSrc = mapToSource(detection.x1, detection.y1, lb);
      const brSrc = mapToSource(detection.x2, detection.y2, lb);
      const tl = mirrorX ? mapToScreen(tlSrc.x, tlSrc.y, lb) : tlSrc;
      const br = mirrorX ? mapToScreen(brSrc.x, brSrc.y, lb) : brSrc;
      point.box = { x1: tl.x, y1: tl.y, x2: br.x, y2: br.y };
    }

    return point;
  }

  return { detect, dispose() {} };
}

export default createEarlobeTracker;

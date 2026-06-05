/**
 * Browser library: detect left and/or right earlobe from a trained 2-class pose model.
 *
 * @example
 * const tracker = await createEarlobeTracker({ modelUrl: '/models/best.onnx' });
 * const { left, right } = await tracker.detect(videoElement);
 */

import defaultCfg from '../config/tracking.json';
import { decodeBothEars } from './decoder.js';
import { letterboxToTensor, mapToScreen, mapToSource } from './letterbox.js';
import { createSession, runInference } from './onnx-engine.js';

const IMGSZ = 640;

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

export async function createEarlobeTracker(options) {
  const cfg = {
    boxConfThreshold: options.boxConfThreshold ?? defaultCfg.boxConfThreshold,
    kptConfThreshold: options.kptConfThreshold ?? defaultCfg.kptConfThreshold,
    earlobeKptIndex: options.earlobeKptIndex ?? defaultCfg.earlobeKptIndex,
    leftClassId: options.leftClassId ?? defaultCfg.leftClassId ?? 0,
    rightClassId: options.rightClassId ?? defaultCfg.rightClassId ?? 1,
    numClasses: options.numClasses ?? defaultCfg.numClasses ?? 2,
  };
  const mirrorX = options.mirrorX !== false;

  const { session, inputName } = await createSession(options.modelUrl);
  const offscreen = document.createElement('canvas');
  const tensorBuf = new Float32Array(1 * 3 * 640 * 640);

  async function detect(frameSource) {
    let lb;
    if (frameSource instanceof HTMLVideoElement) {
      if (!frameSource.videoWidth) return { left: null, right: null };
      lb = letterboxToTensor(frameSource, offscreen, tensorBuf);
    } else if (frameSource instanceof HTMLCanvasElement) {
      if (!frameSource.width) return { left: null, right: null };
      lb = letterboxFromCanvas(frameSource, offscreen, tensorBuf);
    } else {
      throw new Error('detect() requires HTMLVideoElement or HTMLCanvasElement');
    }

    const { data, dims } = await runInference(session, inputName, lb.tensor);
    const ears = decodeBothEars(data, dims, cfg);
    return {
      left: mapEar(ears.left, lb, mirrorX),
      right: mapEar(ears.right, lb, mirrorX),
    };
  }

  /** @param {'left'|'right'} side */
  async function detectSide(frameSource, side) {
    const r = await detect(frameSource);
    return side === 'left' ? r.left : r.right;
  }

  return { detect, detectSide, dispose() {} };
}

export default createEarlobeTracker;

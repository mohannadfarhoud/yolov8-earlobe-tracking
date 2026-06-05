const IMGSZ = 640;

/**
 * @typedef {{ tensor: Float32Array, gain: number, padX: number, padY: number, srcW: number, srcH: number, capW: number, capH: number }} LetterboxResult
 */

function drawLetterbox(ctx, source, srcW, srcH) {
  const gain = Math.min(IMGSZ / srcW, IMGSZ / srcH);
  const newW = Math.round(srcW * gain);
  const newH = Math.round(srcH * gain);
  const padX = (IMGSZ - newW) / 2;
  const padY = (IMGSZ - newH) / 2;
  ctx.fillStyle = '#114';
  ctx.fillRect(0, 0, IMGSZ, IMGSZ);
  ctx.drawImage(source, 0, 0, srcW, srcH, padX, padY, newW, newH);
  return { gain, padX, padY };
}

function imageDataToTensor(data, tensor) {
  const size = 1 * 3 * IMGSZ * IMGSZ;
  const out = tensor && tensor.length === size ? tensor : new Float32Array(size);
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
  return out;
}

/**
 * Downscale video to capture canvas (saves mobile RAM/CPU), then letterbox to 640.
 * @param {HTMLVideoElement} video
 * @param {HTMLCanvasElement} offscreen
 * @param {HTMLCanvasElement} captureCanvas
 * @param {Float32Array} [tensor]
 * @param {{ maxCaptureSide?: number }} [opts]
 */
export function letterboxToTensor(video, offscreen, captureCanvas, tensor, opts = {}) {
  const fullW = video.videoWidth;
  const fullH = video.videoHeight;
  const maxCapture = opts.maxCaptureSide ?? 960;
  const capScale = Math.min(1, maxCapture / Math.max(fullW, fullH));
  const capW = Math.max(1, Math.round(fullW * capScale));
  const capH = Math.max(1, Math.round(fullH * capScale));

  const capCtx = captureCanvas.getContext('2d', { willReadFrequently: true });
  captureCanvas.width = capW;
  captureCanvas.height = capH;
  capCtx.drawImage(video, 0, 0, capW, capH);

  const ctx = offscreen.getContext('2d', { willReadFrequently: true });
  offscreen.width = IMGSZ;
  offscreen.height = IMGSZ;
  const { gain, padX, padY } = drawLetterbox(ctx, captureCanvas, capW, capH);

  const { data } = ctx.getImageData(0, 0, IMGSZ, IMGSZ);
  return {
    tensor: imageDataToTensor(data, tensor),
    gain,
    padX,
    padY,
    srcW: fullW,
    srcH: fullH,
    capW,
    capH,
  };
}

/**
 * Letterbox a canvas directly (no downscale step).
 */
export function letterboxFromCanvas(canvas, offscreen, tensor) {
  const srcW = canvas.width;
  const srcH = canvas.height;
  const ctx = offscreen.getContext('2d', { willReadFrequently: true });
  offscreen.width = IMGSZ;
  offscreen.height = IMGSZ;
  const { gain, padX, padY } = drawLetterbox(ctx, canvas, srcW, srcH);
  const { data } = ctx.getImageData(0, 0, IMGSZ, IMGSZ);
  return {
    tensor: imageDataToTensor(data, tensor),
    gain,
    padX,
    padY,
    srcW,
    srcH,
    capW: srcW,
    capH: srcH,
  };
}

/**
 * Map point from letterboxed 640 space to full video pixels.
 */
export function mapToSource(x, y, meta) {
  const capX = (x - meta.padX) / meta.gain;
  const capY = (y - meta.padY) / meta.gain;
  const scaleX = meta.srcW / meta.capW;
  const scaleY = meta.srcH / meta.capH;
  return {
    x: Math.max(0, Math.min(meta.srcW, capX * scaleX)),
    y: Math.max(0, Math.min(meta.srcH, capY * scaleY)),
  };
}

/**
 * Map source pixels to mirrored screen coordinates (video is scaleX(-1)).
 */
export function mapToScreen(x, y, meta) {
  return {
    x: meta.srcW - x,
    y,
  };
}

export { IMGSZ };

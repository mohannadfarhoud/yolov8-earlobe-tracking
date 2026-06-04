const IMGSZ = 640;

/**
 * Ultralytics-style letterbox metadata for inverse coordinate mapping.
 * @typedef {{ tensor: Float32Array, gain: number, padX: number, padY: number, srcW: number, srcH: number }} LetterboxResult
 */

/**
 * Draw video frame into offscreen canvas with letterbox padding to IMGSZ.
 * Reuses `tensor` buffer when provided (length 1*3*640*640).
 * @param {HTMLVideoElement} video
 * @param {HTMLCanvasElement} offscreen
 * @param {Float32Array} [tensor]
 * @returns {LetterboxResult}
 */
export function letterboxToTensor(video, offscreen, tensor) {
  const srcW = video.videoWidth;
  const srcH = video.videoHeight;
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
  ctx.drawImage(video, 0, 0, srcW, srcH, padX, padY, newW, newH);

  const { data } = ctx.getImageData(0, 0, IMGSZ, IMGSZ);
  const size = 1 * 3 * IMGSZ * IMGSZ;
  const out = tensor && tensor.length === size ? tensor : new Float32Array(size);

  let p = 0;
  const plane = IMGSZ * IMGSZ;
  for (let y = 0; y < IMGSZ; y++) {
    for (let x = 0; x < IMGSZ; x++) {
      const i = (y * IMGSZ + x) * 4;
      const r = data[i] / 255;
      const g = data[i + 1] / 255;
      const b = data[i + 2] / 255;
      out[p] = r;
      out[p + plane] = g;
      out[p + 2 * plane] = b;
      p++;
    }
  }

  return { tensor: out, gain, padX, padY, srcW, srcH };
}

/**
 * Map point from letterboxed 640 space to source video pixels.
 */
export function mapToSource(x, y, meta) {
  const sx = (x - meta.padX) / meta.gain;
  const sy = (y - meta.padY) / meta.gain;
  return {
    x: Math.max(0, Math.min(meta.srcW, sx)),
    y: Math.max(0, Math.min(meta.srcH, sy)),
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

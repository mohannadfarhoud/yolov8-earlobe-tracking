/**
 * Time-based exponential smoothing for earlobe screen coordinates.
 * Runs cheaply every animation frame while ONNX inference stays throttled.
 */

export class EarPointSmoother {
  /**
   * @param {{ tauMs?: number, hystOn?: number, hystOff?: number }} [opts]
   */
  constructor(opts = {}) {
    this.tauMs = opts.tauMs ?? 70;
    this.hystOn = opts.hystOn ?? 0.55;
    this.hystOff = opts.hystOff ?? 0.45;
    this.visible = false;
    this.x = 0;
    this.y = 0;
    this.confidence = 0;
    this.xRaw = 0;
    this.yRaw = 0;
    this._lastT = performance.now();
  }

  /**
   * @param {{ x: number, y: number, confidence: number, xRaw?: number, yRaw?: number } | null} raw
   * @param {number} [now]
   */
  update(raw, now = performance.now()) {
    const dt = Math.min(120, Math.max(0, now - this._lastT));
    this._lastT = now;

    if (!raw) {
      if (this.visible) this.visible = false;
      return null;
    }

    const conf = raw.confidence;
    if (!this.visible) {
      if (conf < this.hystOn) return null;
      this.visible = true;
      this.x = raw.x;
      this.y = raw.y;
      this.xRaw = raw.xRaw ?? raw.x;
      this.yRaw = raw.yRaw ?? raw.y;
      this.confidence = conf;
      return this._out();
    }

    if (conf < this.hystOff) {
      this.visible = false;
      return null;
    }

    const alpha = 1 - Math.exp(-dt / this.tauMs);
    this.x += (raw.x - this.x) * alpha;
    this.y += (raw.y - this.y) * alpha;
    this.xRaw += ((raw.xRaw ?? raw.x) - this.xRaw) * alpha;
    this.yRaw += ((raw.yRaw ?? raw.y) - this.yRaw) * alpha;
    this.confidence = conf;
    return this._out();
  }

  _out() {
    return {
      x: this.x,
      y: this.y,
      confidence: this.confidence,
      xRaw: this.xRaw,
      yRaw: this.yRaw,
    };
  }
}

/**
 * @param {{ left, right }} raw
 * @param {{ left: EarPointSmoother, right: EarPointSmoother }} smoothers
 * @param {number} [now]
 */
export function smoothBothEars(raw, smoothers, now) {
  return {
    left: smoothers.left.update(raw.left, now),
    right: smoothers.right.update(raw.right, now),
  };
}

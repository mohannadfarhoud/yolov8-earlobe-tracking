/**
 * Throttled async inference: render every frame, infer when prior run finished + interval elapsed.
 */
export class InferenceLoop {
  /**
   * @param {{ intervalMs: number, onInfer: () => Promise<void>, onFrame: () => void }} opts
   */
  constructor(opts) {
    this.intervalMs = opts.intervalMs;
    this.onInfer = opts.onInfer;
    this.onFrame = opts.onFrame;
    this.running = false;
    this.busy = false;
    this.lastInfer = 0;
    this._raf = 0;
  }

  start() {
    if (this.running) return;
    this.running = true;
    const tick = (t) => {
      if (!this.running) return;
      this.onFrame();
      if (!this.busy && t - this.lastInfer >= this.intervalMs) {
        this.busy = true;
        this.lastInfer = t;
        this.onInfer()
          .catch((e) => console.error('inference', e))
          .finally(() => {
            this.busy = false;
          });
      }
      this._raf = requestAnimationFrame(tick);
    };
    this._raf = requestAnimationFrame(tick);
  }

  stop() {
    this.running = false;
    cancelAnimationFrame(this._raf);
  }
}

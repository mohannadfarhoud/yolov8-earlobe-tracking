/**
 * Main-thread client for earlobe-worker.js (module worker).
 */
export class EarlobeWorkerClient {
  /**
   * @param {string | URL} [workerUrl]
   */
  constructor(workerUrl) {
    const url = workerUrl || new URL('./earlobe-worker.js', import.meta.url);
    this.worker = new Worker(url, { type: 'module' });
    this._seq = 0;
    /** @type {Map<number, { resolve, reject }>} */
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
        p.resolve(msg.ears);
      }
    }
  }

  /**
   * @param {{ modelUrl: string, cfg: object, mobile?: boolean, useGpu?: boolean }} opts
   */
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

  /**
   * @param {Float32Array} tensor
   * @returns {Promise<{ left: { x, y, conf } | null, right: { x, y, conf } | null }>}
   */
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

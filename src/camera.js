/**
 * Webcam capture with lifecycle handling and canvas/video dimension sync.
 * Requires secure context (https or http://localhost).
 */

const DEFAULT_CONSTRAINTS = {
  audio: false,
  video: {
    width: { ideal: 1280 },
    height: { ideal: 720 },
    facingMode: 'user',
  },
};

export class Camera {
  /**
   * @param {HTMLVideoElement} video
   * @param {HTMLElement} statusEl
   * @param {(size: { width: number, height: number }) => void} [onResize]
   */
  constructor(video, statusEl, onResize) {
    this.video = video;
    this.statusEl = statusEl;
    this.onResize = onResize;
    this.stream = null;
  }

  setStatus(message, isError = false) {
    this.statusEl.textContent = message;
    this.statusEl.classList.toggle('ok', !isError && message.length > 0);
  }

  syncSize() {
    const w = this.video.videoWidth;
    const h = this.video.videoHeight;
    if (w && h && this.onResize) {
      this.onResize({ width: w, height: h });
    }
  }

  async start() {
    if (!navigator.mediaDevices?.getUserMedia) {
      this.setStatus('Camera API not available. Use HTTPS or localhost.', true);
      throw new Error('getUserMedia unavailable');
    }

    try {
      this.stream = await navigator.mediaDevices.getUserMedia(DEFAULT_CONSTRAINTS);
    } catch (err) {
      const name = err instanceof Error ? err.name : 'Error';
      if (name === 'NotAllowedError') {
        this.setStatus('Camera permission denied.', true);
      } else if (name === 'NotFoundError') {
        this.setStatus('No camera device found.', true);
      } else {
        this.setStatus(`Camera error: ${name}`, true);
      }
      throw err;
    }

    this.video.srcObject = this.stream;
    this.video.addEventListener('loadedmetadata', () => this.syncSize());
    window.addEventListener('resize', () => this.syncSize());

    await this.video.play();
    this.syncSize();
    this.setStatus('Camera active', false);
  }

  stop() {
    if (this.stream) {
      for (const track of this.stream.getTracks()) track.stop();
      this.stream = null;
    }
    this.video.srcObject = null;
  }
}

/**
 * Match canvas pixel dimensions to video intrinsic size and CSS layout.
 * @param {HTMLCanvasElement[]} canvases
 * @param {{ width: number, height: number }} size
 * @param {HTMLElement} stage
 */
export function syncCanvases(canvases, size, stage) {
  for (const canvas of canvases) {
    canvas.width = size.width;
    canvas.height = size.height;
  }
  stage.style.aspectRatio = `${size.width} / ${size.height}`;
}

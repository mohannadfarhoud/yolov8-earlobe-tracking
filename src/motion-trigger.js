import { isMobileDevice } from './device.js';

/**
 * MediaPipe Face Landmarker index hints (478-point model).
 * @see https://github.com/google/mediapipe/blob/master/mediapipe/modules/face_geometry/data/canonical_face_model_uv_visualization.png
 */
export const MEDIAPIPE_LANDMARKS = {
  noseTip: 1,
  leftEar: 234,
  rightEar: 454,
  chin: 152,
};

/**
 * Fires when face landmarks move enough to warrant an earlobe ONNX recalculation.
 * Feed landmarks from MediaPipe Face Landmarker on each frame.
 */
export class MotionTrigger {
  /**
   * @param {{
   *   landmarkIndices?: number[],
   *   threshold?: number,
   *   mobileThreshold?: number,
   * }} [opts]
   */
  constructor(opts = {}) {
    this.indices = opts.landmarkIndices ?? [
      MEDIAPIPE_LANDMARKS.noseTip,
      MEDIAPIPE_LANDMARKS.leftEar,
      MEDIAPIPE_LANDMARKS.rightEar,
    ];
    const mobile = isMobileDevice();
    this.threshold = opts.threshold ?? (mobile ? 0.008 : 0.006);
    if (opts.mobileThreshold != null && mobile) {
      this.threshold = opts.mobileThreshold;
    }
    this._prev = null;
  }

  reset() {
    this._prev = null;
  }

  /**
   * @param {Array<{ x: number, y: number, z?: number }> | null | undefined} landmarks
   * @returns {boolean} true when movement exceeds threshold
   */
  check(landmarks) {
    if (!landmarks?.length) {
      this._prev = null;
      return false;
    }

    const pts = this.indices
      .map((i) => landmarks[i])
      .filter((p) => p && Number.isFinite(p.x) && Number.isFinite(p.y));
    if (!pts.length) {
      this._prev = null;
      return false;
    }

    if (!this._prev) {
      this._prev = pts.map((p) => ({ x: p.x, y: p.y }));
      return true;
    }

    let maxDist = 0;
    for (let i = 0; i < pts.length; i++) {
      const dx = pts[i].x - this._prev[i].x;
      const dy = pts[i].y - this._prev[i].y;
      maxDist = Math.max(maxDist, Math.hypot(dx, dy));
    }

    this._prev = pts.map((p) => ({ x: p.x, y: p.y }));
    return maxDist >= this.threshold;
  }
}

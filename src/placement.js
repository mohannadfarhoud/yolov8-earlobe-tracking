/**
 * Screen-space placement: lerp smoothing, bbox scale, confidence hysteresis.
 */

export class PlacementState {
  /**
   * @param {{ lerpFactor: number, showHysteresis: { on: number, off: number } }} cfg
   */
  constructor(cfg) {
    this.lerp = cfg.lerpFactor;
    this.hystOn = cfg.showHysteresis.on;
    this.hystOff = cfg.showHysteresis.off;
    this.visible = false;
    this.x = 0;
    this.y = 0;
    this.scale = 1;
  }

  /**
   * @param {{ x: number, y: number, conf: number } | null} earlobe
   * @param {{ x1: number, y1: number, x2: number, y2: number } | null} box
   * @param {{ srcW: number, srcH: number }} meta
   */
  update(earlobe, box, meta) {
    if (!earlobe || !box) {
      if (this.visible && (earlobe?.conf ?? 0) < this.hystOff) {
        this.visible = false;
      }
      return { visible: this.visible, x: this.x, y: this.y, scale: this.scale };
    }

    const conf = earlobe.conf;
    if (!this.visible && conf >= this.hystOn) this.visible = true;
    else if (this.visible && conf < this.hystOff) this.visible = false;

    const targetX = earlobe.x;
    const targetY = earlobe.y;
    const bw = box.x2 - box.x1;
    const targetScale = Math.max(0.15, Math.min(2, bw / 80));

    if (!this.visible) {
      this.x = targetX;
      this.y = targetY;
      this.scale = targetScale;
    } else {
      this.x += (targetX - this.x) * this.lerp;
      this.y += (targetY - this.y) * this.lerp;
      this.scale += (targetScale - this.scale) * this.lerp;
    }

    return { visible: this.visible, x: this.x, y: this.y, scale: this.scale };
  }
}

/**
 * Normalized device coordinates from screen pixel coords.
 */
export function pixelsToNdc(x, y, width, height) {
  return {
    x: (x / width) * 2 - 1,
    y: -(y / height) * 2 + 1,
  };
}

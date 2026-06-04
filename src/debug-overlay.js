/**
 * Optional debug overlay: bounding box + earlobe keypoint (toggle via ?debug=1).
 */

/**
 * @param {CanvasRenderingContext2D} ctx
 * @param {{ x: number, y: number, conf: number } | null} earlobe
 * @param {{ x1: number, y1: number, x2: number, y2: number } | null} box
 * @param {{ srcW: number, srcH: number }} meta
 * @param {(x: number, y: number) => { x: number, y: number }} mapFn
 */
export function drawDebug(ctx, earlobe, box, meta, mapFn) {
  ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);
  if (!earlobe && !box) return;

  if (box) {
    const tl = mapFn(box.x1, box.y1);
    const br = mapFn(box.x2, box.y2);
    ctx.strokeStyle = '#0f0';
    ctx.lineWidth = 2;
    ctx.strokeRect(tl.x, tl.y, br.x - tl.x, br.y - tl.y);
  }

  if (earlobe) {
    const p = mapFn(earlobe.x, earlobe.y);
    ctx.fillStyle = '#f0f';
    ctx.beginPath();
    ctx.arc(p.x, p.y, 6, 0, Math.PI * 2);
    ctx.fill();
    ctx.fillStyle = '#fff';
    ctx.font = '12px monospace';
    ctx.fillText(earlobe.conf.toFixed(2), p.x + 8, p.y - 8);
  }
}

export function isDebugEnabled() {
  return new URLSearchParams(location.search).has('debug');
}

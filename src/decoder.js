/**
 * YOLOv8-pose output decode, NMS, and detection selection.
 */

const IMGSZ = 640;

function iou(a, b) {
  const x1 = Math.max(a.x1, b.x1);
  const y1 = Math.max(a.y1, b.y1);
  const x2 = Math.min(a.x2, b.x2);
  const y2 = Math.min(a.y2, b.y2);
  const inter = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
  const areaA = (a.x2 - a.x1) * (a.y2 - a.y1);
  const areaB = (b.x2 - b.x1) * (b.y2 - b.y1);
  return inter / (areaA + areaB - inter + 1e-6);
}

/**
 * @param {number[][]} output [channels][anchors] after transpose
 * @param {number} numClasses
 * @param {number} numKpts
 * @param {number} confThresh
 */
function decodeCandidates(output, numClasses, numKpts, confThresh) {
  const channels = output.length;
  const numAnchors = output[0].length;
  const candidates = [];
  const kptDims = numKpts * 3;
  const kptStart = 4 + numClasses;

  for (let a = 0; a < numAnchors; a++) {
    let bestCls = 0;
    let bestScore = 0;
    for (let c = 0; c < numClasses; c++) {
      const s = output[4 + c][a];
      if (s > bestScore) {
        bestScore = s;
        bestCls = c;
      }
    }
    if (bestScore < confThresh) continue;

    const cx = output[0][a];
    const cy = output[1][a];
    const w = output[2][a];
    const h = output[3][a];
    const x1 = cx - w / 2;
    const y1 = cy - h / 2;
    const x2 = cx + w / 2;
    const y2 = cy + h / 2;

    const kpts = [];
    for (let k = 0; k < numKpts; k++) {
      const base = kptStart + k * 3;
      kpts.push({
        x: output[base][a],
        y: output[base + 1][a],
        conf: output[base + 2][a],
      });
    }

    candidates.push({
      x1,
      y1,
      x2,
      y2,
      score: bestScore,
      classId: bestCls,
      kpts,
    });
  }
  return candidates;
}

function nms(boxes, thresh) {
  boxes.sort((a, b) => b.score - a.score);
  const kept = [];
  const suppressed = new Set();
  for (let i = 0; i < boxes.length; i++) {
    if (suppressed.has(i)) continue;
    kept.push(boxes[i]);
    for (let j = i + 1; j < boxes.length; j++) {
      if (suppressed.has(j)) continue;
      if (iou(boxes[i], boxes[j]) > thresh) suppressed.add(j);
    }
  }
  return kept;
}

/**
 * @param {import('./decoder.js').Detection[]} detections
 * @param {string} rule
 * @param {number} frameW
 * @param {number} frameH
 */
export function selectDetection(detections, rule, frameW = IMGSZ, frameH = IMGSZ) {
  if (!detections.length) return null;
  const cx0 = frameW / 2;
  const cy0 = frameH / 2;

  if (rule === 'largest_box') {
    return detections.reduce((best, d) => {
      const area = (d.x2 - d.x1) * (d.y2 - d.y1);
      const bestArea = (best.x2 - best.x1) * (best.y2 - best.y1);
      return area > bestArea ? d : best;
    });
  }

  if (rule === 'center_nearest') {
    return detections.reduce((best, d) => {
      const mx = (d.x1 + d.x2) / 2;
      const my = (d.y1 + d.y2) / 2;
      const bmx = (best.x1 + best.x2) / 2;
      const bmy = (best.y1 + best.y2) / 2;
      const dist = (mx - cx0) ** 2 + (my - cy0) ** 2;
      const bdist = (bmx - cx0) ** 2 + (bmy - cy0) ** 2;
      return dist < bdist ? d : best;
    });
  }

  return detections[0];
}

/**
 * @typedef {{ x1: number, y1: number, x2: number, y2: number, score: number, classId: number, kpts: { x: number, y: number, conf: number }[] }} Detection
 */

/**
 * @param {Float32Array} raw - ONNX output flattened or 3D [1, C, N]
 * @param {number[]} shape
 * @param {{ numClasses?: number, numKpts?: number, boxConfThreshold: number, kptConfThreshold: number, earlobeKptIndex: number, selectionRule: string, nmsIou?: number }} cfg
 * @returns {{ detection: Detection | null, earlobe: { x: number, y: number, conf: number } | null }}
 */
export function decodePoseOutput(raw, shape, cfg) {
  let channels;
  let anchors;
  if (shape.length === 3) {
    channels = shape[1];
    anchors = shape[2];
  } else {
    return { detection: null, earlobe: null };
  }

  const numKpts = cfg.numKpts ?? Math.max(1, Math.floor((channels - 4 - (cfg.numClasses ?? 1)) / 3));
  const numClasses = cfg.numClasses ?? channels - 4 - numKpts * 3;
  if (numClasses < 1) return { detection: null, earlobe: null };

  const output = [];
  for (let c = 0; c < channels; c++) {
    const row = new Float32Array(anchors);
    for (let a = 0; a < anchors; a++) {
      row[a] = raw[c * anchors + a];
    }
    output.push(Array.from(row));
  }

  let candidates = decodeCandidates(output, numClasses, numKpts, cfg.boxConfThreshold);
  candidates = nms(candidates, cfg.nmsIou ?? 0.45);
  const det = selectDetection(candidates, cfg.selectionRule);
  if (!det) return { detection: null, earlobe: null };

  const kpt = det.kpts[cfg.earlobeKptIndex];
  if (!kpt || kpt.conf < cfg.kptConfThreshold) {
    return { detection: det, earlobe: null };
  }
  return { detection: det, earlobe: { x: kpt.x, y: kpt.y, conf: kpt.conf } };
}

function bestForClass(detections, classId) {
  const subset = detections.filter((d) => d.classId === classId);
  if (!subset.length) return null;
  return subset.reduce((a, b) => (a.score >= b.score ? a : b));
}

/**
 * Decode left and right earlobe detections (class 0 / class 1).
 * @returns {{ left: { x, y, conf, detection } | null, right: { x, y, conf, detection } | null }}
 */
export function decodeBothEars(raw, shape, cfg) {
  let channels;
  let anchors;
  if (shape.length !== 3) return { left: null, right: null };
  channels = shape[1];
  anchors = shape[2];

  const numClasses = cfg.numClasses ?? 2;
  const numKpts = cfg.numKpts ?? 1;
  const output = [];
  for (let c = 0; c < channels; c++) {
    const row = new Float32Array(anchors);
    for (let a = 0; a < anchors; a++) row[a] = raw[c * anchors + a];
    output.push(Array.from(row));
  }

  let candidates = decodeCandidates(output, numClasses, numKpts, cfg.boxConfThreshold);
  const leftId = cfg.leftClassId ?? 0;
  const rightId = cfg.rightClassId ?? 1;
  const leftNms = nms(candidates.filter((d) => d.classId === leftId), cfg.nmsIou ?? 0.45);
  const rightNms = nms(candidates.filter((d) => d.classId === rightId), cfg.nmsIou ?? 0.45);
  const leftDet = bestForClass(leftNms, leftId);
  const rightDet = bestForClass(rightNms, rightId);

  function toEar(det) {
    if (!det) return null;
    const kpt = det.kpts[cfg.earlobeKptIndex ?? 0];
    if (!kpt || kpt.conf < cfg.kptConfThreshold) return null;
    return { x: kpt.x, y: kpt.y, conf: kpt.conf, detection: det };
  }

  return { left: toEar(leftDet), right: toEar(rightDet) };
}

import trackingCfg from '../config/tracking.json';
import { Camera, syncCanvases } from './camera.js';
import { decodePoseOutput } from './decoder.js';
import { drawDebug, isDebugEnabled } from './debug-overlay.js';
import { InferenceLoop } from './inference-loop.js';
import { letterboxToTensor, mapToScreen, mapToSource } from './letterbox.js';
import { createSession, runInference } from './onnx-engine.js';
import { PlacementState } from './placement.js';
import { EarringScene } from './three-scene.js';

const MODEL_URL = '/models/best.onnx';
const EARRING_URL = '/assets/earring.glb';

const video = document.getElementById('webcam');
const threeCanvas = document.getElementById('three-canvas');
const debugCanvas = document.getElementById('debug-canvas');
const stage = document.getElementById('stage');
const statusEl = document.getElementById('status');

const offscreen = document.createElement('canvas');
const tensorBuf = new Float32Array(1 * 3 * 640 * 640);
const debug = isDebugEnabled();
const placement = new PlacementState(trackingCfg);

let letterboxMeta = null;
let lastDetection = null;
let lastEarlobe640 = null;
let session = null;
let inputName = 'images';
let fpsLast = performance.now();
let fpsFrames = 0;

function mapEarlobeToScreen(x640, y640) {
  if (!letterboxMeta) return { x: 0, y: 0, conf: 0 };
  const src = mapToSource(x640, y640, letterboxMeta);
  const screen = mapToScreen(src.x, src.y, letterboxMeta);
  return screen;
}

function mapBoxToScreen(det) {
  if (!det || !letterboxMeta) return null;
  const tl = mapEarlobeToScreen(det.x1, det.y1);
  const br = mapEarlobeToScreen(det.x2, det.y2);
  return { x1: tl.x, y1: tl.y, x2: br.x, y2: br.y };
}

async function initModel() {
  statusEl.textContent = 'Loading ONNX model…';
  try {
    const s = await createSession(MODEL_URL);
    session = s.session;
    inputName = s.inputName;
    statusEl.textContent = 'Model loaded. Starting camera…';
    statusEl.classList.add('ok');
  } catch (e) {
    statusEl.textContent =
      'Model load failed. Train and copy best.onnx to public/models/. ' + (e?.message ?? e);
    console.error(e);
    throw e;
  }
}

async function runLoop() {
  const earringScene = new EarringScene(threeCanvas, EARRING_URL);
  await earringScene.load();

  const camera = new Camera(video, statusEl, (size) => {
    syncCanvases([threeCanvas, debugCanvas], size, stage);
    earringScene.resize(size.width, size.height);
  });

  await camera.start();

  const loop = new InferenceLoop({
    intervalMs: trackingCfg.inferenceIntervalMs,
    onFrame: () => {
      fpsFrames++;
      const now = performance.now();
      if (now - fpsLast >= 1000) {
        if (debug) console.info(`FPS ~${fpsFrames}`);
        fpsFrames = 0;
        fpsLast = now;
      }

      let screenEar = null;
      let screenBox = null;
      if (lastEarlobe640 && lastDetection) {
        screenEar = {
          ...mapEarlobeToScreen(lastEarlobe640.x, lastEarlobe640.y),
          conf: lastEarlobe640.conf,
        };
        screenBox = mapBoxToScreen(lastDetection);
      }

      const place = placement.update(screenEar, screenBox, letterboxMeta ?? { srcW: 1, srcH: 1, gain: 1 });
      earringScene.render(place);

      if (debug) {
        const ctx = debugCanvas.getContext('2d');
        drawDebug(ctx, screenEar, screenBox, letterboxMeta ?? { srcW: 1, srcH: 1 }, (x, y) =>
          mapEarlobeToScreen(x, y)
        );
      }
    },
    onInfer: async () => {
      if (!session || video.readyState < 2) return;
      const lb = letterboxToTensor(video, offscreen, tensorBuf);
      letterboxMeta = lb;
      const { data, dims } = await runInference(session, inputName, lb.tensor);
      const { detection, earlobe } = decodePoseOutput(data, dims, {
        boxConfThreshold: trackingCfg.boxConfThreshold,
        kptConfThreshold: trackingCfg.kptConfThreshold,
        earlobeKptIndex: trackingCfg.earlobeKptIndex,
        selectionRule: trackingCfg.selectionRule,
      });
      lastDetection = detection;
      lastEarlobe640 = earlobe;
    },
  });

  loop.start();
}

initModel()
  .then(runLoop)
  .catch(() => {
    /* status set in handlers */
  });

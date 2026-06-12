/**
 * Web Worker: side-ear ONNX inference (4 keypoints).
 */
import { decodeSideEar } from './decoder.js';
import { createSession, runInference } from './onnx-engine.js';

let session = null;
let inputName = '';
let cfg = {};
const tensorReuse = {};

function toPointMsg(pt) {
  if (!pt) return null;
  return { x: pt.x, y: pt.y, conf: pt.conf };
}

self.onmessage = async (ev) => {
  const msg = ev.data;
  try {
    if (msg.type === 'init') {
      cfg = msg.cfg || {};
      const created = await createSession(msg.modelUrl, {
        mobile: !!msg.mobile,
        useGpu: !!msg.useGpu,
      });
      session = created.session;
      inputName = created.inputName;
      self.postMessage({ type: 'ready' });
      return;
    }

    if (msg.type === 'infer') {
      if (!session) throw new Error('worker not initialized');
      const tensor = new Float32Array(msg.tensor);
      const { data, dims } = await runInference(session, inputName, tensor, tensorReuse);
      const points = decodeSideEar(data, dims, cfg);
      self.postMessage({
        type: 'result',
        id: msg.id,
        points: {
          earlobe: toPointMsg(points.earlobe),
          earPos1: toPointMsg(points.earPos1),
          earPos2: toPointMsg(points.earPos2),
          earPos3: toPointMsg(points.earPos3),
        },
      });
      return;
    }

    if (msg.type === 'dispose') {
      if (session) {
        try {
          await session.release();
        } catch (_) {}
        session = null;
      }
      self.close();
    }
  } catch (e) {
    self.postMessage({
      type: 'error',
      id: msg.id,
      message: e && e.message ? e.message : String(e),
    });
  }
};

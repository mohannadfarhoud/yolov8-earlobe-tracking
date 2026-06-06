/**
 * Web Worker: ONNX inference + decode off the main UI thread.
 */
import { decodeBothEars } from './decoder.js';
import { createSession, runInference } from './onnx-engine.js';

let session = null;
let inputName = '';
let cfg = {};
const tensorReuse = {};

function toEarMsg(ear) {
  if (!ear) return null;
  return { x: ear.x, y: ear.y, conf: ear.conf };
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
      if (!session) {
        throw new Error('worker not initialized');
      }
      const tensor = new Float32Array(msg.tensor);
      const { data, dims } = await runInference(session, inputName, tensor, tensorReuse);
      const ears = decodeBothEars(data, dims, cfg);
      self.postMessage({
        type: 'result',
        id: msg.id,
        ears: {
          left: toEarMsg(ears.left),
          right: toEarMsg(ears.right),
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

import * as ort from 'onnxruntime-web';

const WASM_CDN = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.0/dist/';

/**
 * Configure ORT WASM paths (CDN fallback per plan guardrails).
 */
export function configureOrt() {
  ort.env.wasm.wasmPaths = WASM_CDN;
}

/**
 * @param {string} modelUrl
 * @returns {Promise<{ session: ort.InferenceSession, inputName: string }>}
 */
export async function createSession(modelUrl) {
  configureOrt();
  const providers = ['webgpu', 'webgl', 'wasm'];
  let lastError;

  for (const ep of providers) {
    try {
      const session = await ort.InferenceSession.create(modelUrl, {
        executionProviders: [ep],
      });
      const inputName = session.inputNames[0];
      console.info(`ONNX session ready (EP: ${ep}), input: ${inputName}`);
      return { session, inputName };
    } catch (e) {
      lastError = e;
      console.warn(`EP ${ep} failed:`, e);
    }
  }

  throw lastError ?? new Error('Failed to create ONNX session');
}

/**
 * @param {ort.InferenceSession} session
 * @param {string} inputName
 * @param {Float32Array} tensor
 */
export async function runInference(session, inputName, tensor) {
  const feeds = {
    [inputName]: new ort.Tensor('float32', tensor, [1, 3, 640, 640]),
  };
  const result = await session.run(feeds);
  const outName = session.outputNames[0];
  const tensorOut = result[outName];
  return {
    data: tensorOut.data,
    dims: tensorOut.dims,
  };
}

import * as ort from 'onnxruntime-web';

const WASM_CDN = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.0/dist/';

/**
 * Configure ORT WASM paths. Mobile uses single-thread WASM (much less RAM).
 * @param {{ mobile?: boolean }} [opts]
 */
export function configureOrt(opts = {}) {
  ort.env.wasm.wasmPaths = WASM_CDN;
  ort.env.wasm.numThreads = 1;
  ort.env.wasm.proxy = false;
  if (opts.mobile) {
    ort.env.wasm.simd = true;
    ort.env.logLevel = 'warning';
  }
}

/**
 * @param {string} modelUrl
 * @param {{ useGpu?: boolean, mobile?: boolean }} [opts]
 * @returns {Promise<{ session: ort.InferenceSession, inputName: string }>}
 */
export async function createSession(modelUrl, opts = {}) {
  configureOrt(opts);
  const mobile = !!opts.mobile;
  const providers = !mobile && opts.useGpu ? ['webgpu', 'webgl', 'wasm'] : ['wasm'];
  let lastError;

  for (const ep of providers) {
    try {
      const session = await ort.InferenceSession.create(modelUrl, {
        executionProviders: [ep],
        graphOptimizationLevel: 'all',
      });
      const inputName = session.inputNames[0];
      console.info(`ONNX session ready (EP: ${ep}, mobile: ${mobile}), input: ${inputName}`);
      return { session, inputName };
    } catch (e) {
      lastError = e;
      console.warn(`EP ${ep} failed:`, e);
    }
  }

  throw lastError ?? new Error('Failed to create ONNX session');
}

function disposeTensor(t) {
  if (t && typeof t.dispose === 'function') {
    try {
      t.dispose();
    } catch (_) {}
  }
}

/**
 * @param {ort.InferenceSession} session
 * @param {string} inputName
 * @param {Float32Array} tensor
 * @param {{ inputTensor?: ort.Tensor }} reuse
 */
export async function runInference(session, inputName, tensor, reuse = {}) {
  if (!reuse.inputTensor) {
    reuse.inputTensor = new ort.Tensor('float32', tensor, [1, 3, 640, 640]);
  }

  const feeds = { [inputName]: reuse.inputTensor };
  const result = await session.run(feeds);
  const outName = session.outputNames[0];
  const tensorOut = result[outName];
  const data =
    tensorOut.data instanceof Float32Array
      ? tensorOut.data
      : new Float32Array(tensorOut.data);
  const dims = tensorOut.dims.slice();

  for (const t of Object.values(result)) disposeTensor(t);

  return { data, dims };
}

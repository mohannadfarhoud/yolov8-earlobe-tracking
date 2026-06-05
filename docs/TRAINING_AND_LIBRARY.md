# Training + earlobe library (no try-on)

## Overview

1. **Train once** (Python, RTX GPU) → `public/models/best.onnx`
2. **Use in your webpage** (JavaScript) → `createEarlobeTracker()` → `{ x, y, confidence }`

---

## Training web UI (easiest)

```powershell
pip install -r requirements.txt
python python/training_server.py
```

Open **http://127.0.0.1:8000** and use the tabs:

1. **Annotate** — Setup folder, upload images, click earlobe on each face, Save & next  
2. **Train** — Prepare for training → Start training  
3. **Export** — Export web library (ZIP + folder)  
4. **GPU** — Optional CUDA check  

---

## Step 1 — Create dataset folders

On your training PC:

```powershell
$DATA = "C:\earlobe-tracking\earlobe"
mkdir "$DATA\images\train", "$DATA\images\val", "$DATA\labels\train", "$DATA\labels\val"
```

Add annotated pairs (Roboflow / CVAT → YOLO Pose export):

- `images\train\photo001.jpg` + `labels\train\photo001.txt`
- Same for `val\` (at least a few images)

---

## Step 2 — data.yaml

In the repo (`yolov8-earlobe-tracking\data.yaml`):

```yaml
path: C:/earlobe-tracking/earlobe
train: images/train
val: images/val
nc: 1
names:
  0: earlobe
kpt_shape: [1, 3]
```

Validate:

```powershell
cd C:\earlobe-tracking\yolov8-earlobe-tracking
.\.venv\Scripts\Activate.ps1
python python/validate_dataset.py --data data.yaml
```

Must print **OK** with image/label counts > 0.

---

## Step 3 — Train and export

```powershell
python python/check_env.py
python python/train.py --data data.yaml
python python/verify_onnx.py
```

Output: `public\models\best.onnx`

---

## Step 4 — Use the library in your page

### Install dependency

```bash
npm install onnxruntime-web
```

### Copy these files into your project

- `src/earlobe-tracker.js`
- `src/onnx-engine.js`
- `src/letterbox.js`
- `src/decoder.js`
- `config/tracking.json`

Or import from this repo if you use Vite/npm link.

### API (safe — won't crash the tab)

```javascript
import { createEarlobeTracker } from './earlobe-tracker.js';

const tracker = await createEarlobeTracker({
  modelUrl: '/models/best.onnx',
  mirrorX: true,
  useGpu: false, // default; WebGPU can crash some GPUs
});

// Use startLoop — one inference at a time, throttled (~30ms)
tracker.startLoop(yourVideoElement, ({ left, right }) => {
  if (left) { /* left.x, left.y, left.confidence */ }
  if (right) { /* right.x, right.y, right.confidence */ }
});

// When leaving the page:
tracker.dispose();
```

**Do not** call `detect()` inside `requestAnimationFrame` without awaiting — overlapping ONNX runs can freeze or crash the browser.

### Mobile phones

Phones have much less RAM than a PC. The library auto-detects mobile and:

- Limits camera to ~480p and 15 FPS
- Runs inference ~6–7×/sec; display is smoothed at 60 FPS (no extra ONNX cost)
- Uses WASM only (no WebGPU)
- Pauses when the tab is hidden

```javascript
import { createEarlobeTracker, getMobileCameraConstraints } from './earlobe-tracker.js';

const stream = await navigator.mediaDevices.getUserMedia(getMobileCameraConstraints());
video.srcObject = stream;
await video.play();

const tracker = await createEarlobeTracker({ modelUrl: './models/best.onnx' });
tracker.startLoop(video, onResult); // mobile-safe by default
```

Serve your page over **HTTPS** (required for camera on many phones). Close other tabs before loading — the ONNX model + WASM runtime need ~50–80 MB.

### Standalone export (no npm)

After **Export** in the training UI, open `export/web-library/README.txt`. Serve the folder with any static server and open `example.html` (includes ONNX Runtime CDN import map).

### Test demo in this repo

```powershell
npm run dev
```

Open `http://localhost:5173/examples/earlobe-only.html`

---

## Minimum dataset size

| Goal | Train images |
|------|----------------|
| First test | 50–100 |
| Usable | 300–800+ |

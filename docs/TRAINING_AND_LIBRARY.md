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

## Step 4 — Export library for your UI app

Your UI application is **separate**. This repo only ships an abstract detection library.

```powershell
python python/export_library.py --model public\models\best.onnx
```

Copy `export/web-library/` into your project. See **[LIBRARY.md](LIBRARY.md)** for the full API contract.

### Install dependency (in your UI app)

```bash
npm install onnxruntime-web
```

### API (your UI app — no MediaPipe required)

```javascript
import { createEarlobeTracker, getMobileCameraConstraints } from './earlobe-tracker.js';

const tracker = await createEarlobeTracker({
  modelUrl: '/models/best.onnx',
  mirrorX: true,
  useWorker: true,       // default — ONNX off UI thread
  minIntervalMs: 50,     // scan every 50ms (lower = faster; try 33 on PC, 50–66 on phone)
});

tracker.startLoop(video, ({ left, right }) => {
  // runs every animation frame (smooth); ONNX scans at minIntervalMs
  if (left) { /* left.x, left.y, left.confidence */ }
});

tracker.dispose();
```

`minIntervalMs` controls how often ONNX runs. Display is still smooth at 60 FPS.

### Web Worker (default)

`useWorker: true` loads `earlobe-worker.js` so ONNX does not block your UI.

**Do not** call `detect()` in a tight loop — use `startLoop()`.

MediaPipe and other face logic belong in **your UI** — not in this library. Optional `startMotionDriven()` exists if you pass landmarks from your own code; see `examples/mediapipe-trigger.html` in the repo only.

### Mobile phones

Phones have much less RAM than a PC. The library auto-detects mobile and:

- Limits camera to ~480p and 15 FPS
- Runs inference ~10×/sec; display follows at 60 FPS with 3× snappier smoothing (no extra ONNX cost)
- Uses WASM only (no WebGPU)
- Pauses when the tab is hidden

```javascript
import { createEarlobeTracker, getMobileCameraConstraints } from './earlobe-tracker.js';

const stream = await navigator.mediaDevices.getUserMedia(getMobileCameraConstraints());
video.srcObject = stream;
await video.play();

const tracker = await createEarlobeTracker({ modelUrl: './models/best.onnx', minIntervalMs: 50 });
tracker.startLoop(video, onResult);
```

Serve your page over **HTTPS** (required for camera on many phones). Close other tabs before loading — the ONNX model + WASM runtime need ~50–80 MB.

### Test export

```powershell
cd export/web-library
python -m http.server 8080
```

Open `http://localhost:8080/example.html`

---

## Minimum dataset size

| Goal | Train images |
|------|----------------|
| First test | 50–100 |
| Usable | 300–800+ |

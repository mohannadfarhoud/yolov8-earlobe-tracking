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

### API — MediaPipe motion trigger (recommended)

ONNX runs only when MediaPipe detects face movement. Display stays smooth at 60 FPS.

```javascript
import { createEarlobeTracker } from './earlobe-tracker.js';

const tracker = await createEarlobeTracker({
  modelUrl: '/models/best.onnx',
  mirrorX: true,
  minIntervalMs: 50, // safety cooldown between ONNX runs
});

const motionCtrl = tracker.startMotionDriven(video, ({ left, right }) => {
  // smooth feedback every frame — update your UI here
});

// Inside your existing MediaPipe Face Landmarker callback:
function onMediaPipeResults(results) {
  if (results.faceLandmarks?.[0]) {
    motionCtrl.onLandmarks(results.faceLandmarks[0]);
  } else {
    motionCtrl.motionTrigger.reset();
  }
}

// Or trigger manually when your app decides:
motionCtrl.requestDetect();

motionCtrl.stop();
tracker.dispose();
```

Demo: `examples/mediapipe-trigger.html` (repo) or `example-mediapipe.html` (export).

### API — interval fallback

```javascript
tracker.startLoop(video, ({ left, right }) => { ... });
```

**Do not** call `detect()` inside a tight loop without throttling — overlapping ONNX runs can crash the browser.

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

const tracker = await createEarlobeTracker({ modelUrl: './models/best.onnx' });
const motionCtrl = tracker.startMotionDriven(video, onResult);
// wire motionCtrl.onLandmarks() to your MediaPipe callback
```

Serve your page over **HTTPS** (required for camera on many phones). Close other tabs before loading — the ONNX model + WASM runtime need ~50–80 MB.

### Standalone export (no npm)

After **Export** in the training UI, open `export/web-library/README.txt`. Serve the folder with any static server and open `example.html` (includes ONNX Runtime CDN import map).

### Test demo in this repo

```powershell
npm run dev
```

Open `http://localhost:5173/examples/mediapipe-trigger.html` (motion trigger) or `earlobe-only.html` (interval)

---

## Minimum dataset size

| Goal | Train images |
|------|----------------|
| First test | 50–100 |
| Usable | 300–800+ |

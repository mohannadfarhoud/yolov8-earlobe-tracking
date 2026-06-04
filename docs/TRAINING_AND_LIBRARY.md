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

### API

```javascript
import { createEarlobeTracker } from './earlobe-tracker.js';

const tracker = await createEarlobeTracker({
  modelUrl: '/models/best.onnx',
  mirrorX: true, // set false if video is not mirrored
});

// On each frame (throttle ~30ms):
const point = await tracker.detect(yourVideoElement);
if (point) {
  // point.x, point.y — use in your UI (CSS, canvas, etc.)
  // point.confidence — gate visibility
  // point.xRaw, point.yRaw — non-mirrored video pixels
}
```

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

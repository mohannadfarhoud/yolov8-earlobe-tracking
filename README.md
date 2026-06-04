# Earlobe tracking (train + browser library)

Train a YOLOv8-pose model on your earlobe dataset, export `best.onnx`, then use **`createEarlobeTracker()`** in any webpage to get `{ x, y, confidence }`. All inference runs in the browser.

**Full guide:** [docs/TRAINING_AND_LIBRARY.md](docs/TRAINING_AND_LIBRARY.md)

**Library entry:** `src/earlobe-tracker.js`  
**Demo (no try-on):** `examples/earlobe-only.html` after `npm run dev`

---

## Quick start

### 1. Dataset + train (RTX PC)

**Training UI (recommended):**

```powershell
pip install gradio
python python/training_ui.py
```

Open **http://127.0.0.1:7860** — create folders, save `data.yaml`, validate, train, verify ONNX.

**Or command line:**

```powershell
python python/setup_dataset_dirs.py --root C:\earlobe-tracking\earlobe
# Add images + labels, copy data.template.yaml → data.yaml
python python/validate_dataset.py --data data.yaml
python python/train.py --data data.yaml
```

### 2. Library in your page

```javascript
import { createEarlobeTracker } from './src/earlobe-tracker.js';
const tracker = await createEarlobeTracker({ modelUrl: '/models/best.onnx' });
const point = await tracker.detect(videoElement);
```

---

## Optional: 3D try-on demo

The repo also includes a Three.js earring overlay (`npm run dev` → main `index.html`). You can ignore that if you only need coordinates.

## Prerequisites

- Python 3.10+ with CUDA PyTorch (RTX GPU recommended)
- Node.js 18+
- Annotated YOLO pose dataset (`data.yaml`) — see [docs/DATASET_CONTRACT.md](docs/DATASET_CONTRACT.md)

## Setup

```bash
pip install -r requirements.txt
npm install
```

Copy `data.example.yaml` to `data.yaml` and set `path`, `train`, `val` to your dataset.

## Phase 0.5 — Validate dataset

Expected layout:

```text
project/
  data.yaml
  images/train/*.jpg   labels/train/*.txt
  images/val/*.jpg     labels/val/*.txt
```

```bash
python python/validate_dataset.py --data data.yaml
```

If images are only in `images/` and labels in `labels/` (no train/val yet):

```powershell
mkdir images\train, images\val, labels\train, labels\val
# Or auto-split ~85% train / 15% val from flat folders:
python python/split_train_val.py --root .
```

## Phase 1 — Train & export

```bash
python python/check_env.py
python python/train.py --data data.yaml
python python/verify_onnx.py
```

Exports `public/models/best.onnx` for the web app. Re-run `verify_onnx.py` after each export to refresh [docs/MODEL_IO.md](docs/MODEL_IO.md).

Export only (existing weights):

```bash
python python/train.py --skip-train --weights runs/pose/earlobe/weights/best.pt
```

## Phase 2–4 — Dev server

Place optional `public/assets/earring.glb` (hook near origin; adjust pivot in `src/three-scene.js`).

```bash
npm run dev
```

Open `http://localhost:5173` (secure context required for camera). Debug overlay: `http://localhost:5173?debug=1`.

## Project layout

| Path | Purpose |
|------|---------|
| `python/` | Training, validation, ONNX verify |
| `config/tracking.json` | Thresholds, kpt index, selection rule |
| `src/` | Vite frontend (camera, ORT, decoder, Three.js) |
| `public/models/best.onnx` | Exported model (gitignored; copy after train) |
| `public/assets/earring.glb` | 3D earring asset |
| [plan.txt](plan.txt) | Engineering plan |
| [docs/QA_CHECKLIST.md](docs/QA_CHECKLIST.md) | End-to-end tests |

## ONNX vs WASM

The **model** is ONNX (`best.onnx`). **onnxruntime-web** uses WASM binaries as its runtime (CDN fallback in `src/onnx-engine.js`), not as the model format.

## License

Provide your own dataset and earring model license as applicable.

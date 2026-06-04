# Real-Time Earlobe Tracking & 3D Earring Try-On

Local webcam app: custom YOLOv8-pose ONNX finds the earlobe; Three.js renders an earring overlay. **All inference runs in the browser** — no uploads.

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

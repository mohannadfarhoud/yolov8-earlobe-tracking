# Earlobe tracking

Two parts:

1. **Train** a YOLOv8-pose model → `best.onnx` (Python + web UI in this repo)
2. **Use** the exported browser library in **your own UI app** → earlobe `{ x, y, confidence }`

The library is abstract: no MediaPipe, no try-on UI, no overlays — only detection.

**Library API:** [docs/LIBRARY.md](docs/LIBRARY.md)  
**Training guide:** [docs/TRAINING_AND_LIBRARY.md](docs/TRAINING_AND_LIBRARY.md)

---

## Library (for your UI app)

```powershell
python python/export_library.py --model public\models\best.onnx
```

Copy `export/web-library/` into your project.

```javascript
import { createEarlobeTracker } from './web-library/src/earlobe-tracker.js';

const tracker = await createEarlobeTracker({
  modelUrl: './web-library/models/best.onnx',
  minIntervalMs: 50,
});

tracker.startLoop(video, ({ left, right }) => {
  // your UI updates here
});
```

---

## Train (this repo)

```powershell
pip install -r requirements.txt
python python/training_server.py
```

Open **http://127.0.0.1:8000** — annotate, train, export library.

Or CLI:

```powershell
python python/train.py --data data.yaml
python python/export_library.py --model public\models\best.onnx
```

---

## Optional demos in this repo

| Path | Purpose |
|------|---------|
| `export/web-library/example.html` | Library integration reference |
| `examples/earlobe-only.html` | Dev demo (`npm run dev`) |
| `index.html` | Three.js try-on demo (ignore if you only need coordinates) |

---

## Layout

| Path | Purpose |
|------|---------|
| `src/earlobe-tracker.js` | **Library entry** |
| `python/` | Training, annotation, export |
| `training-web/` | Training server UI |
| `export/web-library/` | Packaged library for your app |

## Prerequisites

- Python 3.10+ with CUDA (training)
- Node.js 18+ (optional dev server)
- Annotated dataset — [docs/DATASET_CONTRACT.md](docs/DATASET_CONTRACT.md)

## License

Provide your own dataset and model license as applicable.

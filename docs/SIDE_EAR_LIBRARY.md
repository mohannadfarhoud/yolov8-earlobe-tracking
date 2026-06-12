# Side-ear library (4 keypoints)

Separate from front-face earlobe training. For **profile/side ear** images.

## Keypoints

| Index | Name | Use |
|-------|------|-----|
| 0 | `earlobe` | Earlobe position |
| 1 | `earPos1` | Earring anchor 1 |
| 2 | `earPos2` | Earring anchor 2 |
| 3 | `earPos3` | Earring anchor 3 |

## Train

1. Start server: `python python/training_server.py`
2. Open **http://127.0.0.1:8000/side-ear/**
3. Use a **separate dataset folder** (e.g. `C:\earlobe-tracking\side-ear`)
4. Annotate 4 points per image → Train → Export

CLI:

```powershell
python python/train.py --data data.side-ear.yaml --name side-ear --onnx-out public/models/side-ear.onnx
python python/export_side_ear_library.py --model public/models/side-ear.onnx
```

## Export output

- Folder: `export/side-ear-library/`
- Zip: `export/side-ear-web-library.zip`
- Model: `models/side-ear.onnx`

## UI integration

```javascript
import { createSideEarTracker, getVideoFps } from './side-ear-library/src/side-ear-tracker.js';

const fps = getVideoFps(video) ?? 15;
const tracker = await createSideEarTracker({
  modelUrl: './side-ear-library/models/side-ear.onnx',
  fps,
});

tracker.startLoop(video, ({ earlobe, earPos1, earPos2, earPos3 }) => {
  // your earring UI
});
```

## vs front-face library

| | Front-face | Side-ear |
|--|------------|----------|
| URL | `/` | `/side-ear/` |
| Dataset | `data.yaml` | `data.side-ear.yaml` |
| Keypoints | 1 (earlobe) | 4 |
| Classes | 2 (L/R) | 1 |
| Export | `export/web-library/` | `export/side-ear-library/` |
| Tracker | `createEarlobeTracker` | `createSideEarTracker` |

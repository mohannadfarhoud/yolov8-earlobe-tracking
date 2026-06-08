# Earlobe browser library — API contract

Abstract library: **FPS-based scan cycles**. Returns earlobe coordinates only. No MediaPipe, no UI overlays.

## Export

```powershell
python python/export_library.py --model public\models\best.onnx
```

Copy `export/web-library/` into your UI app.

## Integration (your UI app)

```javascript
import { createEarlobeTracker, getVideoFps, getMobileCameraConstraints } from './web-library/src/earlobe-tracker.js';

// 1. Start camera (your UI)
const stream = await navigator.mediaDevices.getUserMedia(getMobileCameraConstraints());
video.srcObject = stream;
await video.play();

// 2. Read FPS from camera (your UI passes it to the library)
const fps = getVideoFps(video) ?? 15;

// 3. Init with FPS — scan interval = 1000 / fps
const tracker = await createEarlobeTracker({
  modelUrl: './web-library/models/best.onnx',
  mirrorX: true,
  fps,                    // required: from camera or fixed (e.g. 15 phone, 30 desktop)
  useWorker: true,
});

// 4. Start — one clean scan per FPS cycle
tracker.startLoop(video, ({ left, right }) => {
  // your UI: update earring position
});

// 5. Cleanup
tracker.stopLoop();
await tracker.dispose();
```

## Init options

| Option | Description |
|--------|-------------|
| `modelUrl` | Path to `best.onnx` |
| `fps` | **Scan rate.** Interval = `1000 / fps` ms per cycle |
| `minIntervalMs` | Alternative to `fps` (use one, not both) |
| `mirrorX` | `true` for selfie camera (default) |
| `useWorker` | ONNX off UI thread (default `true`) |

## Scan behaviour

- **`requestVideoFrameCallback`** — aligned to real video frames (clean cycles)
- **One scan at a time** — skips frame if previous scan still running
- **`syncFpsFromVideo`** — on `startLoop()`, reads camera FPS from video track automatically

## Result

```javascript
{ left: { x, y, confidence, xRaw, yRaw } | null, right: { ... } | null }
```

## UI app changes (migration)

### Remove
- MediaPipe `onLandmarks()` wiring to earlobe library
- `startMotionDriven()` — removed from library
- `minIntervalMs` tuning without `fps` (use `fps` instead)
- Manual `detect()` loops

### Add
- Pass `fps` to `createEarlobeTracker()` from `getVideoFps(video)` after camera starts
- Use only `tracker.startLoop(video, callback)`

### Keep
- Your MediaPipe (if any) for **other** UI features — not connected to earlobe library
- Earring rendering in your callback from `{ left, right }`

## Training

See [TRAINING_AND_LIBRARY.md](TRAINING_AND_LIBRARY.md).

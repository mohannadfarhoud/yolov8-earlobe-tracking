# Model I/O

Run after export to regenerate from the real ONNX graph:

```bash
python python/verify_onnx.py --model public/models/best.onnx
```

## Expected (YOLOv8-pose @ 640)

| Tensor | Typical name | Shape | Notes |
|--------|--------------|-------|-------|
| Input | `images` | `[1, 3, 640, 640]` | float32, RGB, NCHW, values 0–1 |
| Output | `output0` | `[1, 56, 8400]` or variant | 4 box + 1 class + 51 kpt dims for COCO; custom nc/kpt may differ |

Confirm names and shapes with `verify_onnx.py` — do not hard-code output layout until export exists.

## Frontend

- Load: `public/models/best.onnx` via `onnxruntime-web`
- Preprocess: `src/letterbox.js`
- Decode: `src/decoder.js` (NMS + earlobe keypoint index 0)

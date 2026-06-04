# Dataset Contract

Training, validation, and the browser decoder must share this schema.

## data.yaml requirements

- `path`: **dataset root** — the folder that contains both `images/` and `labels/` (not the `images/` folder alone)
- `train` / `val`: paths **under images/**, e.g. `images/train` and `images/val`

Example layout:

```text
yolov8-earlobe-tracking/
  data.yaml
  images/train/*.jpg
  images/val/*.jpg
  labels/train/*.txt
  labels/val/*.txt
```

Example `data.yaml`:

```yaml
path: C:/earlobe-tracking/yolov8-earlobe-tracking
train: images/train
val: images/val
```
- `nc`: number of classes (typically `1` for earlobe)
- `names`: class index to name map (e.g. `0: earlobe`)
- `kpt_shape`: `[num_keypoints, 3]` — e.g. `[1, 3]` for one earlobe (x, y, visibility)

## Label format (YOLO pose)

Each line: `class cx cy w h kpt1_x kpt1_y kpt1_v ...` (normalized 0–1).

- **Earlobe keypoint index:** `0` (see `config/tracking.json`)
- Visibility: `0` = not labeled, `1` = occluded, `2` = visible

## Detection selection (multi-box scenes)

When multiple detections pass NMS, pick the winner using `config/tracking.json` → `selectionRule`:

| Rule | Behavior |
|------|----------|
| `highest_conf` | Largest box confidence (default) |
| `largest_box` | Largest bounding-box area |
| `center_nearest` | Closest box center to frame center |

## Left vs right ear

Use a single class `earlobe` with one keypoint per instance. If both ears appear, `selectionRule` chooses one detection; tune rule or crop framing for try-on UX.

## Validation

```bash
python python/validate_dataset.py --data path/to/data.yaml
```

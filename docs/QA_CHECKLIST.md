# QA Checklist (Phase 5)

## Python pipeline

- [ ] `python python/check_env.py` — CUDA available (or accept CPU warning)
- [ ] `python python/validate_dataset.py --data data.yaml` — no label format errors
- [ ] `python python/train.py --data data.yaml` — val metrics improve; `runs/pose/earlobe/weights/best.pt` exists
- [ ] `public/models/best.onnx` exists after train/export
- [ ] `python python/verify_onnx.py` — smoke test passes; `docs/MODEL_IO.md` updated

## Browser (Chrome or Edge first)

- [ ] `npm install` && `npm run dev` — app loads on `http://localhost:5173`
- [ ] Camera permission granted; status shows "Camera active"
- [ ] With model present: no load error; inference runs (check console)
- [ ] `?debug=1` — green box + magenta keypoint track face movement
- [ ] Earring visible above confidence threshold; hides below hysteresis off
- [ ] FPS ~30+ with default `inferenceIntervalMs` (33ms); log in console when debug on

## Regression (optional)

- [ ] 5–10 val images inferenced in Python; keypoints within expected pixel bounds

## Privacy

- [ ] Confirm no frames or model data leave the device (all local inference)

"""Helpers for web UI image upload and YOLO pose earlobe labels (left + right classes)."""

from __future__ import annotations

import random
import shutil
from pathlib import Path
from typing import Optional

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
DEFAULT_BOX_W = 0.18
DEFAULT_BOX_H = 0.22

CLASS_LEFT = 0
CLASS_RIGHT = 1


def ensure_dataset_layout(root: Path) -> None:
    for split in ("train", "val"):
        (root / "images" / split).mkdir(parents=True, exist_ok=True)
        (root / "labels" / split).mkdir(parents=True, exist_ok=True)


def dataset_yaml_config(root: Path) -> dict:
    return {
        "path": str(root).replace("\\", "/"),
        "train": "images/train",
        "val": "images/val",
        "nc": 2,
        "names": {0: "left_earlobe", 1: "right_earlobe"},
        "kpt_shape": [1, 3],
    }


def _format_line(class_id: int, kpt_x: float, kpt_y: float, box_w: float, box_h: float) -> str:
    cx = max(0.0, min(1.0, kpt_x))
    cy = max(0.0, min(1.0, kpt_y))
    w = max(0.02, min(1.0, box_w))
    h = max(0.02, min(1.0, box_h))
    return f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f} {cx:.6f} {cy:.6f} 2"


def list_images(root: Path) -> list[dict]:
    ensure_dataset_layout(root)
    items: list[dict] = []
    for split in ("train", "val"):
        img_dir = root / "images" / split
        if not img_dir.is_dir():
            continue
        for img in sorted(img_dir.iterdir()):
            if img.suffix.lower() not in IMAGE_EXTS:
                continue
            label = root / "labels" / split / f"{img.stem}.txt"
            ears = read_ear_labels(root, split, img.stem)
            has_label = ears["left"] is not None or ears["right"] is not None
            items.append(
                {
                    "split": split,
                    "filename": img.name,
                    "stem": img.stem,
                    "annotated": has_label,
                    "has_left": ears["left"] is not None,
                    "has_right": ears["right"] is not None,
                    "path": str(img),
                }
            )
    return items


def read_ear_labels(root: Path, split: str, stem: str) -> dict:
    """Return normalized keypoints for left (class 0) and right (class 1)."""
    label_path = root / "labels" / split / f"{stem}.txt"
    result = {"left": None, "right": None}
    if not label_path.is_file():
        return result
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if len(parts) < 8:
            continue
        cls = int(float(parts[0]))
        kx, ky = float(parts[5]), float(parts[6])
        entry = {"kx": kx, "ky": ky}
        if cls == CLASS_LEFT:
            result["left"] = entry
        elif cls == CLASS_RIGHT:
            result["right"] = entry
    return result


def write_ear_labels(
    root: Path,
    split: str,
    stem: str,
    left: Optional[dict],
    right: Optional[dict],
    image_width: int,
    image_height: int,
) -> Path:
    """
    left/right: {x, y} in pixels or None if not visible in this photo.
    """
    if image_width < 1 or image_height < 1:
        raise ValueError("invalid image dimensions")
    if left is None and right is None:
        raise ValueError("mark at least one earlobe (left or right)")

    lines: list[str] = []
    if left is not None:
        lines.append(
            _format_line(
                CLASS_LEFT,
                left["x"] / image_width,
                left["y"] / image_height,
                DEFAULT_BOX_W,
                DEFAULT_BOX_H,
            )
        )
    if right is not None:
        lines.append(
            _format_line(
                CLASS_RIGHT,
                right["x"] / image_width,
                right["y"] / image_height,
                DEFAULT_BOX_W,
                DEFAULT_BOX_H,
            )
        )

    label_path = root / "labels" / split / f"{stem}.txt"
    label_path.parent.mkdir(parents=True, exist_ok=True)
    label_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return label_path


def split_train_to_val(root: Path, val_ratio: float = 0.15, seed: int = 42) -> dict:
    train_img = root / "images" / "train"
    annotated = [
        p
        for p in train_img.iterdir()
        if p.suffix.lower() in IMAGE_EXTS
        and (root / "labels" / "train" / f"{p.stem}.txt").is_file()
    ]
    if not annotated:
        return {"moved": 0, "message": "No annotated train images to split"}
    random.seed(seed)
    random.shuffle(annotated)
    n_val = max(1, int(len(annotated) * val_ratio))
    moved = 0
    for img in annotated[:n_val]:
        lbl = root / "labels" / "train" / f"{img.stem}.txt"
        dest_img = root / "images" / "val" / img.name
        dest_lbl = root / "labels" / "val" / f"{img.stem}.txt"
        if dest_img.exists():
            continue
        shutil.move(str(img), str(dest_img))
        if lbl.is_file():
            shutil.move(str(lbl), str(dest_lbl))
        moved += 1
    return {"moved": moved, "message": f"Moved {moved} pairs to val"}


def export_web_library(root_project: Path, model_path: Optional[Path] = None) -> dict:
    onnx_src = model_path or (root_project / "public" / "models" / "best.onnx")
    has_model = onnx_src.is_file()

    export_dir = root_project / "export" / "web-library"
    if export_dir.exists():
        shutil.rmtree(export_dir)
    (export_dir / "models").mkdir(parents=True)
    (export_dir / "src").mkdir(parents=True)
    (export_dir / "config").mkdir(parents=True)
    if has_model:
        shutil.copy2(onnx_src, export_dir / "models" / "best.onnx")
    else:
        (export_dir / "models" / "COPY_best.onnx_HERE.txt").write_text(
            "Copy your trained model here as best.onnx\n"
            "Train in the web UI (tab 2–3) or run: python python/train.py --data data.yaml\n"
            "Output: public/models/best.onnx\n",
            encoding="utf-8",
        )

    copies = [
        ("src/earlobe-tracker.js", "src/earlobe-tracker.js"),
        ("src/onnx-engine.js", "src/onnx-engine.js"),
        ("src/letterbox.js", "src/letterbox.js"),
        ("src/decoder.js", "src/decoder.js"),
        ("src/tracking-defaults.js", "src/tracking-defaults.js"),
        ("src/device.js", "src/device.js"),
        ("src/smoothing.js", "src/smoothing.js"),
        ("src/motion-trigger.js", "src/motion-trigger.js"),
        ("src/earlobe-worker.js", "src/earlobe-worker.js"),
        ("src/worker-client.js", "src/worker-client.js"),
        ("config/tracking.json", "config/tracking.json"),
        ("examples/standalone-export.html", "example.html"),
    ]
    for src_rel, dest_rel in copies:
        src = root_project / src_rel
        if src.is_file():
            shutil.copy2(src, export_dir / dest_rel)

    readme = export_dir / "README.txt"
    readme.write_text(
        "Earlobe detection library (for your UI app)\n"
        "=========================================\n\n"
        "Abstract library — coordinates only. No MediaPipe, no overlays.\n"
        "Copy this folder into your project. Full API: docs/LIBRARY.md in repo.\n\n"
        "Integration (FPS-based — no MediaPipe):\n"
        "  import { createEarlobeTracker, getVideoFps } from './src/earlobe-tracker.js';\n"
        "  const fps = getVideoFps(video) ?? 15;\n"
        "  const tracker = await createEarlobeTracker({ modelUrl: './models/best.onnx', fps });\n"
        "  tracker.startLoop(video, ({ left, right }) => { /* your UI */ });\n\n"
        "Files required: all of src/ (including earlobe-worker.js)\n"
        "onnxruntime-web: npm install OR import map in example.html\n\n"
        "Test: python -m http.server 8080 → http://localhost:8080/example.html\n",
        encoding="utf-8",
    )
    return {
        "export_dir": str(export_dir),
        "has_model": has_model,
        "model_path": str(onnx_src) if has_model else None,
    }

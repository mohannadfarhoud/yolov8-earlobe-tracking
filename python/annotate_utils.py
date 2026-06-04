"""Helpers for web UI image upload and YOLO pose earlobe labels."""

from __future__ import annotations

import random
import shutil
from pathlib import Path

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
# Default normalized box size around earlobe click
DEFAULT_BOX_W = 0.18
DEFAULT_BOX_H = 0.22


def ensure_dataset_layout(root: Path) -> None:
    for split in ("train", "val"):
        (root / "images" / split).mkdir(parents=True, exist_ok=True)
        (root / "labels" / split).mkdir(parents=True, exist_ok=True)


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
            items.append(
                {
                    "split": split,
                    "filename": img.name,
                    "stem": img.stem,
                    "annotated": label.is_file() and label.stat().st_size > 0,
                    "path": str(img),
                }
            )
    return items


def write_label(
    root: Path,
    split: str,
    stem: str,
    kpt_x: float,
    kpt_y: float,
    box_w: float = DEFAULT_BOX_W,
    box_h: float = DEFAULT_BOX_H,
) -> Path:
    """Write YOLO pose line: class cx cy w h kx ky visibility."""
    cx = max(0.0, min(1.0, kpt_x))
    cy = max(0.0, min(1.0, kpt_y))
    w = max(0.02, min(1.0, box_w))
    h = max(0.02, min(1.0, box_h))
    line = f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f} {cx:.6f} {cy:.6f} 2\n"
    label_path = root / "labels" / split / f"{stem}.txt"
    label_path.parent.mkdir(parents=True, exist_ok=True)
    label_path.write_text(line, encoding="utf-8")
    return label_path


def read_label(root: Path, split: str, stem: str) -> dict | None:
    label_path = root / "labels" / split / f"{stem}.txt"
    if not label_path.is_file():
        return None
    parts = label_path.read_text(encoding="utf-8").strip().split()
    if len(parts) < 8:
        return None
    return {
        "kx": float(parts[5]),
        "ky": float(parts[6]),
        "cx": float(parts[1]),
        "cy": float(parts[2]),
    }


def split_train_to_val(root: Path, val_ratio: float = 0.15, seed: int = 42) -> dict:
    """Move a fraction of annotated train images (and labels) to val."""
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


def export_web_library(root_project: Path) -> dict:
    """Copy ONNX + JS library files into export/web-library/."""
    onnx_src = root_project / "public" / "models" / "best.onnx"
    if not onnx_src.is_file():
        raise FileNotFoundError("best.onnx not found — train first")

    export_dir = root_project / "export" / "web-library"
    if export_dir.exists():
        shutil.rmtree(export_dir)
    (export_dir / "models").mkdir(parents=True)
    (export_dir / "src").mkdir(parents=True)
    (export_dir / "config").mkdir(parents=True)
    shutil.copy2(onnx_src, export_dir / "models" / "best.onnx")

    copies = [
        ("src/earlobe-tracker.js", "src/earlobe-tracker.js"),
        ("src/onnx-engine.js", "src/onnx-engine.js"),
        ("src/letterbox.js", "src/letterbox.js"),
        ("src/decoder.js", "src/decoder.js"),
        ("config/tracking.json", "config/tracking.json"),
    ]
    for src_rel, dest_rel in copies:
        src = root_project / src_rel
        if src.is_file():
            shutil.copy2(src, export_dir / dest_rel)

    readme = export_dir / "README.txt"
    readme.write_text(
        "Earlobe web library export\n"
        "==========================\n\n"
        "Copy this folder into your website.\n\n"
        "  npm install onnxruntime-web\n\n"
        "Usage:\n"
        "  import { createEarlobeTracker } from './src/earlobe-tracker.js';\n"
        "  const tracker = await createEarlobeTracker({\n"
        "    modelUrl: './models/best.onnx',\n"
        "  });\n"
        "  const point = await tracker.detect(videoElement);\n",
        encoding="utf-8",
    )
    return {"export_dir": str(export_dir)}

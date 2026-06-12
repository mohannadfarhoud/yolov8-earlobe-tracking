"""Side-ear dataset: 4 keypoints per image (earlobe + 3 earring anchors)."""

from __future__ import annotations

import random
import shutil
from pathlib import Path
from typing import Optional

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
DEFAULT_BOX_W = 0.28
DEFAULT_BOX_H = 0.35
CLASS_SIDE_EAR = 0
SIDE_EAR_KPT_COUNT = 4
KPT_NAMES = ("earlobe", "ear_pos_1", "ear_pos_2", "ear_pos_3")


def ensure_dataset_layout(root: Path) -> None:
    for split in ("train", "val"):
        (root / "images" / split).mkdir(parents=True, exist_ok=True)
        (root / "labels" / split).mkdir(parents=True, exist_ok=True)


def dataset_yaml_config(root: Path) -> dict:
    return {
        "path": str(root).replace("\\", "/"),
        "train": "images/train",
        "val": "images/val",
        "nc": 1,
        "names": {0: "side_ear"},
        "kpt_shape": [SIDE_EAR_KPT_COUNT, 3],
    }


def _format_line(class_id: int, points: list[dict], box_w: float, box_h: float) -> str:
    if len(points) != SIDE_EAR_KPT_COUNT:
        raise ValueError(f"expected {SIDE_EAR_KPT_COUNT} points, got {len(points)}")
    xs = [max(0.0, min(1.0, p["x"])) for p in points]
    ys = [max(0.0, min(1.0, p["y"])) for p in points]
    cx = sum(xs) / len(xs)
    cy = sum(ys) / len(ys)
    w = max(0.02, min(1.0, box_w))
    h = max(0.02, min(1.0, box_h))
    parts = [str(class_id), f"{cx:.6f}", f"{cy:.6f}", f"{w:.6f}", f"{h:.6f}"]
    for x, y in zip(xs, ys):
        parts.extend([f"{x:.6f}", f"{y:.6f}", "2"])
    return " ".join(parts)


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
            pts = read_side_ear_labels(root, split, img.stem)
            items.append(
                {
                    "split": split,
                    "filename": img.name,
                    "stem": img.stem,
                    "annotated": pts is not None,
                    "path": str(img),
                }
            )
    return items


def read_side_ear_labels(root: Path, split: str, stem: str) -> Optional[list[dict]]:
    """Return 4 normalized keypoints or None."""
    label_path = root / "labels" / split / f"{stem}.txt"
    if not label_path.is_file():
        return None
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        expected = 1 + 4 + SIDE_EAR_KPT_COUNT * 3
        if len(parts) < expected:
            continue
        if int(float(parts[0])) != CLASS_SIDE_EAR:
            continue
        pts = []
        base = 5
        for i in range(SIDE_EAR_KPT_COUNT):
            off = base + i * 3
            pts.append({"kx": float(parts[off]), "ky": float(parts[off + 1])})
        return pts
    return None


def write_side_ear_labels(
    root: Path,
    split: str,
    stem: str,
    points: list[dict],
    image_width: int,
    image_height: int,
) -> Path:
    if image_width < 1 or image_height < 1:
        raise ValueError("invalid image dimensions")
    if len(points) != SIDE_EAR_KPT_COUNT:
        raise ValueError(f"mark all {SIDE_EAR_KPT_COUNT} points")

    norm = [{"x": p["x"] / image_width, "y": p["y"] / image_height} for p in points]
    line = _format_line(CLASS_SIDE_EAR, norm, DEFAULT_BOX_W, DEFAULT_BOX_H)
    label_path = root / "labels" / split / f"{stem}.txt"
    label_path.parent.mkdir(parents=True, exist_ok=True)
    label_path.write_text(line + "\n", encoding="utf-8")
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


def export_side_ear_web_library(root_project: Path, model_path: Optional[Path] = None) -> dict:
    onnx_src = model_path or (root_project / "public" / "models" / "side-ear.onnx")
    has_model = onnx_src.is_file()

    export_dir = root_project / "export" / "side-ear-library"
    if export_dir.exists():
        shutil.rmtree(export_dir)
    (export_dir / "models").mkdir(parents=True)
    (export_dir / "src").mkdir(parents=True)
    (export_dir / "config").mkdir(parents=True)

    model_name = "side-ear.onnx"
    if has_model:
        shutil.copy2(onnx_src, export_dir / "models" / model_name)
    else:
        (export_dir / "models" / "COPY_side-ear.onnx_HERE.txt").write_text(
            "Copy your trained side-ear model here as side-ear.onnx\n"
            "Train at http://127.0.0.1:8000/side-ear/ or:\n"
            "  python python/train.py --data data.side-ear.yaml --name side-ear "
            "--onnx-out public/models/side-ear.onnx\n",
            encoding="utf-8",
        )

    copies = [
        ("src/side-ear-tracker.js", "src/side-ear-tracker.js"),
        ("src/side-ear-worker.js", "src/side-ear-worker.js"),
        ("src/onnx-engine.js", "src/onnx-engine.js"),
        ("src/letterbox.js", "src/letterbox.js"),
        ("src/decoder.js", "src/decoder.js"),
        ("src/side-ear-defaults.js", "src/side-ear-defaults.js"),
        ("src/device.js", "src/device.js"),
        ("src/smoothing.js", "src/smoothing.js"),
        ("src/worker-client.js", "src/worker-client.js"),
        ("config/side-ear.json", "config/side-ear.json"),
        ("examples/side-ear-export.html", "example.html"),
    ]
    for src_rel, dest_rel in copies:
        src = root_project / src_rel
        if src.is_file():
            shutil.copy2(src, export_dir / dest_rel)

    readme = export_dir / "README.txt"
    readme.write_text(
        "Side-ear detection library (4 points)\n"
        "=====================================\n\n"
        "For profile/side ear images. Returns earlobe + 3 earring anchor positions.\n\n"
        "Integration:\n"
        "  import { createSideEarTracker, getVideoFps } from './src/side-ear-tracker.js';\n"
        "  const fps = getVideoFps(video) ?? 15;\n"
        "  const tracker = await createSideEarTracker({\n"
        "    modelUrl: './models/side-ear.onnx',\n"
        "    fps,\n"
        "  });\n"
        "  tracker.startLoop(video, ({ earlobe, earPos1, earPos2, earPos3 }) => { /* UI */ });\n\n"
        "See docs/SIDE_EAR_LIBRARY.md in the repo.\n",
        encoding="utf-8",
    )
    return {
        "export_dir": str(export_dir),
        "has_model": has_model,
        "model_path": str(onnx_src) if has_model else None,
    }

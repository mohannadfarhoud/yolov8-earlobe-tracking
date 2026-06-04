#!/usr/bin/env python3
"""
Move paired images/labels from flat folders into train/val splits.

Usage (from dataset root):
  python python/split_train_val.py --root . --val-ratio 0.15
"""

from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."), help="Dataset root")
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    root = args.root.resolve()
    images_flat = root / "images"
    labels_flat = root / "labels"

    pairs: list[tuple[Path, Path]] = []
    if not images_flat.is_dir():
        raise SystemExit(f"No folder: {images_flat}")

    for img in images_flat.iterdir():
        if img.suffix.lower() not in IMAGE_EXTS:
            continue
        lbl = labels_flat / f"{img.stem}.txt"
        if lbl.is_file():
            pairs.append((img, lbl))

    if not pairs:
        raise SystemExit(
            f"No image+label pairs in {images_flat} + {labels_flat}.\n"
            "Put .jpg in images/ and matching .txt in labels/ first."
        )

    random.seed(args.seed)
    random.shuffle(pairs)
    n_val = max(1, int(len(pairs) * args.val_ratio))
    val_set = set(range(n_val))

    for split in ("train", "val"):
        (root / "images" / split).mkdir(parents=True, exist_ok=True)
        (root / "labels" / split).mkdir(parents=True, exist_ok=True)

    for i, (img, lbl) in enumerate(pairs):
        split = "val" if i < n_val else "train"
        shutil.move(str(img), str(root / "images" / split / img.name))
        shutil.move(str(lbl), str(root / "labels" / split / lbl.name))

    print(f"Split {len(pairs)} pairs: train={len(pairs) - n_val}, val={n_val}")


if __name__ == "__main__":
    main()

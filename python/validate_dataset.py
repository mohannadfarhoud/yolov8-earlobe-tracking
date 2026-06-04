#!/usr/bin/env python3
"""Validate YOLO pose data.yaml and label files against the project dataset contract."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

EXPECTED_KPT_COUNT = 1
EXPECTED_DIMS_PER_KPT = 3
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def load_yaml(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_split(root: Path, entry: str | list) -> Path:
    if isinstance(entry, list):
        return (root / entry[0]).resolve()
    return (root / entry).resolve()


def resolve_labels_dir(images_dir: Path) -> Path:
    """
    YOLO layout: .../images/train/*.jpg -> .../labels/train/*.txt
    Works when path is the dataset root OR when path is the images/ folder.
    """
    parts = images_dir.parts
    if "images" in parts:
        idx = len(parts) - 1 - list(reversed(parts)).index("images")
        label_parts = list(parts[:idx]) + ["labels"] + list(parts[idx + 1 :])
        return Path(*label_parts)

    # path points at images/ as root: .../images/train -> .../labels/train
    if images_dir.parent.name == "images":
        return images_dir.parent.parent / "labels" / images_dir.name

    # Fallback: sibling labels/ next to images/
    return images_dir.parent.parent / "labels" / images_dir.name


def count_images(images_dir: Path) -> int:
    if not images_dir.is_dir():
        return 0
    return sum(1 for p in images_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS)


def count_label_lines(label_dir: Path) -> tuple[int, list[str]]:
    errors: list[str] = []
    labels = 0
    if not label_dir.is_dir():
        return 0, [f"missing labels directory: {label_dir}"]

    for txt in label_dir.glob("*.txt"):
        labels += 1
        lines = [ln.strip() for ln in txt.read_text(encoding="utf-8").splitlines() if ln.strip()]
        for i, line in enumerate(lines):
            parts = line.split()
            expected = 1 + 4 + EXPECTED_KPT_COUNT * EXPECTED_DIMS_PER_KPT
            if len(parts) != expected:
                errors.append(
                    f"{txt.name}:{i + 1} expected {expected} fields, got {len(parts)}"
                )
    return labels, errors


def validate(data_yaml: Path) -> int:
    if not data_yaml.is_file():
        print(f"ERROR: data.yaml not found: {data_yaml}")
        return 1

    cfg = load_yaml(data_yaml)
    required = ("path", "train", "val", "nc", "names", "kpt_shape")
    missing = [k for k in required if k not in cfg]
    if missing:
        print(f"ERROR: data.yaml missing keys: {missing}")
        return 1

    kpt_shape = cfg["kpt_shape"]
    if list(kpt_shape) != [EXPECTED_KPT_COUNT, EXPECTED_DIMS_PER_KPT]:
        print(
            f"WARN: kpt_shape {kpt_shape} != expected [{EXPECTED_KPT_COUNT}, {EXPECTED_DIMS_PER_KPT}]"
        )

    root = Path(cfg["path"])
    if not root.is_absolute():
        root = (data_yaml.parent / root).resolve()

    print(f"Dataset root (path): {root}")
    print(f"Classes (nc={cfg['nc']}): {cfg['names']}")
    print(f"Keypoint shape: {kpt_shape}")

    exit_code = 0
    for split in ("train", "val"):
        images_dir = resolve_split(root, cfg[split])
        labels_dir = resolve_labels_dir(images_dir)
        n_images = count_images(images_dir)
        n_labels, errs = count_label_lines(labels_dir)

        print(f"  {split}:")
        print(f"    images: {images_dir} -> {n_images} image files")
        print(f"    labels: {labels_dir} -> {n_labels} label files")

        parts_lower = [p.lower() for p in images_dir.parts]
        if "labels" in parts_lower and "images" not in parts_lower:
            print(
                f"    ERROR: data.yaml '{split}' points at labels/, not images/. "
                f"Use images/{split} with path = dataset root (parent of images/ and labels/)."
            )
            exit_code = 1

        if not images_dir.is_dir():
            print(f"    ERROR: missing images directory: {images_dir}")
            exit_code = 1
        if not labels_dir.is_dir():
            print(f"    ERROR: missing labels directory: {labels_dir}")
            exit_code = 1
        if n_labels == 0:
            print(f"    ERROR: no .txt labels found in {labels_dir}")
            exit_code = 1
        for e in errs[:10]:
            print(f"    ERROR: {e}")
            exit_code = 1
        if len(errs) > 10:
            print(f"    ... and {len(errs) - 10} more label format errors")
            exit_code = 1

    print("\nEarlobe keypoint index: 0 (config/tracking.json)")
    print("Selection rule: highest_conf (config/tracking.json)")

    if exit_code != 0:
        print("\nRecommended data.yaml (dataset root = project folder, NOT images/):")
        print("  path: C:/earlobe-tracking/yolov8-earlobe-tracking")
        print("  train: images/train")
        print("  val: images/val")
        print("\nExpected folders:")
        print("  images/train/*.jpg   labels/train/*.txt")
        print("  images/val/*.jpg     labels/val/*.txt")
        return 1

    print("\nOK: dataset structure and labels validated.")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate YOLO pose dataset")
    parser.add_argument("--data", type=Path, required=True, help="Path to data.yaml")
    args = parser.parse_args()
    sys.exit(validate(args.data.resolve()))


if __name__ == "__main__":
    main()

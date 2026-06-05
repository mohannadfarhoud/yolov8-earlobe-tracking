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


def scan_for_assets(root: Path) -> None:
    """Print where images/labels actually live under the dataset root."""
    print("\nScanning dataset root for images and labels...")
    if not root.is_dir():
        print(f"  ERROR: dataset root does not exist: {root}")
        return

    def count_in_dir(d: Path, exts: set[str]) -> int:
        if not d.is_dir():
            return 0
        return sum(1 for p in d.iterdir() if p.is_file() and p.suffix.lower() in exts)

    for name in sorted(root.iterdir()):
        if not name.is_dir():
            continue
        n_img = count_in_dir(name, IMAGE_EXTS) if name.name == "images" else 0
        n_lbl = count_in_dir(name, {".txt"}) if name.name == "labels" else 0
        if name.name == "images":
            for sub in sorted(name.iterdir()):
                if sub.is_dir():
                    ni = count_in_dir(sub, IMAGE_EXTS)
                    if ni:
                        print(f"  Found {ni} images in: {sub}")
            if n_img:
                print(f"  Found {n_img} images in: {name} (root of images/, not in train/val)")
        elif name.name == "labels":
            for sub in sorted(name.iterdir()):
                if sub.is_dir():
                    nl = count_in_dir(sub, {".txt"})
                    if nl:
                        print(f"  Found {nl} labels in: {sub}")
            if n_lbl:
                print(f"  Found {n_lbl} labels in: {name} (root of labels/, not in train/val)")
        else:
            ni = count_in_dir(name, IMAGE_EXTS)
            nl = count_in_dir(name, {".txt"})
            if ni or nl:
                print(f"  Found {ni} images, {nl} labels in: {name}")


def count_label_lines(label_dir: Path) -> tuple[int, list[str]]:
    errors: list[str] = []
    labels = 0
    if not label_dir.is_dir():
        return 0, []

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
        elif n_labels == 0:
            print(f"    ERROR: no .txt labels found in {labels_dir}")
            exit_code = 1
        for e in errs[:10]:
            print(f"    ERROR: {e}")
            exit_code = 1
        if len(errs) > 10:
            print(f"    ... and {len(errs) - 10} more label format errors")
            exit_code = 1

    if cfg.get("nc") != 2:
        print(f"WARN: nc={cfg.get('nc')} — expected nc=2 (left_earlobe, right_earlobe)")
    print("\nClasses: 0=left_earlobe, 1=right_earlobe (one or two lines per image)")

    if exit_code != 0:
        scan_for_assets(root)
        print("\nYour data.yaml paths look correct. Create missing folders and add files:")
        print("  mkdir images\\train, images\\val, labels\\train, labels\\val")
        print("  Move ~80-90% of pairs into train/, rest into val/ (same base names).")
        print("\nEach image needs a matching label, e.g.:")
        print("  images/train/photo001.jpg  <->  labels/train/photo001.txt")
        print("\nIf files are only in images/ and labels/ (no train/val subfolders), run:")
        print("  PowerShell: see README 'Split dataset into train/val' or move files manually.")
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

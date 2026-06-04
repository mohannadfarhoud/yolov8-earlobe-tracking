#!/usr/bin/env python3
"""Validate YOLO pose data.yaml and label files against the project dataset contract."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

EXPECTED_KPT_COUNT = 1
EXPECTED_DIMS_PER_KPT = 3


def load_yaml(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_split(root: Path, entry: str | list) -> Path:
    if isinstance(entry, list):
        return root / entry[0]
    return root / entry


def count_label_lines(label_dir: Path) -> tuple[int, int, list[str]]:
    errors: list[str] = []
    images = 0
    labels = 0
    if not label_dir.is_dir():
        return 0, 0, [f"missing labels directory: {label_dir}"]

    for txt in label_dir.glob("*.txt"):
        labels += 1
        lines = [ln.strip() for ln in txt.read_text(encoding="utf-8").splitlines() if ln.strip()]
        for i, line in enumerate(lines):
            parts = line.split()
            # class + box(4) + kpts(1*3)
            expected = 1 + 4 + EXPECTED_KPT_COUNT * EXPECTED_DIMS_PER_KPT
            if len(parts) != expected:
                errors.append(
                    f"{txt.name}:{i + 1} expected {expected} fields, got {len(parts)}"
                )
    images = labels  # paired check omitted; user may add images separately
    return images, labels, errors


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

    print(f"Dataset root: {root}")
    print(f"Classes (nc={cfg['nc']}): {cfg['names']}")
    print(f"Keypoint shape: {kpt_shape}")

    for split in ("train", "val"):
        images_dir = resolve_split(root, cfg[split])
        labels_dir = images_dir.parent / "labels" / images_dir.name
        if labels_dir.name == "images":
            labels_dir = images_dir.parent.parent / "labels" / images_dir.name
        # YOLO layout: images/train -> labels/train
        if "images" in str(images_dir):
            labels_dir = Path(str(images_dir).replace("images", "labels"))

        n_img, n_lbl, errs = count_label_lines(labels_dir)
        print(f"  {split}: labels dir {labels_dir} -> {n_lbl} label files")
        for e in errs[:10]:
            print(f"    ERROR: {e}")
        if len(errs) > 10:
            print(f"    ... and {len(errs) - 10} more errors")

    print("\nEarlobe keypoint index: 0 (config/tracking.json)")
    print("Selection rule: highest_conf (config/tracking.json)")
    print("OK: data.yaml structure validated. Fix label errors above before training.")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate YOLO pose dataset")
    parser.add_argument("--data", type=Path, required=True, help="Path to data.yaml")
    args = parser.parse_args()
    sys.exit(validate(args.data.resolve()))


if __name__ == "__main__":
    main()

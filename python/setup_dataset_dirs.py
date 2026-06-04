#!/usr/bin/env python3
"""Create standard YOLO train/val folder structure."""

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("."),
        help="Dataset root (contains images/ and labels/)",
    )
    args = parser.parse_args()
    root = args.root.resolve()
    for split in ("train", "val"):
        (root / "images" / split).mkdir(parents=True, exist_ok=True)
        (root / "labels" / split).mkdir(parents=True, exist_ok=True)
    print(f"Created under {root}:")
    print("  images/train  images/val  labels/train  labels/val")
    print("Add .jpg + matching .txt label files, then validate with validate_dataset.py")


if __name__ == "__main__":
    main()

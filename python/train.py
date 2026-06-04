#!/usr/bin/env python3
"""Train YOLOv8-pose on earlobe keypoints and export best.onnx for the web app."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from ultralytics import YOLO

from validate_dataset import validate

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA = ROOT / "data.yaml"
ONNX_OUT = ROOT / "public" / "models" / "best.onnx"
WEIGHTS_DIR = ROOT / "runs" / "pose" / "earlobe"


def train(
    data: Path,
    epochs: int = 100,
    imgsz: int = 640,
    batch: int = 16,
    device: str | int = 0,
    seed: int = 42,
    patience: int = 20,
) -> Path:
    model = YOLO("yolov8n-pose.pt")
    results = model.train(
        data=str(data),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        seed=seed,
        patience=patience,
        project=str(ROOT / "runs" / "pose"),
        name="earlobe",
        exist_ok=True,
        save=True,
        val=True,
    )
    best_pt = Path(results.save_dir) / "weights" / "best.pt"
    if not best_pt.is_file():
        best_pt = WEIGHTS_DIR / "weights" / "best.pt"
    if not best_pt.is_file():
        raise FileNotFoundError("best.pt not found after training")
    print(f"Training complete. best.pt: {best_pt}")
    return best_pt


def export_onnx(
    weights: Path,
    imgsz: int = 640,
    opset: int = 12,
    simplify: bool = True,
) -> Path:
    model = YOLO(str(weights))
    exported = model.export(
        format="onnx",
        imgsz=imgsz,
        opset=opset,
        simplify=simplify,
        dynamic=False,
    )
    src = Path(exported)
    ONNX_OUT.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, ONNX_OUT)
    print(f"ONNX copied to: {ONNX_OUT}")
    return ONNX_OUT


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default="0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--opset", type=int, default=12)
    parser.add_argument("--skip-train", action="store_true", help="Only export from existing best.pt")
    parser.add_argument(
        "--weights",
        type=Path,
        default=WEIGHTS_DIR / "weights" / "best.pt",
    )
    args = parser.parse_args()

    if args.skip_train:
        weights = args.weights
        if not weights.is_file():
            raise SystemExit(f"Weights not found: {weights}")
    else:
        if not args.data.is_file():
            raise SystemExit(
                f"data.yaml not found: {args.data}\n"
                "Copy data.example.yaml to data.yaml and point path/train/val at your dataset."
            )
        data_path = args.data.resolve()
        if validate(data_path) != 0:
            raise SystemExit(
                "Dataset validation failed. Fix images/labels folders before training.\n"
                "Run: python python/validate_dataset.py --data data.yaml"
            )
        weights = train(
            data=data_path,
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
            seed=args.seed,
            patience=args.patience,
        )

    export_onnx(weights, imgsz=args.imgsz, opset=args.opset)


if __name__ == "__main__":
    main()

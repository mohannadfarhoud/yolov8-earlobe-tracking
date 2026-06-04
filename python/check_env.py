#!/usr/bin/env python3
"""Verify CUDA and GPU availability for local training."""

import sys

import torch


def main() -> int:
    cuda = torch.cuda.is_available()
    print(f"torch.cuda.is_available(): {cuda}")
    if cuda:
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA version: {torch.version.cuda}")
    else:
        print("WARN: CUDA not available — training will fall back to CPU (slow).")
        print("Install a CUDA-enabled PyTorch build for your RTX GPU.")

    try:
        from ultralytics import YOLO  # noqa: F401

        print("ultralytics: OK")
    except ImportError:
        print("ERROR: ultralytics not installed. Run: pip install -r requirements.txt")
        return 1

    return 0 if cuda else 0  # warn but do not hard-fail on CPU


if __name__ == "__main__":
    sys.exit(main())

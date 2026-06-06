"""Export web-library folder + zip for frontend integration."""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from annotate_utils import export_web_library  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Export earlobe web library for frontend")
    parser.add_argument(
        "--model",
        type=Path,
        default=None,
        help="Path to best.onnx (default: public/models/best.onnx)",
    )
    parser.add_argument("--no-zip", action="store_true", help="Skip creating zip archive")
    args = parser.parse_args()

    info = export_web_library(ROOT, model_path=args.model)
    export_dir = Path(info["export_dir"])

    zip_path = None
    if not args.no_zip:
        zip_base = str(export_dir.parent / "earlobe-web-library")
        old_zip = Path(zip_base + ".zip")
        if old_zip.is_file():
            old_zip.unlink()
        zip_path = shutil.make_archive(zip_base, "zip", export_dir)

    print("Export complete")
    print(f"  Folder: {export_dir}")
    if zip_path:
        print(f"  Zip:    {zip_path}")
    if info["has_model"]:
        print(f"  Model:  {info['model_path']}")
    else:
        print("  Model:  MISSING — copy best.onnx to export/web-library/models/")
        print("           (train first, or pass --model C:\\path\\to\\best.onnx)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

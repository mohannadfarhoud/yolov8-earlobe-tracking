"""Export side-ear web library + zip."""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from side_ear_utils import export_side_ear_web_library  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Export side-ear web library")
    parser.add_argument(
        "--model",
        type=Path,
        default=None,
        help="Path to side-ear.onnx (default: public/models/side-ear.onnx)",
    )
    parser.add_argument("--no-zip", action="store_true")
    args = parser.parse_args()

    info = export_side_ear_web_library(ROOT, model_path=args.model)
    export_dir = Path(info["export_dir"])

    zip_path = None
    if not args.no_zip:
        zip_base = str(export_dir.parent / "side-ear-web-library")
        old_zip = Path(zip_base + ".zip")
        if old_zip.is_file():
            old_zip.unlink()
        zip_path = shutil.make_archive(zip_base, "zip", export_dir)

    print("Side-ear export complete")
    print(f"  Folder: {export_dir}")
    if zip_path:
        print(f"  Zip:    {zip_path}")
    if not info["has_model"]:
        print("  WARN: no ONNX model copied — train first")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

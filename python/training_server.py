#!/usr/bin/env python3
"""
Training web UI backend + static page.

  python python/training_server.py
  → http://127.0.0.1:8000
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import threading
from pathlib import Path
from typing import Optional

import yaml
from annotate_utils import (
    dataset_yaml_config,
    ensure_dataset_layout,
    export_web_library,
    list_images,
    read_ear_labels,
    split_train_to_val,
    write_ear_labels,
)
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
WEB_DIR = ROOT / "training-web"
DATA_YAML = ROOT / "data.yaml"
ONNX_PATH = ROOT / "public" / "models" / "best.onnx"
PYTHON = sys.executable


def run_cmd(script: str, *args: str) -> tuple[int, str]:
    proc = subprocess.run(
        [PYTHON, str(ROOT / "python" / script), *args],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    text = ((proc.stdout or "") + (proc.stderr or "")).strip()
    return proc.returncode, text or "(no output)"


class TrainState:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.running = False
        self.lines: list[str] = []
        self.returncode: int | None = None

    def reset(self) -> None:
        with self.lock:
            self.running = True
            self.lines = []
            self.returncode = None

    def append(self, line: str) -> None:
        with self.lock:
            self.lines.append(line.rstrip())
            if len(self.lines) > 500:
                self.lines = self.lines[-400:]

    def finish(self, code: int) -> None:
        with self.lock:
            self.running = False
            self.returncode = code

    def snapshot(self) -> dict:
        with self.lock:
            return {
                "running": self.running,
                "lines": list(self.lines),
                "returncode": self.returncode,
            }


train_state = TrainState()


class DatasetRoot(BaseModel):
    root: str = Field(..., description="Dataset root folder")


class TrainParams(BaseModel):
    epochs: int = 100
    batch: int = 16
    imgsz: int = 640
    device: str = "0"
    patience: int = 20


class EarPoint(BaseModel):
    x: float
    y: float


class SaveAnnotation(BaseModel):
    root: str
    split: str = "train"
    filename: str
    image_width: int
    image_height: int
    left: Optional[EarPoint] = None
    right: Optional[EarPoint] = None


def resolve_root(path: str) -> Path:
    root = Path(path.strip()).resolve()
    if not root.is_dir():
        root.mkdir(parents=True, exist_ok=True)
    return root


def safe_image_path(root: Path, split: str, filename: str) -> Path:
    if split not in ("train", "val"):
        raise HTTPException(400, "split must be train or val")
    img = (root / "images" / split / Path(filename).name).resolve()
    if not str(img).startswith(str(root.resolve())):
        raise HTTPException(403, "invalid path")
    if not img.is_file():
        raise HTTPException(404, "image not found")
    return img


def save_data_yaml_for_root(root: Path) -> None:
    cfg = dataset_yaml_config(root)
    with DATA_YAML.open("w", encoding="utf-8") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)


app = FastAPI(title="Earlobe Training")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def api_health():
    return {"ok": True, "python": sys.executable, "project": str(ROOT)}


@app.get("/api/environment")
def api_environment():
    """Run GPU check in-process (subprocess + torch import can hang/silence on Windows)."""
    lines: list[str] = []
    try:
        import torch

        cuda = torch.cuda.is_available()
        lines.append(f"torch.cuda.is_available(): {cuda}")
        if cuda:
            lines.append(f"GPU: {torch.cuda.get_device_name(0)}")
            lines.append(f"CUDA version: {torch.version.cuda}")
        else:
            lines.append("WARN: CUDA not available — training will use CPU (very slow).")
            lines.append("Install CUDA PyTorch: pip install torch --index-url https://download.pytorch.org/whl/cu124")
        try:
            from ultralytics import YOLO  # noqa: F401

            lines.append("ultralytics: OK")
        except ImportError:
            lines.append("ERROR: ultralytics not installed — pip install -r requirements.txt")
            return {"ok": False, "output": "\n".join(lines)}
        lines.append(f"Python: {sys.executable}")
        return {"ok": True, "output": "\n".join(lines)}
    except Exception as e:
        lines.append(f"ERROR: {e}")
        return {"ok": False, "output": "\n".join(lines)}


@app.post("/api/dataset/setup")
def api_dataset_setup(body: DatasetRoot):
    root = Path(body.root.strip()).resolve()
    for split in ("train", "val"):
        (root / "images" / split).mkdir(parents=True, exist_ok=True)
        (root / "labels" / split).mkdir(parents=True, exist_ok=True)
    return {
        "ok": True,
        "message": f"Created folders under {root}",
        "paths": [
            str(root / "images" / "train"),
            str(root / "images" / "val"),
            str(root / "labels" / "train"),
            str(root / "labels" / "val"),
        ],
    }


@app.post("/api/dataset/config")
def api_dataset_config(body: DatasetRoot):
    root = resolve_root(body.root)
    ensure_dataset_layout(root)
    save_data_yaml_for_root(root)
    return {"ok": True, "path": str(DATA_YAML), "root": str(root)}


@app.post("/api/annotate/init")
def api_annotate_init(body: DatasetRoot):
    root = resolve_root(body.root)
    ensure_dataset_layout(root)
    save_data_yaml_for_root(root)
    items = list_images(root)
    return {
        "ok": True,
        "root": str(root),
        "total": len(items),
        "annotated": sum(1 for i in items if i["annotated"]),
    }


@app.post("/api/annotate/upload")
async def api_annotate_upload(
    root: str = Query(...),
    files: list[UploadFile] = File(...),
):
    dataset = resolve_root(root)
    ensure_dataset_layout(dataset)
    train_dir = dataset / "images" / "train"
    saved: list[str] = []
    for uf in files:
        if not uf.filename:
            continue
        ext = Path(uf.filename).suffix.lower()
        if ext not in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
            continue
        dest = train_dir / Path(uf.filename).name
        n = 1
        while dest.exists():
            dest = train_dir / f"{Path(uf.filename).stem}_{n}{ext}"
            n += 1
        content = await uf.read()
        dest.write_bytes(content)
        saved.append(dest.name)
    items = list_images(dataset)
    return {
        "ok": True,
        "saved": saved,
        "total": len(items),
        "annotated": sum(1 for i in items if i["annotated"]),
    }


@app.get("/api/annotate/images")
def api_annotate_images(root: str = Query(...)):
    dataset = resolve_root(root)
    return {"ok": True, "images": list_images(dataset)}


@app.get("/api/annotate/image/{split}/{filename}")
def api_annotate_image(split: str, filename: str, root: str = Query(...)):
    dataset = resolve_root(root)
    path = safe_image_path(dataset, split, filename)
    return FileResponse(path)


@app.get("/api/annotate/label/{split}/{stem}")
def api_annotate_label(split: str, stem: str, root: str = Query(...)):
    dataset = resolve_root(root)
    ears = read_ear_labels(dataset, split, stem)
    return {"ok": True, "ears": ears}


@app.post("/api/annotate/save")
def api_annotate_save(body: SaveAnnotation):
    dataset = resolve_root(body.root)
    if body.image_width < 1 or body.image_height < 1:
        raise HTTPException(400, "invalid image dimensions")
    if body.left is None and body.right is None:
        raise HTTPException(400, "mark at least one earlobe (left or right)")
    img_path = safe_image_path(dataset, body.split, body.filename)
    left = {"x": body.left.x, "y": body.left.y} if body.left else None
    right = {"x": body.right.x, "y": body.right.y} if body.right else None
    write_ear_labels(
        dataset,
        body.split,
        img_path.stem,
        left,
        right,
        body.image_width,
        body.image_height,
    )
    items = list_images(dataset)
    return {
        "ok": True,
        "label": f"labels/{body.split}/{img_path.stem}.txt",
        "total": len(items),
        "annotated": sum(1 for i in items if i["annotated"]),
    }


@app.post("/api/annotate/split-val")
def api_annotate_split_val(body: DatasetRoot, ratio: float = Query(0.15)):
    dataset = resolve_root(body.root)
    result = split_train_to_val(dataset, val_ratio=ratio)
    save_data_yaml_for_root(dataset)
    return {"ok": True, **result, "images": list_images(dataset)}


@app.post("/api/annotate/prepare-train")
def api_annotate_prepare_train(body: DatasetRoot):
    """Split val, save data.yaml, validate — run before training."""
    dataset = resolve_root(body.root)
    ensure_dataset_layout(dataset)
    val_count = len(list((dataset / "images" / "val").glob("*")))
    if val_count == 0:
        split_train_to_val(dataset)
    save_data_yaml_for_root(dataset)
    code, text = run_cmd("validate_dataset.py", "--data", str(DATA_YAML))
    return {"ok": code == 0, "output": text}


@app.post("/api/dataset/validate")
def api_dataset_validate():
    if not DATA_YAML.is_file():
        raise HTTPException(400, "data.yaml missing — save dataset config first")
    code, text = run_cmd("validate_dataset.py", "--data", str(DATA_YAML))
    return {"ok": code == 0, "output": text}


@app.get("/api/model/status")
def api_model_status():
    pt = ROOT / "runs" / "pose" / "earlobe" / "weights" / "best.pt"
    return {
        "best_pt": str(pt),
        "best_pt_exists": pt.is_file(),
        "best_onnx": str(ONNX_PATH),
        "best_onnx_exists": ONNX_PATH.is_file(),
    }


@app.post("/api/model/verify")
def api_model_verify():
    if not ONNX_PATH.is_file():
        raise HTTPException(400, "best.onnx not found — train first")
    code, text = run_cmd("verify_onnx.py")
    return {"ok": code == 0, "output": text}


@app.post("/api/model/export-library")
def api_model_export_library():
    if not ONNX_PATH.is_file():
        raise HTTPException(400, "best.onnx not found — train first")
    try:
        info = export_web_library(ROOT)
        export_dir = Path(info["export_dir"])
        zip_base = str(export_dir.parent / "earlobe-web-library")
        if Path(zip_base + ".zip").is_file():
            Path(zip_base + ".zip").unlink()
        archive = shutil.make_archive(zip_base, "zip", export_dir)
        code, text = run_cmd("verify_onnx.py")
        return {
            "ok": True,
            "export_dir": str(export_dir),
            "zip": archive,
            "verify": text,
            "message": "Copy export/web-library/ into your site, or download the zip.",
        }
    except FileNotFoundError as e:
        raise HTTPException(400, str(e)) from e


def _train_worker(params: TrainParams) -> None:
    cmd = [
        PYTHON,
        str(ROOT / "python" / "train.py"),
        "--data",
        str(DATA_YAML),
        "--epochs",
        str(params.epochs),
        "--batch",
        str(params.batch),
        "--imgsz",
        str(params.imgsz),
        "--device",
        params.device.strip(),
        "--patience",
        str(params.patience),
    ]
    try:
        proc = subprocess.Popen(
            cmd,
            cwd=str(ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            train_state.append(line)
        proc.wait()
        train_state.append(f"\n--- Finished (exit {proc.returncode}) ---")
        if ONNX_PATH.is_file():
            train_state.append(f"ONNX ready: {ONNX_PATH}")
        train_state.finish(proc.returncode or 0)
    except Exception as e:
        train_state.append(f"ERROR: {e}")
        train_state.finish(1)


@app.post("/api/train/start")
def api_train_start(params: TrainParams):
    if not DATA_YAML.is_file():
        raise HTTPException(400, "data.yaml missing — save dataset config first")
    snap = train_state.snapshot()
    if snap["running"]:
        raise HTTPException(409, "Training already running")
    train_state.reset()
    thread = threading.Thread(target=_train_worker, args=(params,), daemon=True)
    thread.start()
    return {"ok": True, "message": "Training started"}


@app.get("/api/train/status")
def api_train_status():
    return train_state.snapshot()


@app.get("/")
def index():
    index_file = WEB_DIR / "index.html"
    if not index_file.is_file():
        raise HTTPException(404, "training-web/index.html not found")
    return FileResponse(index_file)


@app.get("/assets/app.js")
def asset_js():
    return FileResponse(WEB_DIR / "app.js", media_type="application/javascript")


@app.get("/api/model/download-library")
def api_model_download_library():
    zip_path = ROOT / "export" / "earlobe-web-library.zip"
    if not zip_path.is_file():
        raise HTTPException(404, "Run Export web library first")
    return FileResponse(zip_path, filename="earlobe-web-library.zip", media_type="application/zip")


@app.get("/assets/styles.css")
def asset_css():
    return FileResponse(WEB_DIR / "styles.css", media_type="text/css")


def main() -> None:
    import uvicorn

    print("Earlobe training UI: http://127.0.0.1:8000")
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")


if __name__ == "__main__":
    main()

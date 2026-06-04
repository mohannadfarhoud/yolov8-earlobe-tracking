#!/usr/bin/env python3
"""
Training web UI backend + static page.

  python python/training_server.py
  → http://127.0.0.1:8000
"""

from __future__ import annotations

import asyncio
import subprocess
import sys
import threading
from pathlib import Path

import yaml
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent.parent
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


app = FastAPI(title="Earlobe Training")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/environment")
def api_environment():
    code, text = run_cmd("check_env.py")
    return {"ok": code == 0, "output": text}


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
    root = Path(body.root.strip()).resolve()
    cfg = {
        "path": str(root).replace("\\", "/"),
        "train": "images/train",
        "val": "images/val",
        "nc": 1,
        "names": {0: "earlobe"},
        "kpt_shape": [1, 3],
    }
    with DATA_YAML.open("w", encoding="utf-8") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
    return {"ok": True, "path": str(DATA_YAML), "yaml": yaml.dump(cfg, sort_keys=False)}


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


if WEB_DIR.is_dir():
    app.mount("/assets", StaticFiles(directory=WEB_DIR), name="assets")


def main() -> None:
    import uvicorn

    print("Earlobe training UI: http://127.0.0.1:8000")
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")


if __name__ == "__main__":
    main()

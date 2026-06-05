#!/usr/bin/env python3
"""
Local web UI for dataset setup, validation, training, and ONNX export.

Run from project root:
  python python/training_ui.py
Then open http://127.0.0.1:7860
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import gradio as gr
import yaml

ROOT = Path(__file__).resolve().parent.parent
DATA_YAML = ROOT / "data.yaml"
ONNX_PATH = ROOT / "public" / "models" / "best.onnx"
PYTHON = sys.executable


def _run_script(script: str, *args: str) -> str:
    cmd = [PYTHON, str(ROOT / "python" / script), *args]
    proc = subprocess.run(
        cmd,
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    out = (proc.stdout or "") + (proc.stderr or "")
    if proc.returncode != 0:
        out += f"\n[exit code {proc.returncode}]"
    return out.strip() or "(no output)"


def check_gpu() -> str:
    return _run_script("check_env.py")


def write_data_yaml(dataset_root: str) -> str:
    root = Path(dataset_root.strip()).resolve()
    cfg = {
        "path": str(root).replace("\\", "/"),
        "train": "images/train",
        "val": "images/val",
        "nc": 2,
        "names": {0: "left_earlobe", 1: "right_earlobe"},
        "kpt_shape": [1, 3],
    }
    with DATA_YAML.open("w", encoding="utf-8") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
    return f"Saved {DATA_YAML}\n\n{yaml.dump(cfg, sort_keys=False)}"


def setup_folders(dataset_root: str) -> str:
    root = Path(dataset_root.strip()).resolve()
    for split in ("train", "val"):
        (root / "images" / split).mkdir(parents=True, exist_ok=True)
        (root / "labels" / split).mkdir(parents=True, exist_ok=True)
    return (
        f"Created folders under {root}:\n"
        "  images/train, images/val\n"
        "  labels/train, labels/val\n\n"
        "Add .jpg images and matching .txt label files, then Validate."
    )


def validate_dataset() -> str:
    if not DATA_YAML.is_file():
        return f"ERROR: {DATA_YAML} not found. Save data.yaml first (Dataset tab)."
    return _run_script("validate_dataset.py", "--data", str(DATA_YAML))


def model_status() -> str:
    lines = []
    pt = ROOT / "runs" / "pose" / "earlobe" / "weights" / "best.pt"
    lines.append(f"best.pt:  {'OK — ' + str(pt) if pt.is_file() else 'missing'}")
    lines.append(f"best.onnx: {'OK — ' + str(ONNX_PATH) if ONNX_PATH.is_file() else 'missing (train first)'}")
    return "\n".join(lines)


def verify_onnx() -> str:
    if not ONNX_PATH.is_file():
        return "ERROR: best.onnx not found. Run training first."
    return _run_script("verify_onnx.py")


def run_train(epochs: int, batch: int, imgsz: int, device: str, patience: int, progress=gr.Progress()):
    if not DATA_YAML.is_file():
        yield "ERROR: Save data.yaml first (Dataset tab)."
        return

    progress(0, desc="Starting training…")
    cmd = [
        PYTHON,
        str(ROOT / "python" / "train.py"),
        "--data",
        str(DATA_YAML),
        "--epochs",
        str(int(epochs)),
        "--batch",
        str(int(batch)),
        "--imgsz",
        str(int(imgsz)),
        "--device",
        str(device).strip(),
        "--patience",
        str(int(patience)),
    ]
    proc = subprocess.Popen(
        cmd,
        cwd=str(ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    log_lines: list[str] = []
    assert proc.stdout is not None
    for line in proc.stdout:
        log_lines.append(line.rstrip())
        if len(log_lines) > 200:
            log_lines.pop(0)
        yield "\n".join(log_lines)

    proc.wait()
    log_lines.append(f"\n--- Finished (exit {proc.returncode}) ---")
    log_lines.append(model_status())
    if proc.returncode == 0 and ONNX_PATH.is_file():
        log_lines.append("\nReady for browser library: public/models/best.onnx")
    yield "\n".join(log_lines)


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="Earlobe training", theme=gr.themes.Soft()) as app:
        gr.Markdown(
            "# Earlobe model training\n"
            "Local UI for your RTX PC. Training uses **Python/CUDA**; the browser library uses **best.onnx** after export."
        )

        with gr.Tab("1 — GPU"):
            gr.Markdown("Confirm CUDA and GPU are available before training.")
            gpu_log = gr.Textbox(label="Environment", lines=8, interactive=False)
            gr.Button("Check GPU", variant="primary").click(check_gpu, outputs=gpu_log)

        with gr.Tab("2 — Dataset"):
            dataset_root = gr.Textbox(
                label="Dataset root folder",
                placeholder=r"C:\earlobe-tracking\earlobe",
                info="Must contain images/train, images/val, labels/train, labels/val",
            )
            ds_log = gr.Textbox(label="Log", lines=12, interactive=False)
            with gr.Row():
                gr.Button("Create train/val folders").click(setup_folders, inputs=dataset_root, outputs=ds_log)
                gr.Button("Save data.yaml", variant="primary").click(
                    write_data_yaml, inputs=dataset_root, outputs=ds_log
                )
            gr.Button("Validate dataset", variant="secondary").click(validate_dataset, outputs=ds_log)

        with gr.Tab("3 — Train"):
            gr.Markdown("Runs `train.py` then exports `public/models/best.onnx`. This can take a long time.")
            with gr.Row():
                epochs = gr.Slider(10, 300, value=100, step=10, label="Epochs")
                batch = gr.Slider(1, 32, value=16, step=1, label="Batch size")
            with gr.Row():
                imgsz = gr.Dropdown([416, 640], value=640, label="Image size")
                device = gr.Textbox(value="0", label="Device (0 = first GPU)")
                patience = gr.Slider(5, 50, value=20, step=5, label="Early stop patience")
            train_log = gr.Textbox(label="Training log", lines=20, interactive=False, max_lines=30)
            gr.Button("Start training", variant="primary").click(
                run_train,
                inputs=[epochs, batch, imgsz, device, patience],
                outputs=train_log,
            )

        with gr.Tab("4 — Model"):
            status = gr.Textbox(label="Files", lines=4, interactive=False)
            gr.Button("Refresh status").click(model_status, outputs=status)
            onnx_log = gr.Textbox(label="Verify log", lines=10, interactive=False)
            gr.Button("Verify ONNX", variant="primary").click(verify_onnx, outputs=onnx_log)
            app.load(model_status, outputs=status)

        gr.Markdown(
            "---\n"
            "**Next:** copy `public/models/best.onnx` to your web app and use `src/earlobe-tracker.js`.\n"
            "Demo: `npm run dev` → `/examples/earlobe-only.html`"
        )

    return app


def main() -> None:
    app = build_ui()
    app.launch(server_name="127.0.0.1", server_port=7860, show_error=True)


if __name__ == "__main__":
    main()

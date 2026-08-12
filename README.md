---
title: YOLOv8 Vehicle Tracking & Counting System
emoji: 🚗
colorFrom: blue
colorTo: indigo
sdk: streamlit
sdk_version: "1.35.0"
app_file: app.py
pinned: false
license: mit
---

# Vehicle Counting — YOLOv8 Custom Training

This repository contains code and experiments for training a custom YOLOv8 model for vehicle counting and tracking.

Contents
- `train_v1.py`, `train_v2.py`, `run_v2.py` — training entrypoints
- `augment_dataset.py` — dataset augmentation utilities
- `compare_models.py` — compare model metrics and outputs
- `app.py` — quick demo / inference script
- `yolov8n.pt` — base model weights (downloaded or provided)
- `runs/` — training outputs, logs, metrics, and weights

Quickstart
1. Create a Python environment and install dependencies (example):

```bash
python -m venv .venv
source .venv/Scripts/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt  # if provided, otherwise install ultralytics, opencv-python, etc.
```

2. Train a model (example):

```bash
python train_v2.py
```

3. Inspect results in `runs/` — each run has `results.csv`, `weights/`, and evaluation outputs.

Viewing metrics
- See `runs/v2_augmented/results.csv` for per-epoch metrics (precision, recall, mAP, losses, learning rate).

Notes
- This repo expects YOLO/Ultralytics-style training scripts. Adjust dataset YAML paths in the `train_*.py` scripts if needed.
- If you want, I can add a `requirements.txt`, example dataset YAML, or a short demo notebook.

License
- Add your preferred license file.

Latest Results
- Source: [runs/v2_augmented/results.csv](runs/v2_augmented/results.csv#L20)
- Latest epoch (20) summary: precision=0.95974, recall=0.87903, mAP50=0.94304, mAP50-95=0.71158, val losses (box/cls/dfl)=0.99343 / 0.73173 / 1.28592

v1 vs v2 Comparison
| Metric | v1 (validation) | v2 (test) |
|---|---:|---:|
| Precision | 0.94032 | 0.89815 |
| Recall | 0.87950 | 0.83079 |
| mAP50 | 0.95474 | 0.90098 |
| mAP50-95 | 0.71011 | 0.61648 |

Sources: [runs/v1_metrics.json](runs/v1_metrics.json#L1) (validation) and [runs/v2_metrics.json](runs/v2_metrics.json#L1) (test). Metrics are shown to three-five decimal places as reported.

Streamlit Deployment
- Install dependencies and run the demo/inference app with Streamlit.

```bash
python -m venv .venv
# Windows activate: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

- If you need GPU-aware PyTorch, install `torch` using the instructions at https://pytorch.org/ for the appropriate CUDA/toolkit version before `pip install -r requirements.txt`.
- If `app.py` is not a Streamlit app, replace `app.py` with your Streamlit entrypoint (for example `streamlit_app.py`).

Streamlit Cloud runtime note
- Streamlit Cloud uses the Python runtime declared in `runtime.txt` (if present). Some binary packages such as OpenCV may not have prebuilt wheels for the newest Python versions. If you see `ModuleNotFoundError: No module named 'cv2'` on deploy, ensure the repo contains `runtime.txt` with a compatible version (example included: `python-3.10.12`).

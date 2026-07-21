# CV Starter

Live object detection using YOLO (Ultralytics) with webcam input, accelerated on Apple Silicon (MPS).

## Prerequisites

- macOS with Apple Silicon (M-series)
- [uv](https://docs.astral.sh/uv/) installed (`brew install uv` or `curl -LsSf https://astral.sh/uv/install.sh | sh`)

## Setup

```bash
git clone git@github.com:cltj/cv-starter.git
cd cv-starter
uv sync
```

### Grant camera permission

The first time you run a script that opens the webcam, macOS will prompt you to grant
camera access to your terminal app. Approve it — you can manage this later under
**System Settings → Privacy & Security → Camera**.

### Verify everything works

```bash
uv run python src/check_setup.py
```

You should see:
- ✅ MPS available (GPU acceleration)
- ✅ Webcam OK
- ✅ Ultralytics YOLO loaded

## Run the live detector

```bash
uv run python main.py
```

A window will open showing your webcam feed with real-time object detection (bounding boxes + labels). Press **q** to quit.

## Project structure

```
cv-starter/
├── main.py              # live object detector
├── pyproject.toml
├── .gitignore
└── src/
    └── check_setup.py   # environment verification script
```

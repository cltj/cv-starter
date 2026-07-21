"""
Run this after installing dependencies to confirm everything is wired up
correctly before we build the actual detector.

Checks:
1. PyTorch can see the Apple Silicon MPS backend
2. OpenCV can open the laptop webcam and grab a frame
3. Ultralytics is importable and can load a tiny pretrained model
"""

import sys


def check_torch_mps():
    import torch
    print(f"PyTorch version: {torch.__version__}")
    if torch.backends.mps.is_available():
        print("✅ MPS (Apple Silicon GPU acceleration) is available")
        return "mps"
    else:
        print("⚠️  MPS not available — will fall back to CPU (still fine for testing)")
        return "cpu"


def check_webcam():
    import cv2
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Could not open webcam (index 0). Check System Settings > Privacy > Camera "
              "and make sure your terminal/IDE has camera permission.")
        return False
    ret, frame = cap.read()
    cap.release()
    if not ret or frame is None:
        print("❌ Webcam opened but no frame was captured.")
        return False
    h, w = frame.shape[:2]
    print(f"✅ Webcam OK — captured a frame at {w}x{h}")
    return True


def check_ultralytics(device):
    from ultralytics import YOLO
    print("Downloading/loading a small pretrained YOLO model (first run only)...")
    model = YOLO("yolo11n.pt")  # nano model, ~5MB, fast to fetch and load
    model.to(device)
    print(f"✅ Ultralytics YOLO loaded successfully on '{device}'")
    return True


if __name__ == "__main__":
    print("=== Environment check ===\n")

    device = check_torch_mps()
    print()
    webcam_ok = check_webcam()
    print()
    yolo_ok = check_ultralytics(device)

    print("\n=== Summary ===")
    if webcam_ok and yolo_ok:
        print("Everything is set up. Ready to build the real detector.")
        sys.exit(0)
    else:
        print("Something needs attention before we continue — see warnings above.")
        sys.exit(1)

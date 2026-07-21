# Edge Deployment — YOLO on Microcontrollers & SBCs

## ESP8266 / ESP32 — Too weak for YOLO

| Spec | ESP8266 | ESP32-CAM | YOLO nano needs |
|------|---------|-----------|-----------------|
| CPU | 80MHz, single core | 240MHz, dual core | Multi-core GHz |
| RAM | 80KB | 520KB | ~500MB+ |
| Storage | 4MB flash | 4-16MB flash | ~6MB model alone |

These chips **cannot run YOLO**. They can capture and stream frames (ESP32-CAM + OV2640 camera ~$3) over WiFi to a device that runs inference.

## Raspberry Pi — Feasible but slow

| | Pi 4 (4GB) | Pi 5 (8GB) | Apple M4 Mac |
|--|-----------|-----------|-------------|
| CPU | 1.5GHz quad-core ARM | 2.4GHz quad-core ARM | 4.4GHz 10-core |
| GPU for ML | None (CPU only) | None (CPU only) | MPS |
| RAM | 4GB | 8GB | 64GB |
| YOLO nano estimate | ~1-3 FPS | ~3-6 FPS | ~27 FPS |
| Price | ~$55 | ~$80 | — |

Tips for Pi:
- Nano model only (`yolo11n.pt`)
- `imgsz=320` or `imgsz=160`
- Export to NCNN or ONNX for optimized CPU inference (PyTorch is slow on ARM)

## Coral USB Accelerator — the edge ML secret weapon

~$25-35, plugs into Pi via USB, dedicated ML chip (4 TOPS).
With a TFLite-exported YOLO model: **15-25 FPS** on a Pi.

## Budget edge setup (~$50-100)

| Component | Price | Role |
|-----------|-------|------|
| Raspberry Pi 5 (8GB) | ~$80 | Runs inference |
| Pi Camera Module 3 | ~$25 | Direct CSI connection |
| or ESP32-CAM | ~$8 | Captures + streams over WiFi |
| Coral USB Accelerator | ~$30 | Optional — boosts to 15-25 FPS |

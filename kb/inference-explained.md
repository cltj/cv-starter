# What Determines Inference Time

## The formula

```
Inference time ≈ total operations / hardware throughput
```

## Total operations (what the model needs to do)

Determined by:
- **Parameters** — number of weights in the model (nano: ~3M, medium: ~25M)
- **Input size** — pixels to process (640×640 = 410K pixels, 320×320 = 102K)
- **Architecture** — how many multiply-accumulate ops per layer

YOLO nano does roughly 6-8 GFLOPS (billion floating point operations) per frame. This is fixed for a given model + input size.

## Hardware throughput (how fast the chip does it)

| Hardware | Throughput | YOLO nano inference |
|----------|-----------|-------------------|
| M4 Max (MPS/GPU) | ~53 TOPS | ~25ms |
| M4 Max (CPU only) | ~5-8 TOPS | ~80ms |
| Raspberry Pi 5 (CPU) | ~0.05 TOPS | ~200-300ms |
| Coral USB Accelerator | 4 TOPS | ~30-50ms |
| ESP32 | ~0.001 TOPS | Impossible |

TOPS = Trillion Operations Per Second

## The levers you can pull

### 1. Smaller model
Fewer parameters = fewer operations per frame.

| Model | Parameters | GFLOPS | Accuracy tradeoff |
|-------|-----------|--------|-------------------|
| YOLO nano | ~3M | ~8 | Baseline |
| Custom pruned | ~1M | ~3 | Lower, but often good enough |

### 2. Smaller input
Fewer pixels = fewer operations. Roughly scales with area.

| Input size | Pixels | Relative speed |
|-----------|--------|---------------|
| 640×640 | 410K | 1x (baseline) |
| 320×320 | 102K | ~3-4x faster |
| 160×160 | 26K | ~10-15x faster |

### 3. Model optimization (biggest win on weak hardware)

**Quantization** — convert model weights from 32-bit floats to 8-bit integers:
- Same model, same architecture
- Roughly same accuracy (within 1-2%)
- 2-4x faster inference
- How people run YOLO at 10-15 FPS on a Pi

**Export formats** optimized for specific hardware:

| Format | Best for | Why faster |
|--------|---------|------------|
| PyTorch (.pt) | GPU (MPS, CUDA) | Default, good GPU support |
| ONNX | Cross-platform CPU | Optimized CPU ops |
| NCNN | ARM/mobile | Tuned for ARM instruction set |
| TFLite | Coral/edge TPU | INT8, hardware-specific |
| CoreML | Apple devices | Uses Neural Engine directly |

### 4. Better hardware
More TOPS = faster inference. Add a Coral USB ($30) to a Pi and go from 3 FPS to 15-25 FPS.

## Combining levers

These multiply. On a Raspberry Pi 5:

| Setup | Estimated FPS |
|-------|--------------|
| PyTorch, nano, 640 | ~3 |
| NCNN, nano, 320 | ~10-15 |
| NCNN, nano, 320, INT8 quantized | ~15-25 |
| + Coral USB accelerator | ~25-40 |

The same model that runs at 3 FPS can reach 25+ FPS with the right optimizations — no accuracy change needed.

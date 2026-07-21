# Understanding Resource Metrics

## CPU % — the confusing one

`psutil` reports CPU % relative to **a single core**:
- 100% = 1 core fully busy
- 250% = 2.5 cores fully busy

This is misleading because it doesn't tell you how much of the machine you're using.

**Fix: normalize to total capacity.**

| Machine | Cores | Max psutil % | 250% psutil = | Real usage |
|---------|-------|-------------|---------------|------------|
| M4 Max | 16 | 1600% | 250/1600 | 15.6% |
| Raspberry Pi 5 | 4 | 400% | 250/400 | 62.5% |
| ESP32 | 2 | 200% | impossible | — |

Formula: `actual_usage = psutil_cpu_pct / cpu_count`

## Inference time (ms) — the portable metric

How long one YOLO call takes. This is the most useful metric for comparing across hardware because the workload is identical — only the hardware speed changes.

| Hardware | YOLO nano inference | Meaning |
|----------|-------------------|---------|
| M4 Max (MPS) | ~25ms | 40 FPS max |
| M4 Max (CPU only) | ~80ms | 12 FPS max |
| Raspberry Pi 5 | ~200-300ms | 3-5 FPS max |

## Memory (MB) — the fixed cost

Loading a YOLO model takes a fixed amount of RAM regardless of whether it's running or idle:
- YOLO nano: ~400-600MB
- YOLO medium: ~1-2GB

This is why memory is the first bottleneck on small devices — if the model doesn't fit, nothing else matters.

## GPU time (seconds) — the real cost

Total seconds spent on YOLO inference. This is what you'd pay for in cloud GPU billing.

Example from our testing:
- 13-minute session, 13% YOLO usage
- 134 seconds of actual GPU inference time
- Motion gating saved ~900 seconds of GPU time vs running every frame

## Summary: which metrics matter when?

| Question | Metric to check |
|----------|----------------|
| "Can this device run it?" | Memory (does model fit?) + inference time (fast enough?) |
| "How much of my machine am I using?" | CPU % of total |
| "What will this cost in the cloud?" | GPU time (seconds) |
| "Is the motion gate saving resources?" | YOLO calls vs total frames, GPU total seconds |
| "Will this work on smaller hardware?" | Inference time (scales with hardware speed) |

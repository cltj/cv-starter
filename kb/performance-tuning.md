# Performance Tuning — YOLO on Apple M4 (64GB RAM)

Model: `yolo11n.pt` (nano), MPS acceleration

## Benchmarks

| Camera Resolution | YOLO imgsz | Threaded Capture | FPS | Detection Quality |
|-------------------|-----------|-----------------|-----|-------------------|
| 1920×1080 | 640 (default) | No | ~12 | Best — reliably detects small and distant objects, high confidence scores |
| 640×480 | 640 (default) | No | ~23 | Good — slight drop on small/distant objects, solid for most use cases |
| 640×480 | 320 | No | ~30 | Reduced — misses smaller objects, lower confidence, bounding boxes less precise |
| 640×480 | 640 (default) | Yes | ~27 | Good — same quality as non-threaded, smooth real-time experience |
| 640×480 | 320 | Yes | ~44 | Reduced — misses smaller objects, but very high throughput |

## The tradeoff: FPS vs detection quality

Lower image fidelity (smaller camera resolution and/or smaller YOLO processing size) gives higher FPS, which is useful for:
- Real-time tracking where responsiveness matters more than precision (e.g. following a person or pet moving quickly)
- Running on weaker hardware or leaving GPU headroom for other tasks
- Interactive demos where visual smoothness is important

But at the expense of:
- Detection accuracy — smaller objects get missed entirely
- Confidence scores — the model is less certain about what it sees
- Bounding box precision — boxes may be offset or poorly sized
- Range — objects need to be closer to the camera to be detected reliably

## Rule of thumb

- **Accuracy matters** (e.g. inventory scanning, security): use highest resolution you can while staying above 10 FPS
- **Responsiveness matters** (e.g. live tracking, interactive): target 20+ FPS, lower resolution as needed
- **Sweet spot on M4**: 640×480 camera, imgsz=640, threaded capture (~27 FPS) — good detection quality with smooth real-time performance

## Threaded capture

Reading frames in a background thread removes the camera wait from the inference loop, giving a significant FPS boost (~30 → ~44 in our tests). Good for live detection, but be aware:
- **Dropped frames** — frames captured during inference are silently discarded (fine for live view, bad for frame-by-frame analysis)
- **Thread safety** — frame reads/writes happen across threads without locks (works in practice but not strictly safe)
- **CPU overhead** — camera runs at full speed even when frames aren't consumed

**Verdict**: Use it for live/interactive detection. Avoid for offline video processing where every frame matters.

## Other ways to increase FPS

- **Larger model variant** (`s`, `m`, `l`, `x`) trades FPS for better accuracy at same resolution
- **Skip frames** — run inference every 2nd or 3rd frame, display the last annotated result in between

# Why Classical CV Filters Matter

## The core idea

YOLO is powerful but expensive (~30ms per frame, needs GPU). Classical filters are nearly free (~0.1ms, CPU only). Use filters for the simple stuff, YOLO for the hard stuff.

## Pattern: filters + YOLO together

```
Frame → cheap filter → worth analyzing?
    No  → skip
    Yes → run YOLO
```

## Practical examples

| Filter | Use case | Why not just YOLO? |
|---|---|---|
| Background subtraction | Only run YOLO when something moves | Saves 90%+ compute when scene is static |
| Color isolation | Find the red fire extinguisher on a wall | YOLO detects "object" not "red object" |
| Edge detection | Check if a circuit board has broken traces | YOLO detects objects, not fine structural defects |
| Blur | Clean up noisy input before feeding to YOLO | Better accuracy on cheap cameras |
| Contours | Measure actual shape/size of a detected object | YOLO gives a bounding box, contours give the real outline |

## Why this matters at scale

- **Edge devices** (Raspberry Pi): can't afford to run YOLO on every frame. Filters act as a gate.
- **Multi-camera systems**: 100 cameras × 30 FPS = 3000 YOLO calls/sec. Filtering first might cut that to 300.
- **Battery-powered devices**: less compute = longer battery life.
- **Cost**: cloud GPU inference costs money per call. Fewer calls = lower bill.

## Rule of thumb

- Can a simple filter answer the question? Use the filter.
- Do you need to know *what* something is? Use YOLO.
- Best results often come from combining both.

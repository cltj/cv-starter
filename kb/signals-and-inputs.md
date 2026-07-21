# Signals & Input Sources

Every sensor produces a signal. The real power comes from combining them.

## Signal Categories

### Vision (what you see)
- **Camera (RGB)** — standard video/image, what we're using now
- **Infrared/thermal** — sees heat, works in darkness
- **Depth cameras (LiDAR, stereo)** — distance to objects
- **Satellite imagery** — large-scale patterns over time
- **Radar** — sees through weather, walls, darkness

### Audio (what you hear)
- **Microphone** — speech, ambient sounds, machinery noise
- **Ultrasonic** — distance sensing (like parking sensors)
- **Echo/sonar** — underwater or enclosed space mapping

### Physical environment
- Temperature, humidity, air quality, pressure
- Light level, UV index
- Vibration/accelerometer
- Weight/force/strain sensors

### Human input
- Keyboard/mouse activity
- Touch sensors, gestures
- Screen capture
- Eye tracking

### Location & movement
- GPS, compass
- IMU (accelerometer + gyroscope)
- Wheel encoders, odometry

### Binary/state
- Switches, buttons, relays
- Door open/closed, motion detected
- Liquid level, flow rate

## Combining Signals

A single signal is useful. Multiple signals together are transformative:

| Single signal | Combined with | Unlocks |
|---|---|---|
| Camera sees a person | Door sensor + time | Access control |
| Microphone hears a machine | Vibration + temperature | Predictive maintenance |
| Camera sees a shelf | Weight sensor | Accurate inventory |
| Camera sees a room | Thermal + air quality | Smart building management |
| Satellite imagery | Weather + soil moisture | Precision agriculture |
| Camera sees traffic | Radar + GPS | Autonomous driving |
| Microphone hears speech | Camera (lip tracking) | Robust speech recognition |

## What You Can Explore With Just a Laptop

| Signal | How | Library |
|---|---|---|
| Camera | Webcam (done) | OpenCV |
| Camera + filters | Edge detection, color isolation, blur, background subtraction | OpenCV |
| Microphone | Built-in mic | PyAudio, librosa |
| Screen capture | What's on your display | mss, Pillow |
| Keyboard/mouse | Activity patterns | pynput |
| Weather data | API calls | requests (Open-Meteo, free) |

## Camera Filters (Classical Computer Vision)

Before deep learning, computer vision relied on hand-crafted filters. Understanding these helps you know when you need YOLO vs when a simple filter is enough:

- **Edge detection** (Canny, Sobel) — find object boundaries
- **Color filtering** (HSV thresholds) — isolate objects by color
- **Background subtraction** — detect what's moving vs what's static
- **Blur/smoothing** (Gaussian, median) — reduce noise
- **Morphological operations** (erode, dilate) — clean up binary masks
- **Contour detection** — find shapes after filtering
- **Optical flow** — track motion between frames

These are fast, lightweight, and don't need a GPU. A combination of classical filters + YOLO is often better than either alone.

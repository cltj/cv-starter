"""
Motion-triggered object detector.

Uses background subtraction as a cheap gate — YOLO only runs when motion is detected.
Saves GPU when the scene is static (e.g. you're away from your desk).

Controls:
    q    Quit
"""

import csv
import os
import time
import threading
from collections import Counter
from datetime import datetime, timezone

import cv2
import psutil
from ultralytics import YOLO

CLASSES = ["person", "laptop", "keyboard", "mouse", "cell phone", "cup",
           "bottle", "book", "scissors", "remote", "clock", "vase"]

LOG_INTERVAL = 5.0
LOG_MIN_CONFIDENCE = 0.5
CHANGE_DEBOUNCE_FRAMES = 5
LOG_DIR = "data/detections"

# Percentage of frame pixels that must change to trigger YOLO
MOTION_THRESHOLD = 0.5


class CameraStream:
    def __init__(self, src=0):
        self.cap = cv2.VideoCapture(src)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.ret, self.frame = self.cap.read()
        self.stopped = False
        self.thread = threading.Thread(target=self._update, daemon=True)
        self.thread.start()

    def _update(self):
        while not self.stopped:
            self.ret, self.frame = self.cap.read()

    def read(self):
        return self.ret, self.frame

    def stop(self):
        self.stopped = True
        self.thread.join()
        self.cap.release()


def init_log_files():
    os.makedirs(LOG_DIR, exist_ok=True)
    session_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    det_path = os.path.join(LOG_DIR, f"motion_{session_id}.csv")
    det_f = open(det_path, "w", newline="")
    det_writer = csv.writer(det_f)
    det_writer.writerow([
        "timestamp", "object_class", "confidence",
        "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2",
    ])

    metrics_path = os.path.join(LOG_DIR, f"metrics_{session_id}.csv")
    metrics_f = open(metrics_path, "w", newline="")
    metrics_writer = csv.writer(metrics_f)
    metrics_writer.writerow([
        "timestamp", "motion_pct", "yolo_ran", "inference_ms",
        "fps", "objects_detected", "cpu_pct", "memory_mb",
        "yolo_total_sec",
    ])

    print(f"Logging detections to {det_path}")
    print(f"Logging metrics to {metrics_path}")
    return det_writer, det_f, metrics_writer, metrics_f


def get_signature(detections):
    return tuple(sorted(Counter(d[0] for d in detections).items()))


def main():
    model = YOLO("yolo11n.pt")

    class_indices = None
    if CLASSES is not None:
        name_to_id = {v: k for k, v in model.names.items()}
        class_indices = [name_to_id[c] for c in CLASSES if c in name_to_id]

    stream = CameraStream(0)
    if not stream.cap.isOpened():
        print("Error: Could not open webcam")
        return

    csv_writer, log_file, metrics_writer, metrics_file = init_log_files()
    bg_sub = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50)

    cv2.namedWindow("Motion Detector", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Motion Detector", 960, 540)

    print("Motion-triggered detector running. Press 'q' to quit.")

    process = psutil.Process()

    prev_time = time.time()
    last_log_time = 0.0
    last_logged_signature = ()
    pending_signature = None
    pending_detections = []
    pending_count = 0
    last_detections = []
    yolo_calls = 0
    total_frames = 0
    yolo_total_sec = 0.0
    last_inference_ms = 0.0

    while True:
        ret, frame = stream.read()
        if not ret:
            break

        total_frames += 1

        # Check for motion
        fg_mask = bg_sub.apply(frame)
        motion_pct = (fg_mask > 200).sum() / fg_mask.size * 100

        now = time.time()
        fps = 1.0 / (now - prev_time) if (now - prev_time) > 0 else 0
        prev_time = now

        if motion_pct > MOTION_THRESHOLD:
            # Motion detected — run YOLO
            t0 = time.perf_counter()
            results = model(frame, classes=class_indices, verbose=False)
            last_inference_ms = (time.perf_counter() - t0) * 1000
            yolo_total_sec += last_inference_ms / 1000
            annotated = results[0].plot()
            yolo_calls += 1

            detections = []
            for box in results[0].boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                if conf >= LOG_MIN_CONFIDENCE:
                    cls_name = results[0].names[cls_id]
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    detections.append((cls_name, conf, x1, y1, x2, y2))
            last_detections = detections
            status = f"YOLO | motion: {motion_pct:.1f}%"
            status_color = (0, 255, 0)
        else:
            # No motion — skip YOLO, show raw frame
            last_inference_ms = 0.0
            annotated = frame.copy()
            detections = last_detections
            status = f"IDLE | motion: {motion_pct:.1f}%"
            status_color = (0, 165, 255)

        # Logging (same debounce logic as main.py)
        signature = get_signature(detections)
        time_to_log = (now - last_log_time) >= LOG_INTERVAL
        changed = False
        if signature != last_logged_signature:
            if signature == pending_signature:
                pending_count += 1
                pending_detections = detections
                if pending_count >= CHANGE_DEBOUNCE_FRAMES:
                    changed = True
            else:
                pending_signature = signature
                pending_detections = detections
                pending_count = 1
        else:
            pending_signature = None
            pending_count = 0

        if changed:
            ts = datetime.now(timezone.utc).isoformat()
            for cls_name, conf, x1, y1, x2, y2 in pending_detections:
                csv_writer.writerow([ts, cls_name, f"{conf:.3f}",
                                     f"{x1:.1f}", f"{y1:.1f}", f"{x2:.1f}", f"{y2:.1f}"])
            log_file.flush()
            last_log_time = now
            last_logged_signature = signature
            pending_signature = None
            pending_count = 0
        elif time_to_log:
            ts = datetime.now(timezone.utc).isoformat()
            for cls_name, conf, x1, y1, x2, y2 in detections:
                csv_writer.writerow([ts, cls_name, f"{conf:.3f}",
                                     f"{x1:.1f}", f"{y1:.1f}", f"{x2:.1f}", f"{y2:.1f}"])
            log_file.flush()
            last_log_time = now
            last_logged_signature = signature

        # Stats
        yolo_pct = (yolo_calls / total_frames * 100) if total_frames > 0 else 0

        # Resource usage
        cpu_pct = process.cpu_percent()
        mem_mb = process.memory_info().rss / 1024 / 1024

        # Log metrics every interval
        if time_to_log or changed:
            ts = datetime.now(timezone.utc).isoformat()
            metrics_writer.writerow([
                ts, f"{motion_pct:.2f}",
                1 if motion_pct > MOTION_THRESHOLD else 0,
                f"{last_inference_ms:.1f}", f"{fps:.1f}", len(detections),
                f"{cpu_pct:.1f}", f"{mem_mb:.0f}", f"{yolo_total_sec:.1f}",
            ])
            metrics_file.flush()

        # HUD
        cv2.putText(annotated, f"FPS: {fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
        cv2.putText(annotated, status, (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
        cv2.putText(annotated, f"Inference: {last_inference_ms:.0f}ms | GPU total: {yolo_total_sec:.1f}s", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
        cv2.putText(annotated, f"CPU: {cpu_pct:.0f}% | RAM: {mem_mb:.0f}MB", (10, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)

        cv2.imshow("Motion Detector", annotated)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    log_file.close()
    metrics_file.close()
    stream.stop()
    cv2.destroyAllWindows()
    print(f"\nSession stats:")
    print(f"  Total frames: {total_frames}")
    print(f"  YOLO calls: {yolo_calls} ({yolo_pct:.0f}% of frames)")
    print(f"  Frames skipped: {total_frames - yolo_calls}")
    print(f"  Total GPU inference time: {yolo_total_sec:.1f}s")
    print(f"  Final RAM usage: {process.memory_info().rss / 1024 / 1024:.0f}MB")


if __name__ == "__main__":
    main()

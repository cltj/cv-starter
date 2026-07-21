import csv
import os
import time
import threading
from collections import Counter
from datetime import datetime, timezone

import cv2
from ultralytics import YOLO

# Filter to these classes only. Set to None to detect everything.
# See kb/yolo-coco-classes.md for all 80 available classes.
CLASSES = ["person", "laptop", "keyboard", "mouse", "cell phone", "cup",
           "bottle", "book", "scissors", "remote", "clock", "vase"]

# How often to log detections (seconds). Detections still run every frame for display.
LOG_INTERVAL = 5.0

# Minimum confidence to log a detection
LOG_MIN_CONFIDENCE = 0.5

# Number of consecutive frames a change must persist before logging
CHANGE_DEBOUNCE_FRAMES = 5

# Output directory for detection logs
LOG_DIR = "data/detections"


class CameraStream:
    """Reads frames in a background thread so capture doesn't block inference."""

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


def init_log_file():
    """Create a CSV log file for this session, return the writer and file handle."""
    os.makedirs(LOG_DIR, exist_ok=True)
    session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = os.path.join(LOG_DIR, f"session_{session_id}.csv")
    f = open(filepath, "w", newline="")
    writer = csv.writer(f)
    writer.writerow([
        "timestamp", "object_class", "confidence",
        "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2",
    ])
    print(f"Logging detections to {filepath}")
    return writer, f


def extract_detections(results):
    """Extract a list of (class_name, confidence, bbox) from YOLO results."""
    detections = []
    for box in results[0].boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        if conf < LOG_MIN_CONFIDENCE:
            continue
        cls_name = results[0].names[cls_id]
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        detections.append((cls_name, conf, x1, y1, x2, y2))
    return detections


def get_signature(detections):
    """Get a hashable signature: sorted class counts."""
    return tuple(sorted(Counter(d[0] for d in detections).items()))


def log_detections(writer, detections):
    """Write detections to the CSV log."""
    ts = datetime.now(timezone.utc).isoformat()
    for cls_name, conf, x1, y1, x2, y2 in detections:
        writer.writerow([ts, cls_name, f"{conf:.3f}",
                         f"{x1:.1f}", f"{y1:.1f}", f"{x2:.1f}", f"{y2:.1f}"])


def main():
    model = YOLO("yolo11n.pt")

    # Resolve class names to YOLO class indices
    class_indices = None
    if CLASSES is not None:
        name_to_id = {v: k for k, v in model.names.items()}
        class_indices = [name_to_id[c] for c in CLASSES if c in name_to_id]
        print(f"Filtering to: {CLASSES}")

    stream = CameraStream(0)
    if not stream.cap.isOpened():
        print("Error: Could not open webcam")
        return

    csv_writer, log_file = init_log_file()

    cv2.namedWindow("YOLO Live Detector", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("YOLO Live Detector", 960, 540)

    print(f"Live object detector running. Logging every {LOG_INTERVAL}s. Press 'q' to quit.")

    prev_time = time.time()
    last_log_time = 0.0
    last_logged_signature = ()
    # Debounce: track how many consecutive frames show a new signature
    pending_signature = None
    pending_detections = []
    pending_count = 0

    while True:
        ret, frame = stream.read()
        if not ret:
            break

        results = model(frame, classes=class_indices, verbose=False)
        annotated = results[0].plot()

        # FPS counter
        now = time.time()
        fps = 1.0 / (now - prev_time)
        prev_time = now
        cv2.putText(
            annotated, f"FPS: {fps:.1f}", (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2,
        )

        detections = extract_detections(results)
        signature = get_signature(detections)
        time_to_log = (now - last_log_time) >= LOG_INTERVAL

        # Debounced change detection
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
            log_detections(csv_writer, pending_detections)
            log_file.flush()
            last_log_time = now
            last_logged_signature = signature
            pending_signature = None
            pending_count = 0
        elif time_to_log:
            log_detections(csv_writer, detections)
            log_file.flush()
            last_log_time = now
            last_logged_signature = signature

        cv2.imshow("YOLO Live Detector", annotated)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    log_file.close()
    stream.stop()
    cv2.destroyAllWindows()
    print(f"Session ended. Log saved to {LOG_DIR}/")


if __name__ == "__main__":
    main()

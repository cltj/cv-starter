import time
import threading

import cv2
from ultralytics import YOLO

# Filter to these classes only. Set to None to detect everything.
# See kb/yolo-coco-classes.md for all 80 available classes.
CLASSES = ["laptop", "keyboard", "mouse", "cell phone", "cup", "bottle",
           "book", "scissors", "remote", "clock", "vase"]


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

    cv2.namedWindow("YOLO Live Detector", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("YOLO Live Detector", 960, 540)

    print("Live object detector running. Press 'q' to quit.")

    prev_time = time.time()

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

        cv2.imshow("YOLO Live Detector", annotated)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    stream.stop()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

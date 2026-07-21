import cv2
from ultralytics import YOLO


def main():
    model = YOLO("yolo11n.pt")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return

    cv2.namedWindow("YOLO Live Detector", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("YOLO Live Detector", 960, 540)

    print("Live object detector running. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, verbose=False)
        annotated = results[0].plot()

        cv2.imshow("YOLO Live Detector", annotated)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

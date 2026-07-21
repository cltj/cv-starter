"""
Classical CV filter explorer — cycle through filters on your live webcam.

Controls:
    1-7  Select filter
    q    Quit

Filters:
    1  Original (no filter)
    2  Canny edge detection
    3  Color isolation (blue objects)
    4  Background subtraction (motion detection)
    5  Gaussian blur
    6  Sobel edges (gradient)
    7  Contour detection
"""

import cv2
import numpy as np


def apply_canny(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    return cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)


def apply_color_isolation(frame):
    """Isolate blue objects using HSV color space."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # Blue range in HSV
    lower = np.array([100, 50, 50])
    upper = np.array([130, 255, 255])
    mask = cv2.inRange(hsv, lower, upper)
    return cv2.bitwise_and(frame, frame, mask=mask)


def make_background_subtractor():
    return cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50)


def apply_background_sub(frame, subtractor):
    mask = subtractor.apply(frame)
    return cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)


def apply_blur(frame):
    return cv2.GaussianBlur(frame, (21, 21), 0)


def apply_sobel(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
    magnitude = np.uint8(np.clip(magnitude, 0, 255))
    return cv2.cvtColor(magnitude, cv2.COLOR_GRAY2BGR)


def apply_contours(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    output = frame.copy()
    cv2.drawContours(output, contours, -1, (0, 255, 0), 2)
    return output


FILTERS = {
    ord("1"): ("Original", None),
    ord("2"): ("Canny Edge Detection", "canny"),
    ord("3"): ("Color Isolation (Blue)", "color"),
    ord("4"): ("Background Subtraction", "bgsub"),
    ord("5"): ("Gaussian Blur", "blur"),
    ord("6"): ("Sobel Edges", "sobel"),
    ord("7"): ("Contour Detection", "contours"),
}


def main():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not cap.isOpened():
        print("Error: Could not open webcam")
        return

    cv2.namedWindow("Filter Explorer", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Filter Explorer", 960, 540)

    bg_sub = make_background_subtractor()
    current_filter = "Original"
    current_key = None

    print("Filter Explorer running. Press 1-7 to switch filters, 'q' to quit.")
    print("  1: Original  2: Canny  3: Color (blue)  4: Background sub")
    print("  5: Blur  6: Sobel  7: Contours")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Apply selected filter
        if current_key == "canny":
            output = apply_canny(frame)
        elif current_key == "color":
            output = apply_color_isolation(frame)
        elif current_key == "bgsub":
            output = apply_background_sub(frame, bg_sub)
        elif current_key == "blur":
            output = apply_blur(frame)
        elif current_key == "sobel":
            output = apply_sobel(frame)
        elif current_key == "contours":
            output = apply_contours(frame)
        else:
            output = frame

        # Show filter name
        cv2.putText(
            output, current_filter, (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2,
        )

        cv2.imshow("Filter Explorer", output)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key in FILTERS:
            current_filter, current_key = FILTERS[key]
            print(f"  → {current_filter}")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

"""
Bat tracker: detects a blue object (the bat body) and a wooden/brown tip from
the webcam, then draws a line through them representing the bat.

Controls:
    q or ESC  - quit
    d         - toggle debug masks window

Tuning:
    If detection is poor under your lighting, adjust the HSV ranges below.
"""

import cv2
import numpy as np

# --- HSV color ranges (OpenCV hue is 0-179) -------------------------------
# Blue object (bat body) — dull/muted, so low saturation floor
BLUE_LOWER = np.array([95, 50, 40])
BLUE_UPPER = np.array([135, 255, 255])

# Wooden / brown tip (brown = dark, moderately-saturated orange in HSV).
# Widen/narrow these under your lighting; use 'd' to see the mask live.
BROWN_LOWER = np.array([5, 30, 30])
BROWN_UPPER = np.array([28, 200, 190])

MIN_AREA = 300  # ignore tiny blobs (noise)


def largest_centroid(mask):
    """Return (cx, cy, area) of the largest contour in mask, or None."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    c = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(c)
    if area < MIN_AREA:
        return None
    M = cv2.moments(c)
    if M["m00"] == 0:
        return None
    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])
    return cx, cy, area


def color_mask(hsv, lower, upper):
    mask = cv2.inRange(hsv, lower, upper)
    # clean up: remove specks, fill holes
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask


def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam (index 0).")

    show_debug = False
    print("Running. Press 'q' or ESC to quit, 'd' to toggle debug masks.")

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)  # mirror for natural interaction
        blurred = cv2.GaussianBlur(frame, (5, 5), 0)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

        blue_mask = color_mask(hsv, BLUE_LOWER, BLUE_UPPER)
        brown_mask = color_mask(hsv, BROWN_LOWER, BROWN_UPPER)

        blue = largest_centroid(blue_mask)
        brown = largest_centroid(brown_mask)

        if blue:
            cv2.circle(frame, (blue[0], blue[1]), 8, (255, 0, 0), -1)
        if brown:
            cv2.circle(frame, (brown[0], brown[1]), 8, (19, 69, 139), -1)

        if blue and brown:
            base = np.array([blue[0], blue[1]], dtype=float)
            tip = np.array([brown[0], brown[1]], dtype=float)
            # extend the line a bit beyond the tip so it reads as a full bat
            direction = tip - base
            norm = np.linalg.norm(direction)
            if norm > 1:
                unit = direction / norm
                end = tip + unit * 40  # overshoot past the tip
                cv2.line(frame, tuple(base.astype(int)), tuple(end.astype(int)),
                         (0, 0, 255), 6)
                cv2.putText(frame, "BAT", tuple((base - unit * 20).astype(int)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            cv2.putText(frame, "Show blue body + brown tip", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.imshow("Bat Tracker", frame)
        if show_debug:
            cv2.imshow("Blue mask", blue_mask)
            cv2.imshow("Brown mask", brown_mask)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break
        if key == ord("d"):
            show_debug = not show_debug
            if not show_debug:
                cv2.destroyWindow("Blue mask")
                cv2.destroyWindow("Brown mask")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

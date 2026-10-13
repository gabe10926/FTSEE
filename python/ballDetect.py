#%%
"""
Ball detection + tracking using OpenCV tracker.
"""

import argparse
import os

import cv2
import numpy as np

from yellow_ball import (
    BLUE_HSV_LOWER, BLUE_HSV_UPPER, detect_blue,
    RED_HSV_LOWER_1, RED_HSV_UPPER_1, RED_HSV_LOWER_2, RED_HSV_UPPER_2,
    YELLOW_HSV_LOWER, YELLOW_HSV_UPPER, detect_red, detect_yellow,
)

# BALL THRESHOLDS: BLUE / RED / YELLOW
MIN_BALL_AREA_BLUE_RED = 300
MIN_BALL_AREA_YELLOW = 500
MAX_BALL_AREA = 50000
MAX_BALL_AREA_YELLOW = 120000


def detect_balls(frame):
    """Detect blue, red, and yellow balls."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    blue_mask, blue_contours = detect_blue(frame)

    red_mask, red_contours = detect_red(frame)

    yellow_mask, yellow_contours = detect_yellow(frame)

    balls = []
    for mask, color, min_area in [
        (blue_mask, 'B', MIN_BALL_AREA_BLUE_RED),
        (red_mask, 'R', MIN_BALL_AREA_BLUE_RED),
        (yellow_mask, 'Y', MIN_BALL_AREA_YELLOW),
    ]:
        contours = (
            yellow_contours if color == 'Y' else
            red_contours if color == 'R' else blue_contours
        )
        for cnt in contours:
            area = cv2.contourArea(cnt)
            max_area = MAX_BALL_AREA_YELLOW if color == 'Y' else MAX_BALL_AREA
            if min_area < area < max_area:
                x, y, w, h = cv2.boundingRect(cnt)
                balls.append({
                    'bbox': (x, y, w, h),
                    'color': color,
                    'contour': cnt
                })

    balls.sort(key=lambda b: b['bbox'][0])
    return balls, blue_mask, red_mask, yellow_mask


def get_ball_colors(frame):
    """Return balls left-to-right as a string, e.g. 'BRY'."""
    balls, _, _, _ = detect_balls(frame)
    return ''.join(b['color'] for b in balls)


def find_default_image_source():
    candidates = [
        os.path.join(os.path.dirname(__file__), 'calib_imgs'),
        os.path.join(os.path.dirname(__file__), '..', 'python', 'calib_imgs'),
        os.getcwd(),
    ]
    for base in candidates:
        try:
            if not os.path.isdir(base):
                continue
            files = sorted(
                f for f in os.listdir(base)
                if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))
            )
            if files:
                return os.path.join(base, files[0])
        except OSError:
            pass
    return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='OpenCV ball detector for B/R/Y balls')
    parser.add_argument('--source', type=str, default=None, help='camera index, video path, or image path')
    parser.add_argument('--no-window', action='store_true', help='skip OpenCV windows and print results to the terminal')
    args = parser.parse_args()

    src = args.source
    if src is None:
        src = 0

    try:
        cap = cv2.VideoCapture(int(src))
        if not cap.isOpened():
            raise ValueError
    except (ValueError, TypeError):
        cap = cv2.VideoCapture(src)

    if not cap.isOpened():
        fallback = find_default_image_source()
        if fallback is None:
            print("No camera and no fallback image found. Pass --source /path/to/image.jpg")
            raise SystemExit(1)
        print(f"No camera detected; using fallback image: {fallback}")
        frame = cv2.imread(fallback)
        if frame is None:
            print(f"Could not read fallback image: {fallback}")
            raise SystemExit(1)
        # Single-frame run
        balls, blue_mask, red_mask, yellow_mask = detect_balls(frame)
        colors = ''.join(b['color'] for b in balls)
        print(f"Detected {len(balls)} balls: {colors if colors else 'none'}")
        if args.no_window:
            raise SystemExit(0)
        output = frame.copy()
        for ball in balls:
            x, y, w, h = ball['bbox']
            if ball['color'] == 'B':
                clr = (255, 0, 0)
            elif ball['color'] == 'R':
                clr = (0, 0, 255)
            else:
                clr = (0, 255, 255)
            cv2.rectangle(output, (x, y), (x + w, y + h), clr, 2)
        cv2.putText(output, f"DETECT: {len(balls)} balls - {colors}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.imshow("Ball Detection", output)
        cv2.imshow("Blue Mask", blue_mask)
        cv2.imshow("Red Mask", red_mask)
        cv2.imshow("Yellow Mask", yellow_mask)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        raise SystemExit(0)

    if isinstance(src, str) and os.path.isfile(src):
        frame = cv2.imread(src)
        if frame is not None:
            balls, blue_mask, red_mask, yellow_mask = detect_balls(frame)
            colors = ''.join(b['color'] for b in balls)
            print(f"Detected {len(balls)} balls: {colors if colors else 'none'}")
            if args.no_window:
                cap.release()
                raise SystemExit(0)
            output = frame.copy()
            for ball in balls:
                x, y, w, h = ball['bbox']
                if ball['color'] == 'B':
                    clr = (255, 0, 0)
                elif ball['color'] == 'R':
                    clr = (0, 0, 255)
                else:
                    clr = (0, 255, 255)
                cv2.rectangle(output, (x, y), (x + w, y + h), clr, 2)
            cv2.putText(output, f"DETECT: {len(balls)} balls - {colors}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.imshow("Ball Detection", output)
            cv2.imshow("Blue Mask", blue_mask)
            cv2.imshow("Red Mask", red_mask)
            cv2.imshow("Yellow Mask", yellow_mask)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            cap.release()
            raise SystemExit(0)

    trackers = []  # list of (tracker, color)
    tracking = False

    print("=" * 40)
    print("BALL TRACKING")
    print("'s' = start tracking detected balls")
    print("'r' = reset trackers")
    print("'q' = quit")
    print("=" * 40)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        output = frame.copy()

        if tracking and trackers:
            # Update trackers
            for i, (tracker, color) in enumerate(trackers):
                success, bbox = tracker.update(frame)
                if success:
                    x, y, w, h = [int(v) for v in bbox]
                    if color == 'B':
                        clr = (255, 0, 0)
                    elif color == 'R':
                        clr = (0, 0, 255)
                    else:
                        clr = (0, 255, 255)
                    cv2.rectangle(output, (x, y), (x + w, y + h), clr, 2)
                    cv2.putText(output, f"{color}{i}", (x, y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, clr, 2)

            cv2.putText(output, f"TRACKING {len(trackers)} balls", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        else:
            # Detection mode
            balls, _, _, _ = detect_balls(frame)

            for ball in balls:
                x, y, w, h = ball['bbox']
                if ball['color'] == 'B':
                    clr = (255, 0, 0)
                elif ball['color'] == 'R':
                    clr = (0, 0, 255)
                else:
                    clr = (0, 255, 255)
                cv2.rectangle(output, (x, y), (x + w, y + h), clr, 2)

            cv2.putText(output, f"DETECT: {len(balls)} balls - press 's' to track",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        cv2.imshow("Ball Tracking", output)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            # Start tracking current detections
            balls, _, _, _ = detect_balls(frame)
            trackers = []
            for ball in balls:
                tracker = cv2.TrackerCSRT_create()
                tracker.init(frame, ball['bbox'])
                trackers.append((tracker, ball['color']))
            tracking = True
            print(f"Tracking {len(trackers)} balls")
        elif key == ord('r'):
            trackers = []
            tracking = False
            print("Reset")

    cap.release()
    cv2.destroyAllWindows()

# %%

"""Shared colored-ball detection helpers."""

import cv2
import numpy as np


YELLOW_HSV_LOWER = np.array([18, 100, 80])
YELLOW_HSV_UPPER = np.array([35, 255, 255])
RED_HSV_LOWER_1 = np.array([0, 130, 70])
RED_HSV_UPPER_1 = np.array([8, 255, 255])
RED_HSV_LOWER_2 = np.array([172, 130, 70])
RED_HSV_UPPER_2 = np.array([180, 255, 255])
BLUE_HSV_LOWER = np.array([102, 120, 80])
BLUE_HSV_UPPER = np.array([128, 255, 255])


def _ball_contours(
    mask, min_area, min_circularity, min_solidity,
    min_extent=0.0, max_aspect_ratio=1.6,
):
    mask = cv2.morphologyEx(
        mask, cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
    )
    mask = cv2.morphologyEx(
        mask, cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)),
        iterations=2,
    )
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    selected = []
    for contour in contours:
        area = cv2.contourArea(contour)
        x, y, w, h = cv2.boundingRect(contour)
        if area < min_area or min(w, h) == 0:
            continue
        perimeter = cv2.arcLength(contour, True)
        circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter else 0
        aspect_ratio = max(w, h) / min(w, h)
        hull_area = cv2.contourArea(cv2.convexHull(contour))
        solidity = area / hull_area if hull_area else 0
        extent = area / (w * h)
        if (
            circularity >= min_circularity
            and aspect_ratio <= max_aspect_ratio
            and solidity >= min_solidity
            and extent >= min_extent
        ):
            selected.append(contour)
    return mask, selected


def detect_red(frame):
    """Return a cleaned red mask and contours that look like balls."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask1 = cv2.inRange(hsv, RED_HSV_LOWER_1, RED_HSV_UPPER_1)
    mask2 = cv2.inRange(hsv, RED_HSV_LOWER_2, RED_HSV_UPPER_2)
    mask = cv2.bitwise_or(mask1, mask2)
    return _ball_contours(
        mask, 1200, 0.0, 0.65, min_extent=0.3, max_aspect_ratio=2.2
    )


def detect_blue(frame):
    """Return a cleaned blue mask and contours that look like balls."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, BLUE_HSV_LOWER, BLUE_HSV_UPPER)
    return _ball_contours(
        mask, 1200, 0.0, 0.65, min_extent=0.3, max_aspect_ratio=2.2
    )


def detect_yellow(frame):
    """Return a cleaned yellow mask and contours that look like balls."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, YELLOW_HSV_LOWER, YELLOW_HSV_UPPER)

    mask, ball_contours = _ball_contours(mask, 500, 0.55, 0.8)

    # A ball touching the table can become one large contour. Recover it only
    # when a circle has strong yellow-pixel coverage, not from shape alone.
    if not ball_contours:
        gray = cv2.medianBlur(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), 5)
        circles = cv2.HoughCircles(
            gray, cv2.HOUGH_GRADIENT, dp=1.2, minDist=30,
            param1=80, param2=35, minRadius=10,
            maxRadius=max(10, min(frame.shape[:2]) // 3),
        )
        if circles is not None:
            height, width = mask.shape
            for cx, cy, radius in np.round(circles[0]).astype(int):
                if not (0 <= cx < width and 0 <= cy < height):
                    continue
                yy, xx = np.ogrid[:height, :width]
                circle = (xx - cx) ** 2 + (yy - cy) ** 2 <= radius * radius
                if np.count_nonzero(circle) == 0:
                    continue
                coverage = np.count_nonzero(mask[circle]) / np.count_nonzero(circle)
                if coverage >= 0.7:
                    contour = cv2.ellipse2Poly(
                        (cx, cy), (radius, radius), 0, 0, 360, 5
                    )
                    contour[:, 0] = np.clip(contour[:, 0], 0, width - 1)
                    contour[:, 1] = np.clip(contour[:, 1], 0, height - 1)
                    ball_contours.append(contour.reshape(-1, 1, 2))
                    break

    return mask, ball_contours

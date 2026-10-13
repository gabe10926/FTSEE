"""
SimpleBallTracker — blue/red/yellow ball detection + per-frame tracking.

Usage:
    tracker = SimpleBallTracker()

    # Detection only (per-frame):
    balls = tracker.detect(frame)
    # balls = [{'bbox': (x, y, w, h), 'color': 'B' | 'R' | 'Y'}, ...]

    # Persistent tracking across frames:
    tracker.start_tracking(frame)       # seeds trackers from current detections
    tracked = tracker.update(frame)     # call every frame after
    tracker.reset()                     # clear trackers
"""

import cv2
import numpy as np

from yellow_ball import (
    BLUE_HSV_LOWER, BLUE_HSV_UPPER, detect_blue,
    RED_HSV_LOWER_1, RED_HSV_UPPER_1, RED_HSV_LOWER_2, RED_HSV_UPPER_2,
    YELLOW_HSV_LOWER, YELLOW_HSV_UPPER, detect_red, detect_yellow,
)


class SimpleBallTracker:
    MIN_BALL_AREA_BLUE_RED = 300
    MIN_BALL_AREA_YELLOW = 500
    MAX_BALL_AREA = 50000
    MAX_BALL_AREA_YELLOW = 120000

    def __init__(self):
        self._trackers = []  # list of (cv2.Tracker, color_char)

    def detect(self, frame):
        """Run color detection on a single frame.

        Returns a list of balls sorted left-to-right:
            [{'bbox': (x, y, w, h), 'color': 'B' | 'R' | 'Y'}, ...]
        """
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        blue_mask, blue_contours = detect_blue(frame)

        red_mask, red_contours = detect_red(frame)

        yellow_mask, yellow_contours = detect_yellow(frame)

        balls = []
        for mask, color, min_area in [
            (blue_mask, 'B', self.MIN_BALL_AREA_BLUE_RED),
            (red_mask, 'R', self.MIN_BALL_AREA_BLUE_RED),
            (yellow_mask, 'Y', self.MIN_BALL_AREA_YELLOW),
        ]:
            contours = (
                yellow_contours if color == 'Y' else
                red_contours if color == 'R' else blue_contours
            )
            for cnt in contours:
                area = cv2.contourArea(cnt)
                max_area = self.MAX_BALL_AREA_YELLOW if color == 'Y' else self.MAX_BALL_AREA
                if min_area < area < max_area:
                    x, y, w, h = cv2.boundingRect(cnt)
                    balls.append({'bbox': (x, y, w, h), 'color': color})

        balls.sort(key=lambda b: b['bbox'][0])
        return balls

    def get_ball_colors(self, frame):
        """Return detected ball colors as a left-to-right string, e.g. 'BRY'."""
        return ''.join(b['color'] for b in self.detect(frame))

    def start_tracking(self, frame):
        """Seed CSRT trackers from the current frame's detections."""
        self._trackers = []
        for ball in self.detect(frame):
            tracker = cv2.TrackerCSRT_create()
            tracker.init(frame, ball['bbox'])
            self._trackers.append((tracker, ball['color']))
        return len(self._trackers)

    def update(self, frame):
        """Update all trackers for the new frame.

        Returns a list of currently-tracked balls:
            [{'bbox': (x, y, w, h), 'color': 'B' | 'R' | 'Y', 'id': int}, ...]
        Balls whose tracker loses lock are omitted for this frame.
        """
        tracked = []
        for i, (tracker, color) in enumerate(self._trackers):
            success, bbox = tracker.update(frame)
            if success:
                x, y, w, h = [int(v) for v in bbox]
                tracked.append({'bbox': (x, y, w, h), 'color': color, 'id': i})
        return tracked

    def reset(self):
        """Clear all active trackers."""
        self._trackers = []

    @property
    def is_tracking(self):
        return len(self._trackers) > 0

"""
SimpleBallTracker — green/purple ball detection + per-frame tracking.

Usage:
    tracker = SimpleBallTracker()

    # Detection only (per-frame):
    balls = tracker.detect(frame)
    # balls = [{'bbox': (x, y, w, h), 'color': 'G' | 'P'}, ...]

    # Persistent tracking across frames:
    tracker.start_tracking(frame)       # seeds trackers from current detections
    tracked = tracker.update(frame)     # call every frame after
    tracker.reset()                     # clear trackers
"""

import cv2
import numpy as np


class SimpleBallTracker:
    # GREEN THRESHOLDS (HSV + YCrCb AND)
    GREEN_HSV_LOWER = np.array([65, 100, 40])
    GREEN_HSV_UPPER = np.array([90, 255, 220])
    GREEN_YCRCB_LOWER = np.array([50, 35, 60])
    GREEN_YCRCB_UPPER = np.array([160, 100, 145])

    # PURPLE THRESHOLDS (YCrCb only)
    PURPLE_YCRCB_LOWER = np.array([80, 135, 130])
    PURPLE_YCRCB_UPPER = np.array([170, 175, 180])

    MIN_BALL_AREA = 300
    MAX_BALL_AREA = 20000

    def __init__(self):
        self._trackers = []  # list of (cv2.Tracker, color_char)

    def detect(self, frame):
        """Run color detection on a single frame.

        Returns a list of balls sorted left-to-right:
            [{'bbox': (x, y, w, h), 'color': 'G' | 'P'}, ...]
        """
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)

        green_hsv = cv2.inRange(hsv, self.GREEN_HSV_LOWER, self.GREEN_HSV_UPPER)
        green_ycrcb = cv2.inRange(ycrcb, self.GREEN_YCRCB_LOWER, self.GREEN_YCRCB_UPPER)
        green_mask = cv2.bitwise_and(green_hsv, green_ycrcb)

        purple_mask = cv2.inRange(ycrcb, self.PURPLE_YCRCB_LOWER, self.PURPLE_YCRCB_UPPER)

        balls = []
        for mask, color in [(green_mask, 'G'), (purple_mask, 'P')]:
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if self.MIN_BALL_AREA < area < self.MAX_BALL_AREA:
                    x, y, w, h = cv2.boundingRect(cnt)
                    balls.append({'bbox': (x, y, w, h), 'color': color})

        balls.sort(key=lambda b: b['bbox'][0])
        return balls

    def get_ball_colors(self, frame):
        """Return detected ball colors as a left-to-right string, e.g. 'GPG'."""
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
            [{'bbox': (x, y, w, h), 'color': 'G' | 'P', 'id': int}, ...]
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

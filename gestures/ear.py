import numpy as np
from collections import deque
import time

# MediaPipe landmark indices for EAR calculation
_LEFT_EYE = [33, 160, 158, 133, 153, 144]
_RIGHT_EYE = [263, 387, 385, 362, 380, 373]

# Upper/lower lip landmarks for mouth aspect ratio
_UPPER_LIP = [13, 312, 311, 310, 82, 81, 80]
_LOWER_LIP = [14, 317, 402, 318, 87, 178, 88]
_LEFT_MOUTH = [61]
_RIGHT_MOUTH = [291]


def _ear(landmarks, indices):
    p1, p2, p3, p4, p5, p6 = landmarks[indices, :2]
    v1 = np.linalg.norm(p2 - p6)
    v2 = np.linalg.norm(p3 - p5)
    h = np.linalg.norm(p1 - p4)
    if h < 1e-6:
        return 1.0
    return (v1 + v2) / (2.0 * h)


def _mar(landmarks):
    upper = landmarks[_UPPER_LIP, :2].mean(axis=0)
    lower = landmarks[_LOWER_LIP, :2].mean(axis=0)
    left = landmarks[_LEFT_MOUTH, :2].mean(axis=0)
    right = landmarks[_RIGHT_MOUTH, :2].mean(axis=0)
    v = np.linalg.norm(upper - lower)
    h = np.linalg.norm(left - right)
    if h < 1e-6:
        return 0.0
    return v / h


class BlinkDetector:
    def __init__(self, threshold=0.20, min_frames=2, max_frames=12):
        self._threshold = threshold
        self._min_frames = min_frames
        self._max_frames = max_frames
        self._both_below = 0
        self._last_trigger = 0.0
        self._cooldown = 0.5

    def update(self, landmarks):
        left_ear = _ear(landmarks, _LEFT_EYE)
        right_ear = _ear(landmarks, _RIGHT_EYE)
        both_closed = left_ear < self._threshold and right_ear < self._threshold

        if both_closed:
            self._both_below += 1
        else:
            if self._min_frames <= self._both_below <= self._max_frames:
                if self._can_trigger():
                    self._both_below = 0
                    return 'blink'
            self._both_below = 0

        return None

    def _can_trigger(self):
        now = time.time()
        if now - self._last_trigger >= self._cooldown:
            self._last_trigger = now
            return True
        return False


class MouthOpenTracker:
    """Returns True every frame the mouth is open (for scroll mode hold)."""
    def __init__(self, threshold=0.40, min_frames=3):
        self._threshold = threshold
        self._min_frames = min_frames
        self._above = 0

    def is_open(self, landmarks):
        mar = _mar(landmarks)
        if mar > self._threshold:
            self._above += 1
        else:
            self._above = 0
        return self._above >= self._min_frames

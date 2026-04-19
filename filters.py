import math
import numpy as np
from filterpy.kalman import KalmanFilter


class OneEuroFilter:
   
    def __init__(self, min_cutoff=0.8, beta=0.004, d_cutoff=1.0, freq=30.0):
        self.min_cutoff = min_cutoff
        self.beta       = beta
        self.d_cutoff   = d_cutoff
        self.freq       = freq
        self._x    = None
        self._dx   = 0.0

    @staticmethod
    def _alpha(cutoff, freq):
        tau = 1.0 / (2.0 * math.pi * cutoff)
        te  = 1.0 / freq
        return 1.0 / (1.0 + tau / te)

    def update(self, x):
        if self._x is None:
            self._x = x
            return x

        dx       = (x - self._x) * self.freq
        a_d      = self._alpha(self.d_cutoff, self.freq)
        self._dx = a_d * dx + (1.0 - a_d) * self._dx

        cutoff = self.min_cutoff + self.beta * abs(self._dx)
        a      = self._alpha(cutoff, self.freq)
        self._x = a * x + (1.0 - a) * self._x
        return self._x

    def reset(self):
        self._x  = None
        self._dx = 0.0


class SpikeFilter:
    """Reject single-frame outliers — if a value jumps by more than
    `threshold` units in one frame, hold the previous value instead."""
    def __init__(self, threshold=6.0):
        self.threshold = threshold
        self._prev = None

    def update(self, value):
        if self._prev is None:
            self._prev = value
            return value
        if abs(value - self._prev) > self.threshold:
            return self._prev   # discard outlier frame
        self._prev = value
        return value

    def reset(self):
        self._prev = None


class EMA:
    def __init__(self, alpha=0.3):
        self.alpha  = alpha
        self._value = None

    def update(self, value):
        if self._value is None:
            self._value = value
        else:
            self._value = self.alpha * value + (1 - self.alpha) * self._value
        return self._value

    def reset(self):
        self._value = None


class KalmanCursor:
    def __init__(self, process_noise=0.01, measurement_noise=0.1, dt=1/30, velocity_decay=0.8):
        kf = KalmanFilter(dim_x=4, dim_z=2)
        kf.F = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, velocity_decay, 0],
            [0, 0, 0, velocity_decay],
        ])
        kf.H = np.array([[1, 0, 0, 0],
                         [0, 1, 0, 0]])
        kf.R *= measurement_noise
        kf.Q *= process_noise
        kf.P *= 10
        self._kf          = kf
        self._initialized = False

    def update(self, x, y):
        z = np.array([[x], [y]])
        if not self._initialized:
            self._kf.x    = np.array([[x], [y], [0.0], [0.0]])
            self._initialized = True
        self._kf.predict()
        self._kf.update(z)
        return float(self._kf.x[0]), float(self._kf.x[1])

    def reset(self):
        self._initialized = False


class CursorSettler:
    """Adaptive dead-zone that grows as the cursor holds still.

    When the user stops moving, the gate radius ramps up over time, making the
    cursor progressively stickier and easier to pinpoint.  A deliberate movement
    (exceeding the current gate) instantly resets the gate to its minimum so the
    cursor feels responsive again.
    """
    def __init__(self, gate_min=3.0, gate_max=40.0, ramp_frames=20, centroid_window=10):
        self.gate_min = gate_min
        self.gate_max = gate_max
        self.ramp_frames = ramp_frames
        self.centroid_window = centroid_window
        self._still_frames = 0
        self._anchor_x = None
        self._anchor_y = None
        self._history = []

    def update(self, x, y):
        gate = self.gate_min + (self.gate_max - self.gate_min) * min(self._still_frames / self.ramp_frames, 1.0)

        if self._anchor_x is None:
            self._anchor_x, self._anchor_y = x, y
            self._history = [(x, y)]
            return x, y

        dx = x - self._anchor_x
        dy = y - self._anchor_y
        dist = (dx * dx + dy * dy) ** 0.5

        if dist < gate:
            self._still_frames = min(self._still_frames + 1, self.ramp_frames)
            self._history.append((x, y))
            if len(self._history) > self.centroid_window:
                self._history = self._history[-self.centroid_window:]
            cx = sum(p[0] for p in self._history) / len(self._history)
            cy = sum(p[1] for p in self._history) / len(self._history)
            self._anchor_x, self._anchor_y = cx, cy
            return int(cx), int(cy)
        else:
            self._still_frames = 0
            self._anchor_x, self._anchor_y = x, y
            self._history = [(x, y)]
            return x, y

    def reset(self):
        self._still_frames = 0
        self._anchor_x = None
        self._anchor_y = None
        self._history = []

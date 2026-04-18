import math
import numpy as np
from filterpy.kalman import KalmanFilter


class OneEuroFilter:
    """
    Adaptive low-pass filter for pointer control.
    At rest: aggressively damps jitter (low cutoff).
    During fast movement: nearly transparent (cutoff rises with speed).
    Geurts et al. 2012 — "One Euro Filter".
    """
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

import numpy as np

# Large landmark sets for robust centroids — averaging many points
# reduces per-landmark jitter by sqrt(N).
_LEFT_EYE_IDX = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
_RIGHT_EYE_IDX = [263, 249, 390, 373, 374, 380, 381, 382, 362, 398, 384, 385, 386, 387, 388, 466]
_NOSE_IDX = [6, 197, 195, 168]  # upper nose bridge only — jaw doesn't affect these
# Forehead / brow ridge — stable skull landmarks
_FOREHEAD_IDX = [10, 67, 69, 104, 108, 109, 151, 297, 299, 333, 337, 338]

_SCALE = 90.0


class PoseEstimator:
    """
    Geometric head-pose from large-centroid landmarks.
    Uses only skull-attached points so mouth/jaw gestures don't move the cursor.
    Landmarks are temporally smoothed before pose calculation.
    """

    def __init__(self, landmark_alpha=0.10):
        self._landmark_alpha = landmark_alpha
        self._smooth_landmarks = None

    def estimate(self, landmarks, frame_shape):
        if self._smooth_landmarks is None:
            self._smooth_landmarks = landmarks.copy()
        else:
            a = self._landmark_alpha
            self._smooth_landmarks = a * landmarks + (1 - a) * self._smooth_landmarks

        lm = self._smooth_landmarks

        nose = lm[_NOSE_IDX, :2].mean(axis=0)
        left_eye = lm[_LEFT_EYE_IDX, :2].mean(axis=0)
        right_eye = lm[_RIGHT_EYE_IDX, :2].mean(axis=0)
        forehead = lm[_FOREHEAD_IDX, :2].mean(axis=0)

        eye_mid = (left_eye + right_eye) / 2.0
        eye_vec = right_eye - left_eye
        eye_dist = float(np.linalg.norm(eye_vec))

        if eye_dist < 1e-6:
            return 0.0, 0.0, 0.0

        # Yaw: horizontal offset of nose from eye midpoint
        dx = (nose[0] - eye_mid[0]) / eye_dist
        yaw = float(dx * _SCALE)

        # Pitch: use forehead-to-nose vertical distance relative to eye width
        # More stable than nose-to-eye-mid because forehead doesn't move with expressions
        dy = (nose[1] - forehead[1]) / eye_dist
        # Normalize so that neutral dy maps to ~0 after baseline subtraction
        pitch = float(-dy * _SCALE)

        roll = float(np.degrees(np.arctan2(-eye_vec[1], eye_vec[0])))

        return pitch, yaw, roll

    def reset(self):
        self._smooth_landmarks = None

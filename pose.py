import numpy as np

# Large landmark sets for robust centroids — averaging many points
# reduces per-landmark jitter by sqrt(N).
_LEFT_EYE_IDX = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
_RIGHT_EYE_IDX = [263, 249, 390, 373, 374, 380, 381, 382, 362, 398, 384, 385, 386, 387, 388, 466]
_NOSE_IDX = [6, 197, 195, 168]  # upper nose bridge only — jaw doesn't affect these
# Forehead / brow ridge — stable skull landmarks
_FOREHEAD_IDX = [10, 67, 69, 104, 108, 109, 151, 297, 299, 333, 337, 338]


class PoseEstimator:
    """
    3D geometric head-pose from MediaPipe landmarks.
    Uses full 3D coordinates to decouple pitch/yaw axes.
    Uses only skull-attached points so mouth/jaw gestures don't move the cursor.
    """

    def __init__(self, landmark_alpha=0.45):
        self._landmark_alpha = landmark_alpha
        self._smooth_landmarks = None

    def estimate(self, landmarks, frame_shape):
        if self._smooth_landmarks is None:
            self._smooth_landmarks = landmarks.copy()
        else:
            a = self._landmark_alpha
            self._smooth_landmarks = a * landmarks + (1 - a) * self._smooth_landmarks

        lm = self._smooth_landmarks

        # Use all 3 coordinates (x, y, z)
        nose = lm[_NOSE_IDX, :3].mean(axis=0)
        left_eye = lm[_LEFT_EYE_IDX, :3].mean(axis=0)
        right_eye = lm[_RIGHT_EYE_IDX, :3].mean(axis=0)
        forehead = lm[_FOREHEAD_IDX, :3].mean(axis=0)

        eye_mid = (left_eye + right_eye) / 2.0
        eye_vec_3d = right_eye - left_eye
        eye_dist = float(np.linalg.norm(eye_vec_3d[:2]))  # 2D distance for normalization

        if eye_dist < 1e-6:
            return 0.0, 0.0, 0.0

        # Yaw: angle of nose offset from eye midpoint in the XZ plane
        # Using atan2 gives true angle independent of pitch
        dx = nose[0] - eye_mid[0]
        dz = nose[2] - eye_mid[2]
        yaw = float(np.degrees(np.arctan2(dx, -dz + 1e-6)))

        # Pitch: angle of forehead-to-nose vector in the YZ plane
        # This decouples pitch from yaw by using depth
        fwd_vec = nose - forehead
        pitch = float(np.degrees(np.arctan2(-fwd_vec[1], -fwd_vec[2] + 1e-6)))

        # Roll: rotation around the forward axis (from eye vector)
        roll = float(np.degrees(np.arctan2(-eye_vec_3d[1], eye_vec_3d[0])))

        return pitch, yaw, roll

    def reset(self):
        self._smooth_landmarks = None

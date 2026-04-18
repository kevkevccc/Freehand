import cv2
import numpy as np

_LANDMARK_IDS = [1, 152, 33, 263, 61, 291]

_MODEL_POINTS = np.array([
    (0.0,   0.0,    0.0),
    (0.0,  -63.6,  -12.5),
    (-43.3, 32.7,  -26.0),
    (43.3,  32.7,  -26.0),
    (-28.9, -28.9, -24.1),
    (28.9,  -28.9, -24.1),
], dtype=np.float64)


class PoseEstimator:
    def __init__(self):
        self._prev_rvec = None
        self._prev_tvec = None

    def estimate(self, landmarks, frame_shape):
        h, w = frame_shape[:2]
        focal = w
        camera_matrix = np.array([
            [focal, 0,     w / 2],
            [0,     focal, h / 2],
            [0,     0,     1    ],
        ], dtype=np.float64)

        image_points = np.array([
            (landmarks[i][0] * w, landmarks[i][1] * h)
            for i in _LANDMARK_IDS
        ], dtype=np.float64)

        if self._prev_rvec is not None:
            ok, rvec, tvec = cv2.solvePnP(
                _MODEL_POINTS, image_points, camera_matrix, np.zeros((4, 1)),
                rvec=self._prev_rvec, tvec=self._prev_tvec,
                useExtrinsicGuess=True,
                flags=cv2.SOLVEPNP_ITERATIVE,
            )
        else:
            ok, rvec, tvec = cv2.solvePnP(
                _MODEL_POINTS, image_points, camera_matrix, np.zeros((4, 1)),
                flags=cv2.SOLVEPNP_EPNP,
            )

        if not ok:
            return 0.0, 0.0, 0.0

        self._prev_rvec = rvec.copy()
        self._prev_tvec = tvec.copy()

        rmat, _ = cv2.Rodrigues(rvec)
        face_fwd = rmat @ np.array([0.0, 0.0, 1.0])
        face_up  = rmat @ np.array([0.0, 1.0, 0.0])

        yaw   = float(np.degrees(np.arctan2( face_fwd[0], -face_fwd[2])))
        pitch = float(np.degrees(np.arctan2(-face_fwd[1], -face_fwd[2])))
        roll  = float(np.degrees(np.arctan2( face_up[0],  -face_up[1])))

        return pitch, yaw, roll

    def reset(self):
        self._prev_rvec = None
        self._prev_tvec = None

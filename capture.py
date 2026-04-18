import cv2
import mediapipe as mp
import numpy as np


class FrameCapture:
    def __init__(self, camera_index=0, width=640, height=480, target_fps=30):
        self.camera_index = camera_index
        self.width = width
        self.height = height
        self.target_fps = target_fps
        self._cap = None
        self._face_mesh = None

    def start(self):
        self._cap = cv2.VideoCapture(self.camera_index)
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self._cap.set(cv2.CAP_PROP_FPS, self.target_fps)

        self._face_mesh = mp.solutions.face_mesh.FaceMesh(
            refine_landmarks=True,
            max_num_faces=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

    def read(self):
        ok, frame = self._cap.read()
        if not ok:
            return None, None

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self._face_mesh.process(rgb)

        if not result.multi_face_landmarks:
            return None, None

        lm = result.multi_face_landmarks[0].landmark
        landmarks = np.array([[p.x, p.y, p.z] for p in lm], dtype=np.float32)
        return frame, landmarks

    def release(self):
        if self._cap:
            self._cap.release()
        if self._face_mesh:
            self._face_mesh.close()

import cv2
import mediapipe as mp

mp_face = mp.solutions.face_mesh
face = mp_face.FaceMesh(refine_landmarks=True, max_num_faces=1)
draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
while True:
    ok, frame = cap.read()
    if not ok:
        break
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res = face.process(rgb)
    if res.multi_face_landmarks:
        for lm in res.multi_face_landmarks:
            draw.draw_landmarks(frame, lm, mp_face.FACEMESH_TESSELATION)
    cv2.imshow("test", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()

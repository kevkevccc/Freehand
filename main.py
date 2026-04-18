import argparse
import time

import cv2
from pynput.mouse import Controller as MouseController

from capture import FrameCapture
from pose import PoseEstimator

SCREEN_W = 1792
SCREEN_H = 1120

# Degrees of head rotation that map to full screen edge — tuned to your range
YAW_RANGE   = 28.0   # ±28° yaw   → left/right screen edge
PITCH_UP    =  6.0   # +6°  pitch → top edge
PITCH_DOWN  = 14.0   # -14° pitch → bottom edge

FACE_LOST_FREEZE_S = 1.0  # freeze cursor if face gone longer than this


def yaw_pitch_to_cursor(yaw, pitch):
    x = SCREEN_W / 2 + (yaw / YAW_RANGE) * (SCREEN_W / 2)
    if pitch >= 0:
        y = SCREEN_H / 2 - (pitch / PITCH_UP)   * (SCREEN_H / 2)
    else:
        y = SCREEN_H / 2 - (pitch / PITCH_DOWN)  * (SCREEN_H / 2)
    x = max(0, min(SCREEN_W - 1, x))
    y = max(0, min(SCREEN_H - 1, y))
    return int(x), int(y)


def run_raw(capture, estimator, debug=False):
    mouse = MouseController()
    last_face_time = time.time()
    print("Head mouse running — look around to move cursor. Ctrl-C to quit.")

    try:
        while True:
            frame, landmarks = capture.read()

            if landmarks is None:
                if time.time() - last_face_time > FACE_LOST_FREEZE_S:
                    pass  # freeze: skip mouse.position update
                if debug and frame is not None:
                    cv2.imshow("debug", frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                continue

            last_face_time = time.time()
            pitch, yaw, roll = estimator.estimate(landmarks, frame.shape)
            x, y = yaw_pitch_to_cursor(yaw, pitch)
            mouse.position = (x, y)

            if debug:
                cv2.putText(frame, f"yaw={yaw:+.1f} pitch={pitch:+.1f}",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, f"cursor=({x},{y})",
                            (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.imshow("debug", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

    except KeyboardInterrupt:
        pass
    finally:
        if debug:
            cv2.destroyAllWindows()
        print("\nStopped.")


def run_pose_debug(capture, estimator):
    print("Pose debug — move your head. Ctrl-C to quit.\n")
    try:
        while True:
            frame, landmarks = capture.read()
            if landmarks is None:
                continue
            pitch, yaw, roll = estimator.estimate(landmarks, frame.shape)
            print(f"\rpitch={pitch:+6.1f}  yaw={yaw:+6.1f}  roll={roll:+6.1f}", end="", flush=True)
    except KeyboardInterrupt:
        print()


def parse_args():
    p = argparse.ArgumentParser(description="Head Mouse")
    p.add_argument("--run",        action="store_true", help="Start cursor control")
    p.add_argument("--debug-pose", action="store_true", help="Print pose angles only")
    p.add_argument("--debug",      action="store_true", help="Show camera preview while running")
    return p.parse_args()


def main():
    args = parse_args()
    capture = FrameCapture()
    capture.start()
    estimator = PoseEstimator()

    try:
        if args.debug_pose:
            run_pose_debug(capture, estimator)
        elif args.run:
            run_raw(capture, estimator, debug=args.debug)
        else:
            print("Use --run to start, --debug-pose to check angles, --debug to add preview.")
    finally:
        capture.release()


if __name__ == "__main__":
    main()

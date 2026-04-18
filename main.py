import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import argparse
import time

import cv2
import numpy as np
from pynput.mouse import Controller as MouseController

from capture import FrameCapture
from filters import OneEuroFilter, KalmanCursor
from pose import PoseEstimator
from gestures.ear import BlinkDetector, MouthOpenTracker
from actions import left_click, right_click, scroll
from voice import VoiceTyper

SCREEN_W = 1792
SCREEN_H = 1120

YAW_RANGE  = 7.0
PITCH_UP   =  4.5
PITCH_DOWN =  6.0

DEAD_ZONE          = 0.3
EDGE_EXPONENT      = 1.0  # linear — no edge compression
FACE_LOST_FREEZE_S = 1.0
NEUTRAL_FRAMES     = 60


def capture_neutral(capture, estimator):
    print("Hold your head in a neutral position...", end="", flush=True)
    pitches, yaws = [], []
    while len(pitches) < NEUTRAL_FRAMES:
        frame, landmarks = capture.read()
        if landmarks is None:
            continue
        pitch, yaw, _ = estimator.estimate(landmarks, frame.shape)
        pitches.append(pitch)
        yaws.append(yaw)
    neutral = (float(np.median(pitches)), float(np.median(yaws)))
    print(f"  done  (pitch={neutral[0]:+.1f}  yaw={neutral[1]:+.1f})")
    return neutral


def _apply_dead_zone(v, zone):
    if abs(v) < zone:
        return 0.0
    return v - zone * (1 if v > 0 else -1)


def _power_map(v, exponent):
    """Sublinear mapping: sign(v) * |v|^exponent. Compresses large values."""
    return (1 if v >= 0 else -1) * abs(v) ** exponent


def yaw_pitch_to_cursor(yaw, pitch):
    yaw   = _apply_dead_zone(yaw,   DEAD_ZONE)
    pitch = _apply_dead_zone(pitch, DEAD_ZONE)

    # Normalize to -1..1, apply power curve, then map to pixels
    nx = max(-1.0, min(1.0, yaw / YAW_RANGE))
    nx = _power_map(nx, EDGE_EXPONENT)

    if pitch >= 0:
        ny = max(-1.0, min(1.0, pitch / PITCH_UP))
    else:
        ny = max(-1.0, min(1.0, pitch / PITCH_DOWN))
    ny = _power_map(ny, EDGE_EXPONENT)

    x = SCREEN_W / 2 + nx * (SCREEN_W / 2)
    y = SCREEN_H / 2 - ny * (SCREEN_H / 2)
    return int(max(0, min(SCREEN_W - 1, x))), int(max(0, min(SCREEN_H - 1, y)))


def run_raw(capture, estimator, debug=False):
    neutral_pitch, neutral_yaw = capture_neutral(capture, estimator)

    mouse = MouseController()

    # Pipeline: raw angle → One Euro (jitter removal) → pixel map → Kalman (prediction/smoothing)
    # One Euro: min_cutoff=1.5 gives good jitter reduction at rest,
    # beta=0.007 keeps it responsive without overshooting
    euro_yaw   = OneEuroFilter(min_cutoff=0.15, beta=0.05)
    euro_pitch = OneEuroFilter(min_cutoff=0.15, beta=0.05)
    kalman     = KalmanCursor(process_noise=0.01, measurement_noise=35.0, velocity_decay=0.3)

    blink_detector = BlinkDetector()
    mouth_tracker = MouthOpenTracker()
    scroll_mode = False
    scroll_accumulator = 0.0
    SCROLL_SPEED = 0.4
    SCROLL_DEAD_ZONE = 1.5

    voice = VoiceTyper()
    voice.start()

    last_face_time = time.time()
    prev_x, prev_y = SCREEN_W // 2, SCREEN_H // 2
    VELOCITY_GATE = 8
    print("Head mouse running — look around to move cursor. Ctrl-C to quit.")
    print("  Blink = left click | Hold mouth open = scroll | Voice typing active")

    try:
        while True:
            frame, landmarks = capture.read()

            if landmarks is None:
                if time.time() - last_face_time > FACE_LOST_FREEZE_S:
                    euro_yaw.reset()
                    euro_pitch.reset()
                    kalman.reset()
                    estimator.reset()
                if debug and frame is not None:
                    cv2.imshow("debug", frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                continue

            last_face_time = time.time()
            pitch, yaw, _ = estimator.estimate(landmarks, frame.shape)

            rel_yaw   = yaw   - neutral_yaw
            rel_pitch = pitch - neutral_pitch

            # Distance-adaptive filtering: more smoothing at screen edges
            # where landmark noise is amplified by perspective
            dist = (rel_yaw / YAW_RANGE) ** 2 + (rel_pitch / max(PITCH_UP, PITCH_DOWN)) ** 2
            dist = min(dist, 1.0)  # 0 at center, 1 at edge
            adaptive_cutoff = 0.15 * (1.0 - 0.5 * dist)
            euro_yaw.min_cutoff = adaptive_cutoff
            euro_pitch.min_cutoff = adaptive_cutoff

            smooth_yaw   = euro_yaw.update(rel_yaw)
            smooth_pitch = euro_pitch.update(rel_pitch)

            raw_x, raw_y = yaw_pitch_to_cursor(smooth_yaw, smooth_pitch)
            kx, ky = kalman.update(raw_x, raw_y)
            x, y = int(kx), int(ky)
            x = max(0, min(SCREEN_W - 1, x))
            y = max(0, min(SCREEN_H - 1, y))

            was_scrolling = scroll_mode
            if voice.is_speaking:
                scroll_mode = False
            else:
                scroll_mode = mouth_tracker.is_open(landmarks)

            if scroll_mode:
                scroll_pitch = _apply_dead_zone(smooth_pitch, SCROLL_DEAD_ZONE)
                scroll_accumulator += scroll_pitch * SCROLL_SPEED
                scroll_amount = int(scroll_accumulator)
                if scroll_amount != 0:
                    scroll(scroll_amount)
                    scroll_accumulator -= scroll_amount
            else:
                scroll_accumulator = 0.0
                dx = x - prev_x
                dy = y - prev_y
                if dx * dx + dy * dy >= VELOCITY_GATE * VELOCITY_GATE:
                    mouse.position = (x, y)
                    prev_x, prev_y = x, y

            blink = blink_detector.update(landmarks)
            if blink and not scroll_mode:
                left_click()

            if debug:
                cv2.putText(frame, f"yaw={smooth_yaw:+.1f}  pitch={smooth_pitch:+.1f}",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                mode = "SCROLL" if scroll_mode else "CURSOR"
                cv2.putText(frame, f"cursor=({x},{y})  [{mode}]",
                            (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                if scroll_mode:
                    cv2.putText(frame, f"scroll_pitch={smooth_pitch:+.2f}",
                                (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.putText(frame, f"voice: {voice.status}",
                            (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 0), 2)
                cv2.imshow("debug", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

    except KeyboardInterrupt:
        pass
    finally:
        voice.stop()
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
            pitch, yaw, _ = estimator.estimate(landmarks, frame.shape)
            print(f"\rpitch={pitch:+6.1f}  yaw={yaw:+6.1f}", end="", flush=True)
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

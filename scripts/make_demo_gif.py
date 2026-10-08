#!/usr/bin/env python3
"""
Render assets/demo.gif: pose landmarks, knee angle and a live squat rep count
drawn by this repo's PoseEstimator, SpatialReasoning and RepCounter on the
bundled sample clip (assets/sample_squat.webm, CC BY 3.0, see assets/README.md).

    python scripts/make_demo_gif.py
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from pose_estimator import PoseEstimator  # noqa: E402
from spatial_reasoning import RepCounter, SpatialReasoning  # noqa: E402


def draw_panel(frame, knee_angle, reps, phase):
    """Draw a compact status panel in the top-left corner."""
    overlay = frame.copy()
    cv2.rectangle(overlay, (16, 16), (420, 150), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, f"Squat reps: {reps}", (32, 62), font, 1.2, (255, 255, 255), 3, cv2.LINE_AA)
    cv2.putText(frame, f"Right knee: {knee_angle:5.1f} deg", (32, 104), font, 0.9, (120, 220, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, f"Phase: {phase}", (32, 136), font, 0.8, (180, 255, 180), 2, cv2.LINE_AA)
    return frame


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--input", default=str(ROOT / "assets" / "sample_squat.webm"))
    parser.add_argument("--output", default=str(ROOT / "assets" / "demo.gif"))
    parser.add_argument("--width", type=int, default=480)
    parser.add_argument("--every", type=int, default=3, help="Keep every Nth frame")
    parser.add_argument("--joint", default="right_knee_angle")
    parser.add_argument("--down", type=float, default=100.0)
    parser.add_argument("--up", type=float, default=160.0)
    args = parser.parse_args()

    estimator = PoseEstimator(static_image_mode=False, model_complexity=1)
    spatial = SpatialReasoning()
    counter = RepCounter(smoothing_window=3)

    cap = cv2.VideoCapture(args.input)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    frames, index, detected = [], 0, 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        success, landmarks = estimator.estimate_pose(frame, int(index * 1000 / fps))
        index += 1
        if success:
            detected += 1
            points = {k: np.array([v.x, v.y, v.z, v.visibility]) for k, v in landmarks.items()}
            angles = spatial.compute_all_joint_angles(points)
            angle = angles[args.joint].angle_degrees
            reps, _ = counter.update(angle, min_angle=args.down, max_angle=args.up)
            frame = estimator.draw_landmarks(frame, landmarks, circle_radius=6, line_thickness=4)
            frame = draw_panel(frame, angle, reps, "down" if counter.in_rep else "up")
        if index % args.every:
            continue
        h, w = frame.shape[:2]
        small = cv2.resize(frame, (args.width, int(h * args.width / w)), interpolation=cv2.INTER_AREA)
        frames.append(Image.fromarray(cv2.cvtColor(small, cv2.COLOR_BGR2RGB)))
    estimator.close()

    frames = [f.quantize(colors=64, method=Image.Quantize.MEDIANCUT) for f in frames]
    frames[0].save(
        args.output, save_all=True, append_images=frames[1:],
        duration=int(1000 * args.every / fps), loop=0, optimize=True,
    )
    print(f"{index} frames, pose found in {detected}, reps counted: {counter.get_rep_count()}")
    print(f"Wrote {args.output} ({len(frames)} frames)")


if __name__ == "__main__":
    main()

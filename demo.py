#!/usr/bin/env python3
"""
Real-Time Pose Estimation Demo
Supports webcam or video file input with real-time visualization.
Can run in headless mode (saves output video) if no display is available.
"""

import cv2
import numpy as np
import argparse
import sys
import os
from pathlib import Path

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from pose_estimator import PoseEstimator
from spatial_reasoning import SpatialReasoning, RepCounter
from activity_classifier import ActivityClassifier
from metrics import FPSBenchmark


def run_pose_estimation_demo(
    input_source: str = '0',
    output_video: str = None,
    headless: bool = False,
    activity_classifier: bool = False,
    max_frames: int = None,
    classifier_weights: str = None,
    rep_joint: str = 'left_elbow_angle',
    rep_min: float = 60.0,
    rep_max: float = 170.0,
    model_complexity: int = 1,
):
    """
    Run pose estimation demo on webcam or video file.

    Args:
        input_source: '0' for webcam, or path to video file.
        output_video: Path to save output video (optional).
        headless: If True, don't display window (for servers).
        activity_classifier: If True, run the optional LSTM activity classifier.
        max_frames: Maximum frames to process (None for unlimited).
        classifier_weights: Path to a Keras model trained with ActivityClassifier.train().
        rep_joint: Joint angle used for rep counting (e.g. left_knee_angle for squats).
        rep_min: Angle below which a rep is "down".
        rep_max: Angle above which a rep is completed.
        model_complexity: 0 (lite), 1 (full) or 2 (heavy) pose model.
    """
    print("=" * 60)
    print("Real-Time Human Pose Estimation & Spatial Reasoning")
    print("=" * 60)

    # Initialize components
    print("Initializing pose estimator...")
    pose_estimator = PoseEstimator(
        static_image_mode=False,
        model_complexity=model_complexity,
        smooth_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    print("Initializing spatial reasoning...")
    spatial_reasoning = SpatialReasoning(history_size=30)
    rep_counter = RepCounter(smoothing_window=5)

    print("Initializing FPS benchmark...")
    fps_benchmark = FPSBenchmark(window_size=30)

    activity_clf = None
    if activity_classifier:
        print("Initializing activity classifier...")
        try:
            activity_clf = ActivityClassifier()
            if classifier_weights:
                activity_clf.load_model(classifier_weights)
            else:
                print(
                    "Warning: no --classifier-weights given. The LSTM is untrained, "
                    "so its labels fall back to a simple motion heuristic."
                )
        except Exception as e:
            print(f"Warning: Could not initialize activity classifier: {e}")
            activity_clf = None

    # Open video source
    print(f"Opening video source: {input_source}")
    is_webcam = input_source == '0'
    if is_webcam:
        cap = cv2.VideoCapture(0)
    else:
        cap = cv2.VideoCapture(input_source)

    if not cap.isOpened():
        print(f"Error: Could not open video source {input_source}")
        sys.exit(1)

    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Resolution: {frame_width}x{frame_height}")
    print(f"FPS: {fps}")
    print(f"Total frames: {total_frames}")

    # Setup video writer if output is specified
    video_writer = None
    if output_video:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(
            output_video,
            fourcc,
            fps,
            (frame_width, frame_height),
        )
        print(f"Output video: {output_video}")

    # Main processing loop
    frame_count = 0
    detected_count = 0

    try:
        while True:
            # Check frame limit
            if max_frames and frame_count >= max_frames:
                break

            ret, frame = cap.read()
            if not ret:
                print("End of video reached.")
                break

            frame_count += 1
            frame_start = fps_benchmark.start_frame()

            # Estimate pose (video files use their own timestamps)
            timestamp_ms = None if is_webcam else int(frame_count * 1000 / fps)
            success, landmarks = pose_estimator.estimate_pose(frame, timestamp_ms)

            if success:
                detected_count += 1

                # Convert landmarks dict to numpy array for spatial reasoning
                landmarks_array = {}
                for name, lm in landmarks.items():
                    landmarks_array[name] = np.array([lm.x, lm.y, lm.z, lm.visibility])

                # Compute angles
                angles = spatial_reasoning.compute_all_joint_angles(
                    landmarks_array, use_2d=False
                )

                # Compute velocities
                velocities = spatial_reasoning.compute_joint_velocity(
                    landmarks_array
                )

                # Classify activity
                activity, activity_conf = spatial_reasoning.classify_activity(
                    angles, velocities
                )

                # Update activity classifier if available
                if activity_clf:
                    pose_vector = spatial_reasoning.get_pose_vector(
                        landmarks_array
                    )
                    activity_clf.add_frame_to_sequence(pose_vector)
                    if activity_clf.get_sequence_progress() >= 1.0:
                        activity, activity_conf, _ = activity_clf.predict_activity()

                # Update rep counter on the chosen joint angle
                if rep_joint in angles:
                    rep_count, rep_completed = rep_counter.update(
                        angles[rep_joint].angle_degrees,
                        min_angle=rep_min,
                        max_angle=rep_max,
                    )
                else:
                    rep_count = rep_counter.get_rep_count()

                # Draw landmarks and connections
                frame = pose_estimator.draw_landmarks(
                    frame,
                    landmarks,
                    draw_connections=True,
                )

                # Draw information overlays
                draw_info_overlay(
                    frame,
                    pose_estimator.get_fps(),
                    activity,
                    activity_conf,
                    angles,
                    rep_count,
                    detected_count,
                    frame_count,
                    rep_joint=rep_joint,
                )

            else:
                # No pose detected
                draw_no_detection_overlay(frame, frame_count)

            fps_benchmark.end_frame(frame_start)

            # Write output video
            if video_writer:
                video_writer.write(frame)

            # Display (if not headless)
            if not headless:
                cv2.imshow('Pose Estimation', frame)

                # Press 'q' to quit, 'p' to pause
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("Quit requested by user.")
                    break
                elif key == ord('p'):
                    cv2.waitKey(0)

            # Progress
            if frame_count % 30 == 0:
                detection_rate = (detected_count / frame_count * 100) if frame_count > 0 else 0
                print(
                    f"Frame {frame_count}/{total_frames} | "
                    f"Detection rate: {detection_rate:.1f}% | "
                    f"FPS: {fps_benchmark.get_fps():.1f}"
                )

    except KeyboardInterrupt:
        print("Interrupted by user.")

    finally:
        # Cleanup
        print("Cleaning up...")
        cap.release()
        if video_writer:
            video_writer.release()
        if not headless:
            cv2.destroyAllWindows()
        pose_estimator.close()

        # Print statistics
        stats = fps_benchmark.get_statistics()
        print("\n" + "=" * 60)
        print("FINAL STATISTICS")
        print("=" * 60)
        print(f"Total frames processed: {frame_count}")
        print(f"Frames with pose detected: {detected_count}")
        print(f"Detection rate: {(detected_count / max(frame_count, 1) * 100):.1f}%")
        print(f"Average FPS: {stats.get('fps', 0):.2f}")
        print(f"Avg frame time: {stats.get('avg_frame_time_ms', 0):.2f} ms")
        print(f"Min frame time: {stats.get('min_frame_time_ms', 0):.2f} ms")
        print(f"Max frame time: {stats.get('max_frame_time_ms', 0):.2f} ms")
        print("=" * 60)

        if output_video:
            print(f"Output saved to: {output_video}")


def draw_info_overlay(
    frame,
    fps,
    activity,
    activity_conf,
    angles,
    rep_count,
    detected_count,
    frame_count,
    rep_joint=None,
):
    """Draw text information overlay on frame."""
    h, w = frame.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    color = (0, 255, 0)
    thickness = 2

    # FPS
    cv2.putText(
        frame,
        f"FPS: {fps:.1f}",
        (10, 30),
        font,
        font_scale,
        color,
        thickness,
    )

    # Activity
    cv2.putText(
        frame,
        f"Activity: {activity} ({activity_conf:.2f})",
        (10, 60),
        font,
        font_scale,
        color,
        thickness,
    )

    # Rep count
    cv2.putText(
        frame,
        f"Reps: {rep_count}",
        (10, 90),
        font,
        font_scale,
        color,
        thickness,
    )

    # Angles
    y_offset = 120
    shown = [rep_joint] if rep_joint in angles else []
    shown += [name for name in angles if name not in shown][:4 - len(shown)]
    for joint_name in shown:
        angle_obj = angles[joint_name]
        cv2.putText(
            frame,
            f"{joint_name}: {angle_obj.angle_degrees:.1f} deg",
            (10, y_offset),
            font,
            0.5,
            (100, 200, 255),
            1,
        )
        y_offset += 25

    # Frame counter (top right)
    cv2.putText(
        frame,
        f"Frame: {frame_count}",
        (w - 150, 30),
        font,
        font_scale,
        color,
        thickness,
    )


def draw_no_detection_overlay(frame, frame_count):
    """Draw overlay when no pose is detected."""
    h, w = frame.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX

    cv2.putText(
        frame,
        "No pose detected",
        (w // 2 - 100, h // 2),
        font,
        1.0,
        (0, 0, 255),
        2,
    )


def main():
    """Parse arguments and run demo."""
    parser = argparse.ArgumentParser(
        description="Real-Time Pose Estimation Demo"
    )
    parser.add_argument(
        '--input',
        type=str,
        default='0',
        help="Input source: '0' for webcam or path to video file",
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help="Output video file path",
    )
    parser.add_argument(
        '--headless',
        action='store_true',
        help="Run without display (for servers)",
    )
    parser.add_argument(
        '--classifier',
        action='store_true',
        help="Enable the optional LSTM activity classifier (needs TensorFlow)",
    )
    parser.add_argument(
        '--classifier-weights',
        type=str,
        default=None,
        help="Keras model file trained with ActivityClassifier.train()",
    )
    parser.add_argument(
        '--rep-joint',
        type=str,
        default='left_elbow_angle',
        help="Joint angle to count reps on, e.g. left_knee_angle for squats",
    )
    parser.add_argument('--rep-min', type=float, default=60.0,
                        help="Angle (deg) below which the rep is 'down'")
    parser.add_argument('--rep-max', type=float, default=170.0,
                        help="Angle (deg) above which the rep is completed")
    parser.add_argument('--model-complexity', type=int, default=1, choices=[0, 1, 2],
                        help="Pose model: 0=lite, 1=full, 2=heavy")
    parser.add_argument(
        '--max-frames',
        type=int,
        default=None,
        help="Maximum frames to process",
    )

    args = parser.parse_args()

    # Validate input
    if args.input != '0' and not os.path.exists(args.input):
        print(f"Error: Input file not found: {args.input}")
        sys.exit(1)

    run_pose_estimation_demo(
        input_source=args.input,
        output_video=args.output,
        headless=args.headless,
        activity_classifier=args.classifier,
        max_frames=args.max_frames,
        classifier_weights=args.classifier_weights,
        rep_joint=args.rep_joint,
        rep_min=args.rep_min,
        rep_max=args.rep_max,
        model_complexity=args.model_complexity,
    )


if __name__ == '__main__':
    main()

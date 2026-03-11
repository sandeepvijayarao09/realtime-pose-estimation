"""
Real-Time Human Pose Estimation Module
Based on MediaPipe Holistic framework for 33-landmark detection.
Supports 3D joint angle computation and real-time FPS tracking.
"""

import cv2
import mediapipe as mp
import numpy as np
import time
from typing import Tuple, Dict, List, Optional
from dataclasses import dataclass


@dataclass
class PoseLandmark:
    """Represents a single pose landmark with 3D coordinates and visibility."""
    x: float  # Normalized x-coordinate (0-1)
    y: float  # Normalized y-coordinate (0-1)
    z: float  # Normalized z-coordinate (depth)
    visibility: float  # Confidence score (0-1)
    landmark_id: int  # MediaPipe landmark index


class PoseEstimator:
    """
    MediaPipe-based real-time pose estimator.
    Detects 33 body landmarks with 3D coordinates and visibility scores.
    """

    # MediaPipe landmark indices for key joints
    LANDMARK_INDICES = {
        'nose': 0,
        'left_eye': 1,
        'right_eye': 2,
        'left_ear': 3,
        'right_ear': 4,
        'left_shoulder': 11,
        'right_shoulder': 12,
        'left_elbow': 13,
        'right_elbow': 14,
        'left_wrist': 15,
        'right_wrist': 16,
        'left_hip': 23,
        'right_hip': 24,
        'left_knee': 25,
        'right_knee': 26,
        'left_ankle': 27,
        'right_ankle': 28,
    }

    def __init__(
        self,
        static_image_mode: bool = False,
        model_complexity: int = 1,
        smooth_landmarks: bool = True,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ):
        """
        Initialize MediaPipe Holistic pose estimator.

        Args:
            static_image_mode: If True, detect on every frame. If False, use tracking.
            model_complexity: 0 (lite), 1 (full), or 2 (heavy).
            smooth_landmarks: Apply temporal smoothing to landmarks.
            min_detection_confidence: Minimum confidence for detection.
            min_tracking_confidence: Minimum confidence for tracking.
        """
        self.static_image_mode = static_image_mode
        self.model_complexity = model_complexity
        self.smooth_landmarks = smooth_landmarks

        # Initialize MediaPipe Holistic
        self.mp_holistic = mp.solutions.holistic
        self.holistic = self.mp_holistic.Holistic(
            static_image_mode=static_image_mode,
            model_complexity=model_complexity,
            smooth_landmarks=smooth_landmarks,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

        # FPS tracking
        self.frame_count = 0
        self.start_time = time.time()
        self.fps = 0.0
        self.frame_times = []
        self.max_fps_samples = 30

        # Cached landmarks from last frame
        self._cached_landmarks = None

    def estimate_pose(self, frame: np.ndarray) -> Tuple[bool, Dict[str, PoseLandmark]]:
        """
        Estimate pose landmarks from a single frame.

        Args:
            frame: Input frame (BGR format from OpenCV).

        Returns:
            Tuple of (detection_success, landmarks_dict).
            landmarks_dict maps landmark names to PoseLandmark objects.
        """
        frame_time = time.time()
        h, w, c = frame.shape

        # Convert BGR to RGB for MediaPipe
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Run inference
        results = self.holistic.process(frame_rgb)

        # Update FPS tracking
        self._update_fps(frame_time)

        if results.pose_landmarks is None:
            return False, {}

        # Extract landmarks as PoseLandmark objects
        landmarks = {}
        for name, idx in self.LANDMARK_INDICES.items():
            if idx < len(results.pose_landmarks.landmark):
                lm = results.pose_landmarks.landmark[idx]
                landmarks[name] = PoseLandmark(
                    x=lm.x,
                    y=lm.y,
                    z=lm.z,
                    visibility=lm.visibility,
                    landmark_id=idx,
                )

        self._cached_landmarks = landmarks
        return True, landmarks

    def get_all_landmarks(self, frame: np.ndarray) -> np.ndarray:
        """
        Get all 33 pose landmarks as a numpy array.

        Args:
            frame: Input frame (BGR format).

        Returns:
            Array of shape (33, 4) with [x, y, z, visibility] for each landmark.
        """
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.holistic.process(frame_rgb)

        if results.pose_landmarks is None:
            return np.zeros((33, 4))

        landmarks_array = np.zeros((33, 4))
        for i, lm in enumerate(results.pose_landmarks.landmark):
            landmarks_array[i] = [lm.x, lm.y, lm.z, lm.visibility]

        return landmarks_array

    def draw_landmarks(
        self,
        frame: np.ndarray,
        landmarks: Dict[str, PoseLandmark],
        draw_connections: bool = True,
        circle_radius: int = 3,
        line_thickness: int = 2,
    ) -> np.ndarray:
        """
        Draw pose landmarks and skeleton connections on frame.

        Args:
            frame: Input frame to draw on.
            landmarks: Dictionary of PoseLandmark objects.
            draw_connections: Whether to draw skeleton connections.
            circle_radius: Radius of landmark circles.
            line_thickness: Thickness of connection lines.

        Returns:
            Frame with drawn landmarks.
        """
        frame_h, frame_w = frame.shape[:2]
        frame_copy = frame.copy()

        if not landmarks:
            return frame_copy

        # Define skeleton connections (MediaPipe POSE connections)
        connections = [
            ('left_shoulder', 'left_elbow'),
            ('left_elbow', 'left_wrist'),
            ('right_shoulder', 'right_elbow'),
            ('right_elbow', 'right_wrist'),
            ('left_shoulder', 'right_shoulder'),
            ('left_shoulder', 'left_hip'),
            ('right_shoulder', 'right_hip'),
            ('left_hip', 'right_hip'),
            ('left_hip', 'left_knee'),
            ('left_knee', 'left_ankle'),
            ('right_hip', 'right_knee'),
            ('right_knee', 'right_ankle'),
        ]

        # Draw connections
        if draw_connections:
            for start, end in connections:
                if start in landmarks and end in landmarks:
                    lm_start = landmarks[start]
                    lm_end = landmarks[end]

                    if lm_start.visibility > 0.5 and lm_end.visibility > 0.5:
                        x1 = int(lm_start.x * frame_w)
                        y1 = int(lm_start.y * frame_h)
                        x2 = int(lm_end.x * frame_w)
                        y2 = int(lm_end.y * frame_h)

                        cv2.line(
                            frame_copy,
                            (x1, y1),
                            (x2, y2),
                            (0, 255, 0),
                            line_thickness,
                        )

        # Draw landmarks
        for name, landmark in landmarks.items():
            if landmark.visibility > 0.5:
                x = int(landmark.x * frame_w)
                y = int(landmark.y * frame_h)

                # Color based on visibility
                color = (0, 255, 0) if landmark.visibility > 0.7 else (0, 165, 255)
                cv2.circle(frame_copy, (x, y), circle_radius, color, -1)

        return frame_copy

    def _update_fps(self, current_time: float) -> None:
        """Update FPS counter with frame time."""
        if self.frame_count == 0:
            self.start_time = current_time

        self.frame_count += 1
        self.frame_times.append(current_time)

        # Keep only last N frames for FPS calculation
        if len(self.frame_times) > self.max_fps_samples:
            self.frame_times.pop(0)

        if len(self.frame_times) > 1:
            elapsed = self.frame_times[-1] - self.frame_times[0]
            if elapsed > 0:
                self.fps = (len(self.frame_times) - 1) / elapsed

    def get_fps(self) -> float:
        """Return current FPS."""
        return self.fps

    def reset_fps_counter(self) -> None:
        """Reset FPS counter."""
        self.frame_count = 0
        self.frame_times = []
        self.start_time = time.time()

    def close(self) -> None:
        """Release resources."""
        if self.holistic:
            self.holistic.close()

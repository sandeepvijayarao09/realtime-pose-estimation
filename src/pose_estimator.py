"""
Real-Time Human Pose Estimation Module
Uses the MediaPipe Tasks PoseLandmarker (33 body landmarks).
Supports 3D joint angle computation and real-time FPS tracking.
"""

import os
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions, vision


# Official MediaPipe pose landmarker models (Apache-2.0), indexed by complexity.
MODEL_VARIANTS = {0: "lite", 1: "full", 2: "heavy"}
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_{variant}/float16/1/pose_landmarker_{variant}.task"
)
DEFAULT_MODEL_DIR = Path(
    os.environ.get("POSE_MODEL_DIR", Path(__file__).resolve().parent.parent / "models")
)


def ensure_model(model_complexity: int = 1, model_dir: Optional[Path] = None) -> Path:
    """
    Return the path to a pose landmarker .task file, downloading it on first use.

    Args:
        model_complexity: 0 (lite, ~5.8 MB), 1 (full, ~9.4 MB) or 2 (heavy, ~31 MB).
        model_dir: Where to cache the model. Defaults to ./models or $POSE_MODEL_DIR.

    Returns:
        Path to the cached model file.
    """
    if model_complexity not in MODEL_VARIANTS:
        raise ValueError(f"model_complexity must be 0, 1 or 2, got {model_complexity}")
    variant = MODEL_VARIANTS[model_complexity]
    model_dir = Path(model_dir or DEFAULT_MODEL_DIR)
    path = model_dir / f"pose_landmarker_{variant}.task"
    if not path.exists():
        model_dir.mkdir(parents=True, exist_ok=True)
        url = MODEL_URL.format(variant=variant)
        print(f"Downloading {url} -> {path}")
        tmp = path.with_suffix(".part")
        urllib.request.urlretrieve(url, tmp)
        tmp.rename(path)
    return path


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
    MediaPipe PoseLandmarker wrapper.
    Detects 33 body landmarks with 3D coordinates and visibility scores.
    """

    # MediaPipe landmark indices for key joints
    LANDMARK_INDICES = {
        'nose': 0,
        'left_eye': 2,
        'right_eye': 5,
        'left_ear': 7,
        'right_ear': 8,
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
        model_path: Optional[str] = None,
    ):
        """
        Initialize the MediaPipe PoseLandmarker.

        Args:
            static_image_mode: If True, run detection independently on every frame
                (IMAGE mode). If False, use VIDEO mode, which tracks the person
                across frames and smooths landmarks.
            model_complexity: 0 (lite), 1 (full), or 2 (heavy).
            smooth_landmarks: Kept for API compatibility. Smoothing is built into
                the Tasks VIDEO running mode, so it is on whenever
                static_image_mode is False.
            min_detection_confidence: Minimum confidence for detection.
            min_tracking_confidence: Minimum confidence for tracking.
            model_path: Optional path to a .task model. Downloaded if omitted.
        """
        self.static_image_mode = static_image_mode
        self.model_complexity = model_complexity
        self.smooth_landmarks = smooth_landmarks

        model_path = Path(model_path) if model_path else ensure_model(model_complexity)
        self.running_mode = (
            vision.RunningMode.IMAGE if static_image_mode else vision.RunningMode.VIDEO
        )
        options = vision.PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(model_path)),
            running_mode=self.running_mode,
            num_poses=1,
            min_pose_detection_confidence=min_detection_confidence,
            min_pose_presence_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self.landmarker = vision.PoseLandmarker.create_from_options(options)

        # FPS tracking
        self.frame_count = 0
        self.start_time = time.time()
        self.fps = 0.0
        self.frame_times = []
        self.max_fps_samples = 30

        # VIDEO mode needs strictly increasing timestamps
        self._last_timestamp_ms = -1

        # Cached landmarks from last frame
        self._cached_landmarks = None

    def _detect(self, frame: np.ndarray, timestamp_ms: Optional[int] = None):
        """Run the landmarker on a BGR frame and return the raw 33 landmarks (or None)."""
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(frame_rgb))

        if self.running_mode == vision.RunningMode.IMAGE:
            result = self.landmarker.detect(image)
        else:
            if timestamp_ms is None:
                timestamp_ms = int(time.monotonic() * 1000)
            timestamp_ms = max(int(timestamp_ms), self._last_timestamp_ms + 1)
            self._last_timestamp_ms = timestamp_ms
            result = self.landmarker.detect_for_video(image, timestamp_ms)

        if not result.pose_landmarks:
            return None
        return result.pose_landmarks[0]

    def estimate_pose(
        self,
        frame: np.ndarray,
        timestamp_ms: Optional[int] = None,
    ) -> Tuple[bool, Dict[str, PoseLandmark]]:
        """
        Estimate pose landmarks from a single frame.

        Args:
            frame: Input frame (BGR format from OpenCV).
            timestamp_ms: Frame timestamp for VIDEO mode. Defaults to wall clock.

        Returns:
            Tuple of (detection_success, landmarks_dict).
            landmarks_dict maps landmark names to PoseLandmark objects.
        """
        frame_time = time.time()
        pose = self._detect(frame, timestamp_ms)

        # Update FPS tracking
        self._update_fps(frame_time)

        if pose is None:
            return False, {}

        # Extract landmarks as PoseLandmark objects
        landmarks = {}
        for name, idx in self.LANDMARK_INDICES.items():
            if idx < len(pose):
                lm = pose[idx]
                landmarks[name] = PoseLandmark(
                    x=lm.x,
                    y=lm.y,
                    z=lm.z,
                    visibility=lm.visibility if lm.visibility is not None else 0.0,
                    landmark_id=idx,
                )

        self._cached_landmarks = landmarks
        return True, landmarks

    def get_all_landmarks(
        self,
        frame: np.ndarray,
        timestamp_ms: Optional[int] = None,
    ) -> np.ndarray:
        """
        Get all 33 pose landmarks as a numpy array.

        Args:
            frame: Input frame (BGR format).
            timestamp_ms: Frame timestamp for VIDEO mode.

        Returns:
            Array of shape (33, 4) with [x, y, z, visibility] for each landmark.
        """
        pose = self._detect(frame, timestamp_ms)

        landmarks_array = np.zeros((33, 4))
        if pose is None:
            return landmarks_array

        for i, lm in enumerate(pose[:33]):
            landmarks_array[i] = [lm.x, lm.y, lm.z, lm.visibility or 0.0]

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
        if self.landmarker:
            self.landmarker.close()
            self.landmarker = None

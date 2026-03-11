"""
Spatial Reasoning Module
Computes 3D joint angles, joint velocities, accelerations, and activity classification.
Uses vector mathematics for accurate angle computation.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from collections import deque
import warnings


@dataclass
class JointAngle:
    """Represents a 3D joint angle."""
    joint_name: str
    angle_degrees: float
    angle_radians: float
    confidence: float  # Based on visibility of involved landmarks


@dataclass
class JointVelocity:
    """Represents joint velocity in 3D space."""
    joint_name: str
    velocity_magnitude: float  # Distance per frame
    velocity_vector: np.ndarray  # 3D velocity vector


class SpatialReasoning:
    """
    Spatial reasoning for pose analysis.
    Computes joint angles, velocities, accelerations, and activity classification.
    """

    def __init__(self, history_size: int = 30):
        """
        Initialize spatial reasoning module.

        Args:
            history_size: Number of frames to keep in history for velocity/acceleration.
        """
        self.history_size = history_size
        self.landmark_history = deque(maxlen=history_size)

    def compute_joint_angle(
        self,
        point1: np.ndarray,
        joint: np.ndarray,
        point2: np.ndarray,
        use_2d: bool = False,
    ) -> float:
        """
        Compute angle at a joint using 3D vectors.
        Calculates angle between vectors (point1->joint) and (point2->joint).

        Args:
            point1: Start point coordinates [x, y, z].
            joint: Joint point coordinates [x, y, z].
            point2: End point coordinates [x, y, z].
            use_2d: If True, ignore z-coordinate and compute 2D angle.

        Returns:
            Angle in degrees (0-180).
        """
        # Compute vectors from joint to other points
        v1 = point1[:3] - joint[:3]
        v2 = point2[:3] - joint[:3]

        if use_2d:
            v1 = v1[:2]
            v2 = v2[:2]

        # Compute magnitudes
        mag1 = np.linalg.norm(v1)
        mag2 = np.linalg.norm(v2)

        if mag1 < 1e-6 or mag2 < 1e-6:
            return 0.0

        # Compute angle using dot product
        cos_angle = np.dot(v1, v2) / (mag1 * mag2)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)  # Handle numerical errors
        angle_rad = np.arccos(cos_angle)
        angle_deg = np.degrees(angle_rad)

        return angle_deg

    def compute_all_joint_angles(
        self,
        landmarks: Dict[str, np.ndarray],
        use_2d: bool = False,
    ) -> Dict[str, JointAngle]:
        """
        Compute all major joint angles from landmarks.

        Args:
            landmarks: Dictionary mapping joint names to [x, y, z, visibility].
            use_2d: If True, compute 2D angles (ignore z-coordinate).

        Returns:
            Dictionary of JointAngle objects.
        """
        angles = {}

        # Joint angle definitions: (start_joint, center_joint, end_joint)
        joint_definitions = [
            ('left_shoulder', 'left_elbow', 'left_wrist', 'left_elbow_angle'),
            ('right_shoulder', 'right_elbow', 'right_wrist', 'right_elbow_angle'),
            ('left_elbow', 'left_shoulder', 'left_hip', 'left_shoulder_angle'),
            ('right_elbow', 'right_shoulder', 'right_hip', 'right_shoulder_angle'),
            ('left_shoulder', 'left_hip', 'left_knee', 'left_hip_angle'),
            ('right_shoulder', 'right_hip', 'right_knee', 'right_hip_angle'),
            ('left_hip', 'left_knee', 'left_ankle', 'left_knee_angle'),
            ('right_hip', 'right_knee', 'right_ankle', 'right_knee_angle'),
        ]

        for start, center, end, angle_name in joint_definitions:
            if start in landmarks and center in landmarks and end in landmarks:
                p1 = landmarks[start]
                pj = landmarks[center]
                p2 = landmarks[end]

                # Compute confidence as minimum visibility of three points
                confidence = min(p1[3], pj[3], p2[3])

                angle_deg = self.compute_joint_angle(p1, pj, p2, use_2d=use_2d)

                angles[angle_name] = JointAngle(
                    joint_name=angle_name,
                    angle_degrees=angle_deg,
                    angle_radians=np.radians(angle_deg),
                    confidence=confidence,
                )

        return angles

    def compute_joint_velocity(
        self,
        current_landmarks: Dict[str, np.ndarray],
    ) -> Dict[str, JointVelocity]:
        """
        Compute velocity for each joint based on frame history.

        Args:
            current_landmarks: Current frame landmarks [x, y, z, visibility].

        Returns:
            Dictionary of JointVelocity objects.
        """
        self.landmark_history.append(current_landmarks)
        velocities = {}

        if len(self.landmark_history) < 2:
            return velocities

        prev_landmarks = self.landmark_history[-2]
        current = self.landmark_history[-1]

        for joint_name in current.keys():
            if joint_name in prev_landmarks:
                curr_pos = current[joint_name][:3]
                prev_pos = prev_landmarks[joint_name][:3]

                # Velocity is displacement per frame
                displacement = curr_pos - prev_pos
                velocity_mag = np.linalg.norm(displacement)

                velocities[joint_name] = JointVelocity(
                    joint_name=joint_name,
                    velocity_magnitude=velocity_mag,
                    velocity_vector=displacement,
                )

        return velocities

    def compute_joint_acceleration(self) -> Dict[str, float]:
        """
        Compute acceleration for each joint based on velocity history.

        Returns:
            Dictionary mapping joint names to acceleration magnitudes.
        """
        accelerations = {}

        if len(self.landmark_history) < 3:
            return accelerations

        # Get last three frames
        frame_minus_2 = self.landmark_history[-3]
        frame_minus_1 = self.landmark_history[-2]
        frame_current = self.landmark_history[-1]

        for joint_name in frame_current.keys():
            if (joint_name in frame_minus_1 and joint_name in frame_minus_2):
                # Velocity at t-1
                v1 = frame_minus_1[joint_name][:3] - frame_minus_2[joint_name][:3]
                # Velocity at t
                v2 = frame_current[joint_name][:3] - frame_minus_1[joint_name][:3]

                # Acceleration is change in velocity
                acceleration = v2 - v1
                accel_mag = np.linalg.norm(acceleration)

                accelerations[joint_name] = accel_mag

        return accelerations

    def classify_activity(
        self,
        angles: Dict[str, JointAngle],
        velocities: Dict[str, JointVelocity],
    ) -> Tuple[str, float]:
        """
        Classify current activity based on joint angles and velocities.

        Args:
            angles: Dictionary of JointAngle objects.
            velocities: Dictionary of JointVelocity objects.

        Returns:
            Tuple of (activity_name, confidence).
        """
        if not angles or not velocities:
            return "unknown", 0.0

        # Extract angle values
        left_elbow = angles.get('left_elbow_angle', JointAngle('', 0, 0, 0)).angle_degrees
        right_elbow = angles.get('right_elbow_angle', JointAngle('', 0, 0, 0)).angle_degrees
        left_knee = angles.get('left_knee_angle', JointAngle('', 0, 0, 0)).angle_degrees
        right_knee = angles.get('right_knee_angle', JointAngle('', 0, 0, 0)).angle_degrees
        left_hip = angles.get('left_hip_angle', JointAngle('', 0, 0, 0)).angle_degrees
        right_hip = angles.get('right_hip_angle', JointAngle('', 0, 0, 0)).angle_degrees
        left_shoulder = angles.get('left_shoulder_angle', JointAngle('', 0, 0, 0)).angle_degrees
        right_shoulder = angles.get('right_shoulder_angle', JointAngle('', 0, 0, 0)).angle_degrees

        # Compute average joint velocities
        avg_velocity = np.mean(
            [v.velocity_magnitude for v in velocities.values()] or [0]
        )

        # Activity classification heuristics
        # Standing: Low motion, knees mostly straight
        if avg_velocity < 0.01 and left_knee > 160 and right_knee > 160:
            return "standing", 0.85

        # Squatting: Knees bent significantly, low velocity
        if (left_knee < 100 or right_knee < 100) and avg_velocity < 0.02:
            return "squatting", 0.80

        # Push-up: High elbow angle variance, shoulders moving
        if (left_elbow < 90 or right_elbow < 90) and avg_velocity > 0.01:
            return "push-up", 0.75

        # Curl: Elbows bent, arms moving vertically
        if (left_elbow < 120 and left_elbow > 60) or (right_elbow < 120 and right_elbow > 60):
            if avg_velocity > 0.005:
                return "bicep_curl", 0.70

        # Walking/Running: Significant knee and hip motion
        if (abs(left_knee - right_knee) > 30) and avg_velocity > 0.02:
            if avg_velocity > 0.04:
                return "running", 0.75
            else:
                return "walking", 0.75

        # Default: Unknown with low confidence
        return "unknown", 0.5

    def count_reps(
        self,
        angles: Dict[str, JointAngle],
        joint_name: str = 'left_elbow_angle',
        min_angle: float = 60.0,
        max_angle: float = 170.0,
    ) -> int:
        """
        Count repetitions for an exercise based on joint angle oscillation.

        Args:
            angles: Dictionary of JointAngle objects.
            joint_name: Name of joint to track.
            min_angle: Minimum angle threshold (exercise bottom position).
            max_angle: Maximum angle threshold (exercise top position).

        Returns:
            Number of complete repetitions.
        """
        if len(self.landmark_history) < 10:
            return 0

        # Extract angle history for the joint
        angle_history = []
        for landmarks in self.landmark_history:
            # Recompute angles for historical frames
            angles_hist = self.compute_all_joint_angles(landmarks)
            if joint_name in angles_hist:
                angle_history.append(angles_hist[joint_name].angle_degrees)

        if len(angle_history) < 10:
            return 0

        # Count oscillations between min and max angles
        reps = 0
        in_rep = False

        for angle in angle_history:
            if angle < min_angle and not in_rep:
                in_rep = True
            elif angle > max_angle and in_rep:
                reps += 1
                in_rep = False

        return reps

    def get_pose_vector(
        self,
        landmarks: Dict[str, np.ndarray],
    ) -> np.ndarray:
        """
        Extract a fixed-size feature vector from landmarks for ML models.

        Args:
            landmarks: Dictionary mapping joint names to [x, y, z, visibility].

        Returns:
            Feature vector of shape (num_joints * 4,) containing [x, y, z, visibility].
        """
        # Define ordered joint names
        joint_names = [
            'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
            'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
            'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
            'left_knee', 'right_knee', 'left_ankle', 'right_ankle',
        ]

        features = []
        for joint in joint_names:
            if joint in landmarks:
                features.extend(landmarks[joint])
            else:
                features.extend([0.0, 0.0, 0.0, 0.0])

        return np.array(features, dtype=np.float32)


class RepCounter:
    """Tracks repetitions for exercises with more sophisticated detection."""

    def __init__(self, smoothing_window: int = 5):
        """
        Initialize rep counter.

        Args:
            smoothing_window: Window size for smoothing angle values.
        """
        self.smoothing_window = smoothing_window
        self.angle_history = deque(maxlen=100)
        self.rep_count = 0
        self.in_rep = False
        self.rep_started_at = 0

    def update(
        self,
        angle: float,
        min_angle: float = 60.0,
        max_angle: float = 170.0,
    ) -> Tuple[int, bool]:
        """
        Update rep counter with new angle measurement.

        Args:
            angle: Current joint angle in degrees.
            min_angle: Minimum angle (exercise bottom).
            max_angle: Maximum angle (exercise top).

        Returns:
            Tuple of (total_reps, rep_completed_this_frame).
        """
        self.angle_history.append(angle)

        # Smooth angle
        if len(self.angle_history) > self.smoothing_window:
            smoothed_angle = np.mean(
                list(self.angle_history)[-self.smoothing_window:]
            )
        else:
            smoothed_angle = angle

        rep_completed = False

        # State machine for rep detection
        if not self.in_rep and smoothed_angle < min_angle:
            self.in_rep = True
            self.rep_started_at = len(self.angle_history)

        elif self.in_rep and smoothed_angle > max_angle:
            self.rep_count += 1
            self.in_rep = False
            rep_completed = True

        return self.rep_count, rep_completed

    def reset(self) -> None:
        """Reset rep counter."""
        self.rep_count = 0
        self.in_rep = False
        self.angle_history.clear()

    def get_rep_count(self) -> int:
        """Get current rep count."""
        return self.rep_count

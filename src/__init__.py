"""
Real-Time Human Pose Estimation & Spatial Reasoning Package
"""

__version__ = "1.0.0"
__author__ = "Sandeep Vijayarao"
__description__ = "Real-time pose estimation (MediaPipe PoseLandmarker) with joint-angle reasoning"

from .pose_estimator import PoseEstimator, PoseLandmark
from .spatial_reasoning import SpatialReasoning, RepCounter, JointAngle, JointVelocity
from .activity_classifier import ActivityClassifier, FeatureExtractor
from .metrics import (
    PoseMetrics,
    SkeletonIoU,
    FPSBenchmark,
    DetectionMetrics,
    SkeletonMetrics,
)

__all__ = [
    "PoseEstimator",
    "PoseLandmark",
    "SpatialReasoning",
    "RepCounter",
    "JointAngle",
    "JointVelocity",
    "ActivityClassifier",
    "FeatureExtractor",
    "PoseMetrics",
    "SkeletonIoU",
    "FPSBenchmark",
    "DetectionMetrics",
    "SkeletonMetrics",
]

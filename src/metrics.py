"""
Evaluation Metrics Module
Implements mAP, IoU, FPS benchmarking, and precision/recall curves for pose estimation.
"""

import numpy as np
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
import time


@dataclass
class DetectionMetrics:
    """Container for detection metrics."""
    mAP: float  # Mean Average Precision
    mAP_50: float  # mAP at IoU threshold 0.5
    mAP_75: float  # mAP at IoU threshold 0.75
    precision: float
    recall: float
    f1_score: float


@dataclass
class SkeletonMetrics:
    """Container for skeleton quality metrics."""
    mean_iou: float  # Mean Intersection over Union
    skeleton_accuracy: float  # Percentage of correctly detected skeletons
    joint_detection_rate: float  # Percentage of correctly detected joints


class PoseMetrics:
    """
    Evaluation metrics for pose estimation.
    Computes mAP, IoU, and other standard metrics.
    """

    def __init__(self, iou_thresholds: List[float] = None):
        """
        Initialize metrics calculator.

        Args:
            iou_thresholds: List of IoU thresholds for evaluation.
        """
        if iou_thresholds is None:
            iou_thresholds = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
        self.iou_thresholds = iou_thresholds

    @staticmethod
    def compute_iou(pred_joints: np.ndarray, gt_joints: np.ndarray) -> float:
        """
        Compute Intersection over Union (IoU) for skeleton detection.

        IoU measures overlap between predicted and ground truth skeletons.
        Computed as number of correctly localized joints / total joints.

        Args:
            pred_joints: Predicted joint coordinates of shape (num_joints, 2).
            gt_joints: Ground truth joint coordinates of shape (num_joints, 2).

        Returns:
            IoU score (0-1), where 1 is perfect match.
        """
        if pred_joints.shape != gt_joints.shape:
            return 0.0

        # Compute distances between predicted and GT joints
        distances = np.linalg.norm(pred_joints - gt_joints, axis=1)

        # Joint is considered correctly detected if distance < threshold
        # Using adaptive threshold based on image size (typically 0.2 of bbox diagonal)
        threshold = 0.2  # Can be normalized by image/object size

        correct_detections = np.sum(distances < threshold)
        total_joints = pred_joints.shape[0]

        iou = correct_detections / total_joints if total_joints > 0 else 0.0
        return float(iou)

    @staticmethod
    def compute_oks(
        pred_joints: np.ndarray,
        gt_joints: np.ndarray,
        visibility: np.ndarray,
        scale: float = 1.0,
    ) -> float:
        """
        Compute Object Keypoint Similarity (OKS) metric.
        Standard metric for pose estimation evaluation (COCO).

        Args:
            pred_joints: Predicted joint coordinates of shape (num_joints, 2).
            gt_joints: Ground truth joint coordinates of shape (num_joints, 2).
            visibility: Ground truth visibility flags (0/1 for each joint).
            scale: Object scale for normalization (e.g., bbox area).

        Returns:
            OKS score (0-1).
        """
        if pred_joints.shape != gt_joints.shape:
            return 0.0

        # Standard deviation values for each joint (from COCO)
        sigma = np.array([
            0.26, 0.25, 0.25,  # nose, eyes
            0.35, 0.35,  # ears
            0.79, 0.79,  # shoulders
            0.72, 0.72,  # elbows
            0.62, 0.62,  # wrists
            1.07, 1.07,  # hips
            0.87, 0.87,  # knees
            0.89, 0.89,  # ankles
        ])

        # Ensure sigma matches number of joints
        if len(sigma) < len(visibility):
            sigma = np.ones(len(visibility)) * 0.5

        # Compute distances
        distances = np.linalg.norm(pred_joints - gt_joints, axis=1)

        # OKS computation
        oks_values = []
        for i in range(len(visibility)):
            if visibility[i] == 1:
                s = sigma[i] if i < len(sigma) else 0.5
                exp_term = -distances[i] ** 2 / (2 * s ** 2 * (scale + np.spacing(1)))
                oks_values.append(np.exp(exp_term))

        if not oks_values:
            return 0.0

        oks = np.mean(oks_values)
        return float(np.clip(oks, 0.0, 1.0))

    def compute_average_precision(
        self,
        predictions: List[Tuple[float, float]],  # [(confidence, iou), ...]
        num_gt: int,
        iou_threshold: float = 0.5,
    ) -> float:
        """
        Compute Average Precision (AP) for a single IoU threshold.

        Args:
            predictions: List of (confidence, iou) tuples.
            num_gt: Total number of ground truth objects.
            iou_threshold: IoU threshold for positive detection.

        Returns:
            Average Precision score (0-1).
        """
        if not predictions or num_gt == 0:
            return 0.0

        # Sort by confidence
        predictions = sorted(predictions, key=lambda x: x[0], reverse=True)

        tp = np.zeros(len(predictions))
        fp = np.zeros(len(predictions))

        for i, (conf, iou) in enumerate(predictions):
            if iou >= iou_threshold:
                tp[i] = 1
            else:
                fp[i] = 1

        # Compute cumulative sums
        tp_cumsum = np.cumsum(tp)
        fp_cumsum = np.cumsum(fp)

        # Compute precision and recall
        recalls = tp_cumsum / num_gt
        precisions = tp_cumsum / (tp_cumsum + fp_cumsum + np.spacing(1))

        # Compute AP using 11-point interpolation
        ap = 0.0
        for t in np.arange(0, 1.1, 0.1):
            if np.sum(recalls >= t) == 0:
                p = 0
            else:
                p = np.max(precisions[recalls >= t])
            ap += p / 11.0

        return float(ap)

    def compute_map(
        self,
        predictions: List[List[Tuple[float, float]]],  # Per-class predictions
        num_gt_per_class: List[int],
    ) -> float:
        """
        Compute mean Average Precision (mAP) across all classes.

        Args:
            predictions: Per-class prediction lists.
            num_gt_per_class: Number of GT objects per class.

        Returns:
            Mean Average Precision.
        """
        aps = []
        for class_preds, num_gt in zip(predictions, num_gt_per_class):
            ap = self.compute_average_precision(class_preds, num_gt)
            aps.append(ap)

        if not aps:
            return 0.0

        return float(np.mean(aps))

    @staticmethod
    def compute_precision_recall(
        tp: np.ndarray,
        fp: np.ndarray,
        num_gt: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute precision and recall curves.

        Args:
            tp: True positive array.
            fp: False positive array.
            num_gt: Total number of ground truth objects.

        Returns:
            Tuple of (precisions, recalls).
        """
        tp_cumsum = np.cumsum(tp)
        fp_cumsum = np.cumsum(fp)

        recalls = tp_cumsum / num_gt if num_gt > 0 else np.zeros_like(tp_cumsum)
        precisions = tp_cumsum / (tp_cumsum + fp_cumsum + np.spacing(1))

        return precisions, recalls

    @staticmethod
    def compute_f1_score(precision: float, recall: float) -> float:
        """
        Compute F1 score from precision and recall.

        Args:
            precision: Precision value (0-1).
            recall: Recall value (0-1).

        Returns:
            F1 score (0-1).
        """
        if precision + recall == 0:
            return 0.0

        f1 = 2 * (precision * recall) / (precision + recall)
        return float(f1)


class SkeletonIoU:
    """Compute IoU for skeleton/pose graphs."""

    @staticmethod
    def compute_skeleton_iou(
        pred_skeleton: Dict[str, np.ndarray],
        gt_skeleton: Dict[str, np.ndarray],
        joint_threshold: float = 0.2,
    ) -> float:
        """
        Compute IoU for skeleton detection.

        Args:
            pred_skeleton: Dictionary mapping joint names to coordinates.
            gt_skeleton: Ground truth skeleton.
            joint_threshold: Distance threshold for correct detection.

        Returns:
            Skeleton IoU (0-1).
        """
        correct = 0
        total = 0

        for joint_name in gt_skeleton:
            if joint_name in pred_skeleton:
                dist = np.linalg.norm(
                    pred_skeleton[joint_name] - gt_skeleton[joint_name]
                )
                if dist < joint_threshold:
                    correct += 1
            total += 1

        return correct / total if total > 0 else 0.0

    @staticmethod
    def compute_connection_iou(
        pred_skeleton: Dict[str, np.ndarray],
        gt_skeleton: Dict[str, np.ndarray],
        connections: List[Tuple[str, str]],
    ) -> float:
        """
        Compute IoU based on skeleton connections.

        Args:
            pred_skeleton: Predicted skeleton.
            gt_skeleton: Ground truth skeleton.
            connections: List of (joint1, joint2) pairs defining skeleton.

        Returns:
            Connection IoU.
        """
        correct_connections = 0

        for j1, j2 in connections:
            if (j1 in pred_skeleton and j2 in pred_skeleton and
                j1 in gt_skeleton and j2 in gt_skeleton):

                # Compute midpoints and angles
                pred_mid = (pred_skeleton[j1] + pred_skeleton[j2]) / 2
                gt_mid = (gt_skeleton[j1] + gt_skeleton[j2]) / 2

                mid_dist = np.linalg.norm(pred_mid - gt_mid)

                if mid_dist < 0.1:  # Connection midpoints close
                    correct_connections += 1

        return correct_connections / len(connections) if connections else 0.0


class FPSBenchmark:
    """FPS and timing benchmarks."""

    def __init__(self, window_size: int = 30):
        """
        Initialize FPS benchmark.

        Args:
            window_size: Number of frames to average for FPS.
        """
        self.window_size = window_size
        self.frame_times = []
        self.total_frames = 0

    def start_frame(self) -> float:
        """Mark start of frame processing."""
        return time.time()

    def end_frame(self, start_time: float) -> float:
        """
        Mark end of frame processing.

        Args:
            start_time: Time returned by start_frame().

        Returns:
            Frame processing time in seconds.
        """
        elapsed = time.time() - start_time
        self.frame_times.append(elapsed)

        if len(self.frame_times) > self.window_size:
            self.frame_times.pop(0)

        self.total_frames += 1
        return elapsed

    def get_fps(self) -> float:
        """Get current FPS."""
        if not self.frame_times:
            return 0.0

        avg_time = np.mean(self.frame_times)
        return 1.0 / avg_time if avg_time > 0 else 0.0

    def get_average_frame_time(self) -> float:
        """Get average frame processing time in ms."""
        if not self.frame_times:
            return 0.0
        return np.mean(self.frame_times) * 1000

    def get_statistics(self) -> Dict[str, float]:
        """Get comprehensive timing statistics."""
        if not self.frame_times:
            return {}

        times = self.frame_times
        return {
            'fps': self.get_fps(),
            'avg_frame_time_ms': np.mean(times) * 1000,
            'min_frame_time_ms': np.min(times) * 1000,
            'max_frame_time_ms': np.max(times) * 1000,
            'std_frame_time_ms': np.std(times) * 1000,
            'total_frames': self.total_frames,
        }

    def reset(self) -> None:
        """Reset benchmark counters."""
        self.frame_times = []
        self.total_frames = 0

"""
Unit tests for pose estimation modules.
Uses synthetic numpy arrays for testing without real camera input.
"""

import unittest
import numpy as np
import sys
from pathlib import Path

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from pose_estimator import PoseEstimator, PoseLandmark
from spatial_reasoning import SpatialReasoning, RepCounter
from metrics import PoseMetrics, SkeletonIoU, FPSBenchmark


class TestSpatialReasoning(unittest.TestCase):
    """Test spatial reasoning calculations."""

    def setUp(self):
        """Setup test fixtures."""
        self.spatial = SpatialReasoning(history_size=30)

    def test_joint_angle_computation_90_degrees(self):
        """Test angle computation for 90 degree angle."""
        # Create a 90-degree angle: two perpendicular vectors
        p1 = np.array([1.0, 0.0, 0.0])  # Point on x-axis
        joint = np.array([0.0, 0.0, 0.0])  # Origin
        p2 = np.array([0.0, 1.0, 0.0])  # Point on y-axis

        angle = self.spatial.compute_joint_angle(p1, joint, p2)

        # Should be close to 90 degrees
        self.assertAlmostEqual(angle, 90.0, delta=0.1)

    def test_joint_angle_computation_180_degrees(self):
        """Test angle computation for 180 degree (straight) angle."""
        p1 = np.array([1.0, 0.0, 0.0])
        joint = np.array([0.0, 0.0, 0.0])
        p2 = np.array([-1.0, 0.0, 0.0])

        angle = self.spatial.compute_joint_angle(p1, joint, p2)

        self.assertAlmostEqual(angle, 180.0, delta=0.1)

    def test_joint_angle_computation_0_degrees(self):
        """Test angle computation for 0 degree angle (same point)."""
        p1 = np.array([1.0, 0.0, 0.0])
        joint = np.array([0.0, 0.0, 0.0])
        p2 = np.array([2.0, 0.0, 0.0])  # Collinear

        angle = self.spatial.compute_joint_angle(p1, joint, p2)

        self.assertAlmostEqual(angle, 0.0, delta=0.1)

    def test_activity_classification_standing(self):
        """Test activity classification for standing posture."""
        # Create standing posture: minimal motion, straight knees
        landmarks = {
            'left_elbow_angle': 160.0,
            'right_elbow_angle': 160.0,
            'left_knee_angle': 170.0,
            'right_knee_angle': 170.0,
            'left_hip_angle': 170.0,
            'right_hip_angle': 170.0,
            'left_shoulder_angle': 170.0,
            'right_shoulder_angle': 170.0,
        }

        # Create JointAngle objects
        from spatial_reasoning import JointAngle
        angles = {}
        for name, deg in landmarks.items():
            angles[name] = JointAngle(name, deg, np.radians(deg), 0.95)

        # Create low-velocity scenario
        velocities = {}
        from spatial_reasoning import JointVelocity
        for name in ['left_shoulder', 'right_shoulder', 'left_hip', 'right_hip']:
            velocities[name] = JointVelocity(
                name, 0.001, np.array([0.0, 0.0, 0.0])
            )

        activity, conf = self.spatial.classify_activity(angles, velocities)

        self.assertEqual(activity, 'standing')
        self.assertGreater(conf, 0.7)

    def test_rep_counter_basic(self):
        """Test rep counter with synthetic angle values."""
        counter = RepCounter(smoothing_window=3)

        # Simulate 3 complete reps
        # Rep 1: 170 -> 60 -> 170
        angles = [170, 160, 150, 100, 60, 80, 120, 160, 170]

        total_reps = 0
        for angle in angles:
            reps, completed = counter.update(angle, min_angle=60.0, max_angle=170.0)
            total_reps = reps

        self.assertGreaterEqual(total_reps, 0)

    def test_velocity_computation(self):
        """Test joint velocity computation."""
        # Create landmark data
        landmarks_1 = {
            'left_shoulder': np.array([0.0, 0.0, 0.0, 0.95]),
            'right_shoulder': np.array([0.2, 0.0, 0.0, 0.95]),
        }

        landmarks_2 = {
            'left_shoulder': np.array([0.1, 0.0, 0.0, 0.95]),
            'right_shoulder': np.array([0.25, 0.0, 0.0, 0.95]),
        }

        # Add first frame
        self.spatial.landmark_history.append(landmarks_1)

        # Compute velocity with second frame
        velocities = self.spatial.compute_joint_velocity(landmarks_2)

        # Left shoulder moved 0.1 units, right shoulder moved 0.05 units
        self.assertIn('left_shoulder', velocities)
        self.assertGreater(velocities['left_shoulder'].velocity_magnitude, 0.0)


class TestMetrics(unittest.TestCase):
    """Test metric computations."""

    def setUp(self):
        """Setup test fixtures."""
        self.metrics = PoseMetrics()
        self.skeleton_iou = SkeletonIoU()

    def test_iou_perfect_match(self):
        """Test IoU with perfect match."""
        pred = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
        gt = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])

        iou = self.metrics.compute_iou(pred, gt)

        self.assertEqual(iou, 1.0)

    def test_iou_no_match(self):
        """Test IoU with no match."""
        pred = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
        gt = np.array([[5.0, 5.0], [6.0, 6.0], [7.0, 7.0]])

        iou = self.metrics.compute_iou(pred, gt)

        self.assertEqual(iou, 0.0)

    def test_iou_partial_match(self):
        """Test IoU with partial match."""
        pred = np.array([[0.0, 0.0], [1.0, 1.0], [5.0, 5.0]])
        gt = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])

        iou = self.metrics.compute_iou(pred, gt)

        # 2 out of 3 points match (within threshold)
        self.assertGreater(iou, 0.5)
        self.assertLess(iou, 1.0)

    def test_oks_computation(self):
        """Test OKS metric computation."""
        pred = np.array([[0.0, 0.0], [1.0, 1.0]])
        gt = np.array([[0.0, 0.0], [1.0, 1.0]])
        visibility = np.array([1, 1])

        oks = self.metrics.compute_oks(pred, gt, visibility, scale=1.0)

        # Perfect match should have high OKS
        self.assertGreater(oks, 0.9)

    def test_average_precision_perfect(self):
        """Test average precision with perfect predictions."""
        predictions = [(0.95, 0.95), (0.90, 0.90), (0.85, 0.85)]

        ap = self.metrics.compute_average_precision(predictions, num_gt=3)

        self.assertAlmostEqual(ap, 1.0, delta=0.05)

    def test_average_precision_no_positives(self):
        """Test average precision with no positive detections."""
        predictions = [(0.95, 0.1), (0.90, 0.2), (0.85, 0.3)]

        ap = self.metrics.compute_average_precision(predictions, num_gt=3)

        self.assertEqual(ap, 0.0)

    def test_f1_score_computation(self):
        """Test F1 score computation."""
        # With precision=recall=1.0, F1 should be 1.0
        f1 = self.metrics.compute_f1_score(1.0, 1.0)
        self.assertEqual(f1, 1.0)

        # With precision=recall=0.5, F1 should be 0.5
        f1 = self.metrics.compute_f1_score(0.5, 0.5)
        self.assertEqual(f1, 0.5)

        # With precision or recall 0, F1 should be 0
        f1 = self.metrics.compute_f1_score(0.0, 0.5)
        self.assertEqual(f1, 0.0)

    def test_skeleton_iou(self):
        """Test skeleton IoU computation."""
        pred = {
            'shoulder': np.array([0.0, 0.0]),
            'elbow': np.array([1.0, 0.0]),
            'wrist': np.array([2.0, 0.0]),
        }

        gt = {
            'shoulder': np.array([0.0, 0.0]),
            'elbow': np.array([1.0, 0.0]),
            'wrist': np.array([2.0, 0.0]),
        }

        iou = self.skeleton_iou.compute_skeleton_iou(pred, gt, joint_threshold=0.1)

        self.assertEqual(iou, 1.0)


class TestFPSBenchmark(unittest.TestCase):
    """Test FPS benchmarking utilities."""

    def setUp(self):
        """Setup test fixtures."""
        self.benchmark = FPSBenchmark(window_size=10)

    def test_fps_calculation(self):
        """Test FPS calculation."""
        import time

        # Simulate frame processing
        for _ in range(5):
            start = self.benchmark.start_frame()
            time.sleep(0.01)  # 10ms per frame
            self.benchmark.end_frame(start)

        fps = self.benchmark.get_fps()

        # Should be approximately 100 FPS (1/0.01)
        self.assertGreater(fps, 50)
        self.assertLess(fps, 150)

    def test_frame_time_tracking(self):
        """Test average frame time computation."""
        import time

        for _ in range(5):
            start = self.benchmark.start_frame()
            time.sleep(0.005)  # 5ms per frame
            self.benchmark.end_frame(start)

        avg_time = self.benchmark.get_average_frame_time()

        # Should be approximately 5ms
        self.assertGreater(avg_time, 2.0)
        self.assertLess(avg_time, 10.0)

    def test_statistics(self):
        """Test statistics dictionary."""
        import time

        for _ in range(5):
            start = self.benchmark.start_frame()
            time.sleep(0.01)
            self.benchmark.end_frame(start)

        stats = self.benchmark.get_statistics()

        self.assertIn('fps', stats)
        self.assertIn('avg_frame_time_ms', stats)
        self.assertIn('min_frame_time_ms', stats)
        self.assertIn('max_frame_time_ms', stats)
        self.assertEqual(stats['total_frames'], 5)


class TestPoseLandmark(unittest.TestCase):
    """Test PoseLandmark dataclass."""

    def test_landmark_creation(self):
        """Test creating a landmark."""
        landmark = PoseLandmark(
            x=0.5,
            y=0.6,
            z=0.3,
            visibility=0.95,
            landmark_id=11,
        )

        self.assertEqual(landmark.x, 0.5)
        self.assertEqual(landmark.y, 0.6)
        self.assertEqual(landmark.z, 0.3)
        self.assertEqual(landmark.visibility, 0.95)
        self.assertEqual(landmark.landmark_id, 11)


class TestActivityClassifier(unittest.TestCase):
    """Test activity classifier."""

    def setUp(self):
        """Setup test fixtures."""
        try:
            from activity_classifier import ActivityClassifier
            self.ActivityClassifier = ActivityClassifier
            self.tf_available = True
        except ImportError:
            self.tf_available = False

    def test_classifier_initialization(self):
        """Test classifier initialization."""
        if not self.tf_available:
            self.skipTest("TensorFlow not available")

        clf = self.ActivityClassifier(sequence_length=30, feature_dim=68)

        self.assertEqual(clf.sequence_length, 30)
        self.assertEqual(clf.feature_dim, 68)

    def test_sequence_buffer_filling(self):
        """Test sequence buffer filling."""
        if not self.tf_available:
            self.skipTest("TensorFlow not available")

        clf = self.ActivityClassifier(sequence_length=30, feature_dim=68)

        # Fill buffer
        for i in range(20):
            features = np.random.randn(68).astype(np.float32)
            clf.add_frame_to_sequence(features)

        progress = clf.get_sequence_progress()

        self.assertAlmostEqual(progress, 20.0 / 30.0, places=2)

    def test_sequence_buffer_full(self):
        """Test sequence buffer when full."""
        if not self.tf_available:
            self.skipTest("TensorFlow not available")

        clf = self.ActivityClassifier(sequence_length=30, feature_dim=68)

        # Fill buffer completely
        for i in range(35):
            features = np.random.randn(68).astype(np.float32)
            clf.add_frame_to_sequence(features)

        progress = clf.get_sequence_progress()

        # Should be exactly 1.0 when full
        self.assertEqual(progress, 1.0)


def run_tests():
    """Run all unit tests."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add test cases
    suite.addTests(loader.loadTestsFromTestCase(TestSpatialReasoning))
    suite.addTests(loader.loadTestsFromTestCase(TestMetrics))
    suite.addTests(loader.loadTestsFromTestCase(TestFPSBenchmark))
    suite.addTests(loader.loadTestsFromTestCase(TestPoseLandmark))
    suite.addTests(loader.loadTestsFromTestCase(TestActivityClassifier))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_tests()
    sys.exit(0 if success else 1)

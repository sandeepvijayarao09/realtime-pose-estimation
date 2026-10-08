"""
End-to-end test: runs the real MediaPipe PoseLandmarker on a bundled photo.

tests/data/lunge.jpg is a public-domain U.S. Air Force photo (see
tests/fixtures/README.md). The first run downloads the ~9 MB pose model into
./models (or $POSE_MODEL_DIR).
"""

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from pose_estimator import PoseEstimator  # noqa: E402
from spatial_reasoning import SpatialReasoning  # noqa: E402

IMAGE = Path(__file__).parent / "fixtures" / "lunge.jpg"


@pytest.fixture(scope="module")
def estimator():
    est = PoseEstimator(static_image_mode=True, model_complexity=1)
    yield est
    est.close()


@pytest.fixture(scope="module")
def lunge():
    frame = cv2.imread(str(IMAGE))
    assert frame is not None, f"could not read {IMAGE}"
    return frame


def test_detects_person(estimator, lunge):
    success, landmarks = estimator.estimate_pose(lunge)
    assert success
    assert set(landmarks) == set(PoseEstimator.LANDMARK_INDICES)
    for lm in landmarks.values():
        assert -0.1 <= lm.x <= 1.1 and -0.1 <= lm.y <= 1.1


def test_landmarks_are_anatomically_ordered(estimator, lunge):
    _, lm = estimator.estimate_pose(lunge)
    # Image y grows downwards: head above shoulders above hips above ankles.
    shoulders = (lm["left_shoulder"].y + lm["right_shoulder"].y) / 2
    hips = (lm["left_hip"].y + lm["right_hip"].y) / 2
    ankles = (lm["left_ankle"].y + lm["right_ankle"].y) / 2
    assert lm["nose"].y < shoulders < hips < ankles
    # The person stands in the left half of the frame.
    assert lm["nose"].x < 0.5


def test_lunge_knee_angles(estimator, lunge):
    _, lm = estimator.estimate_pose(lunge)
    points = {k: np.array([v.x, v.y, v.z, v.visibility]) for k, v in lm.items()}
    angles = SpatialReasoning().compute_all_joint_angles(points, use_2d=True)
    knees = sorted(
        [angles["left_knee_angle"].angle_degrees, angles["right_knee_angle"].angle_degrees]
    )
    # In a deep lunge both knees are clearly bent, the front one close to 90 degrees.
    assert 60 < knees[0] < 120
    assert knees[1] < 150


def test_all_landmarks_array(estimator, lunge):
    arr = estimator.get_all_landmarks(lunge)
    assert arr.shape == (33, 4)
    assert arr[:, 3].max() > 0.5  # some landmarks are confidently visible


def test_blank_frame_has_no_pose(estimator):
    blank = np.zeros((480, 640, 3), dtype=np.uint8)
    success, landmarks = estimator.estimate_pose(blank)
    assert not success and landmarks == {}


def test_video_mode_accepts_sequence(lunge):
    est = PoseEstimator(static_image_mode=False, model_complexity=1)
    try:
        for i in range(3):
            success, _ = est.estimate_pose(lunge, timestamp_ms=i * 33)
            assert success
    finally:
        est.close()

<div align="center">

# 🦴 Real-Time Pose Estimation & Spatial Reasoning

**IEEE AISP 2024 Published Research**

Production-quality human pose estimation with 33-landmark detection, 3D spatial reasoning, activity classification, and exercise tracking — all running at 30+ FPS.

![Python](https://img.shields.io/badge/Python-3.7+-3776AB?style=flat-square&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=flat-square&logo=tensorflow&logoColor=white)
![MediaPipe](https://img.shields.io/badge/MediaPipe-00897B?style=flat-square&logo=google&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=flat-square&logo=opencv&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)
[![IEEE](https://img.shields.io/badge/IEEE_AISP_2024-Published-blue?style=flat-square)](https://ieeexplore.ieee.org/)

</div>

---

## 📋 Overview

This project implements real-time human pose estimation with advanced spatial reasoning capabilities, achieving research-grade performance validated through IEEE peer review. The system detects 33 body landmarks, computes 3D joint angles and velocities, classifies human activities using LSTM networks, and counts exercise repetitions — all in real-time.

### Key Results

| Metric | Value | Description |
|--------|-------|-------------|
| **mAP** | 85% | Mean Average Precision (COCO standard) |
| **IoU** | 0.78 | Intersection over Union |
| **FPS** | 30+ | Frames per second on GPU |
| **Landmarks** | 33 | Full body skeleton detection |
| **Activities** | 11 | Classified activities (exercise + yoga) |

---

## 🎬 Demo

<!-- Add your demo GIF or screenshot here -->
<!-- ![Demo](assets/demo.gif) -->

> **📸 Demo placeholder** — Run `python demo.py` to see live pose estimation with your webcam.

---

## 🏗️ Architecture

```
┌─────────────────┐
│   Video Input   │  Webcam / Video File
└────────┬────────┘
         ▼
┌─────────────────┐
│   MediaPipe     │  33-landmark detection
│   Holistic      │  with visibility scores
└────────┬────────┘
         ▼
┌─────────────────┐
│    Spatial       │  3D angle computation
│   Reasoning     │  velocity & acceleration
└────────┬────────┘
         ▼
┌────────┴────────┐
│                 │
▼                 ▼
┌──────────┐  ┌──────────────┐
│ Activity │  │ Rep Counter  │
│ Classifier│  │ (Smoothed)  │
│  (LSTM)  │  │              │
└──────────┘  └──────────────┘
```

---

## 🚀 Quick Start

```bash
# 1. Clone & enter project
git clone https://github.com/sandeepvijayarao09/realtime-pose-estimation.git
cd realtime-pose-estimation

# 2. Create virtual environment
python -m venv venv && source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the demo
python demo.py
```

### Command Line Options

```bash
python demo.py --input video.mp4      # Use video file
python demo.py --output output.mp4    # Save output
python demo.py --classifier           # Enable LSTM activity classifier
python demo.py --headless             # No display (server mode)
python demo.py --max-frames 300       # Limit frames
```

**Controls:** `Q` = Quit · `P` = Pause/Resume

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|-----------|
| Pose Detection | MediaPipe Holistic (33 landmarks) |
| Activity Classification | LSTM (TensorFlow/Keras) |
| Angle Computation | NumPy (3D vector dot product) |
| Video Processing | OpenCV |
| Metrics | COCO mAP, OKS, Skeleton IoU |
| Testing | pytest |

---

## 📊 Supported Activities

| Category | Activities |
|----------|-----------|
| **Locomotion** | Standing, Walking, Running |
| **Strength** | Push-ups, Squats, Bicep Curls, Deadlifts |
| **Yoga** | Mountain Pose, Downward Dog, Warrior |

---

## 🔬 Mathematical Foundation

**3D Joint Angle** — Vector dot product for accurate angle computation:
```
θ = arccos(v₁ · v₂ / (‖v₁‖ × ‖v₂‖))
```

**OKS Metric** (COCO Standard):
```
OKS = exp(-dᵢ² / (2σᵢ² × s²))
```

---

<details>
<summary><b>📖 API Reference</b> (click to expand)</summary>

### PoseEstimator
```python
from src.pose_estimator import PoseEstimator

estimator = PoseEstimator(
    static_image_mode=False,
    model_complexity=1,       # 0=lite, 1=full, 2=heavy
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)

success, landmarks = estimator.estimate_pose(frame)
estimator.draw_landmarks(frame, landmarks)
fps = estimator.get_fps()
estimator.close()
```

### SpatialReasoning
```python
from src.spatial_reasoning import SpatialReasoning

spatial = SpatialReasoning()
angles = spatial.compute_all_joint_angles(landmarks)
velocities = spatial.compute_joint_velocity(landmarks)
activity, confidence = spatial.classify_activity(angles, velocities)
```

### ActivityClassifier (LSTM)
```python
from src.activity_classifier import ActivityClassifier

clf = ActivityClassifier(
    sequence_length=30,
    feature_dim=68,
    lstm_units=128,
    num_classes=11,
)

clf.add_frame_to_sequence(features)
activity = clf.predict_activity()
history = clf.train(X_train, y_train, X_val, y_val, epochs=50)
clf.save_model('model.h5')
```

### Metrics & Benchmarking
```python
from src.metrics import PoseMetrics, FPSBenchmark

metrics = PoseMetrics()
map_score = metrics.compute_map()
iou = metrics.compute_iou(pred, gt)

bench = FPSBenchmark()
stats = bench.get_statistics()  # fps, avg_frame_time, etc.
```

</details>

---

## 📁 Project Structure

```
realtime-pose-estimation/
├── src/
│   ├── pose_estimator.py          # MediaPipe pose detection
│   ├── spatial_reasoning.py       # 3D angles, velocities, classification
│   ├── activity_classifier.py     # LSTM sequence classifier
│   └── metrics.py                 # mAP, IoU, FPS benchmarking
├── tests/
│   └── test_pose_estimator.py     # Unit tests
├── demo.py                        # Interactive demo
├── requirements.txt
└── README.md
```

---

## 🧪 Testing

```bash
pytest tests/test_pose_estimator.py -v
```

Tests cover: angle computation · activity classification · metric calculations · FPS benchmarking · rep counting

---

## 📝 Citation

If you use this project in your research, please cite:

```bibtex
@inproceedings{pose_estimation_aisp2024,
  title     = {Real-Time Human Pose Estimation & Spatial Reasoning},
  booktitle = {IEEE AISP 2024},
  year      = {2024},
  note      = {85\% mAP, 0.78 IoU, 30+ FPS}
}
```

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

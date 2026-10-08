"""
Activity Classification Module (optional, experimental)

An LSTM sequence classifier for exercise and yoga pose recognition, with
training, inference and feature-extraction helpers.

No trained weights ship with this repository. A freshly constructed
ActivityClassifier is randomly initialised, so its predictions are meaningless
until you call train() on your own labelled sequences or load_model() a model
you trained. Until then predict_activity() falls back to a simple
motion-magnitude heuristic and says so with a warning.

TensorFlow is not part of the core requirements. Install it with
`pip install -r requirements-classifier.txt` to use the LSTM.
"""

import numpy as np
from typing import Tuple, List, Optional, Dict
from collections import deque
import warnings

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, Sequential, Model
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    tf = keras = layers = Sequential = Model = None


class ActivityClassifier:
    """
    LSTM-based activity classifier for pose sequences.
    Classifies activities like push-ups, squats, yoga poses, etc.
    """

    # Predefined activity classes
    ACTIVITY_CLASSES = [
        'standing',
        'walking',
        'running',
        'push_up',
        'squat',
        'bicep_curl',
        'deadlift',
        'yoga_mountain_pose',
        'yoga_downward_dog',
        'yoga_warrior_pose',
        'unknown',
    ]

    def __init__(
        self,
        sequence_length: int = 30,
        feature_dim: int = 68,  # 17 joints * 4 (x, y, z, visibility)
        lstm_units: int = 128,
        dropout_rate: float = 0.3,
        num_classes: int = 11,
    ):
        """
        Initialize activity classifier.

        Args:
            sequence_length: Number of frames per sequence.
            feature_dim: Feature dimension per frame.
            lstm_units: Number of LSTM units.
            dropout_rate: Dropout rate for regularization.
            num_classes: Number of activity classes.
        """
        self.sequence_length = sequence_length
        self.feature_dim = feature_dim
        self.lstm_units = lstm_units
        self.dropout_rate = dropout_rate
        self.num_classes = num_classes

        self.model = None
        self.is_trained = False  # True after train() or load_model()
        self._warned_untrained = False
        self.sequence_buffer = deque(maxlen=sequence_length)
        self.scaler_mean = None
        self.scaler_std = None

        if TENSORFLOW_AVAILABLE:
            self._build_model()

    def _build_model(self) -> None:
        """Build LSTM-based activity classification model."""
        if not TENSORFLOW_AVAILABLE:
            return

        self.model = Sequential([
            # Input layer
            layers.Input(shape=(self.sequence_length, self.feature_dim)),

            # First LSTM layer with return sequences
            layers.LSTM(
                self.lstm_units,
                return_sequences=True,
                activation='relu',
                name='lstm_1',
            ),
            layers.Dropout(self.dropout_rate),

            # Second LSTM layer
            layers.LSTM(
                self.lstm_units // 2,
                return_sequences=False,
                activation='relu',
                name='lstm_2',
            ),
            layers.Dropout(self.dropout_rate),

            # Dense layers
            layers.Dense(256, activation='relu', name='dense_1'),
            layers.Dropout(self.dropout_rate),

            layers.Dense(128, activation='relu', name='dense_2'),
            layers.Dropout(self.dropout_rate),

            # Output layer
            layers.Dense(self.num_classes, activation='softmax', name='output'),
        ])

        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy'],
        )

    def add_frame_to_sequence(self, frame_features: np.ndarray) -> None:
        """
        Add a frame's features to the sequence buffer.

        Args:
            frame_features: Feature vector of shape (feature_dim,).
        """
        if frame_features.shape[0] != self.feature_dim:
            raise ValueError(
                f"Expected feature dimension {self.feature_dim}, "
                f"got {frame_features.shape[0]}"
            )
        self.sequence_buffer.append(frame_features)

    def predict_activity(
        self,
        use_fallback: bool = False,
    ) -> Tuple[str, float, np.ndarray]:
        """
        Predict activity from current sequence buffer.

        Args:
            use_fallback: If True, use fallback heuristic classifier.

        Returns:
            Tuple of (activity_name, confidence, class_probabilities).
        """
        if len(self.sequence_buffer) < self.sequence_length:
            return 'unknown', 0.0, np.zeros(self.num_classes)

        if use_fallback or self.model is None or not TENSORFLOW_AVAILABLE:
            return self._fallback_classify()

        if not self.is_trained:
            if not self._warned_untrained:
                warnings.warn(
                    "ActivityClassifier has no trained weights; using the motion "
                    "heuristic instead. Call train() or load_model() first."
                )
                self._warned_untrained = True
            return self._fallback_classify()

        # Prepare sequence
        sequence = np.array(list(self.sequence_buffer), dtype=np.float32)
        sequence = np.expand_dims(sequence, axis=0)  # Add batch dimension

        # Normalize
        if self.scaler_mean is not None and self.scaler_std is not None:
            sequence = (sequence - self.scaler_mean) / (self.scaler_std + 1e-8)

        # Predict
        try:
            predictions = self.model.predict(sequence, verbose=0)
            class_idx = np.argmax(predictions[0])
            confidence = float(predictions[0][class_idx])

            return self.ACTIVITY_CLASSES[class_idx], confidence, predictions[0]

        except Exception as e:
            warnings.warn(f"Prediction failed: {e}. Using fallback classifier.")
            return self._fallback_classify()

    def _fallback_classify(self) -> Tuple[str, float, np.ndarray]:
        """
        Fallback heuristic based on overall motion magnitude in the buffer.

        This only separates standing / walking / running by thresholds on the
        landmark standard deviation. The returned "confidence" is a fixed score
        per rule, not a calibrated probability.

        Returns:
            Tuple of (activity_name, rule_score, one-hot-style scores).
        """
        # Extract statistics from sequence
        if len(self.sequence_buffer) == 0:
            return 'unknown', 0.0, np.zeros(self.num_classes)

        sequence = np.array(list(self.sequence_buffer))
        motion = np.mean(np.std(sequence, axis=0))

        # Simple heuristics
        if motion < 0.01:
            activity = 'standing'
            confidence = 0.7
        elif motion < 0.02:
            activity = 'walking'
            confidence = 0.65
        elif motion < 0.05:
            activity = 'running'
            confidence = 0.6
        else:
            activity = 'unknown'
            confidence = 0.5

        # Create dummy probabilities
        probs = np.zeros(self.num_classes)
        if activity in self.ACTIVITY_CLASSES:
            idx = self.ACTIVITY_CLASSES.index(activity)
            probs[idx] = confidence

        return activity, confidence, probs

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        epochs: int = 50,
        batch_size: int = 32,
        verbose: int = 1,
    ) -> Optional[Dict]:
        """
        Train the activity classifier.

        Args:
            X_train: Training sequences of shape (num_samples, sequence_length, feature_dim).
            y_train: Training labels (one-hot encoded).
            X_val: Validation sequences.
            y_val: Validation labels.
            epochs: Number of training epochs.
            batch_size: Batch size.
            verbose: Verbosity level.

        Returns:
            Training history dictionary or None if TensorFlow unavailable.
        """
        if self.model is None or not TENSORFLOW_AVAILABLE:
            warnings.warn("TensorFlow not available. Skipping training.")
            return None

        # Compute normalization statistics
        self.scaler_mean = np.mean(X_train, axis=(0, 1), keepdims=True)
        self.scaler_std = np.std(X_train, axis=(0, 1), keepdims=True)

        # Normalize data
        X_train_norm = (X_train - self.scaler_mean) / (self.scaler_std + 1e-8)
        X_val_norm = (X_val - self.scaler_mean) / (self.scaler_std + 1e-8)

        # Train model
        history = self.model.fit(
            X_train_norm,
            y_train,
            validation_data=(X_val_norm, y_val),
            epochs=epochs,
            batch_size=batch_size,
            verbose=verbose,
        )
        self.is_trained = True

        return history.history

    def evaluate(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray,
    ) -> Tuple[float, float]:
        """
        Evaluate model on test set.

        Args:
            X_test: Test sequences.
            y_test: Test labels.

        Returns:
            Tuple of (test_loss, test_accuracy).
        """
        if self.model is None or not TENSORFLOW_AVAILABLE:
            return 0.0, 0.0

        # Normalize
        X_test_norm = (X_test - self.scaler_mean) / (self.scaler_std + 1e-8)

        loss, accuracy = self.model.evaluate(X_test_norm, y_test, verbose=0)
        return loss, accuracy

    def save_model(self, filepath: str) -> None:
        """Save model weights to file."""
        if self.model is None or not TENSORFLOW_AVAILABLE:
            warnings.warn("Cannot save model. TensorFlow not available.")
            return

        self.model.save(filepath)

    def load_model(self, filepath: str) -> None:
        """Load model weights from file."""
        if not TENSORFLOW_AVAILABLE:
            warnings.warn("TensorFlow not available. Cannot load model.")
            return

        self.model = keras.models.load_model(filepath)
        self.is_trained = True

    def clear_sequence(self) -> None:
        """Clear the sequence buffer."""
        self.sequence_buffer.clear()

    def get_sequence_progress(self) -> float:
        """Get sequence buffer fill progress (0.0 to 1.0)."""
        return len(self.sequence_buffer) / self.sequence_length


class FeatureExtractor:
    """Extract features from pose landmarks for classification."""

    @staticmethod
    def extract_angle_features(
        angles: Dict[str, 'JointAngle'],
    ) -> np.ndarray:
        """
        Extract angle-based features.

        Args:
            angles: Dictionary of joint angles.

        Returns:
            Feature vector of angle values.
        """
        angle_names = [
            'left_elbow_angle',
            'right_elbow_angle',
            'left_shoulder_angle',
            'right_shoulder_angle',
            'left_hip_angle',
            'right_hip_angle',
            'left_knee_angle',
            'right_knee_angle',
        ]

        features = []
        for name in angle_names:
            if name in angles:
                features.append(angles[name].angle_degrees)
            else:
                features.append(0.0)

        return np.array(features, dtype=np.float32)

    @staticmethod
    def extract_velocity_features(
        velocities: Dict[str, 'JointVelocity'],
    ) -> np.ndarray:
        """
        Extract velocity-based features.

        Args:
            velocities: Dictionary of joint velocities.

        Returns:
            Feature vector of velocity magnitudes.
        """
        joint_names = [
            'left_shoulder', 'right_shoulder',
            'left_elbow', 'right_elbow',
            'left_wrist', 'right_wrist',
            'left_hip', 'right_hip',
            'left_knee', 'right_knee',
            'left_ankle', 'right_ankle',
        ]

        features = []
        for name in joint_names:
            if name in velocities:
                features.append(velocities[name].velocity_magnitude)
            else:
                features.append(0.0)

        return np.array(features, dtype=np.float32)

    @staticmethod
    def combine_features(
        landmarks: np.ndarray,
        angles: Optional[np.ndarray] = None,
        velocities: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Combine multiple feature types into single vector.

        Args:
            landmarks: Landmark features.
            angles: Angle features (optional).
            velocities: Velocity features (optional).

        Returns:
            Combined feature vector.
        """
        features = [landmarks]

        if angles is not None:
            features.append(angles)
        if velocities is not None:
            features.append(velocities)

        return np.concatenate(features)

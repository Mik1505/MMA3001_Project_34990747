"""
Machine Learning Quality Inspection and Defect Classification Module.

Implements feature engineering (color moments, flange seal redness, spatial anomaly ratios)
and flexible ML classifiers (Random Forest and Support Vector Machine) for automated
defect detection.
"""

from typing import List
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC


class AdvancedMLDetector:
    """Flexible Machine Learning Detector with spatial feature extraction.

    Extracts multi-domain statistical features from tray regions (flange, cavity,
    striations) and trains an ensemble or kernel classifier to predict Pass/Fail.

    Attributes:
        model_type (str): 'rf' for Random Forest, 'svm' for Support Vector Machine.
        model: Trained scikit-learn classifier.
    """

    def __init__(self, model_type: str = "rf") -> None:
        """Initializes the ML detector.

        Args:
            model_type: Classifier architecture ('rf' or 'svm'). Defaults to 'rf'.
        """
        self.model_type = model_type.lower()
        if self.model_type == "rf":
            self.model = RandomForestClassifier(
                n_estimators=100, max_depth=8, random_state=42
            )
        elif self.model_type == "svm":
            self.model = SVC(kernel="rbf", C=10.0, probability=True, random_state=42)
        else:
            raise ValueError(f"Unsupported model_type: {model_type}. Choose 'rf' or 'svm'.")

    @staticmethod
    def extract_features(img: np.ndarray) -> np.ndarray:
        """Extracts statistical, colorimetric, and spatial feature vectors from an image.

        Features extracted:
            - Global RGB channel means and standard deviations
            - Seal flange perimeter average intensities and redness metric
            - Cavity dark pixel density (tear indicator)
            - Cavity metmyoglobin discoloration density

        Args:
            img: 540x720x3 RGB uint8 array.

        Returns:
            np.ndarray: 12-dimensional float32 feature vector.
        """
        r = img[:, :, 0].astype(np.float32)
        g = img[:, :, 1].astype(np.float32)
        b = img[:, :, 2].astype(np.float32)

        # 1. Global illumination moments
        mean_r, std_r = float(np.mean(r)), float(np.std(r))
        mean_g, std_g = float(np.mean(g)), float(np.std(g))
        mean_b, std_b = float(np.mean(b)), float(np.std(b))

        # 2. Seal perimeter flange characteristics
        flange_r = float(np.mean(r[60:80, 80:640]))
        flange_g = float(np.mean(g[60:80, 80:640]))
        flange_b = float(np.mean(b[60:80, 80:640]))
        flange_redness = flange_r - 0.5 * (flange_g + flange_b)

        # 3. Cavity localized anomaly ratios
        cavity_r = r[80:460, 100:620]
        cavity_g = g[80:460, 100:620]
        cavity_b = b[80:460, 100:620]

        dark_ratio = float(
            np.mean((cavity_r < 40.0) & (cavity_g < 40.0) & (cavity_b < 40.0))
        )
        discolor_ratio = float(
            np.mean(
                (cavity_r < 130.0)
                & (cavity_r > 80.0)
                & (cavity_g < 110.0)
                & (cavity_g > 70.0)
            )
        )

        return np.array(
            [
                mean_r,
                std_r,
                mean_g,
                std_g,
                mean_b,
                std_b,
                flange_r,
                flange_g,
                flange_b,
                flange_redness,
                dark_ratio * 1000.0,
                discolor_ratio * 1000.0,
            ],
            dtype=np.float32,
        )

    def fit(self, images: List[np.ndarray], y: np.ndarray) -> "AdvancedMLDetector":
        """Trains the internal machine learning classifier on extracted features.

        Args:
            images: List of 540x720x3 image arrays.
            y: Array of binary labels (0 = Pass, 1 = Fail).

        Returns:
            AdvancedMLDetector: Self reference.
        """
        X = np.array([self.extract_features(img) for img in images], dtype=np.float32)
        self.model.fit(X, y)
        return self

    def predict(self, img: np.ndarray) -> int:
        """Predicts Pass (0) or Fail (1) for an inspection image.

        Args:
            img: 540x720x3 image array.

        Returns:
            int: Predicted class (0 = Pass, 1 = Fail).
        """
        feats = self.extract_features(img).reshape(1, -1)
        return int(self.model.predict(feats)[0])

    def predict_proba(self, img: np.ndarray) -> float:
        """Computes the estimated probability that the sample is defective (Fail).

        Args:
            img: 540x720x3 image array.

        Returns:
            float: Probability of class 1 (Fail) between 0.0 and 1.0.
        """
        feats = self.extract_features(img).reshape(1, -1)
        return float(self.model.predict_proba(feats)[0, 1])

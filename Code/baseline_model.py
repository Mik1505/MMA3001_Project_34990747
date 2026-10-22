"""
Baseline Rule-Based Quality and Defect Detector.

Implements standard classical computer-vision thresholding and morphological
filtering as an engineering baseline for comparative evaluation.
"""

from typing import Tuple
import numpy as np


class BaselineRuleBasedDetector:
    """Heuristic computer-vision detector based on calibrated color and geometry thresholds.

    Physical Constraint:
        Imperfections smaller than `min_defect_pixels` (default 150 px^2, ~16.6 mm^2)
        are categorized as benign cosmetic noise and will not trigger a line rejection.

    Attributes:
        min_defect_pixels (int): Minimum connected anomaly area to flag a failure.
    """

    def __init__(self, min_defect_pixels: int = 150) -> None:
        """Initializes the baseline detector.

        Args:
            min_defect_pixels: Minimum pixel area threshold for failure classification.
        """
        self.min_defect_pixels = min_defect_pixels

    def predict(self, img: np.ndarray) -> Tuple[int, int, np.ndarray]:
        """Classifies a packaging inspection frame into Pass (0) or Fail (1).

        Args:
            img: 540x720x3 RGB uint8 array.

        Returns:
            Tuple containing:
                - is_fail (int): 0 for Pass, 1 for Fail.
                - defect_pixels (int): Total number of detected defect pixels.
                - defect_mask (np.ndarray): 540x720 binary uint8 detected mask.
        """
        r = img[:, :, 0].astype(np.float32)
        g = img[:, :, 1].astype(np.float32)
        b = img[:, :, 2].astype(np.float32)

        # 1. Dark puncture / film rupture detection within inner tray cavity
        tray_mask = np.zeros((540, 720), dtype=bool)
        tray_mask[80:460, 100:620] = True
        dark_holes = (r < 40.0) & (g < 40.0) & (b < 40.0) & tray_mask

        # 2. Seal flange perimeter contamination detection (meat trapped in sealing borders)
        flange_mask = np.zeros((540, 720), dtype=bool)
        flange_mask[60:80, 80:640] = True
        flange_mask[460:480, 80:640] = True
        seal_contamination = (r > 130.0) & (g < 100.0) & (b < 110.0) & flange_mask

        # 3. Meat discoloration detection (loss of oxymyoglobin redness on meat rashers)
        meat_mask = np.zeros((540, 720), dtype=bool)
        meat_mask[95:445, 120:600] = True
        discolored = (
            (r < 130.0) & (r > 80.0) & (g < 110.0) & (g > 70.0) & (b < 95.0) & meat_mask
        )

        combined_mask = dark_holes | seal_contamination | discolored
        defect_pixel_count = int(np.sum(combined_mask))

        # Enforce physical severity threshold
        is_fail = 1 if defect_pixel_count >= self.min_defect_pixels else 0

        return is_fail, defect_pixel_count, combined_mask.astype(np.uint8)

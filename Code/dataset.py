"""
Synthetic Dataset Generator and Ingestion Pipeline.

Generates ground-truth benchmark datasets representing industrial pork rasher
packaging inspection, including normal packaging and verified failure modes.
"""

from typing import Tuple, List, Dict, Optional
from pathlib import Path
try:
    import cv2
except ImportError:
    cv2 = None

try:
    import yaml
except ImportError:
    yaml = None

import numpy as np


class SyntheticDatasetGenerator:
    """Generates synthetic benchmark datasets for pork rasher packaging inspection.

    Simulates:
        - Normal trays (Pass): Compliant packaging, clear thermoformed film, intact rashers.
        - Packaging tears (Fail): Jagged punctures, optical film breach.
        - Seal contamination (Fail): Foreign meat debris trapped in sealing flange.
        - Meat discoloration (Fail): Grayish-brown surface metmyoglobin oxidation.
        - Out-of-domain edge cases: Corrupted inputs, blank frames, random sensor noise.
    """

    def __init__(self, seed: int = 42, height: int = 540, width: int = 720) -> None:
        """Initializes the dataset generator.

        Args:
            seed: Random seed for deterministic generation. Defaults to 42.
            height: Image height in pixels. Defaults to 540.
            width: Image width in pixels. Defaults to 720.
        """
        self.rng = np.random.RandomState(seed)
        self.h = height
        self.w = width

    def generate_sample(self, label: str) -> Tuple[np.ndarray, np.ndarray, bool]:
        """Generates a single synthetic 540x720x3 image, ground-truth defect mask, and label.

        Args:
            label: Defect category ('pass', 'tear', 'seal_contam', 'discolor', 'unsupported').

        Returns:
            Tuple containing:
                - img (np.ndarray): 540x720x3 RGB uint8 array.
                - mask (np.ndarray): 540x720 binary uint8 defect mask.
                - is_defective (bool): True if sample represents a packaging or quality failure.
        """
        img = np.zeros((self.h, self.w, 3), dtype=np.uint8)
        mask = np.zeros((self.h, self.w), dtype=np.uint8)

        if label == "unsupported":
            if self.rng.rand() > 0.5:
                # Blackout sensor failure
                return img, mask, False
            else:
                # Uncorrelated uniform random noise
                noisy = self.rng.randint(0, 256, (self.h, self.w, 3), dtype=np.uint8)
                return noisy, mask, False

        # Conveyor belt matte finish
        img[:, :] = [45, 48, 52]

        # Thermoformed white polymer tray perimeter
        ty1, tx1, ty2, tx2 = 60, 80, 480, 640
        img[ty1:ty2, tx1:tx2] = [212, 216, 222]

        # Inner tray cavity
        cy1, cx1, cy2, cx2 = 80, 100, 460, 620
        img[cy1:cy2, cx1:cx2] = [172, 176, 182]

        # Packaged pork rashers with alternating lean and intramuscular fat striations
        rasher_y_starts = [100, 170, 240, 310, 380]
        for y0 in rasher_y_starts:
            # Lean muscle (oxymyoglobin red)
            img[y0 : y0 + 45, 120:600] = [192, 72, 82]
            # Intramuscular fat stripe
            img[y0 + 15 : y0 + 26, 120:600] = [232, 222, 208]

        # Add Gaussian sensor noise
        noise = self.rng.normal(0, 3.5, (self.h, self.w, 3)).astype(np.int16)
        img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        is_defective = False

        if label == "tear":
            py = self.rng.randint(150, 380)
            px = self.rng.randint(180, 500)
            radius = self.rng.randint(18, 36)
            yy, xx = np.ogrid[: self.h, : self.w]
            dist = np.sqrt((xx - px) ** 2 + (yy - py) ** 2)
            defect_region = dist <= radius
            img[defect_region] = [22, 22, 28]  # Dark puncture / rupture
            mask[defect_region] = 1
            is_defective = True

        elif label == "seal_contam":
            sy = self.rng.randint(ty1, ty1 + 18)
            sx = self.rng.randint(tx1 + 60, tx2 - 120)
            img[sy : sy + 14, sx : sx + 45] = [168, 52, 62]  # Trapped meat in seal
            mask[sy : sy + 14, sx : sx + 45] = 1
            is_defective = True

        elif label == "discolor":
            dy = self.rng.randint(120, 380)
            dx = self.rng.randint(150, 520)
            yy, xx = np.ogrid[: self.h, : self.w]
            dist = np.sqrt((xx - dx) ** 2 + (yy - dy) ** 2)
            discolor_region = dist <= 32
            img[discolor_region] = [108, 98, 78]  # Metmyoglobin brown oxidation
            mask[discolor_region] = 1
            is_defective = True

        return img, mask, is_defective

    def build_dataset(
        self, n_samples: int = 240
    ) -> Tuple[List[np.ndarray], List[np.ndarray], np.ndarray, List[str]]:
        """Constructs a balanced multi-class dataset for training and verification.

        Args:
            n_samples: Total number of samples to synthesize.

        Returns:
            Tuple containing:
                - images (List[np.ndarray]): List of image arrays.
                - masks (List[np.ndarray]): List of binary ground-truth masks.
                - labels (np.ndarray): 1D array of binary labels (0 = Pass, 1 = Fail).
                - categories (List[str]): List of defect category strings.
        """
        images, masks, labels, categories = [], [], [], []
        defect_classes = ["pass", "tear", "seal_contam", "discolor"]

        for i in range(n_samples):
            cls_name = defect_classes[i % len(defect_classes)]
            img, msk, is_def = self.generate_sample(cls_name)
            images.append(img)
            masks.append(msk)
            labels.append(1 if is_def else 0)
            categories.append(cls_name)

        return images, masks, np.array(labels, dtype=np.int32), categories


class RoboflowDatasetLoader:
    """Ingests real pork rasher images and bounding box annotations from Roboflow Version 4.

    Attributes:
        dataset_dir (Path): Root directory containing Roboflow YOLOv8 export (train, valid, test).
        class_names (Dict[int, str]): Mapping from YOLO class indices to descriptive category names.
    """

    def __init__(self, dataset_dir: str = "pork-rasher-error-packaging-4") -> None:
        """Initializes the dataset loader and loads class mapping from data.yaml.

        Args:
            dataset_dir: Path to directory containing YOLOv8 export.
        """
        self.dataset_dir = Path(dataset_dir)
        yaml_path = self.dataset_dir / "data.yaml"
        self.class_names: Dict[int, str] = {}
        if yaml_path.exists():
            with open(yaml_path, "r") as f:
                data_cfg = yaml.safe_load(f)
                names = data_cfg.get("names", [])
                if isinstance(names, dict):
                    self.class_names = names
                elif isinstance(names, list):
                    self.class_names = {i: name for i, name in enumerate(names)}

    def load_split(
        self, split: str = "train", max_samples: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
        """Loads RGB frames, binary defect masks, binary labels, and category strings for a split.

        Args:
            split: 'train', 'valid', or 'test' partition.
            max_samples: Optional cap on the number of samples loaded.

        Returns:
            Tuple containing:
                - images (np.ndarray): Array of shape (N, 540, 720, 3) uint8.
                - masks (np.ndarray): Array of shape (N, 540, 720) uint8 binary defect masks.
                - labels (np.ndarray): Array of shape (N,) int32 (0 = Pass, 1 = Fail).
                - categories (List[str]): List of N category strings.
        """
        img_dir = self.dataset_dir / split / "images"
        lbl_dir = self.dataset_dir / split / "labels"

        if not img_dir.exists():
            raise FileNotFoundError(f"Image directory not found: {img_dir}")

        img_files = sorted(
            list(img_dir.glob("*.jpg"))
            + list(img_dir.glob("*.png"))
            + list(img_dir.glob("*.jpeg"))
        )
        if max_samples:
            img_files = img_files[:max_samples]

        images, masks, labels, categories = [], [], [], []

        for img_path in img_files:
            bgr = cv2.imread(str(img_path))
            if bgr is None:
                continue
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            if rgb.shape[:2] != (540, 720):
                rgb = cv2.resize(rgb, (720, 540))

            h, w, _ = rgb.shape
            mask = np.zeros((h, w), dtype=np.uint8)
            lbl_path = lbl_dir / f"{img_path.stem}.txt"

            sample_cat = "pass"
            is_defective = 0

            if lbl_path.exists() and lbl_path.stat().st_size > 0:
                with open(lbl_path, "r") as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            is_defective = 1
                            cls_id = int(float(parts[0]))
                            sample_cat = self.class_names.get(cls_id, "defect")
                            x_c, y_c, bw, bh = map(float, parts[1:5])
                            x1 = int((x_c - bw / 2) * w)
                            y1 = int((y_c - bh / 2) * h)
                            x2 = int((x_c + bw / 2) * w)
                            y2 = int((y_c + bh / 2) * h)
                            mask[max(0, y1) : min(h, y2), max(0, x1) : min(w, x2)] = 1

            images.append(rgb)
            masks.append(mask)
            labels.append(is_defective)
            categories.append(sample_cat)

        return (
            np.array(images, dtype=np.uint8),
            np.array(masks, dtype=np.uint8),
            np.array(labels, dtype=np.int32),
            categories,
        )

"""
Computational Profiler, Arithmetic Intensity, and Roofline Analysis Module.

Provides timing routines, throughput metrics (FPS), FLOP estimation,
and operational arithmetic intensity for the inspection models.
"""

from typing import Dict, List, Any, Tuple
import time
import numpy as np


class ComputationalProfiler:
    """Profiles computational latency, memory footprint, and arithmetic intensity."""

    @staticmethod
    def profile_detector(
        detector: Any, sample_images: List[np.ndarray], n_repeats: int = 25
    ) -> Dict[str, float]:
        """Measures per-frame execution latency statistics across an image sample set.

        Args:
            detector: Detection model instance with a .predict() method.
            sample_images: List of 540x720x3 image arrays.
            n_repeats: Number of evaluation iterations.

        Returns:
            Dict containing mean, std, max, 95th-percentile latencies (ms) and throughput (FPS).
        """
        latencies = []
        n_imgs = len(sample_images)

        for i in range(n_repeats):
            img = sample_images[i % n_imgs]
            t0 = time.perf_counter()
            _ = detector.predict(img)
            t1 = time.perf_counter()
            latencies.append((t1 - t0) * 1000.0)

        mean_ms = float(np.mean(latencies))
        return {
            "mean_ms": mean_ms,
            "std_ms": float(np.std(latencies)),
            "max_ms": float(np.max(latencies)),
            "p95_ms": float(np.percentile(latencies, 95)),
            "throughput_fps": 1000.0 / mean_ms if mean_ms > 0 else 0.0,
        }

    @staticmethod
    def estimate_arithmetic_intensity(
        img_shape: Tuple[int, int, int] = (540, 720, 3)
    ) -> Dict[str, float]:
        """Calculates total FLOPs and operational arithmetic intensity (FLOPs/byte).

        For a 540x720x3 image tensor:
            Memory footprint = 540 * 720 * 3 = 1,166,400 bytes (~1.17 MB).

        Baseline Model:
            - Channel threshold comparisons & logical operations: ~12 ops/pixel.
            - Total FLOPs ~ 4.67 MFLOPs.
            - Arithmetic Intensity = 4.67M / 1.166MB ~ 4.0 FLOPs/Byte.

        Machine Learning Model:
            - Regional statistical moments (mean, var, color ratios): ~15 ops/pixel.
            - Random Forest ensemble tree routing (100 trees, depth 8): ~800 ops.
            - Total FLOPs ~ 5.83 MFLOPs.
            - Arithmetic Intensity = 5.83M / 1.166MB ~ 5.0 FLOPs/Byte.

        Both algorithms operate strictly in the memory-bound regime on modern hardware.

        Args:
            img_shape: Tuple of (height, width, channels).

        Returns:
            Dict of FLOPs, memory footprint, and arithmetic intensities.
        """
        num_pixels = img_shape[0] * img_shape[1]
        bytes_transferred = float(num_pixels * img_shape[2])

        flops_base = float(num_pixels * 12)
        intensity_base = flops_base / bytes_transferred

        flops_ml = float(num_pixels * 15 + 800)
        intensity_ml = flops_ml / bytes_transferred

        return {
            "bytes_transferred": bytes_transferred,
            "flops_baseline": flops_base,
            "intensity_baseline": intensity_base,
            "flops_ml": flops_ml,
            "intensity_ml": intensity_ml,
        }

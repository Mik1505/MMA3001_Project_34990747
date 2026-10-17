r"""
2D Numerical Integration and Distributed Defect Sizing Module.

Calculates the distributed physical surface area of packaging defects via 2D
quadrature:
    Area = \iint_\Omega I_{defect}(x, y) \, dx \, dy
Implements Composite 2D Trapezoidal and Composite 2D Simpson's 1/3 rules,
along with spatial grid refinement convergence tests and physical plausibility checks.
"""

from typing import Tuple, List, Dict
import numpy as np


class NumericalIntegrator2D:
    """Computes physical area integrals of detected defect distributions.

    Attributes:
        tray_w (float): Physical tray width in mm (default 240.0 mm).
        tray_h (float): Physical tray height in mm (default 180.0 mm).
        dx (float): Spatial step size along x-axis (mm/pixel).
        dy (float): Spatial step size along y-axis (mm/pixel).
        total_tray_area (float): Bounding physical tray area in mm^2.
    """

    def __init__(
        self,
        tray_width_mm: float = 240.0,
        tray_height_mm: float = 180.0,
        grid_w: int = 720,
        grid_h: int = 540,
    ) -> None:
        """Initializes spatial parameters and grid resolution.

        Args:
            tray_width_mm: Physical horizontal dimension of tray in mm.
            tray_height_mm: Physical vertical dimension of tray in mm.
            grid_w: Number of pixel columns in full resolution.
            grid_h: Number of pixel rows in full resolution.
        """
        self.tray_w = tray_width_mm
        self.tray_h = tray_height_mm
        self.grid_w = grid_w
        self.grid_h = grid_h
        self.dx = tray_width_mm / grid_w
        self.dy = tray_height_mm / grid_h
        self.total_tray_area = tray_width_mm * tray_height_mm

    def trapezoidal_2d(self, mask: np.ndarray) -> float:
        r"""Evaluates 2D composite trapezoidal integration over gridded domain.

        Weights along each dimension: w = [0.5, 1.0, 1.0, ..., 1.0, 0.5].
        2D weight matrix: W_{2D} = w_y \otimes w_x.

        Args:
            mask: 2D binary defect mask array of shape (H, W).

        Returns:
            float: Integrated physical defect area in mm^2.
        """
        H, W = mask.shape
        dx = self.tray_w / W
        dy = self.tray_h / H

        wx = np.ones(W, dtype=np.float64)
        wx[0] = 0.5
        wx[-1] = 0.5

        wy = np.ones(H, dtype=np.float64)
        wy[0] = 0.5
        wy[-1] = 0.5

        w2d = np.outer(wy, wx)
        area = float(np.sum(mask.astype(np.float64) * w2d) * dx * dy)
        return area

    def simpsons_2d(self, mask: np.ndarray) -> float:
        """Evaluates 2D composite Simpson's 1/3 rule.

        Ensures odd grid dimension count along both spatial axes.

        Args:
            mask: 2D binary defect mask array.

        Returns:
            float: Integrated physical defect area in mm^2.
        """
        H, W = mask.shape
        if H % 2 == 0:
            mask = mask[:-1, :]
            H -= 1
        if W % 2 == 0:
            mask = mask[:, :-1]
            W -= 1

        dx = self.tray_w / (W - 1)
        dy = self.tray_h / (H - 1)

        def simpson_weights(n: int) -> np.ndarray:
            w = np.ones(n, dtype=np.float64)
            w[1:-1:2] = 4.0
            w[2:-2:2] = 2.0
            return w / 3.0

        wx = simpson_weights(W)
        wy = simpson_weights(H)
        w2d = np.outer(wy, wx)

        area = float(np.sum(mask.astype(np.float64) * w2d) * dx * dy)
        return area

    def grid_refinement_study(
        self, mask: np.ndarray, levels: Tuple[int, ...] = (1, 2, 4, 8)
    ) -> List[Dict[str, float]]:
        """Conducts systematic spatial resolution refinement study.

        Args:
            mask: Full resolution 540x720 binary defect mask.
            levels: Spatial downsampling factors.

        Returns:
            List of dictionaries containing step sizes, quadrature results, and relative errors.
        """
        records = []
        for s in levels:
            sub = mask[::s, ::s]
            trap_val = self.trapezoidal_2d(sub)
            simp_val = self.simpsons_2d(sub)
            records.append(
                {
                    "subsample_factor": s,
                    "grid_h": sub.shape[0],
                    "grid_w": sub.shape[1],
                    "dx_mm": self.dx * s,
                    "dy_mm": self.dy * s,
                    "trapezoidal_area_mm2": trap_val,
                    "simpsons_area_mm2": simp_val,
                }
            )
        return records

    def validate_integral(self, computed_area: float) -> Tuple[bool, str]:
        """Verifies physical plausibility against the tray bounding envelope.

        Args:
            computed_area: Calculated defect area in mm^2.

        Returns:
            Tuple of (is_valid: bool, reason: str).
        """
        if computed_area < 0.0:
            return False, f"ERR_NEGATIVE_AREA: Area {computed_area} < 0 violating non-negativity."
        if computed_area > self.total_tray_area:
            return (
                False,
                f"ERR_OVERFLOW: Calculated area ({computed_area:.1f} mm^2) exceeds physical tray boundary ({self.total_tray_area:.1f} mm^2).",
            )
        return True, "PASSED: Area satisfies physical geometry envelope."

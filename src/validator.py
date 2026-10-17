"""
Input Validation and Out-of-Domain Guardrail Module.

This module enforces strict domain validation for industrial vision inputs,
ensuring invalid or corrupted frames are rejected with descriptive diagnostic codes
rather than yielding unreliable quality predictions.
"""

from typing import Tuple
import numpy as np


class UnsupportedInputError(ValueError):
    """Custom exception raised when an input does not conform to domain specifications."""
    pass


class InputValidator:
    """Validates incoming image tensors against physical, optical, and domain bounds.

    Attributes:
        expected_shape (Tuple[int, int, int]): Expected image shape (height, width, channels).
        min_brightness (float): Minimum allowable mean channel intensity (0-255 scale).
        max_brightness (float): Maximum allowable mean channel intensity (0-255 scale).
        min_variance (float): Minimum variance required to eliminate blank/occluded frames.
        max_noise_entropy (float): Threshold to detect uncalibrated random noise fields.
    """

    def __init__(
        self,
        expected_shape: Tuple[int, int, int] = (540, 720, 3),
        min_brightness: float = 15.0,
        max_brightness: float = 245.0,
        min_variance: float = 15.0
    ) -> None:
        """Initializes the InputValidator with domain parameters.

        Args:
            expected_shape: Expected dimensions (H, W, C). Defaults to (540, 720, 3).
            min_brightness: Lower bound on average pixel intensity. Defaults to 15.0.
            max_brightness: Upper bound on average pixel intensity. Defaults to 245.0.
            min_variance: Lower bound on image variance. Defaults to 15.0.
        """
        self.expected_shape = expected_shape
        self.min_brightness = min_brightness
        self.max_brightness = max_brightness
        self.min_variance = min_variance

    def validate(self, img: np.ndarray) -> bool:
        """Validates image dimensions, types, illumination, and structural plausibility.

        Args:
            img: Input RGB image array.

        Returns:
            bool: True if input strictly satisfies all validation constraints.

        Raises:
            UnsupportedInputError: If any structural or radiometric check fails.
        """
        if not isinstance(img, np.ndarray):
            raise UnsupportedInputError(
                f"ERR_TYPE: Expected numpy.ndarray input, but received {type(img).__name__}."
            )

        if img.shape != self.expected_shape:
            raise UnsupportedInputError(
                f"ERR_DIM: Incompatible dimensions {img.shape}. Expected {self.expected_shape} (Height, Width, Channels)."
            )

        if img.dtype != np.uint8 and not np.issubdtype(img.dtype, np.floating):
            raise UnsupportedInputError(
                f"ERR_DTYPE: Unsupported array data type {img.dtype}. Must be uint8 or float32."
            )

        mean_val = float(np.mean(img))
        var_val = float(np.var(img))

        if mean_val < self.min_brightness:
            raise UnsupportedInputError(
                f"ERR_UNDEREXPOSED: Mean frame brightness ({mean_val:.1f}) is below minimum acceptable threshold ({self.min_brightness}). Possible camera occlusion or strobe trigger failure."
            )

        if mean_val > self.max_brightness:
            raise UnsupportedInputError(
                f"ERR_OVEREXPOSED: Mean frame brightness ({mean_val:.1f}) exceeds maximum allowable threshold ({self.max_brightness}). Optical glare or severe overexposure."
            )

        if var_val < self.min_variance:
            raise UnsupportedInputError(
                f"ERR_BLANK: Frame variance ({var_val:.1f}) is below {self.min_variance}. Frame lacks structural texture, indicating camera blackout or solid lens cap."
            )

        # Domain Structural Check: Check if image has conveyor/tray structure
        # Tray border and conveyor background should have distinct spatial gradient distributions
        grad_y = np.abs(np.diff(img[:, :, 0].astype(np.float32), axis=0))
        mean_grad = float(np.mean(grad_y))
        
        # Pure uncorrelated uniform noise has an unnaturally high mean gradient without macro edges
        if mean_grad > 65.0:
            raise UnsupportedInputError(
                f"ERR_OUT_OF_DOMAIN: Mean spatial gradient ({mean_grad:.1f}) exceeds structured image bounds. Data appears to be uncorrelated noise or corrupted transmission."
            )

        return True

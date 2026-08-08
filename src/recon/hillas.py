from dataclasses import dataclass
from typing import Any, Optional

import numpy as np


@dataclass
class HillasParameters:
    size: float
    centroid_x: float
    centroid_y: float
    length: float
    width: float
    psi: float
    miss: float
    distance: float
    alpha: float
    asymmetry: float
    third_moment: float = 0.0

    @property
    def skewness(self) -> float:
        """Dimensionless longitudinal skewness (an explicit asymmetry alias)."""
        return self.asymmetry


def _as_finite_vector(values: Any, name: str) -> np.ndarray:
    """Return a one-dimensional float64 array after strict numeric validation."""
    try:
        array = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a one-dimensional numeric array") from exc

    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


def _as_boolean_mask(mask: Any, expected_shape: tuple[int, ...]) -> np.ndarray:
    """Validate a boolean mask, accepting finite binary numeric arrays for compatibility."""
    array = np.asarray(mask)
    if array.shape != expected_shape or array.ndim != 1:
        raise ValueError(f"mask must have shape {expected_shape}")

    if np.issubdtype(array.dtype, np.bool_):
        return array.astype(bool, copy=False)

    if not np.issubdtype(array.dtype, np.number):
        raise ValueError("mask must be boolean or contain only binary numeric values")
    if not np.all(np.isfinite(array)):
        raise ValueError("mask must contain only finite values")
    if not np.all((array == 0) | (array == 1)):
        raise ValueError("mask must contain only boolean or binary values")
    return array.astype(bool)


def _wrap_angle(angle: float) -> float:
    """Wrap an angle to [-pi, pi)."""
    return float((angle + np.pi) % (2.0 * np.pi) - np.pi)


def compute_hillas(
    camera: Any, image: np.ndarray, mask: np.ndarray
) -> Optional[HillasParameters]:
    """Compute charge-weighted Hillas parameters for a cleaned camera image.

    ``psi`` is a *directed* major-axis angle in radians.  The 180-degree
    ambiguity is resolved by pointing away from the camera origin whenever
    the centroid permits that choice.  ``asymmetry`` (also available through
    the more explicit ``skewness`` property) is the standardized third central
    moment along precisely that direction.  ``third_moment`` retains the
    corresponding unstandardized moment in camera-coordinate units cubed.

    Args:
        camera: Object with one-dimensional ``pixel_x`` and ``pixel_y`` arrays.
        image: Finite baseline-subtracted pixel signals (photoelectrons).
            Negative pedestal fluctuations are permitted outside ``mask``;
            selected charges must be nonnegative.
        mask: Boolean mask, or a finite binary array, selecting cleaned pixels.

    Returns:
        Hillas parameters, or ``None`` when fewer than three positive-signal
        pixels survive or their total signal is zero.

    Raises:
        ValueError: If arrays have incompatible shapes or contain invalid data.
    """
    image_array = _as_finite_vector(image, "image")
    mask_array = _as_boolean_mask(mask, image_array.shape)
    if np.any(image_array[mask_array] < 0.0):
        raise ValueError("selected image values must be nonnegative")
    pixel_x = _as_finite_vector(camera.pixel_x, "camera.pixel_x")
    pixel_y = _as_finite_vector(camera.pixel_y, "camera.pixel_y")
    if pixel_x.shape != image_array.shape or pixel_y.shape != image_array.shape:
        raise ValueError("camera coordinates and image must have the same shape")

    # Zero-charge pixels carry no moment information and must not satisfy the
    # minimum-pixel requirement on their own. Negative pedestal fluctuations
    # outside the cleaned mask have already been allowed above.
    selected = mask_array & (image_array > 0.0)
    if np.count_nonzero(selected) < 3:
        return None

    x = pixel_x[selected]
    y = pixel_y[selected]
    signal = image_array[selected]

    size = float(np.sum(signal, dtype=np.float64))
    if not np.isfinite(size):
        raise ValueError("the selected image charge is too large to represent")
    if size <= 0.0:
        return None

    centroid_x = float(np.dot(signal, x) / size)
    centroid_y = float(np.dot(signal, y) / size)

    centered_x = x - centroid_x
    centered_y = y - centroid_y
    covariance_xx = float(np.dot(signal, centered_x * centered_x) / size)
    covariance_yy = float(np.dot(signal, centered_y * centered_y) / size)
    covariance_xy = float(np.dot(signal, centered_x * centered_y) / size)

    difference = covariance_xx - covariance_yy
    discriminant = float(np.hypot(difference, 2.0 * covariance_xy))
    major_variance = max(
        0.5 * (covariance_xx + covariance_yy + discriminant), 0.0
    )
    minor_variance = max(
        0.5 * (covariance_xx + covariance_yy - discriminant), 0.0
    )
    length = float(np.sqrt(major_variance))
    width = float(np.sqrt(minor_variance))

    if np.isclose(discriminant, 0.0, rtol=0.0, atol=np.finfo(float).eps):
        psi = 0.0
    else:
        psi = float(0.5 * np.arctan2(2.0 * covariance_xy, difference))

    # Resolve the eigenvector sign and update psi itself.  Previously only a
    # temporary direction vector was flipped, so plots using psi could point
    # opposite to the axis used for the reported asymmetry.
    direction_x = float(np.cos(psi))
    direction_y = float(np.sin(psi))
    radial_projection = direction_x * centroid_x + direction_y * centroid_y
    if radial_projection < 0.0:
        psi = _wrap_angle(psi + np.pi)
        direction_x = -direction_x
        direction_y = -direction_y

    longitudinal = centered_x * direction_x + centered_y * direction_y
    third_moment = float(np.dot(signal, longitudinal**3) / size)
    if length > 0.0:
        asymmetry = float(third_moment / (length**3))
    else:
        asymmetry = 0.0

    miss = float(
        abs(centroid_x * direction_y - centroid_y * direction_x)
    )
    distance = float(np.hypot(centroid_x, centroid_y))
    if distance > 0.0:
        alpha = float(
            np.degrees(np.arcsin(np.clip(miss / distance, 0.0, 1.0)))
        )
    else:
        alpha = 0.0

    return HillasParameters(
        size=size,
        centroid_x=centroid_x,
        centroid_y=centroid_y,
        length=length,
        width=width,
        psi=psi,
        miss=miss,
        distance=distance,
        alpha=min(abs(alpha), 90.0),
        asymmetry=asymmetry,
        third_moment=third_moment,
    )


def print_hillas(params: HillasParameters) -> None:
    """Print Hillas parameters in a formatted table."""
    if params is None:
        print("HillasParameters: None")
        return

    print("-" * 40)
    print("Hillas Parameters")
    print("-" * 40)
    print(f"{'Size (pe)':<20} : {params.size:.2f}")
    print(f"{'Centroid X (deg)':<20} : {params.centroid_x:.4f}")
    print(f"{'Centroid Y (deg)':<20} : {params.centroid_y:.4f}")
    print(f"{'Length (deg)':<20} : {params.length:.4f}")
    print(f"{'Width (deg)':<20} : {params.width:.4f}")
    print(f"{'Psi (rad)':<20} : {params.psi:.4f}")
    print(f"{'Miss (deg)':<20} : {params.miss:.4f}")
    print(f"{'Distance (deg)':<20} : {params.distance:.4f}")
    print(f"{'Alpha (deg)':<20} : {params.alpha:.4f}")
    print(f"{'Skewness':<20} : {params.skewness:.4e}")
    print(f"{'Third moment':<20} : {params.third_moment:.4e}")
    print("-" * 40)

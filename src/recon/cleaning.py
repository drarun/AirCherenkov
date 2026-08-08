from typing import Any, Optional

import numpy as np


def _validated_image(image: Any) -> np.ndarray:
    try:
        array = np.asarray(image, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("image must be a one-dimensional numeric array") from exc
    if array.ndim != 1:
        raise ValueError("image must be one-dimensional")
    if not np.all(np.isfinite(array)):
        raise ValueError("image must contain only finite values")
    return array


def _validated_threshold(value: Any, name: str) -> float:
    try:
        threshold = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite nonnegative number") from exc
    if not np.isfinite(threshold) or threshold < 0.0:
        raise ValueError(f"{name} must be a finite nonnegative number")
    return threshold


def _validated_neighbors(camera: Any, n_pixels: int) -> np.ndarray:
    neighbors = np.asarray(camera.get_neighbor_matrix())
    if neighbors.shape != (n_pixels, n_pixels):
        raise ValueError(
            f"camera neighbor matrix must have shape {(n_pixels, n_pixels)}"
        )
    if not (
        np.issubdtype(neighbors.dtype, np.bool_)
        or np.issubdtype(neighbors.dtype, np.number)
    ):
        raise ValueError("camera neighbor matrix must be boolean or numeric")
    if np.issubdtype(neighbors.dtype, np.number) and not np.all(
        np.isfinite(neighbors)
    ):
        raise ValueError("camera neighbor matrix must contain only finite values")

    neighbors = neighbors.astype(bool, copy=True)
    np.fill_diagonal(neighbors, False)
    # Camera adjacency is physically undirected.  Treat either reported edge
    # direction as evidence of a connection, which is robust to sparse-matrix
    # construction details while retaining deterministic cleaning.
    return neighbors | neighbors.T


def _validated_geometry(camera: Any, n_pixels: int) -> tuple[np.ndarray, np.ndarray]:
    try:
        pixel_x = np.asarray(camera.pixel_x, dtype=np.float64)
        pixel_y = np.asarray(camera.pixel_y, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("camera coordinates must be numeric arrays") from exc
    if pixel_x.shape != (n_pixels,) or pixel_y.shape != (n_pixels,):
        raise ValueError("camera coordinates and image must have the same shape")
    if not np.all(np.isfinite(pixel_x)) or not np.all(np.isfinite(pixel_y)):
        raise ValueError("camera coordinates must contain only finite values")
    return pixel_x, pixel_y


def _connected_to_seed(
    neighbors: np.ndarray, eligible: np.ndarray, seed: np.ndarray
) -> np.ndarray:
    """Return eligible pixels graph-connected to any seed pixel."""
    connected = seed.copy()
    allowed = eligible | seed
    frontier = seed.copy()
    while np.any(frontier):
        adjacent = np.any(neighbors[frontier], axis=0)
        frontier = adjacent & allowed & ~connected
        connected |= frontier
    return connected


def tail_cut_clean(
    camera: Any,
    image: np.ndarray,
    picture_thresh: float = 5.0,
    boundary_thresh: float = 2.5,
    min_neighbors: int = 1,
) -> np.ndarray:
    """Perform standard two-level tail-cut image cleaning.

    Finite baseline-subtracted camera charges are accepted, including negative
    pedestal fluctuations.  With nonnegative thresholds those fluctuations
    cannot enter the cleaned mask.  NaN, infinity, and malformed camera
    adjacency are rejected.
    """
    image_array = _validated_image(image)
    picture_thresh = _validated_threshold(picture_thresh, "picture_thresh")
    boundary_thresh = _validated_threshold(boundary_thresh, "boundary_thresh")
    if isinstance(min_neighbors, (bool, np.bool_)) or not isinstance(
        min_neighbors, (int, np.integer)
    ):
        raise ValueError("min_neighbors must be a nonnegative integer")
    if min_neighbors < 0:
        raise ValueError("min_neighbors must be a nonnegative integer")

    neighbors = _validated_neighbors(camera, image_array.size)
    is_above_picture = image_array >= picture_thresh
    picture_neighbor_counts = np.sum(
        neighbors & is_above_picture[None, :], axis=1
    )
    picture_mask = is_above_picture & (
        picture_neighbor_counts >= min_neighbors
    )

    is_above_boundary = image_array >= boundary_thresh
    adjacent_to_picture = np.any(neighbors & picture_mask[None, :], axis=1)
    boundary_mask = is_above_boundary & adjacent_to_picture & ~picture_mask
    return picture_mask | boundary_mask


def double_pass_clean(
    camera: Any,
    image: np.ndarray,
    pic1: float = 5.0,
    bnd1: float = 2.5,
    pic2: float = 2.0,
    bnd2: float = 1.0,
    dist_tolerance: float = 0.15,
    longitudinal_tolerance: Optional[float] = None,
) -> np.ndarray:
    """Clean an image twice while recovering only a connected faint fringe.

    The second pass must satisfy three conditions: the lower tail cuts, the
    perpendicular distance from the first-pass major axis, and graph
    connectivity to the first-pass core.  It is also restricted to the core's
    finite longitudinal interval plus ``longitudinal_tolerance`` on each end.
    If omitted, that margin is the larger of ``dist_tolerance`` and two camera
    pixel spacings.  Timing is not applied because it is not part of this API.

    Args:
        camera: Camera geometry with coordinates and a neighbor matrix.
        image: Finite raw pixel amplitudes. Baseline-subtracted pedestal
            fluctuations may be negative.
        pic1, bnd1: First-pass picture and boundary thresholds.
        pic2, bnd2: Second-pass picture and boundary thresholds.
        dist_tolerance: Maximum perpendicular distance from the core axis.
        longitudinal_tolerance: Maximum extension beyond each end of the core.

    Returns:
        Boolean retained-pixel mask.
    """
    image_array = _validated_image(image)
    dist_tolerance = _validated_threshold(dist_tolerance, "dist_tolerance")
    if longitudinal_tolerance is not None:
        longitudinal_tolerance = _validated_threshold(
            longitudinal_tolerance, "longitudinal_tolerance"
        )

    # tail_cut_clean also validates all four threshold arguments.
    mask_pass1 = tail_cut_clean(camera, image_array, pic1, bnd1)
    if np.count_nonzero(mask_pass1) < 3:
        return mask_pass1

    pixel_x, pixel_y = _validated_geometry(camera, image_array.size)
    neighbors = _validated_neighbors(camera, image_array.size)
    x_core = pixel_x[mask_pass1]
    y_core = pixel_y[mask_pass1]
    weights = image_array[mask_pass1]
    weight_sum = float(np.sum(weights, dtype=np.float64))
    if weight_sum <= 0.0:
        return np.zeros_like(mask_pass1)

    centroid_x = float(np.dot(x_core, weights) / weight_sum)
    centroid_y = float(np.dot(y_core, weights) / weight_sum)
    centered_x = x_core - centroid_x
    centered_y = y_core - centroid_y
    covariance_xx = float(np.dot(weights, centered_x**2) / weight_sum)
    covariance_yy = float(np.dot(weights, centered_y**2) / weight_sum)
    covariance_xy = float(np.dot(weights, centered_x * centered_y) / weight_sum)
    axis_angle = 0.5 * np.arctan2(
        2.0 * covariance_xy, covariance_xx - covariance_yy
    )
    axis_x = float(np.cos(axis_angle))
    axis_y = float(np.sin(axis_angle))

    mask_pass2_candidates = tail_cut_clean(camera, image_array, pic2, bnd2)
    all_centered_x = pixel_x - centroid_x
    all_centered_y = pixel_y - centroid_y
    perpendicular_distance = np.abs(
        all_centered_x * axis_y - all_centered_y * axis_x
    )
    longitudinal_position = (
        all_centered_x * axis_x + all_centered_y * axis_y
    )
    core_longitudinal = longitudinal_position[mask_pass1]

    if longitudinal_tolerance is None:
        pixel_spacing = getattr(camera, "pixel_size", None)
        try:
            pixel_spacing = float(pixel_spacing)
        except (TypeError, ValueError):
            pixel_spacing = np.nan
        if not np.isfinite(pixel_spacing) or pixel_spacing <= 0.0:
            edge_rows, edge_columns = np.nonzero(np.triu(neighbors, k=1))
            if edge_rows.size:
                edge_lengths = np.hypot(
                    pixel_x[edge_rows] - pixel_x[edge_columns],
                    pixel_y[edge_rows] - pixel_y[edge_columns],
                )
                positive_lengths = edge_lengths[edge_lengths > 0.0]
                pixel_spacing = (
                    float(np.median(positive_lengths))
                    if positive_lengths.size
                    else dist_tolerance
                )
            else:
                pixel_spacing = dist_tolerance
        longitudinal_tolerance = max(dist_tolerance, 2.0 * pixel_spacing)

    within_axis_band = perpendicular_distance <= dist_tolerance
    within_longitudinal_extent = (
        longitudinal_position >= np.min(core_longitudinal) - longitudinal_tolerance
    ) & (
        longitudinal_position <= np.max(core_longitudinal) + longitudinal_tolerance
    )
    eligible_pass2 = (
        mask_pass2_candidates & within_axis_band & within_longitudinal_extent
    )

    return _connected_to_seed(neighbors, eligible_pass2, mask_pass1)

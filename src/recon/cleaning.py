from dataclasses import dataclass
from typing import Any, Optional, Tuple

import numpy as np


@dataclass
class DoublePassResult:
    """Result of VERITAS-style two-pass FADC trace integration and image cleaning."""
    charges: np.ndarray
    mask: np.ndarray
    arrival_times_ns: np.ndarray
    time_gradient: float
    time_intercept: float
    time_rms: float
    loss1: float
    loss2: float


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


def compute_camera_leakage(
    camera: Any, charges: np.ndarray, mask: np.ndarray
) -> Tuple[float, float]:
    """Compute outer-ring charge leakage fractions (loss1, loss2) for a cleaned image.

    ``loss1`` is the fraction of cleaned image charge contained in the outermost
    hexagonal ring of the camera (pixels with < 6 neighbors). ``loss2`` is the
    fraction contained in the outermost two rings.
    """
    selected = np.asarray(mask, dtype=bool) & (np.asarray(charges) > 0.0)
    if not np.any(selected):
        return 0.0, 0.0
    total_charge = float(np.sum(charges[selected]))
    if total_charge <= 0.0:
        return 0.0, 0.0

    r1 = getattr(camera, "outer_ring1_mask", None)
    r2 = getattr(camera, "outer_ring2_mask", None)
    if r1 is None or r2 is None:
        adj = _validated_neighbors(camera, len(charges))
        r1 = np.sum(adj, axis=1) < 6
        r2 = r1 | np.any(adj[:, r1], axis=1)

    loss1 = float(np.sum(charges[selected & r1]) / total_charge)
    loss2 = float(np.sum(charges[selected & r2]) / total_charge)
    return loss1, loss2


def _fit_directed_axis_and_time_gradient(
    pixel_x: np.ndarray,
    pixel_y: np.ndarray,
    charges: np.ndarray,
    mask: np.ndarray,
    arrival_times_ns: np.ndarray,
) -> Tuple[float, float, float, float, float, float, float]:
    """Compute centroid, radially outward major axis, and weighted linear time gradient."""
    sel = mask & (charges > 0.0)
    w = charges[sel]
    w_sum = float(np.sum(w))
    if w_sum <= 0.0 or np.count_nonzero(sel) < 3:
        return 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0

    x_sel = pixel_x[sel]
    y_sel = pixel_y[sel]
    cx = float(np.dot(w, x_sel) / w_sum)
    cy = float(np.dot(w, y_sel) / w_sum)
    dx = x_sel - cx
    dy = y_sel - cy
    cov_xx = float(np.dot(w, dx * dx) / w_sum)
    cov_yy = float(np.dot(w, dy * dy) / w_sum)
    cov_xy = float(np.dot(w, dx * dy) / w_sum)
    psi = float(0.5 * np.arctan2(2.0 * cov_xy, cov_xx - cov_yy))
    dir_x = float(np.cos(psi))
    dir_y = float(np.sin(psi))
    if dir_x * cx + dir_y * cy < 0.0:
        dir_x = -dir_x
        dir_y = -dir_y

    s_sel = dx * dir_x + dy * dir_y
    t_sel = arrival_times_ns[sel]
    s_mean = float(np.dot(w, s_sel) / w_sum)
    t_mean = float(np.dot(w, t_sel) / w_sum)
    ds = s_sel - s_mean
    dt = t_sel - t_mean
    var_s = float(np.dot(w, ds * ds))
    if var_s > 1e-5:
        m_t = float(np.clip(np.dot(w, ds * dt) / var_s, -40.0, 40.0))
    else:
        m_t = 0.0
    b_t = float(t_mean - m_t * s_mean)
    residuals = t_sel - (m_t * s_sel + b_t)
    t_rms = float(np.sqrt(max(0.0, float(np.dot(w, residuals * residuals) / w_sum))))
    return cx, cy, dir_x, dir_y, m_t, b_t, t_rms


def double_pass_fadc_clean(
    trace: np.ndarray,
    camera: Any,
    bin_width_ns: float = 2.0,
    nsb_rate: float = 2.0,
    window_pass1: int = 8,
    window_pass2: int = 5,
    pic1: float = 4.0,
    bnd1: float = 2.0,
    pic2: float = 1.8,
    bnd2: float = 0.9,
    dist_tolerance: float = 0.22,
) -> DoublePassResult:
    """Perform VERITAS-style two-pass FADC trace integration and image cleaning.

    Pass 1:
      - Integrates each pixel over a wide sliding window (``window_pass1`` bins = 16 ns)
        to capture the pulse regardless of arrival-time gradient across the camera.
      - Extracts each pixel's pulse arrival time ``T0`` (ns) as the baseline-subtracted
        charge-weighted time centroid within the Pass-1 window.
      - Applies Pass-1 tail-cut cleaning and fits a linear arrival-time gradient
        ``t_pred(s) = m_t * s + b_t`` along the preliminary Hillas major axis.

    Pass 2:
      - Places a narrow integration window (``window_pass2`` bins = 10 ns) centered at
        the predicted arrival time ``t_pred(s_i)`` for every pixel along the shower axis,
        reducing NSB noise while following the longitudinal time gradient of distant showers.
      - Applies spatial double-pass cleaning (``double_pass_clean``) to recover the
        faint connected longitudinal fringe of the shower ellipse.
      - Computes the final linear time gradient ``m_t`` (ns/deg), time RMS residual
        ``sigma_t`` (ns), and camera-edge leakage fractions ``(loss1, loss2)``.
    """
    trace_arr = np.asarray(trace, dtype=np.float64)
    pixel_x, pixel_y = _validated_geometry(camera, trace_arr.shape[0])
    n_pixels = trace_arr.shape[0]

    if trace_arr.ndim == 1:
        c_clean = np.maximum(0.0, trace_arr - nsb_rate)
        mask = double_pass_clean(
            camera, c_clean, pic1=pic1, bnd1=bnd1, pic2=pic2, bnd2=bnd2,
            dist_tolerance=dist_tolerance
        )
        loss1, loss2 = compute_camera_leakage(camera, c_clean, mask)
        return DoublePassResult(
            charges=c_clean.astype(np.float32),
            mask=mask,
            arrival_times_ns=np.zeros(n_pixels, dtype=np.float32),
            time_gradient=0.0,
            time_intercept=0.0,
            time_rms=0.0,
            loss1=loss1,
            loss2=loss2,
        )

    n_bins = trace_arr.shape[1]
    w1 = min(max(1, int(window_pass1)), n_bins)
    w2 = min(max(1, int(window_pass2)), n_bins)
    nsb_per_bin = float(nsb_rate) / float(n_bins)

    # Cumulative sum for vectorized O(N_pixels) sliding window integration
    cs = np.pad(np.cumsum(trace_arr, axis=1), ((0, 0), (1, 0)))

    # --- PASS 1: Wide sliding window (w1 bins) ---
    win_sums1 = cs[:, w1:] - cs[:, :-w1]
    k1 = np.argmax(win_sums1, axis=1)
    pidx = np.arange(n_pixels)
    q1_raw = win_sums1[pidx, k1]
    ped1 = nsb_per_bin * w1
    c1 = np.maximum(0.0, q1_raw - ped1)

    # Pulse arrival time T0 (ns) within the Pass-1 window
    bins_idx1 = k1[:, None] + np.arange(w1)[None, :]
    sub_traces1 = np.maximum(0.0, np.take_along_axis(trace_arr, bins_idx1, axis=1) - nsb_per_bin)
    sub_sum1 = np.sum(sub_traces1, axis=1)
    peak_bins = np.argmax(trace_arr, axis=1).astype(np.float64)
    t0_bins = np.where(
        sub_sum1 > 1e-3,
        np.sum(sub_traces1 * bins_idx1, axis=1) / np.maximum(sub_sum1, 1e-6),
        peak_bins,
    )
    arrival_times_ns = t0_bins * float(bin_width_ns)

    # Pass-1 cleaning
    mask_pass1 = tail_cut_clean(camera, c1, picture_thresh=pic1, boundary_thresh=bnd1, min_neighbors=1)
    if np.count_nonzero(mask_pass1) < 3:
        mask_pass1 = tail_cut_clean(camera, c1, picture_thresh=pic1, boundary_thresh=bnd1, min_neighbors=0)

    if np.count_nonzero(mask_pass1) < 3 or np.sum(c1[mask_pass1]) <= 0.0:
        return DoublePassResult(
            charges=c1.astype(np.float32),
            mask=np.zeros(n_pixels, dtype=bool),
            arrival_times_ns=arrival_times_ns.astype(np.float32),
            time_gradient=0.0,
            time_intercept=0.0,
            time_rms=0.0,
            loss1=0.0,
            loss2=0.0,
        )

    # Fit Pass-1 major axis and linear time gradient t_pred(s) = m_t * s + b_t
    cx1, cy1, dir_x1, dir_y1, m_t1, b_t1, _ = _fit_directed_axis_and_time_gradient(
        pixel_x, pixel_y, c1, mask_pass1, arrival_times_ns
    )

    # --- PASS 2: Narrow time-gradient-aligned window (w2 bins) ---
    s_all = (pixel_x - cx1) * dir_x1 + (pixel_y - cy1) * dir_y1
    t_pred_bins = (m_t1 * s_all + b_t1) / float(bin_width_ns)
    k2_grad = np.clip(
        np.rint(t_pred_bins - 0.5 * (w2 - 1)).astype(int), 0, n_bins - w2
    )
    win_sums2 = cs[:, w2:] - cs[:, :-w2]
    q2_grad = win_sums2[pidx, k2_grad]
    q2_peak = np.max(win_sums2, axis=1)
    # Core pixels retain full pulse energy even if the shower core has finite thickness
    q2_raw = np.where(mask_pass1, np.maximum(q2_grad, q2_peak), q2_grad)
    ped2 = nsb_per_bin * w2
    c2 = np.maximum(0.0, q2_raw - ped2)

    # Spatial double-pass cleaning on narrow-window integrated image
    mask_pass2 = double_pass_clean(
        camera,
        c2,
        pic1=pic1 * 0.9,
        bnd1=bnd1 * 0.9,
        pic2=pic2,
        bnd2=bnd2,
        dist_tolerance=dist_tolerance,
    )
    if np.count_nonzero(mask_pass2) < 3:
        mask_pass2 = mask_pass1

    # Recompute final time gradient along Pass-2 directed Hillas axis
    _, _, _, _, m_t2, b_t2, t_rms2 = _fit_directed_axis_and_time_gradient(
        pixel_x, pixel_y, c2, mask_pass2, arrival_times_ns
    )
    loss1, loss2 = compute_camera_leakage(camera, c2, mask_pass2)

    return DoublePassResult(
        charges=c2.astype(np.float32),
        mask=mask_pass2,
        arrival_times_ns=arrival_times_ns.astype(np.float32),
        time_gradient=m_t2,
        time_intercept=b_t2,
        time_rms=t_rms2,
        loss1=loss1,
        loss2=loss2,
    )

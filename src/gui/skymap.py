"""2D Event Count Maps and Li & Ma (1983) Significance Map calculations."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np


CARDINAL_WOBBLE_ANGLES: Dict[str, float] = {
    "N": 0.0,
    "E": 0.5 * np.pi,
    "S": np.pi,
    "W": 1.5 * np.pi,
}


def li_and_ma_significance(
    n_on: Union[float, int, np.ndarray],
    n_off: Union[float, int, np.ndarray],
    alpha: Union[float, np.ndarray],
) -> Union[float, np.ndarray]:
    """Compute statistical significance using Li & Ma (1983), ApJ 272, 317, Eq. (17).

    Parameters
    ----------
    n_on : float or array-like
        Counts observed in the ON (signal + background) region.
    n_off : float or array-like
        Counts observed in the OFF (background control) region.
    alpha : float or array-like
        Ratio of ON to OFF exposure/area (t_on * A_on / (t_off * A_off)).

    Returns
    -------
    float or np.ndarray
        Signed Li & Ma significance in Gaussian standard deviations (sigma).
    """
    is_scalar = np.ndim(n_on) == 0 and np.ndim(n_off) == 0 and np.ndim(alpha) == 0

    on_arr, off_arr, a_arr = np.broadcast_arrays(
        np.asarray(n_on, dtype=np.float64),
        np.asarray(n_off, dtype=np.float64),
        np.asarray(alpha, dtype=np.float64),
    )

    total = on_arr + off_arr
    valid = (total > 0.0) & (a_arr > 0.0) & (on_arr >= 0.0) & (off_arr >= 0.0)

    term1 = np.zeros_like(on_arr, dtype=np.float64)
    term2 = np.zeros_like(on_arr, dtype=np.float64)

    # Term 1: N_on * ln[ ((1 + alpha) / alpha) * (N_on / (N_on + N_off)) ]
    m1 = valid & (on_arr > 0.0)
    if np.any(m1):
        ratio1 = ((1.0 + a_arr[m1]) / a_arr[m1]) * (on_arr[m1] / total[m1])
        ratio1 = np.maximum(ratio1, 1e-300)
        term1[m1] = on_arr[m1] * np.log(ratio1)

    # Term 2: N_off * ln[ (1 + alpha) * (N_off / (N_on + N_off)) ]
    m2 = valid & (off_arr > 0.0)
    if np.any(m2):
        ratio2 = (1.0 + a_arr[m2]) * (off_arr[m2] / total[m2])
        ratio2 = np.maximum(ratio2, 1e-300)
        term2[m2] = off_arr[m2] * np.log(ratio2)

    inside = np.maximum(2.0 * (term1 + term2), 0.0)
    excess_sign = np.where(on_arr >= a_arr * off_arr, 1.0, -1.0)
    # Exact zero when excess is identically zero or inputs invalid
    sig = np.where(valid & (on_arr != a_arr * off_arr), excess_sign * np.sqrt(inside), 0.0)

    if is_scalar:
        return float(sig.item())
    return sig


def _wrap_angle_diff_deg(a_deg: np.ndarray, b_deg: np.ndarray) -> np.ndarray:
    """Wrap (a - b) into [-180, 180) degrees."""
    return (a_deg - b_deg + 180.0) % 360.0 - 180.0


def reconstruct_sky_offsets_for_events(
    events: Sequence[Dict[str, Any]],
    reco_zen_deg: np.ndarray,
    reco_azi_deg: np.ndarray,
    default_wobble_deg: float = 0.5,
) -> Dict[str, Any]:
    """Compute source-centered sky coordinates and de-rotated wobble camera coordinates.

    For gamma events (`label == 1`), `true_zenith_deg` / `true_azimuth_deg` gives the true
    source position per pointing. For diffuse hadron events (`label == 0`), their camera
    offsets `(cam_x, cam_y)` are shifted using the observation's wobble geometry (cardinal
    wobble or ring wobble) so gammas and hadrons populate a unified source-centered FoV map.
    """
    n = len(events)
    if n == 0:
        empty = np.zeros(0, dtype=np.float64)
        return {
            "sky_dx_deg": empty,
            "sky_dy_deg": empty,
            "cam_dx_deg": empty,
            "cam_dy_deg": empty,
            "wob_x_deg": empty,
            "wob_y_deg": empty,
            "wobble_offset_deg": float(default_wobble_deg),
        }

    pnt_zen = np.array(
        [float(ev["zenith_deg"]) if ev.get("zenith_deg") is not None else 30.0 for ev in events],
        dtype=np.float64,
    )
    pnt_azi = np.array(
        [float(ev["azimuth_deg"]) if ev.get("azimuth_deg") is not None else 0.0 for ev in events],
        dtype=np.float64,
    )
    true_zen = np.array(
        [
            float(ev["true_zenith_deg"])
            if ev.get("true_zenith_deg") is not None
            else float(pnt_zen[i])
            for i, ev in enumerate(events)
        ],
        dtype=np.float64,
    )
    true_azi = np.array(
        [
            float(ev["true_azimuth_deg"])
            if ev.get("true_azimuth_deg") is not None
            else float(pnt_azi[i])
            for i, ev in enumerate(events)
        ],
        dtype=np.float64,
    )
    labels = np.array([int(ev["label"]) if ev.get("label") is not None else 1 for ev in events], dtype=np.int64)
    wob_dirs: List[str] = [str(ev.get("wobble_dir") or "N") for ev in events]

    r_zen = np.asarray(reco_zen_deg, dtype=np.float64)
    r_azi = np.asarray(reco_azi_deg, dtype=np.float64)
    sin_zen = np.maximum(np.sin(np.radians(pnt_zen)), 1e-6)

    cam_dx = _wrap_angle_diff_deg(r_azi, pnt_azi) * sin_zen
    cam_dy = r_zen - pnt_zen

    src_cam_dx = _wrap_angle_diff_deg(true_azi, pnt_azi) * sin_zen
    src_cam_dy = true_zen - pnt_zen

    # Determine wobble radius from gamma point-source offsets relative to pointing
    gamma_mask = labels == 1
    if np.any(gamma_mask):
        g_wob_r = np.hypot(src_cam_dx[gamma_mask], src_cam_dy[gamma_mask])
        wobble_offset_deg = float(np.median(g_wob_r))
    else:
        wobble_offset_deg = float(default_wobble_deg)

    # Determine wobble azimuth angles for all events
    phi = np.zeros(n, dtype=np.float64)
    cardinal_list = ("N", "E", "S", "W")
    observed_cardinals = [d for d in wob_dirs if d in CARDINAL_WOBBLE_ANGLES]

    for i in range(n):
        w_dir = wob_dirs[i]
        if labels[i] == 1 and (np.hypot(src_cam_dx[i], src_cam_dy[i]) > 1e-4 or w_dir in CARDINAL_WOBBLE_ANGLES):
            # In generate_training_data.py: dy = theta*cos(phi), dx = theta*sin(phi)
            if np.hypot(src_cam_dx[i], src_cam_dy[i]) > 1e-4:
                phi[i] = float(np.arctan2(src_cam_dx[i], src_cam_dy[i]))
            else:
                phi[i] = CARDINAL_WOBBLE_ANGLES.get(w_dir, 0.0)
        elif w_dir in CARDINAL_WOBBLE_ANGLES:
            phi[i] = CARDINAL_WOBBLE_ANGLES[w_dir]
            src_cam_dx[i] = wobble_offset_deg * np.sin(phi[i])
            src_cam_dy[i] = wobble_offset_deg * np.cos(phi[i])
        else:
            # Diffuse hadron (or unclassified) -> map onto the observation's wobble pointings
            if wobble_offset_deg > 1e-4:
                if observed_cardinals or not any(d == "ring" for d in wob_dirs):
                    c_dir = cardinal_list[i % 4]
                    phi[i] = CARDINAL_WOBBLE_ANGLES[c_dir]
                else:
                    phi[i] = (i * 2.399963229728653) % (2.0 * np.pi)
                src_cam_dx[i] = wobble_offset_deg * np.sin(phi[i])
                src_cam_dy[i] = wobble_offset_deg * np.cos(phi[i])
            else:
                phi[i] = 0.0
                src_cam_dx[i] = 0.0
                src_cam_dy[i] = 0.0

    # Source-centered sky offsets (target source is at (0, 0))
    sky_dx = cam_dx - src_cam_dx
    sky_dy = cam_dy - src_cam_dy

    # De-rotated wobble camera frame: camera center is (0, 0), ON source is at (0, +wobble_offset_deg)
    wob_x = cam_dx * np.cos(phi) - cam_dy * np.sin(phi)
    wob_y = cam_dx * np.sin(phi) + cam_dy * np.cos(phi)

    return {
        "sky_dx_deg": sky_dx,
        "sky_dy_deg": sky_dy,
        "cam_dx_deg": cam_dx,
        "cam_dy_deg": cam_dy,
        "wob_x_deg": wob_x,
        "wob_y_deg": wob_y,
        "wobble_offset_deg": float(wobble_offset_deg),
    }


def _fft_convolve_2d(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Fast 2D same-shape convolution using NumPy FFT (no SciPy signal overhead)."""
    ih, iw = image.shape
    kh, kw = kernel.shape
    fh = int(2 ** np.ceil(np.log2(max(ih + kh - 1, 1))))
    fw = int(2 ** np.ceil(np.log2(max(iw + kw - 1, 1))))
    conv = np.fft.irfft2(np.fft.rfft2(image, s=(fh, fw)) * np.fft.rfft2(kernel, s=(fh, fw)), s=(fh, fw))
    r0 = (kh - 1) // 2
    c0 = (kw - 1) // 2
    out = conv[r0 : r0 + ih, c0 : c0 + iw]
    # Guard floating-point rounding noise near zero
    return np.where(np.abs(out) < 1e-10, 0.0, np.maximum(out, 0.0))


def _reflected_angles_for_radius(r_deg: float, theta_cut_deg: float, max_regions: int = 5) -> List[float]:
    """Return symmetric reflected OFF position angles (rad) around `(0, +r)` excluding ON."""
    if r_deg <= theta_cut_deg * 1.15:
        return []
    sin_arg = min(0.999, (1.05 * theta_cut_deg) / max(r_deg, 1e-6))
    half_ang = float(np.arcsin(sin_arg))
    if half_ang <= 0.0:
        return []
    # Maximum number of non-overlapping disks of radius theta_cut on circle of radius r
    n_total = int(np.floor(np.pi / half_ang))
    if n_total < 2:
        return []
    n_off = min(max_regions, n_total - 1)
    # Place symmetric OFF positions starting opposite or cleanly spaced outside ON
    if n_off == 3 and n_total >= 4:
        return [0.5 * np.pi, np.pi, 1.5 * np.pi]
    angles: List[float] = []
    # Distribute uniformly over [2*half_ang, 2*pi - 2*half_ang]
    lo = 2.0 * half_ang
    hi = 2.0 * np.pi - 2.0 * half_ang
    if n_off == 1:
        angles = [np.pi]
    elif hi > lo + 1e-6:
        angles = list(np.linspace(lo, hi, n_off))
    else:
        angles = [np.pi]
    return angles


def compute_skymap_and_significance(
    sky_dx_deg: np.ndarray,
    sky_dy_deg: np.ndarray,
    true_class: np.ndarray,
    pred_class_score: np.ndarray,
    gamma_cut: float = 0.5,
    theta_cut_deg: float = 0.16,
    fov_half_width_deg: float = 1.8,
    bin_size_deg: float = 0.05,
    bg_method: str = "ring",
    wob_x_deg: Optional[np.ndarray] = None,
    wob_y_deg: Optional[np.ndarray] = None,
    wobble_offset_deg: float = 0.5,
) -> Dict[str, Any]:
    """Build 2D Event Count Maps, Excess Map, Li & Ma Significance Map, and theta^2 curve.

    Parameters
    ----------
    sky_dx_deg, sky_dy_deg : np.ndarray
        Reconstructed sky offsets relative to target source at `(0, 0)` in degrees.
    true_class : np.ndarray
        1.0 for gamma events, 0.0 for hadrons.
    pred_class_score : np.ndarray
        Gamma/hadron classifier scores in `[0, 1]`.
    gamma_cut : float
        Threshold on `pred_class_score` for event selection.
    theta_cut_deg : float
        Top-hat ON aperture integration radius in degrees.
    fov_half_width_deg : float
        Half-width of the square 2D sky map around `(0, 0)` in degrees.
    bin_size_deg : float
        Pixel width in degrees.
    bg_method : str
        One of `"ring"` (Ring Background), `"reflected"` (Reflected Wobble Regions),
        or `"mc_hadron"` (Independent MC Truth Hadron Background).
    wob_x_deg, wob_y_deg : np.ndarray, optional
        De-rotated wobble camera coordinates where camera center is `(0, 0)` and
        the ON source sits at `(0, +wobble_offset_deg)`.
    wobble_offset_deg : float
        Nominal wobble offset angle in degrees.
    """
    sx = np.asarray(sky_dx_deg, dtype=np.float64)
    sy = np.asarray(sky_dy_deg, dtype=np.float64)
    cls_arr = np.asarray(true_class, dtype=np.float64)
    scores = np.asarray(pred_class_score, dtype=np.float64)

    finite = np.isfinite(sx) & np.isfinite(sy) & np.isfinite(cls_arr) & np.isfinite(scores)
    passed = finite & (scores >= float(gamma_cut))

    fov_hw = max(float(fov_half_width_deg), 0.4)
    bin_w = float(np.clip(bin_size_deg, 0.01, 0.25))
    theta_cut = float(np.clip(theta_cut_deg, 0.03, 0.6))
    method = str(bg_method).lower().strip()
    if method not in ("ring", "reflected", "mc_hadron"):
        method = "ring"

    # Build odd-sized pixel grid so central pixel is centered exactly at (0.0, 0.0)
    half_steps = max(int(round(fov_hw / bin_w)), 5)
    n_pix = 2 * half_steps + 1
    edges = np.linspace(-(half_steps + 0.5) * bin_w, (half_steps + 0.5) * bin_w, n_pix + 1)
    centres = 0.5 * (edges[:-1] + edges[1:])
    grid_x, grid_y = np.meshgrid(centres, centres)  # shape (n_pix, n_pix): row=y, col=x

    sel_x = sx[passed]
    sel_y = sy[passed]
    sel_cls = cls_arr[passed]

    sel_gamma = sel_cls == 1.0
    sel_hadron = sel_cls == 0.0

    # np.histogram2d returns shape (len(x), len(y)); transpose to (y, x) to match meshgrid/imshow(origin='lower')
    raw_counts, _, _ = np.histogram2d(sel_y, sel_x, bins=[edges, edges])
    raw_gamma, _, _ = np.histogram2d(sel_y[sel_gamma], sel_x[sel_gamma], bins=[edges, edges])
    raw_hadron, _, _ = np.histogram2d(sel_y[sel_hadron], sel_x[sel_hadron], bins=[edges, edges])

    # Support mask inside FoV where simulation/observations provide acceptance coverage
    r_grid = np.hypot(grid_x, grid_y)
    max_evt_r = float(np.percentile(np.hypot(sx[finite], sy[finite]), 99)) if np.any(finite) else fov_hw
    support_radius = min(float(edges[-1]), max(fov_hw, min(max_evt_r, fov_hw * np.sqrt(2.0))))
    support_mask = (r_grid <= support_radius).astype(np.float64)

    # Top-hat ON integration disk kernel
    k_on_rad_pix = max(1, int(np.ceil(theta_cut / bin_w)))
    ky_on, kx_on = np.mgrid[-k_on_rad_pix : k_on_rad_pix + 1, -k_on_rad_pix : k_on_rad_pix + 1]
    kr_on = np.hypot(kx_on * bin_w, ky_on * bin_w)
    on_kernel = (kr_on <= theta_cut + 1e-9).astype(np.float64)

    n_on_map = _fft_convolve_2d(raw_counts, on_kernel)
    on_support = _fft_convolve_2d(support_mask, on_kernel)

    # Ring geometry outside ON aperture and excluding bright central source disk
    r_in = max(theta_cut + max(0.12, 2.0 * bin_w), 0.32)
    r_out = min(fov_hw * 0.85, r_in + max(0.40, 6.0 * bin_w))
    if r_out <= r_in + bin_w:
        r_out = r_in + 3.0 * bin_w

    k_ring_pix = max(2, int(np.ceil(r_out / bin_w)))
    ky_r, kx_r = np.mgrid[-k_ring_pix : k_ring_pix + 1, -k_ring_pix : k_ring_pix + 1]
    kr_r = np.hypot(kx_r * bin_w, ky_r * bin_w)
    ring_kernel = ((kr_r >= r_in) & (kr_r <= r_out)).astype(np.float64)

    src_excl_radius = max(theta_cut + 0.12, 0.30)
    non_src_mask = (r_grid > src_excl_radius).astype(np.float64) * support_mask

    # Compute OFF maps, alpha maps, and exact source ON/OFF counts
    if wob_x_deg is not None and wob_y_deg is not None and len(wob_x_deg) == len(sx):
        wx = np.asarray(wob_x_deg, dtype=np.float64)
        wy = np.asarray(wob_y_deg, dtype=np.float64)
    else:
        wx = sx.copy()
        wy = sy.copy() + float(wobble_offset_deg)

    sel_wx = wx[passed]
    sel_wy = wy[passed]
    has_wobble = wobble_offset_deg > theta_cut * 1.15

    # Exact unbinned distance to target source (0, 0) for events passing cut
    sel_r2_src = sel_x ** 2 + sel_y ** 2
    sel_r_src = np.sqrt(sel_r2_src)
    in_src_aperture = sel_r_src <= theta_cut
    source_n_on = float(np.sum(in_src_aperture))
    n_gammas_in_cut = int(np.sum(in_src_aperture & sel_gamma))
    n_hadrons_in_cut = int(np.sum(in_src_aperture & sel_hadron))

    # Ring background 2D maps (used directly for "ring", and for outer sky pixels in "reflected")
    n_off_ring_map = _fft_convolve_2d(raw_counts * non_src_mask, ring_kernel)
    off_ring_support = _fft_convolve_2d(non_src_mask, ring_kernel)
    alpha_ring_map = np.divide(
        on_support,
        off_ring_support,
        out=np.zeros_like(on_support, dtype=np.float64),
        where=off_ring_support > 0.0,
    )

    # Build theta^2 radial bins out to max_theta2
    max_theta2 = max((theta_cut * 3.0) ** 2, 0.25)
    n_t2_bins = 25
    t2_edges = np.linspace(0.0, max_theta2, n_t2_bins + 1)
    t2_centres = 0.5 * (t2_edges[:-1] + t2_edges[1:])
    t2_on_counts, _ = np.histogram(sel_r2_src, bins=t2_edges)
    t2_on_counts = t2_on_counts.astype(np.float64)

    if method == "ring":
        n_off_map = n_off_ring_map
        alpha_map = alpha_ring_map

        # Exact unbinned source ring counts & theta^2 profile
        in_src_ring = (sel_r_src >= r_in) & (sel_r_src <= r_out)
        source_n_off = float(np.sum(in_src_ring))
        ring_area = np.pi * (r_out ** 2 - r_in ** 2)
        on_area = np.pi * (theta_cut ** 2)
        source_alpha = float(on_area / max(ring_area, 1e-12))

        # Background density per deg^2 in the OFF ring -> expected background in each equal-d(theta^2) bin
        d_theta2 = float(t2_edges[1] - t2_edges[0])
        bin_area_deg2 = np.pi * d_theta2  # dArea = pi * d(theta^2)
        t2_alpha = bin_area_deg2 / max(ring_area, 1e-12)
        t2_off_scaled = np.full(n_t2_bins, source_n_off * t2_alpha, dtype=np.float64)

    elif method == "reflected":
        n_off_map = n_off_ring_map.copy()
        alpha_map = alpha_ring_map.copy()

        off_angles = _reflected_angles_for_radius(float(wobble_offset_deg), theta_cut, max_regions=5)
        if off_angles and has_wobble:
            n_regions = len(off_angles)
            source_alpha = 1.0 / float(n_regions)
            off_region_counts = 0.0
            t2_off_accum = np.zeros(n_t2_bins, dtype=np.float64)
            # In wobble coords (wx, wy), ON source is at (0, +R); phi_ref=0 is ON.
            # Notice cardinal N is at (x=0, y=+R), E at (+R, 0), S at (0, -R), W at (-R, 0):
            for ang in off_angles:
                ox = float(wobble_offset_deg) * np.sin(ang)
                oy = float(wobble_offset_deg) * np.cos(ang)
                r2_off = (sel_wx - ox) ** 2 + (sel_wy - oy) ** 2
                off_region_counts += float(np.sum(r2_off <= (theta_cut ** 2)))
                c_t2, _ = np.histogram(r2_off, bins=t2_edges)
                t2_off_accum += c_t2.astype(np.float64)
            source_n_off = float(off_region_counts)
            t2_off_scaled = t2_off_accum * source_alpha
        else:
            # On-axis / small wobble offset -> fallback cleanly to ring background around (0, 0)
            in_src_ring = (sel_r_src >= r_in) & (sel_r_src <= r_out)
            source_n_off = float(np.sum(in_src_ring))
            ring_area = np.pi * (r_out ** 2 - r_in ** 2)
            on_area = np.pi * (theta_cut ** 2)
            source_alpha = float(on_area / max(ring_area, 1e-12))
            bin_area_deg2 = np.pi * float(t2_edges[1] - t2_edges[0])
            t2_off_scaled = np.full(n_t2_bins, source_n_off * (bin_area_deg2 / max(ring_area, 1e-12)), dtype=np.float64)

        # Anchor central source neighborhood on 2D map to reflected-region estimate
        src_pix_mask = r_grid <= 0.55 * bin_w
        if np.any(src_pix_mask):
            n_off_map[src_pix_mask] = source_n_off
            alpha_map[src_pix_mask] = source_alpha
            n_on_map[src_pix_mask] = source_n_on

    else:  # "mc_hadron" — independent MC hadron counts outside each ON aperture
        hadron_on_map = _fft_convolve_2d(raw_hadron, on_kernel)
        total_hadrons_in_fov = float(np.sum(raw_hadron * support_mask))
        total_support_pix = float(np.sum(support_mask))

        # Hadron counts strictly outside the ON aperture -> disjoint Poisson control sample
        n_off_map = np.maximum(total_hadrons_in_fov - hadron_on_map, 0.0)
        off_support_mc = np.maximum(total_support_pix - on_support, 0.0)
        alpha_map = np.divide(
            on_support,
            off_support_mc,
            out=np.zeros_like(on_support, dtype=np.float64),
            where=off_support_mc > 0.0,
        )

        # Exact unbinned source MC hadron statistics
        in_fov_hadron = sel_hadron & (np.hypot(sel_x, sel_y) <= support_radius) & (np.abs(sel_x) <= edges[-1]) & (np.abs(sel_y) <= edges[-1])
        out_on_hadron = in_fov_hadron & (~in_src_aperture)
        source_n_off = float(np.sum(out_on_hadron))
        fov_area = max(total_support_pix * (bin_w ** 2), np.pi * (theta_cut ** 2) + 1e-6)
        on_area = np.pi * (theta_cut ** 2)
        source_alpha = float(on_area / max(fov_area - on_area, 1e-12))

        bin_area_deg2 = np.pi * float(t2_edges[1] - t2_edges[0])
        hadron_density_per_deg2 = float(np.sum(in_fov_hadron)) / max(fov_area, 1e-12)
        t2_off_scaled = np.full(n_t2_bins, hadron_density_per_deg2 * bin_area_deg2, dtype=np.float64)

    bkg_map = alpha_map * n_off_map
    excess_map = n_on_map - bkg_map
    significance_map = li_and_ma_significance(n_on_map, n_off_map, alpha_map)

    source_n_bkg = float(source_alpha * source_n_off)
    source_n_excess = float(source_n_on - source_n_bkg)
    source_sig = float(li_and_ma_significance(source_n_on, source_n_off, source_alpha))

    # 1D significance distribution histogram (all map bins vs off-source background bins)
    sig_vals_all = significance_map[support_mask > 0.5]
    sig_vals_off = significance_map[(support_mask > 0.5) & (r_grid > src_excl_radius)]
    s_lo = min(-5.0, float(np.min(sig_vals_all)) if len(sig_vals_all) else -5.0)
    s_hi = max(8.0, float(np.max(sig_vals_all)) + 0.5 if len(sig_vals_all) else 8.0)
    sig_edges = np.linspace(s_lo, s_hi, 36)
    sig_centres = 0.5 * (sig_edges[:-1] + sig_edges[1:])
    sig_hist_all, _ = np.histogram(sig_vals_all, bins=sig_edges)
    sig_hist_off, _ = np.histogram(sig_vals_off, bins=sig_edges)

    return {
        "x_edges_deg": edges,
        "y_edges_deg": edges,
        "bin_centres_deg": centres,
        "raw_counts_map": raw_counts,
        "raw_gamma_map": raw_gamma,
        "raw_hadron_map": raw_hadron,
        "n_on_map": n_on_map,
        "n_off_map": n_off_map,
        "alpha_map": alpha_map,
        "bkg_map": bkg_map,
        "excess_map": excess_map,
        "significance_map": significance_map,
        "support_mask": support_mask,
        "source_n_on": source_n_on,
        "source_n_off": source_n_off,
        "source_alpha": source_alpha,
        "source_n_bkg": source_n_bkg,
        "source_n_excess": source_n_excess,
        "source_significance_lima": source_sig,
        "peak_significance_lima": float(np.max(significance_map)) if significance_map.size else 0.0,
        "min_significance_lima": float(np.min(significance_map)) if significance_map.size else 0.0,
        "n_events_total": int(np.sum(finite)),
        "n_events_passing": int(np.sum(passed)),
        "n_gammas_passing": int(np.sum(sel_gamma)),
        "n_hadrons_passing": int(np.sum(sel_hadron)),
        "n_gammas_in_theta_cut": n_gammas_in_cut,
        "n_hadrons_in_theta_cut": n_hadrons_in_cut,
        "theta_cut_deg": theta_cut,
        "theta2_cut_deg2": theta_cut ** 2,
        "theta2_edges_deg2": t2_edges,
        "theta2_centres_deg2": t2_centres,
        "theta2_on_counts": t2_on_counts,
        "theta2_off_scaled_counts": t2_off_scaled,
        "sig_hist_edges": sig_edges,
        "sig_hist_centres": sig_centres,
        "sig_hist_all": sig_hist_all,
        "sig_hist_off_only": sig_hist_off,
        "bg_method": method,
        "gamma_cut": float(gamma_cut),
        "fov_half_width_deg": fov_hw,
        "bin_size_deg": bin_w,
        "wobble_offset_deg": float(wobble_offset_deg),
        "ring_r_in_deg": float(r_in),
        "ring_r_out_deg": float(r_out),
    }

"""
Stereoscopic reconstruction of atmospheric Cherenkov showers.
Implements the standard geometric Shower Detector Plane (SDP) intersection method
(Hofmann et al. 1999, Astroparticle Physics 12, 135-152) for arrival direction
and core impact position reconstruction.
"""
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from sim.camera import Camera
from recon.hillas import compute_hillas, HillasParameters


@dataclass
class StereoReconstructionResult:
    """Result of stereoscopic shower reconstruction."""
    reco_zenith_deg: float
    reco_azimuth_deg: float
    reco_impact_x: float
    reco_impact_y: float
    n_telescopes_used: int
    n_pairs_used: int
    total_size: float
    hillas_by_tel: Dict[int, HillasParameters]
    reco_direction_vector: np.ndarray


def cam_to_sky_vector(
    u_deg: float, v_deg: float, zenith_rad: float, azimuth_rad: float
) -> np.ndarray:
    """Convert camera angular coordinates (degrees) to a 3D unit vector in the observatory frame.
    
    Observatory frame: +z points vertically upward, +x and +y form the horizontal plane.
    Telescope optical axis points at (zenith_rad, azimuth_rad).
    """
    u = np.radians(u_deg)
    v = np.radians(v_deg)
    vec_cam = np.array([np.tan(u), np.tan(v), 1.0], dtype=np.float64)
    vec_cam /= np.linalg.norm(vec_cam)

    Ry = np.array([
        [np.cos(zenith_rad), 0.0, np.sin(zenith_rad)],
        [0.0, 1.0, 0.0],
        [-np.sin(zenith_rad), 0.0, np.cos(zenith_rad)]
    ], dtype=np.float64)

    Rz = np.array([
        [np.cos(azimuth_rad), -np.sin(azimuth_rad), 0.0],
        [np.sin(azimuth_rad), np.cos(azimuth_rad), 0.0],
        [0.0, 0.0, 1.0]
    ], dtype=np.float64)

    return Rz @ (Ry @ vec_cam)


def tailcut_clean(
    charges: np.ndarray,
    camera: Camera,
    pic_th: float = 4.0,
    bnd_th: float = 2.0,
    ped: float = 2.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Perform two-level tail-cut image cleaning.
    
    Args:
        charges: Raw integrated pixel charges (PE).
        camera: Camera geometry with pixel coordinates.
        pic_th: Picture threshold above pedestal (PE).
        bnd_th: Boundary threshold above pedestal (PE).
        ped: NSB baseline/pedestal to subtract (PE).
        
    Returns:
        (c_clean, mask): Baseline-subtracted charges and boolean survival mask.
    """
    c_clean = np.maximum(0.0, charges - ped)
    pic_mask = c_clean >= pic_th

    # Expand to neighbors exceeding boundary threshold
    bnd_candidates = c_clean >= bnd_th
    adj = camera.get_neighbor_matrix()
    has_pic_neighbor = np.any(adj & pic_mask[None, :], axis=1)

    clean_mask = pic_mask | (bnd_candidates & has_pic_neighbor)
    return c_clean, clean_mask


def reconstruct_event_geometric(
    fadc_traces: np.ndarray,
    telescopes: List[Any],
    camera: Camera,
    pointing_zenith_deg: float = 30.0,
    pointing_azimuth_deg: float = 0.0,
    z_obs: float = 1275.0,
    min_size: float = 50.0,
    max_wl_ratio: float = 0.85,
) -> Optional[StereoReconstructionResult]:
    """Reconstruct shower direction and ground impact core geometrically using SDP intersection.
    
    Args:
        fadc_traces: Array of FADC traces with shape (n_telescopes, n_pixels, n_time_bins).
        telescopes: List of telescope objects with x_tel, y_tel, z_tel.
        camera: Camera geometry object.
        pointing_zenith_deg: Telescope pointing zenith angle (degrees).
        pointing_azimuth_deg: Telescope pointing azimuth angle (degrees).
        z_obs: Observatory ground plane altitude (meters).
        min_size: Minimum cleaned image size (PE) to include telescope.
        max_wl_ratio: Maximum width/length ratio (rejects near-circular images with ill-defined axes).
        
    Returns:
        StereoReconstructionResult if >= 2 valid telescope images, else None.
    """
    n_tel = len(telescopes)
    zenith_rad = np.radians(pointing_zenith_deg)
    # Simulation convention: optical axis is azimuth_deg + 180°
    azimuth_rad = np.radians(pointing_azimuth_deg + 180.0)

    planes = []
    hillas_by_tel = {}
    total_size = 0.0

    for t_idx in range(n_tel):
        trace = fadc_traces[t_idx]
        raw_charge = np.sum(trace, axis=-1)
        c_clean, mask = tailcut_clean(raw_charge, camera)

        if np.sum(mask) < 3:
            continue

        h = compute_hillas(camera, c_clean, mask)
        if h is None or h.size < min_size:
            continue

        wl = h.width / max(h.length, 1e-4)
        if wl > max_wl_ratio:
            continue

        hillas_by_tel[t_idx] = h
        total_size += h.size

        # Two points along major axis in camera coordinates
        cx, cy = h.centroid_x, h.centroid_y
        delta = 0.5 * h.length
        psi = h.psi

        p1 = cam_to_sky_vector(cx - delta * np.cos(psi), cy - delta * np.sin(psi), zenith_rad, azimuth_rad)
        p2 = cam_to_sky_vector(cx + delta * np.cos(psi), cy + delta * np.sin(psi), zenith_rad, azimuth_rad)

        normal = np.cross(p1, p2)
        norm_mag = np.linalg.norm(normal)
        if norm_mag < 1e-6:
            continue
        normal /= norm_mag

        tel_pos = np.array([telescopes[t_idx].x_tel, telescopes[t_idx].y_tel, telescopes[t_idx].z_tel], dtype=np.float64)
        planes.append({
            'tel_idx': t_idx,
            'pos': tel_pos,
            'normal': normal,
            'size': h.size,
            'wl': wl
        })

    if len(planes) < 2:
        return None

    dir_vectors = []
    dir_weights = []
    core_points = []
    core_weights = []

    for i in range(len(planes)):
        for j in range(i + 1, len(planes)):
            p_i, p_j = planes[i], planes[j]
            d = np.cross(p_i['normal'], p_j['normal'])
            sin_ang = np.linalg.norm(d)
            if sin_ang < 0.087:  # < 5° opening angle between planes
                continue

            d /= sin_ang
            if d[2] < 0:
                d = -d  # Ensure upward pointing towards sky source

            w = (sin_ang**2) / (1.0 / p_i['size'] + 1.0 / p_j['size']) * max(0.01, 1.0 - p_i['wl'])**2 * max(0.01, 1.0 - p_j['wl'])**2
            dir_vectors.append(d * w)
            dir_weights.append(w)

            # Ground core intersection: n_i . (R_core - R_i) = 0
            n_i = p_i['normal']
            n_j = p_j['normal']
            b_i = np.dot(n_i, p_i['pos']) - n_i[2] * z_obs
            b_j = np.dot(n_j, p_j['pos']) - n_j[2] * z_obs
            A = np.array([[n_i[0], n_i[1]], [n_j[0], n_j[1]]], dtype=np.float64)
            det = np.linalg.det(A)
            if abs(det) > 0.05:
                try:
                    core_xy = np.linalg.solve(A, np.array([b_i, b_j]))
                    if np.hypot(core_xy[0], core_xy[1]) < 350.0:  # within physical array bounds
                        cw = w * (abs(det)**2)
                        core_points.append(core_xy)
                        core_weights.append(cw)
                except np.linalg.LinAlgError:
                    pass

    if not dir_vectors or not core_points:
        return None

    # Weighted direction vector
    reco_dir = np.sum(dir_vectors, axis=0)
    reco_dir /= np.linalg.norm(reco_dir)

    # Weighted core point
    tot_cw = sum(core_weights)
    reco_core = sum(p * w for p, w in zip(core_points, core_weights)) / tot_cw

    # Convert direction vector back to sky angles (zenith, azimuth)
    reco_zenith = np.degrees(np.arccos(np.clip(reco_dir[2], -1.0, 1.0)))
    reco_azimuth = (np.degrees(np.arctan2(reco_dir[1], reco_dir[0])) - 180.0) % 360.0
    if reco_azimuth > 180.0:
        reco_azimuth -= 360.0

    return StereoReconstructionResult(
        reco_zenith_deg=float(reco_zenith),
        reco_azimuth_deg=float(reco_azimuth),
        reco_impact_x=float(reco_core[0]),
        reco_impact_y=float(reco_core[1]),
        n_telescopes_used=len(planes),
        n_pairs_used=len(dir_vectors),
        total_size=float(total_size),
        hillas_by_tel=hillas_by_tel,
        reco_direction_vector=reco_dir,
    )

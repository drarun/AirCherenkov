"""
GPU/CPU backend abstraction for AirCherenkov.

Provides a unified interface that uses PyTorch CUDA when available,
falling back to CPU tensors. All heavy array operations go through
this module so the rest of the codebase stays backend-agnostic.
"""

import numpy as np

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

# ── Device selection ──────────────────────────────────────────────────────────

def get_device():
    """Returns the best available torch device, or None if torch is unavailable."""
    if not HAS_TORCH:
        return None
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')

def device_info():
    """Returns a human-readable string describing the compute backend."""
    if not HAS_TORCH:
        return "NumPy (CPU only — install PyTorch for GPU acceleration)"
    dev = get_device()
    if dev.type == 'cuda':
        name = torch.cuda.get_device_name(0)
        mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        return f"PyTorch CUDA — {name} ({mem:.1f} GB)"
    return "PyTorch CPU (no CUDA GPU detected)"

# ── Array conversion utilities ────────────────────────────────────────────────

def to_tensor(arr, device=None):
    """Convert a numpy array (or list) to a torch tensor on the given device."""
    if not HAS_TORCH:
        return np.asarray(arr, dtype=np.float64)
    if device is None:
        device = get_device()
    if isinstance(arr, torch.Tensor):
        return arr.to(device)
    return torch.tensor(np.asarray(arr), dtype=torch.float64, device=device)

def to_numpy(tensor):
    """Convert a torch tensor back to a numpy array."""
    if isinstance(tensor, np.ndarray):
        return tensor
    return tensor.detach().cpu().numpy()

# ── Cherenkov pool computation (the main GPU kernel) ──────────────────────────

def compute_cherenkov_pool_gpu(seg_x1, seg_y1, seg_z1,
                                seg_x2, seg_y2, seg_z2,
                                seg_px, seg_py, seg_pz,
                                seg_energy, photon_yield_factor,
                                max_photons_per_segment=50000):
    """
    Fully vectorized Cherenkov photon generation on GPU (or CPU fallback).
    
    Takes flat arrays of track segment endpoints and energies,
    returns dict of numpy arrays with photon emission and ground positions.
    
    Parameters
    ----------
    seg_x1, seg_y1, seg_z1 : array-like
        Start positions of each track segment (meters).
    seg_x2, seg_y2, seg_z2 : array-like
        End positions of each track segment (meters).
    seg_px, seg_py, seg_pz : array-like
        Particle direction at each segment (normalized).
    seg_energy : array-like
        Particle energy for each segment (GeV).
    photon_yield_factor : float
        Multiplicative factor on the Frank-Tamm yield.
    max_photons_per_segment : int
        Cap on photons per segment to prevent OOM.
        
    Returns
    -------
    dict with keys 'x_emit', 'y_emit', 'z_emit', 'x_ground', 'y_ground',
    each a numpy array.
    """
    H = 7640.0
    eta_0 = 0.000293
    m_e = 0.000511

    device = get_device()

    if HAS_TORCH and device is not None:
        return _cherenkov_torch(seg_x1, seg_y1, seg_z1,
                                seg_x2, seg_y2, seg_z2,
                                seg_px, seg_py, seg_pz,
                                seg_energy, photon_yield_factor,
                                max_photons_per_segment,
                                H, eta_0, m_e, device)
    else:
        return _cherenkov_numpy(seg_x1, seg_y1, seg_z1,
                                seg_x2, seg_y2, seg_z2,
                                seg_px, seg_py, seg_pz,
                                seg_energy, photon_yield_factor,
                                max_photons_per_segment,
                                H, eta_0, m_e)


def _cherenkov_torch(seg_x1, seg_y1, seg_z1,
                      seg_x2, seg_y2, seg_z2,
                      seg_px, seg_py, seg_pz,
                      seg_energy, photon_yield_factor,
                      max_photons_per_segment,
                      H, eta_0, m_e, device):
    """Cherenkov pool computation using PyTorch tensors with chunked processing.
    
    Uses float32 for photon positions to halve memory usage, and applies
    proportional thinning if the total photon count exceeds a budget
    (similar to CORSIKA thinning).
    """
    DTYPE = torch.float32
    MAX_TOTAL_PHOTONS = 50_000_000  # 50M photon budget (~1 GB total)
    
    # Compute segment properties on GPU (these are small — one per track segment)
    x1 = torch.as_tensor(seg_x1, dtype=DTYPE, device=device)
    y1 = torch.as_tensor(seg_y1, dtype=DTYPE, device=device)
    z1 = torch.as_tensor(seg_z1, dtype=DTYPE, device=device)
    x2 = torch.as_tensor(seg_x2, dtype=DTYPE, device=device)
    y2 = torch.as_tensor(seg_y2, dtype=DTYPE, device=device)
    z2 = torch.as_tensor(seg_z2, dtype=DTYPE, device=device)
    px = torch.as_tensor(seg_px, dtype=DTYPE, device=device)
    py = torch.as_tensor(seg_py, dtype=DTYPE, device=device)
    pz = torch.as_tensor(seg_pz, dtype=DTYPE, device=device)
    energies = torch.as_tensor(seg_energy, dtype=DTYPE, device=device)

    # Midpoints and segment lengths
    z_mid = (z1 + z2) / 2.0
    x_mid = (x1 + x2) / 2.0
    y_mid = (y1 + y2) / 2.0
    ds = torch.sqrt((x2-x1)**2 + (y2-y1)**2 + (z2-z1)**2)

    # Atmospheric optics
    eta_z = eta_0 * torch.exp(-z_mid / H)
    E_thresh = m_e / torch.sqrt(2 * eta_z)
    theta_c = torch.sqrt(2 * eta_z)

    # Photon counts per segment
    n_phot = (ds * theta_c**2 * 37000 * photon_yield_factor).to(torch.int64)
    n_phot = torch.clamp(n_phot, max=max_photons_per_segment)

    # Valid mask
    valid = (z_mid > 0) & (energies > E_thresh) & (n_phot > 0)
    
    if not torch.any(valid):
        empty = np.array([], dtype=np.float32)
        return {k: empty for k in ['x_emit', 'y_emit', 'z_emit', 'x_ground', 'y_ground']}

    x_mid_v = x_mid[valid].cpu()
    y_mid_v = y_mid[valid].cpu()
    z_mid_v = z_mid[valid].cpu()
    theta_c_v = theta_c[valid].cpu()
    px_v = px[valid].cpu()
    py_v = py[valid].cpu()
    pz_v = pz[valid].cpu()
    n_phot_v = n_phot[valid].cpu()

    # Free GPU memory from the segment-level computation
    del x1, y1, z1, x2, y2, z2, px, py, pz, energies, z_mid, x_mid, y_mid, ds
    del eta_z, E_thresh, theta_c, n_phot, valid
    torch.cuda.empty_cache()

    total_photons = int(torch.sum(n_phot_v).item())
    
    # Apply proportional thinning if over budget
    if total_photons > MAX_TOTAL_PHOTONS:
        thin_factor = MAX_TOTAL_PHOTONS / total_photons
        n_phot_v = (n_phot_v.float() * thin_factor).to(torch.int64)
        n_phot_v = torch.clamp(n_phot_v, min=0)
        # Remove segments that got thinned to zero
        keep = n_phot_v > 0
        x_mid_v = x_mid_v[keep]
        y_mid_v = y_mid_v[keep]
        z_mid_v = z_mid_v[keep]
        theta_c_v = theta_c_v[keep]
        px_v = px_v[keep]
        py_v = py_v[keep]
        pz_v = pz_v[keep]
        n_phot_v = n_phot_v[keep]
        total_photons = int(torch.sum(n_phot_v).item())

    n_segments = len(n_phot_v)
    
    # Determine chunk size based on available GPU memory
    gpu_mem = torch.cuda.get_device_properties(0).total_memory
    safe_mem = int(gpu_mem * 0.6)  # Use 60% of VRAM
    bytes_per_photon = 100  # Updated estimate for extra arrays
    max_photons_per_chunk = safe_mem // bytes_per_photon

    # Accumulate results on CPU
    result_xe, result_ye, result_ze = [], [], []
    result_xg, result_yg = [], []

    # Process segments in chunks that fit in GPU memory
    seg_start = 0
    while seg_start < n_segments:
        # Find how many segments fit in this chunk
        cumsum = torch.cumsum(n_phot_v[seg_start:], dim=0)
        fits = cumsum <= max_photons_per_chunk
        if not torch.any(fits):
            # Single segment exceeds chunk — process it alone
            seg_end = seg_start + 1
        else:
            seg_end = seg_start + int(fits.sum().item())
        
        # Slice this chunk's segment data
        chunk_x = x_mid_v[seg_start:seg_end].to(device)
        chunk_y = y_mid_v[seg_start:seg_end].to(device)
        chunk_z = z_mid_v[seg_start:seg_end].to(device)
        chunk_theta_c = theta_c_v[seg_start:seg_end].to(device)
        chunk_px = px_v[seg_start:seg_end].to(device)
        chunk_py = py_v[seg_start:seg_end].to(device)
        chunk_pz = pz_v[seg_start:seg_end].to(device)
        chunk_n = n_phot_v[seg_start:seg_end].to(device)

        # Generate photons for this chunk on GPU
        xe = torch.repeat_interleave(chunk_x, chunk_n)
        ye = torch.repeat_interleave(chunk_y, chunk_n)
        ze = torch.repeat_interleave(chunk_z, chunk_n)
        tc = torch.repeat_interleave(chunk_theta_c, chunk_n)
        dx = torch.repeat_interleave(chunk_px, chunk_n)
        dy = torch.repeat_interleave(chunk_py, chunk_n)
        dz = torch.repeat_interleave(chunk_pz, chunk_n)

        # Normalize direction vector
        norm = torch.sqrt(dx**2 + dy**2 + dz**2)
        dx = dx / norm
        dy = dy / norm
        dz = dz / norm

        # Orthonormal basis
        ref_x = torch.ones_like(dx)
        ref_y = torch.zeros_like(dx)
        ref_z = torch.zeros_like(dx)

        parallel_x = torch.abs(dx) > 0.9
        ref_x[parallel_x] = 0.0
        ref_y[parallel_x] = 1.0

        e1x = dy * ref_z - dz * ref_y
        e1y = dz * ref_x - dx * ref_z
        e1z = dx * ref_y - dy * ref_x
        
        e1_norm = torch.sqrt(e1x**2 + e1y**2 + e1z**2)
        e1x = e1x / e1_norm
        e1y = e1y / e1_norm
        e1z = e1z / e1_norm

        e2x = dy * e1z - dz * e1y
        e2y = dz * e1x - dx * e1z
        e2z = dx * e1y - dy * e1x

        chunk_total = int(chunk_n.sum().item())
        phis = torch.rand(chunk_total, dtype=DTYPE, device=device) * (2 * torch.pi)

        cos_tc = torch.cos(tc)
        sin_tc = torch.sin(tc)
        cos_phi = torch.cos(phis)
        sin_phi = torch.sin(phis)

        photon_dx = dx * cos_tc + e1x * sin_tc * cos_phi + e2x * sin_tc * sin_phi
        photon_dy = dy * cos_tc + e1y * sin_tc * cos_phi + e2y * sin_tc * sin_phi
        photon_dz = dz * cos_tc + e1z * sin_tc * cos_phi + e2z * sin_tc * sin_phi

        # Trace to ground z=0
        t = -ze / photon_dz
        valid_photon = (photon_dz < 0) & (t >= 0)

        xg = xe + t * photon_dx
        yg = ye + t * photon_dy

        # Filter out photons that never reach ground
        xg = xg[valid_photon]
        yg = yg[valid_photon]
        xe_v = xe[valid_photon]
        ye_v = ye[valid_photon]
        ze_v = ze[valid_photon]

        # Transfer to CPU and append
        result_xe.append(xe_v.cpu().numpy())
        result_ye.append(ye_v.cpu().numpy())
        result_ze.append(ze_v.cpu().numpy())
        result_xg.append(xg.cpu().numpy())
        result_yg.append(yg.cpu().numpy())

        # Free GPU memory for next chunk
        del xe, ye, ze, tc, dx, dy, dz, ref_x, ref_y, ref_z, e1x, e1y, e1z
        del e2x, e2y, e2z, phis, cos_tc, sin_tc, cos_phi, sin_phi
        del photon_dx, photon_dy, photon_dz, t, valid_photon, xg, yg
        del xe_v, ye_v, ze_v
        del chunk_x, chunk_y, chunk_z, chunk_theta_c, chunk_px, chunk_py, chunk_pz, chunk_n
        torch.cuda.empty_cache()

        seg_start = seg_end

    return {
        'x_emit': np.concatenate(result_xe) if result_xe else np.array([], dtype=np.float32),
        'y_emit': np.concatenate(result_ye) if result_ye else np.array([], dtype=np.float32),
        'z_emit': np.concatenate(result_ze) if result_ze else np.array([], dtype=np.float32),
        'x_ground': np.concatenate(result_xg) if result_xg else np.array([], dtype=np.float32),
        'y_ground': np.concatenate(result_yg) if result_yg else np.array([], dtype=np.float32),
    }


def _cherenkov_numpy(seg_x1, seg_y1, seg_z1,
                      seg_x2, seg_y2, seg_z2,
                      seg_px, seg_py, seg_pz,
                      seg_energy, photon_yield_factor,
                      max_photons_per_segment,
                      H, eta_0, m_e):
    """Cherenkov pool computation using NumPy (CPU fallback)."""
    
    x1 = np.array(seg_x1)
    y1 = np.array(seg_y1)
    z1 = np.array(seg_z1)
    x2 = np.array(seg_x2)
    y2 = np.array(seg_y2)
    z2 = np.array(seg_z2)
    px = np.array(seg_px)
    py = np.array(seg_py)
    pz = np.array(seg_pz)
    energies = np.array(seg_energy)

    z_mid = (z1 + z2) / 2.0
    x_mid = (x1 + x2) / 2.0
    y_mid = (y1 + y2) / 2.0
    ds = np.sqrt((x2-x1)**2 + (y2-y1)**2 + (z2-z1)**2)

    eta_z = eta_0 * np.exp(-z_mid / H)
    E_thresh = m_e / np.sqrt(2 * eta_z)
    theta_c = np.sqrt(2 * eta_z)

    n_phot = (ds * theta_c**2 * 37000 * photon_yield_factor).astype(int)
    n_phot = np.minimum(n_phot, max_photons_per_segment)

    valid = (z_mid > 0) & (energies > E_thresh) & (n_phot > 0)
    
    if not np.any(valid):
        empty = np.array([])
        return {k: empty for k in ['x_emit', 'y_emit', 'z_emit', 'x_ground', 'y_ground']}

    x_mid_v = x_mid[valid]
    y_mid_v = y_mid[valid]
    z_mid_v = z_mid[valid]
    theta_c_v = theta_c[valid]
    px_v = px[valid]
    py_v = py[valid]
    pz_v = pz[valid]
    n_phot_v = n_phot[valid]

    total_photons = int(np.sum(n_phot_v))

    xe = np.repeat(x_mid_v, n_phot_v)
    ye = np.repeat(y_mid_v, n_phot_v)
    ze = np.repeat(z_mid_v, n_phot_v)
    tc = np.repeat(theta_c_v, n_phot_v)
    dx = np.repeat(px_v, n_phot_v)
    dy = np.repeat(py_v, n_phot_v)
    dz = np.repeat(pz_v, n_phot_v)

    norm = np.sqrt(dx**2 + dy**2 + dz**2)
    dx = dx / norm
    dy = dy / norm
    dz = dz / norm

    ref_x = np.ones_like(dx)
    ref_y = np.zeros_like(dx)
    ref_z = np.zeros_like(dx)

    parallel_x = np.abs(dx) > 0.9
    ref_x[parallel_x] = 0.0
    ref_y[parallel_x] = 1.0

    e1x = dy * ref_z - dz * ref_y
    e1y = dz * ref_x - dx * ref_z
    e1z = dx * ref_y - dy * ref_x

    e1_norm = np.sqrt(e1x**2 + e1y**2 + e1z**2)
    e1x = e1x / e1_norm
    e1y = e1y / e1_norm
    e1z = e1z / e1_norm

    e2x = dy * e1z - dz * e1y
    e2y = dz * e1x - dx * e1z
    e2z = dx * e1y - dy * e1x

    phis = np.random.uniform(0, 2 * np.pi, total_photons)

    cos_tc = np.cos(tc)
    sin_tc = np.sin(tc)
    cos_phi = np.cos(phis)
    sin_phi = np.sin(phis)

    photon_dx = dx * cos_tc + e1x * sin_tc * cos_phi + e2x * sin_tc * sin_phi
    photon_dy = dy * cos_tc + e1y * sin_tc * cos_phi + e2y * sin_tc * sin_phi
    photon_dz = dz * cos_tc + e1z * sin_tc * cos_phi + e2z * sin_tc * sin_phi

    t = -ze / photon_dz
    valid_photon = (photon_dz < 0) & (t >= 0)

    xg = xe + t * photon_dx
    yg = ye + t * photon_dy

    return {
        'x_emit': xe[valid_photon],
        'y_emit': ye[valid_photon],
        'z_emit': ze[valid_photon],
        'x_ground': xg[valid_photon],
        'y_ground': yg[valid_photon],
    }


# ── Ray-tracing computation (GPU-accelerated pixel lookup) ────────────────────

def ray_trace_gpu(cherenkov_photons, pixel_x, pixel_y, pixel_size,
                  x_tel, y_tel, z_tel, mirror_radius,
                  mirror_reflectivity, quantum_efficiency):
    """
    GPU-accelerated ray-tracing: projects photons to focal plane and bins
    into camera pixels using torch.cdist instead of scipy KDTree.
    
    Parameters
    ----------
    cherenkov_photons : dict
        Must contain 'x_emit', 'y_emit', 'z_emit', 'x_ground', 'y_ground'.
    pixel_x, pixel_y : array-like
        Camera pixel positions in degrees.
    pixel_size : float
        Pixel diameter in degrees.
    x_tel, y_tel, z_tel : float
        Telescope position on ground (meters).
    mirror_radius : float
        Mirror radius (meters).
    mirror_reflectivity, quantum_efficiency : float
        Optical efficiency factors.
        
    Returns
    -------
    signal : numpy array of shape (n_pixels,)
        Number of photoelectrons per pixel (signal only, no noise).
    """
    device = get_device()
    
    if not HAS_TORCH or device is None or device.type != 'cuda':
        return _ray_trace_numpy(cherenkov_photons, pixel_x, pixel_y, pixel_size,
                                x_tel, y_tel, z_tel, mirror_radius,
                                mirror_reflectivity, quantum_efficiency)
    
    return _ray_trace_torch(cherenkov_photons, pixel_x, pixel_y, pixel_size,
                            x_tel, y_tel, z_tel, mirror_radius,
                            mirror_reflectivity, quantum_efficiency, device)


def _ray_trace_torch(cherenkov_photons, pixel_x, pixel_y, pixel_size,
                      x_tel, y_tel, z_tel, mirror_radius,
                      mirror_reflectivity, quantum_efficiency, device):
    """Ray-tracing using torch.cdist for GPU-accelerated nearest-neighbor."""
    
    n_pixels = len(pixel_x)
    
    xg = torch.tensor(cherenkov_photons['x_ground'], dtype=torch.float64, device=device)
    yg = torch.tensor(cherenkov_photons['y_ground'], dtype=torch.float64, device=device)
    xe = torch.tensor(cherenkov_photons['x_emit'], dtype=torch.float64, device=device)
    ye = torch.tensor(cherenkov_photons['y_emit'], dtype=torch.float64, device=device)
    ze = torch.tensor(cherenkov_photons['z_emit'], dtype=torch.float64, device=device)
    
    # Project to z_tel
    frac = (z_tel - ze) / (-ze)
    x_hit = xe + frac * (xg - xe)
    y_hit = ye + frac * (yg - ye)
    
    # Mirror hit test
    dist_sq = (x_hit - x_tel)**2 + (y_hit - y_tel)**2
    hit_mask = dist_sq <= mirror_radius**2
    
    if not torch.any(hit_mask):
        return np.zeros(n_pixels)
    
    xe = xe[hit_mask]
    ye = ye[hit_mask]
    ze = ze[hit_mask]
    x_hit = x_hit[hit_mask]
    y_hit = y_hit[hit_mask]
    
    # Survival filter (mirror reflectivity * quantum efficiency)
    survival_prob = mirror_reflectivity * quantum_efficiency
    survived = torch.rand(x_hit.shape[0], device=device) < survival_prob
    
    if not torch.any(survived):
        return np.zeros(n_pixels)
    
    xe = xe[survived]
    ye = ye[survived]
    ze = ze[survived]
    x_hit = x_hit[survived]
    y_hit = y_hit[survived]
    
    # Project to focal plane
    dx = x_hit - xe
    dy = y_hit - ye
    dz = z_tel - ze
    
    u_deg = torch.rad2deg(torch.atan2(-dx, -dz))
    v_deg = torch.rad2deg(torch.atan2(-dy, -dz))
    
    # GPU nearest-neighbor via torch.cdist
    # Process in chunks to avoid OOM on large photon counts
    pix_coords = torch.tensor(np.column_stack((pixel_x, pixel_y)),
                               dtype=torch.float64, device=device)  # (n_pixels, 2)
    
    signal = torch.zeros(n_pixels, dtype=torch.float64, device=device)
    
    chunk_size = 500_000  # Process photons in chunks to limit GPU memory
    n_photons = u_deg.shape[0]
    
    for start in range(0, n_photons, chunk_size):
        end = min(start + chunk_size, n_photons)
        photon_coords = torch.stack((u_deg[start:end], v_deg[start:end]), dim=1)  # (chunk, 2)
        
        # cdist: (chunk, n_pixels) distance matrix
        dists = torch.cdist(photon_coords, pix_coords)  # (chunk, n_pixels)
        
        # Find nearest pixel for each photon
        min_dists, min_indices = torch.min(dists, dim=1)  # (chunk,)
        
        # Filter by pixel acceptance radius
        valid = min_dists < (pixel_size / 2.0)
        valid_indices = min_indices[valid]
        
        # Bin into pixels
        signal.scatter_add_(0, valid_indices, torch.ones_like(valid_indices, dtype=torch.float64))
    
    return signal.cpu().numpy()


def _ray_trace_numpy(cherenkov_photons, pixel_x, pixel_y, pixel_size,
                      x_tel, y_tel, z_tel, mirror_radius,
                      mirror_reflectivity, quantum_efficiency):
    """Ray-tracing using scipy KDTree (CPU fallback)."""
    from scipy.spatial import KDTree
    
    n_pixels = len(pixel_x)
    
    xg = cherenkov_photons['x_ground']
    yg = cherenkov_photons['y_ground']
    xe = cherenkov_photons['x_emit']
    ye = cherenkov_photons['y_emit']
    ze = cherenkov_photons['z_emit']
    
    frac = (z_tel - ze) / (-ze)
    x_hit = xe + frac * (xg - xe)
    y_hit = ye + frac * (yg - ye)
    
    dist_to_center_sq = (x_hit - x_tel)**2 + (y_hit - y_tel)**2
    hit_mask = dist_to_center_sq <= mirror_radius**2
    
    if not np.any(hit_mask):
        return np.zeros(n_pixels)
    
    xe = xe[hit_mask]
    ye = ye[hit_mask]
    ze = ze[hit_mask]
    x_hit = x_hit[hit_mask]
    y_hit = y_hit[hit_mask]
    
    survival_prob = mirror_reflectivity * quantum_efficiency
    survived_mask = np.random.rand(len(x_hit)) < survival_prob
    
    if not np.any(survived_mask):
        return np.zeros(n_pixels)
    
    xe = xe[survived_mask]
    ye = ye[survived_mask]
    ze = ze[survived_mask]
    x_hit = x_hit[survived_mask]
    y_hit = y_hit[survived_mask]
    
    dx = x_hit - xe
    dy = y_hit - ye
    dz = z_tel - ze
    
    u_deg = np.degrees(np.arctan2(-dx, -dz))
    v_deg = np.degrees(np.arctan2(-dy, -dz))
    
    tree = KDTree(np.column_stack((pixel_x, pixel_y)))
    points = np.column_stack((u_deg, v_deg))
    distances, indices = tree.query(points)
    
    valid_hits = distances < (pixel_size / 2.0)
    valid_indices = indices[valid_hits]
    
    signal = np.zeros(n_pixels)
    np.add.at(signal, valid_indices, 1.0)
    
    return signal

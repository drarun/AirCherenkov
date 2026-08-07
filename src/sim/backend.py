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

def get_device(requested='auto'):
    """Resolve an explicitly requested Torch device.

    ``requested`` may be ``"auto"`` (the historical CUDA-then-CPU policy),
    ``"cpu"``, ``"cuda"``, or a :class:`torch.device`.  Asking for CUDA on a
    host where it is unavailable is an error rather than a silent CPU fallback.
    Calls without an argument retain the original behavior.
    """
    if requested is None:
        requested = 'auto'

    if not HAS_TORCH:
        if isinstance(requested, str) and requested.lower() in ('auto', 'cpu'):
            return None
        raise RuntimeError("PyTorch is required for the requested compute device")

    if isinstance(requested, str):
        normalized = requested.lower()
        if normalized == 'auto':
            return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if normalized not in ('cpu', 'cuda'):
            raise ValueError("device must be 'auto', 'cpu', 'cuda', or torch.device")
        device = torch.device(normalized)
    elif isinstance(requested, torch.device):
        device = requested
    else:
        raise ValueError("device must be 'auto', 'cpu', 'cuda', or torch.device")

    if device.type not in ('cpu', 'cuda'):
        raise ValueError("Only CPU and CUDA devices are supported")
    if device.type == 'cuda':
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")
        if device.index is not None and device.index >= torch.cuda.device_count():
            raise RuntimeError(f"CUDA device index {device.index} is not available")
    return device

def device_info(requested='auto'):
    """Returns a human-readable string describing the compute backend."""
    if not HAS_TORCH:
        return "NumPy (CPU only — install PyTorch for GPU acceleration)"
    dev = get_device(requested)
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


def _resolve_requested_device(requested):
    # Calling get_device() without an argument for auto also preserves callers
    # that monkeypatch the historical zero-argument helper.
    if requested is None or (
        isinstance(requested, str) and requested.lower() == 'auto'
    ):
        return get_device()
    return get_device(requested)

# ── Cherenkov pool computation (the main GPU kernel) ──────────────────────────

def compute_cherenkov_pool_gpu(seg_x1, seg_y1, seg_z1,
                                seg_x2, seg_y2, seg_z2,
                                seg_px, seg_py, seg_pz,
                                seg_energy, photon_yield_factor,
                                max_photons_per_segment=50000,
                                seg_event_id=None,
                                target_photons_per_packet=1.0,
                                max_packets_per_segment=None,
                                device='auto', generator=None):
    """Generate weighted Cherenkov photon packets and trace them to ground.

    The five historical position arrays are retained.  ``weight`` gives the
    physical photon multiplicity represented by each returned packet, so
    limiting packet count no longer silently loses light.  If
    ``seg_event_id`` is supplied, a corresponding int32 ``event_id`` array is
    returned as well.

    Packet emission locations and azimuths are stratified within each segment.
    The local refractive index and the electron beta-dependent Cherenkov angle
    are evaluated at each packet's emission altitude.  The packet weights are
    a stratified quadrature of the segment's Frank--Tamm yield.

    ``max_photons_per_segment`` remains as the legacy positional packet cap.
    New code may use ``max_packets_per_segment`` to state that intent directly.
    """
    if not np.isfinite(photon_yield_factor) or photon_yield_factor < 0:
        raise ValueError("photon_yield_factor must be finite and non-negative")
    if not np.isfinite(target_photons_per_packet) or target_photons_per_packet <= 0:
        raise ValueError("target_photons_per_packet must be finite and greater than zero")

    packet_cap = (
        max_photons_per_segment
        if max_packets_per_segment is None
        else max_packets_per_segment
    )
    if not isinstance(packet_cap, (int, np.integer)) or packet_cap <= 0:
        raise ValueError("max_packets_per_segment must be a positive integer")

    resolved_device = _resolve_requested_device(device)
    if HAS_TORCH and resolved_device is not None:
        return _cherenkov_packets_torch(
            seg_x1, seg_y1, seg_z1,
            seg_x2, seg_y2, seg_z2,
            seg_px, seg_py, seg_pz,
            seg_energy, photon_yield_factor,
            int(packet_cap), seg_event_id,
            float(target_photons_per_packet), resolved_device, generator,
        )

    return _cherenkov_packets_numpy(
        seg_x1, seg_y1, seg_z1,
        seg_x2, seg_y2, seg_z2,
        seg_px, seg_py, seg_pz,
        seg_energy, photon_yield_factor,
        int(packet_cap), seg_event_id,
        float(target_photons_per_packet), generator,
    )


_PHOTON_POSITION_KEYS = ('x_emit', 'y_emit', 'z_emit', 'x_ground', 'y_ground')


def _empty_photon_packets(include_event_id=False):
    result = {
        key: np.empty(0, dtype=np.float32)
        for key in (*_PHOTON_POSITION_KEYS, 'weight')
    }
    if include_event_id:
        result['event_id'] = np.empty(0, dtype=np.int32)
    return result


def _validate_generator(generator, device):
    if generator is None:
        return
    if not isinstance(generator, torch.Generator):
        raise TypeError("generator must be a torch.Generator")
    generator_device = torch.device(generator.device)
    if generator_device.type != device.type:
        raise ValueError(
            f"generator is on {generator_device.type}, but computation is on {device.type}"
        )


def _packet_chunk_limit(device):
    """Conservative bound for packet construction temporaries."""
    if device.type == 'cuda':
        free_bytes, _ = torch.cuda.mem_get_info(device)
        return max(1, min(5_000_000, int(free_bytes * 0.35) // 128))
    return 1_000_000


def _cherenkov_packets_torch(seg_x1, seg_y1, seg_z1,
                               seg_x2, seg_y2, seg_z2,
                               seg_px, seg_py, seg_pz,
                               seg_energy, photon_yield_factor,
                               max_packets_per_segment, seg_event_id,
                               target_photons_per_packet, device, generator):
    dtype = torch.float32
    _validate_generator(generator, device)

    raw_values = (
        seg_x1, seg_y1, seg_z1, seg_x2, seg_y2, seg_z2,
        seg_px, seg_py, seg_pz, seg_energy,
    )
    tensors = [
        torch.as_tensor(value, dtype=dtype, device=device).reshape(-1)
        for value in raw_values
    ]
    lengths = {tensor.numel() for tensor in tensors}
    if len(lengths) != 1:
        raise ValueError("All segment arrays must have the same length")
    n_segments = tensors[0].numel()
    include_event_id = seg_event_id is not None
    if n_segments == 0:
        return _empty_photon_packets(include_event_id)

    if include_event_id:
        event_ids = torch.as_tensor(
            seg_event_id, dtype=torch.int32, device=device
        ).reshape(-1)
        if event_ids.numel() != n_segments:
            raise ValueError("seg_event_id must have the same length as segment arrays")
    else:
        event_ids = None

    x1, y1, z1, x2, y2, z2, px, py, pz, energies = tensors
    delta = torch.stack((x2 - x1, y2 - y1, z2 - z1), dim=1)
    segment_length = torch.linalg.vector_norm(delta, dim=1)
    z_mid = 0.5 * (z1 + z2)

    # Electron beta and midpoint yield are used only to choose packet count.
    # The actual local angle and yield are recomputed at each packet altitude.
    scale_height = 7640.0
    eta_0 = 0.000293
    electron_mass_gev = 0.000511
    beta_sq = torch.clamp(
        1.0 - (electron_mass_gev / torch.clamp(energies, min=electron_mass_gev)) ** 2,
        min=0.0,
        max=1.0,
    )
    n_mid = 1.0 + eta_0 * torch.exp(-torch.clamp(z_mid, min=0.0) / scale_height)
    sin2_mid = torch.clamp(
        1.0 - 1.0 / torch.clamp(beta_sq * n_mid.square(), min=1e-20),
        min=0.0,
        max=1.0,
    )
    estimated_yield = (
        segment_length * sin2_mid * 37000.0 * float(photon_yield_factor)
    )
    candidate = (
        (segment_length > 0)
        & (torch.maximum(z1, z2) > 0)
        & (estimated_yield > 0)
    )
    if not bool(torch.any(candidate)):
        return _empty_photon_packets(include_event_id)

    x1 = x1[candidate]
    y1 = y1[candidate]
    z1 = z1[candidate]
    x2 = x2[candidate]
    y2 = y2[candidate]
    z2 = z2[candidate]
    segment_length = segment_length[candidate]
    beta_sq = beta_sq[candidate]
    estimated_yield = estimated_yield[candidate]
    directions = torch.stack((px[candidate], py[candidate], pz[candidate]), dim=1)
    directions = directions / torch.clamp(
        torch.linalg.vector_norm(directions, dim=1, keepdim=True), min=1e-12
    )
    if event_ids is not None:
        event_ids = event_ids[candidate]

    packet_counts = torch.ceil(
        estimated_yield / target_photons_per_packet
    ).clamp(min=1, max=max_packets_per_segment).to(torch.int64)
    cumulative = torch.cumsum(packet_counts, dim=0)
    total_packets = int(cumulative[-1].item())
    if total_packets == 0:
        return _empty_photon_packets(include_event_id)

    packet_limit = _packet_chunk_limit(device)
    packet_chunks = []
    event_chunks = []
    segment_start = 0
    base_packets = 0
    n_valid_segments = packet_counts.numel()

    with torch.no_grad():
        while segment_start < n_valid_segments:
            target_end = base_packets + packet_limit
            segment_end = int(
                torch.searchsorted(
                    cumulative, torch.tensor(target_end, device=device), right=True
                ).item()
            )
            segment_end = max(segment_start + 1, min(segment_end, n_valid_segments))

            counts = packet_counts[segment_start:segment_end]
            local_segment = torch.repeat_interleave(
                torch.arange(counts.numel(), device=device), counts
            )
            n_chunk = local_segment.numel()
            offsets = torch.cumsum(counts, dim=0) - counts
            ordinal = torch.arange(n_chunk, device=device) - torch.repeat_interleave(
                offsets, counts
            )
            count_per_packet = counts[local_segment]

            longitudinal_jitter = torch.rand(
                n_chunk, dtype=dtype, device=device, generator=generator
            )
            fraction = (
                ordinal.to(dtype) + longitudinal_jitter
            ) / count_per_packet.to(dtype)

            sx1 = x1[segment_start:segment_end]
            sy1 = y1[segment_start:segment_end]
            sz1 = z1[segment_start:segment_end]
            sx2 = x2[segment_start:segment_end]
            sy2 = y2[segment_start:segment_end]
            sz2 = z2[segment_start:segment_end]
            xe = sx1[local_segment] + fraction * (sx2 - sx1)[local_segment]
            ye = sy1[local_segment] + fraction * (sy2 - sy1)[local_segment]
            ze = sz1[local_segment] + fraction * (sz2 - sz1)[local_segment]

            local_n = 1.0 + eta_0 * torch.exp(
                -torch.clamp(ze, min=0.0) / scale_height
            )
            local_beta_sq = beta_sq[segment_start:segment_end][local_segment]
            local_sin2 = torch.clamp(
                1.0 - 1.0 / torch.clamp(
                    local_beta_sq * local_n.square(), min=1e-20
                ),
                min=0.0,
                max=1.0,
            )
            cos_theta = torch.rsqrt(torch.clamp(
                local_beta_sq * local_n.square(), min=1.0
            ))
            cos_theta = torch.clamp(cos_theta, min=-1.0, max=1.0)
            sin_theta = torch.sqrt(local_sin2)

            chunk_directions = directions[segment_start:segment_end]
            reference = torch.zeros_like(chunk_directions)
            reference[:, 0] = 1.0
            parallel = torch.abs(chunk_directions[:, 0]) > 0.9
            reference[parallel, 0] = 0.0
            reference[parallel, 1] = 1.0
            basis1 = torch.linalg.cross(chunk_directions, reference, dim=1)
            basis1 = basis1 / torch.clamp(
                torch.linalg.vector_norm(basis1, dim=1, keepdim=True), min=1e-12
            )
            basis2 = torch.linalg.cross(chunk_directions, basis1, dim=1)

            azimuth_jitter = torch.rand(
                n_chunk, dtype=dtype, device=device, generator=generator
            )
            phi = 2.0 * torch.pi * (
                ordinal.to(dtype) + azimuth_jitter
            ) / count_per_packet.to(dtype)
            direction = chunk_directions[local_segment]
            e1 = basis1[local_segment]
            e2 = basis2[local_segment]
            photon_direction = (
                direction * cos_theta[:, None]
                + e1 * (sin_theta * torch.cos(phi))[:, None]
                + e2 * (sin_theta * torch.sin(phi))[:, None]
            )

            distance_to_ground = -ze / photon_direction[:, 2]
            reaches_ground = (
                (ze > 0)
                & (photon_direction[:, 2] < 0)
                & torch.isfinite(distance_to_ground)
                & (distance_to_ground >= 0)
                & (local_sin2 > 0)
            )
            x_ground = xe + distance_to_ground * photon_direction[:, 0]
            y_ground = ye + distance_to_ground * photon_direction[:, 1]

            # Stratified quadrature of the Frank--Tamm yield along the segment.
            # This preserves the local altitude/beta dependence instead of
            # forcing every long segment to use its midpoint refractivity.
            packet_weight = (
                segment_length[segment_start:segment_end][local_segment]
                * 37000.0
                * float(photon_yield_factor)
                * local_sin2
                / count_per_packet.to(dtype)
            )
            packet_values = torch.stack(
                (xe, ye, ze, x_ground, y_ground, packet_weight), dim=1
            )[reaches_ground]
            packet_chunks.append(packet_values.cpu().numpy())
            if event_ids is not None:
                chunk_events = event_ids[segment_start:segment_end][local_segment]
                event_chunks.append(chunk_events[reaches_ground].cpu().numpy())

            segment_start = segment_end
            base_packets = int(cumulative[segment_end - 1].item())

    if not packet_chunks:
        return _empty_photon_packets(include_event_id)
    packed = np.concatenate(packet_chunks, axis=0).astype(np.float32, copy=False)
    result = {
        key: packed[:, column]
        for column, key in enumerate((*_PHOTON_POSITION_KEYS, 'weight'))
    }
    if include_event_id:
        result['event_id'] = np.concatenate(event_chunks).astype(np.int32, copy=False)
    return result


def _cherenkov_packets_numpy(seg_x1, seg_y1, seg_z1,
                               seg_x2, seg_y2, seg_z2,
                               seg_px, seg_py, seg_pz,
                               seg_energy, photon_yield_factor,
                               max_packets_per_segment, seg_event_id,
                               target_photons_per_packet, generator):
    """NumPy fallback with the same weighted-packet contract."""
    arrays = [
        np.asarray(value, dtype=np.float32).reshape(-1)
        for value in (
            seg_x1, seg_y1, seg_z1, seg_x2, seg_y2, seg_z2,
            seg_px, seg_py, seg_pz, seg_energy,
        )
    ]
    if len({len(value) for value in arrays}) != 1:
        raise ValueError("All segment arrays must have the same length")
    include_event_id = seg_event_id is not None
    if len(arrays[0]) == 0:
        return _empty_photon_packets(include_event_id)
    if generator is not None and not isinstance(generator, np.random.Generator):
        raise TypeError("Without PyTorch, generator must be numpy.random.Generator")
    rng = generator if generator is not None else np.random

    x1, y1, z1, x2, y2, z2, px, py, pz, energies = arrays
    ds = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2 + (z2 - z1) ** 2)
    beta_sq = np.clip(1.0 - (0.000511 / np.maximum(energies, 0.000511)) ** 2, 0, 1)
    n_mid = 1.0 + 0.000293 * np.exp(-np.maximum(0.5 * (z1 + z2), 0) / 7640.0)
    sin2_mid = np.clip(1.0 - 1.0 / np.maximum(beta_sq * n_mid ** 2, 1e-20), 0, 1)
    estimated = ds * sin2_mid * 37000.0 * photon_yield_factor
    valid = (ds > 0) & (np.maximum(z1, z2) > 0) & (estimated > 0)
    if not np.any(valid):
        return _empty_photon_packets(include_event_id)

    x1, y1, z1, x2, y2, z2, px, py, pz, ds, beta_sq, estimated = [
        value[valid]
        for value in (x1, y1, z1, x2, y2, z2, px, py, pz, ds, beta_sq, estimated)
    ]
    events = None
    if include_event_id:
        events = np.asarray(seg_event_id, dtype=np.int32).reshape(-1)
        if len(events) != len(valid):
            raise ValueError("seg_event_id must have the same length as segment arrays")
        events = events[valid]

    counts = np.clip(
        np.ceil(estimated / target_photons_per_packet), 1, max_packets_per_segment
    ).astype(np.int64)
    segment_index = np.repeat(np.arange(len(counts)), counts)
    offsets = np.cumsum(counts) - counts
    ordinal = np.arange(len(segment_index)) - np.repeat(offsets, counts)
    packet_count = counts[segment_index]
    fraction = (ordinal + rng.random(len(segment_index))) / packet_count
    xe = x1[segment_index] + fraction * (x2 - x1)[segment_index]
    ye = y1[segment_index] + fraction * (y2 - y1)[segment_index]
    ze = z1[segment_index] + fraction * (z2 - z1)[segment_index]

    local_n = 1.0 + 0.000293 * np.exp(-np.maximum(ze, 0) / 7640.0)
    local_sin2 = np.clip(
        1.0 - 1.0 / np.maximum(beta_sq[segment_index] * local_n ** 2, 1e-20), 0, 1
    )
    cos_theta = np.clip(
        1.0 / np.sqrt(np.maximum(beta_sq[segment_index] * local_n ** 2, 1.0)), -1, 1
    )
    sin_theta = np.sqrt(local_sin2)

    directions = np.column_stack((px, py, pz))
    directions /= np.maximum(np.linalg.norm(directions, axis=1, keepdims=True), 1e-12)
    reference = np.zeros_like(directions)
    reference[:, 0] = 1
    parallel = np.abs(directions[:, 0]) > 0.9
    reference[parallel] = (0, 1, 0)
    basis1 = np.cross(directions, reference)
    basis1 /= np.maximum(np.linalg.norm(basis1, axis=1, keepdims=True), 1e-12)
    basis2 = np.cross(directions, basis1)
    phi = 2 * np.pi * (ordinal + rng.random(len(segment_index))) / packet_count
    photon_direction = (
        directions[segment_index] * cos_theta[:, None]
        + basis1[segment_index] * (sin_theta * np.cos(phi))[:, None]
        + basis2[segment_index] * (sin_theta * np.sin(phi))[:, None]
    )
    distance = -ze / photon_direction[:, 2]
    reaches_ground = (
        (ze > 0) & (photon_direction[:, 2] < 0) & np.isfinite(distance)
        & (distance >= 0) & (local_sin2 > 0)
    )
    weight = (
        ds[segment_index]
        * 37000.0
        * photon_yield_factor
        * local_sin2
        / packet_count
    )
    packed = np.column_stack((
        xe, ye, ze,
        xe + distance * photon_direction[:, 0],
        ye + distance * photon_direction[:, 1],
        weight,
    ))[reaches_ground].astype(np.float32, copy=False)
    result = {
        key: packed[:, column]
        for column, key in enumerate((*_PHOTON_POSITION_KEYS, 'weight'))
    }
    if include_event_id:
        result['event_id'] = events[segment_index][reaches_ground].astype(np.int32, copy=False)
    return result


# ── Ray-tracing computation (GPU-accelerated pixel lookup) ────────────────────

def _camera_axial_coordinates(pixel_x, pixel_y, pixel_size):
    """Recover integer axial coordinates from the camera's pixel centers."""
    if pixel_size <= 0:
        raise ValueError("pixel_size must be greater than zero")

    pixel_x = np.asarray(pixel_x, dtype=np.float64)
    pixel_y = np.asarray(pixel_y, dtype=np.float64)
    height = np.sqrt(3.0) / 2.0 * pixel_size

    axial_r = np.rint(pixel_y / height).astype(np.int64)
    axial_q = np.rint((pixel_x / pixel_size) - axial_r / 2.0).astype(np.int64)

    reconstructed_x = pixel_size * (axial_q + axial_r / 2.0)
    reconstructed_y = height * axial_r
    tolerance = max(1e-9, pixel_size * 1e-6)
    if not (
        np.allclose(pixel_x, reconstructed_x, atol=tolerance, rtol=0.0)
        and np.allclose(pixel_y, reconstructed_y, atol=tolerance, rtol=0.0)
    ):
        raise ValueError("Camera pixels do not lie on the expected hexagonal grid")

    n_rings = int(np.max(np.abs(np.concatenate([
        axial_q, axial_r, axial_q + axial_r,
    ])))) if len(axial_q) else 0
    return axial_q, axial_r, n_rings


def ray_trace_gpu(cherenkov_photons, pixel_x, pixel_y, pixel_size,
                  x_tel, y_tel, z_tel, mirror_radius,
                  mirror_reflectivity, quantum_efficiency, *,
                  n_time_bins=16, bin_width_ns=2.0, nsb_rate=2.0,
                  pedestal_std=0.5, saturation_limit=250.0,
                  low_gain_factor=10.0, shower_start_altitude=20000.0,
                  device='auto', generator=None):
    """Compatibility wrapper for one telescope.

    The implementation is the same shared Torch core used by
    :func:`ray_trace_array` on both CPU and CUDA.  Its return shapes remain
    ``(n_pixels, n_time_bins)`` and ``(n_pixels,)``.
    """
    traces, gain_flags = ray_trace_array(
        cherenkov_photons,
        pixel_x, pixel_y, pixel_size,
        [x_tel], [y_tel], [z_tel], [mirror_radius],
        [mirror_reflectivity], [quantum_efficiency],
        n_time_bins=n_time_bins,
        bin_width_ns=bin_width_ns,
        nsb_rate=nsb_rate,
        pedestal_std=pedestal_std,
        saturation_limit=saturation_limit,
        low_gain_factor=low_gain_factor,
        shower_start_altitude=shower_start_altitude,
        device=device,
        generator=generator,
    )
    return traces[0], gain_flags[0]


def ray_trace_array(cherenkov_photons, pixel_x, pixel_y, pixel_size,
                    telescope_x, telescope_y, telescope_z, mirror_radius,
                    mirror_reflectivity, quantum_efficiency, *,
                    n_time_bins=16, bin_width_ns=2.0, nsb_rate=2.0,
                    pedestal_std=0.5, saturation_limit=250.0,
                    low_gain_factor=10.0, shower_start_altitude=20000.0,
                    device='auto', generator=None):
    """Trace a photon pool through several telescopes sharing one camera.

    Telescope parameters may be scalars or one-dimensional arrays; scalar
    values broadcast to the longest array.  Weighted packets are converted to
    photoelectrons with binomial detection statistics.  ``nsb_rate`` is the
    mean number of NSB photoelectrons per pixel over the complete window.
    ``pedestal_std`` is the RMS of the integrated window, distributed as
    independent Gaussian noise across its time bins.

    Returns
    -------
    traces, gain_flags : numpy.ndarray
        Float32 arrays shaped ``(telescopes, pixels, time_bins)`` and
        ``(telescopes, pixels)`` respectively.
    """
    if not HAS_TORCH:
        raise RuntimeError("PyTorch is required for ray tracing")
    resolved_device = _resolve_requested_device(device)
    _validate_generator(generator, resolved_device)
    _validate_detector_configuration(
        n_time_bins, bin_width_ns, nsb_rate, pedestal_std,
        saturation_limit, low_gain_factor,
    )

    pixel_q, pixel_r, n_rings = _camera_axial_coordinates(
        pixel_x, pixel_y, pixel_size
    )
    n_pixels = len(pixel_q)
    if n_pixels == 0:
        raise ValueError("Camera must contain at least one pixel")

    telescope_parameters = _broadcast_telescope_parameters(
        resolved_device,
        telescope_x=telescope_x,
        telescope_y=telescope_y,
        telescope_z=telescope_z,
        mirror_radius=mirror_radius,
        mirror_reflectivity=mirror_reflectivity,
        quantum_efficiency=quantum_efficiency,
    )
    _validate_telescope_parameters(telescope_parameters)

    with torch.no_grad():
        traces, gain_flags = _ray_trace_torch_core(
            cherenkov_photons,
            pixel_q, pixel_r, n_rings, float(pixel_size),
            telescope_parameters,
            int(n_time_bins), float(bin_width_ns), float(nsb_rate),
            float(pedestal_std), float(saturation_limit),
            float(low_gain_factor), float(shower_start_altitude),
            resolved_device, generator,
        )
    return (
        traces.detach().cpu().numpy().astype(np.float32, copy=False),
        gain_flags.detach().cpu().numpy().astype(np.float32, copy=False),
    )


def _validate_detector_configuration(n_time_bins, bin_width_ns, nsb_rate,
                                     pedestal_std, saturation_limit,
                                     low_gain_factor):
    if not isinstance(n_time_bins, (int, np.integer)) or n_time_bins <= 0:
        raise ValueError("n_time_bins must be a positive integer")
    for name, value, allow_zero in (
        ('bin_width_ns', bin_width_ns, False),
        ('nsb_rate', nsb_rate, True),
        ('pedestal_std', pedestal_std, True),
        ('saturation_limit', saturation_limit, False),
        ('low_gain_factor', low_gain_factor, False),
    ):
        if not np.isfinite(value) or value < 0 or (not allow_zero and value == 0):
            qualifier = 'non-negative' if allow_zero else 'greater than zero'
            raise ValueError(f"{name} must be finite and {qualifier}")


def _broadcast_telescope_parameters(device, **parameters):
    tensors = {
        name: torch.as_tensor(value, dtype=torch.float32, device=device).reshape(-1)
        for name, value in parameters.items()
    }
    if any(value.numel() == 0 for value in tensors.values()):
        raise ValueError("Telescope parameter arrays cannot be empty")
    n_telescopes = max(value.numel() for value in tensors.values())
    for name, value in tuple(tensors.items()):
        if value.numel() == 1:
            tensors[name] = value.expand(n_telescopes)
        elif value.numel() != n_telescopes:
            raise ValueError(
                f"{name} must be scalar or contain {n_telescopes} values"
            )
    return tensors


def _validate_telescope_parameters(parameters):
    for name, value in parameters.items():
        if not bool(torch.all(torch.isfinite(value))):
            raise ValueError(f"{name} must contain only finite values")
    if bool(torch.any(parameters['mirror_radius'] <= 0)):
        raise ValueError("mirror_radius must be greater than zero")
    for name in ('mirror_reflectivity', 'quantum_efficiency'):
        value = parameters[name]
        if bool(torch.any((value < 0) | (value > 1))):
            raise ValueError(f"{name} values must lie in [0, 1]")


def _photon_tensor(photons, key, n_packets, device, *, default=None,
                   dtype=None):
    if key not in photons:
        if default is None:
            raise KeyError(f"cherenkov_photons is missing required key {key!r}")
        value = default
    else:
        value = photons[key]
    tensor = torch.as_tensor(
        value, dtype=dtype or torch.float32, device=device
    ).reshape(-1)
    if tensor.numel() == 1 and n_packets != 1:
        tensor = tensor.expand(n_packets)
    elif tensor.numel() != n_packets:
        raise ValueError(f"Photon field {key!r} must contain {n_packets} values")
    return tensor


def _detected_packet_charge(weights, probability, generator):
    """Unbiased packet-to-photoelectron conversion, exact for integer weight."""
    weights = torch.clamp(weights, min=0.0)
    probability = torch.clamp(probability, min=0.0, max=1.0)
    whole_photons = torch.floor(weights)
    detected = torch.binomial(whole_photons, probability, generator=generator)
    fractional_probability = (weights - whole_photons) * probability
    detected += (
        torch.rand(
            fractional_probability.shape,
            dtype=fractional_probability.dtype,
            device=fractional_probability.device,
            generator=generator,
        ) < fractional_probability
    ).to(detected.dtype)
    return detected


def _ray_trace_torch_core(cherenkov_photons, pixel_q, pixel_r, n_rings,
                          pixel_size, telescope, n_time_bins, bin_width_ns,
                          nsb_rate, pedestal_std, saturation_limit,
                          low_gain_factor, shower_start_altitude,
                          device, generator):
    """Shared CPU/CUDA geometry, timing, noise, and gain implementation."""
    n_telescopes = telescope['telescope_x'].numel()
    n_pixels = len(pixel_q)
    traces = torch.zeros(
        (n_telescopes, n_pixels, n_time_bins),
        dtype=torch.float32, device=device,
    )

    grid_size = 2 * n_rings + 1
    lookup = torch.full(
        (grid_size, grid_size), -1, dtype=torch.int64, device=device
    )
    pixel_q_t = torch.as_tensor(pixel_q + n_rings, dtype=torch.int64, device=device)
    pixel_r_t = torch.as_tensor(pixel_r + n_rings, dtype=torch.int64, device=device)
    lookup[pixel_q_t, pixel_r_t] = torch.arange(
        n_pixels, dtype=torch.int64, device=device
    )

    if 'x_ground' not in cherenkov_photons:
        raise KeyError("cherenkov_photons is missing required key 'x_ground'")
    n_packets = len(cherenkov_photons['x_ground'])
    hit_telescopes = []
    hit_pixels = []
    hit_times = []
    hit_charges = []

    if n_packets:
        xe = _photon_tensor(cherenkov_photons, 'x_emit', n_packets, device)
        ye = _photon_tensor(cherenkov_photons, 'y_emit', n_packets, device)
        ze = _photon_tensor(cherenkov_photons, 'z_emit', n_packets, device)
        xg = _photon_tensor(cherenkov_photons, 'x_ground', n_packets, device)
        yg = _photon_tensor(cherenkov_photons, 'y_ground', n_packets, device)
        weights = _photon_tensor(
            cherenkov_photons, 'weight', n_packets, device, default=1.0
        )
        if bool(torch.any(~torch.isfinite(weights))) or bool(torch.any(weights < 0)):
            raise ValueError("Photon weights must be finite and non-negative")

        if 'emission_time_ns' in cherenkov_photons:
            emission_time = _photon_tensor(
                cherenkov_photons, 'emission_time_ns', n_packets, device
            )
        else:
            packet_start_altitude = _photon_tensor(
                cherenkov_photons,
                'shower_start_altitude',
                n_packets,
                device,
                default=shower_start_altitude,
            )
            emission_time = (packet_start_altitude - ze) / 0.299792458

        pair_limit = _ray_pair_chunk_limit(device)
        packet_chunk_size = max(1, pair_limit // n_telescopes)
        height = np.sqrt(3.0) * 0.5 * pixel_size

        for start in range(0, n_packets, packet_chunk_size):
            end = min(start + packet_chunk_size, n_packets)
            c_xe, c_ye, c_ze = xe[start:end], ye[start:end], ze[start:end]
            c_xg, c_yg = xg[start:end], yg[start:end]

            tel_z = telescope['telescope_z'][:, None]
            denominator = -c_ze[None, :]
            projection = (tel_z - c_ze[None, :]) / denominator
            x_hit = c_xe[None, :] + projection * (c_xg - c_xe)[None, :]
            y_hit = c_ye[None, :] + projection * (c_yg - c_ye)[None, :]
            mirror_distance_sq = (
                (x_hit - telescope['telescope_x'][:, None]).square()
                + (y_hit - telescope['telescope_y'][:, None]).square()
            )
            mirror_hit = (
                torch.isfinite(x_hit)
                & torch.isfinite(y_hit)
                & (c_ze[None, :] > tel_z)
                & (mirror_distance_sq <= telescope['mirror_radius'][:, None].square())
            )
            telescope_index, local_packet_index = torch.nonzero(
                mirror_hit, as_tuple=True
            )
            if telescope_index.numel() == 0:
                continue

            hit_x = x_hit[telescope_index, local_packet_index]
            hit_y = y_hit[telescope_index, local_packet_index]
            emit_x = c_xe[local_packet_index]
            emit_y = c_ye[local_packet_index]
            emit_z = c_ze[local_packet_index]
            hit_z = telescope['telescope_z'][telescope_index]
            dx = hit_x - emit_x
            dy = hit_y - emit_y
            dz = hit_z - emit_z

            u_deg = torch.rad2deg(torch.atan2(-dx, -dz))
            v_deg = torch.rad2deg(torch.atan2(-dy, -dz))
            r_float = v_deg / height
            q_float = (u_deg / pixel_size) - 0.5 * r_float
            cube_x = q_float
            cube_z = r_float
            cube_y = -cube_x - cube_z
            rounded_x = torch.round(cube_x)
            rounded_y = torch.round(cube_y)
            rounded_z = torch.round(cube_z)
            diff_x = torch.abs(rounded_x - cube_x)
            diff_y = torch.abs(rounded_y - cube_y)
            diff_z = torch.abs(rounded_z - cube_z)
            adjust_x = (diff_x > diff_y) & (diff_x > diff_z)
            adjust_y = (~adjust_x) & (diff_y > diff_z)
            adjust_z = ~(adjust_x | adjust_y)
            rounded_x = torch.where(adjust_x, -rounded_y - rounded_z, rounded_x)
            rounded_y = torch.where(adjust_y, -rounded_x - rounded_z, rounded_y)
            rounded_z = torch.where(adjust_z, -rounded_x - rounded_y, rounded_z)
            q_int = rounded_x.to(torch.int64)
            r_int = rounded_z.to(torch.int64)
            in_camera = (
                (q_int >= -n_rings) & (q_int <= n_rings)
                & (r_int >= -n_rings) & (r_int <= n_rings)
                & (q_int + r_int >= -n_rings)
                & (q_int + r_int <= n_rings)
            )
            if not bool(torch.any(in_camera)):
                continue

            telescope_index = telescope_index[in_camera]
            local_packet_index = local_packet_index[in_camera]
            q_int = q_int[in_camera]
            r_int = r_int[in_camera]
            pixel_index = lookup[q_int + n_rings, r_int + n_rings]
            mapped = pixel_index >= 0
            if not bool(torch.any(mapped)):
                continue
            telescope_index = telescope_index[mapped]
            local_packet_index = local_packet_index[mapped]
            pixel_index = pixel_index[mapped]
            emit_x = c_xe[local_packet_index]
            emit_y = c_ye[local_packet_index]
            emit_z = c_ze[local_packet_index]
            hit_x = x_hit[telescope_index, local_packet_index]
            hit_y = y_hit[telescope_index, local_packet_index]
            hit_z = telescope['telescope_z'][telescope_index]
            photon_distance = torch.sqrt(
                (hit_x - emit_x).square()
                + (hit_y - emit_y).square()
                + (hit_z - emit_z).square()
            )
            arrival_time = (
                emission_time[start:end][local_packet_index]
                + photon_distance / (0.299792458 / 1.0003)
            )

            probability = (
                telescope['mirror_reflectivity'][telescope_index]
                * telescope['quantum_efficiency'][telescope_index]
            )
            charge = _detected_packet_charge(
                weights[start:end][local_packet_index], probability, generator
            )
            detected = charge > 0
            if not bool(torch.any(detected)):
                continue
            hit_telescopes.append(telescope_index[detected])
            hit_pixels.append(pixel_index[detected])
            hit_times.append(arrival_time[detected])
            hit_charges.append(charge[detected])

    if hit_telescopes:
        telescope_index = torch.cat(hit_telescopes)
        pixel_index = torch.cat(hit_pixels)
        arrival_time = torch.cat(hit_times)
        charge = torch.cat(hit_charges)
        # All telescopes in an array share one event clock. Using a separate
        # first-photon origin for each telescope would erase array timing and
        # make a telescope-coincidence window meaningless.
        trace_start = torch.min(arrival_time) - bin_width_ns
        time_bin = torch.floor(
            (arrival_time - trace_start) / bin_width_ns
        ).to(torch.int64)
        in_window = (time_bin >= 0) & (time_bin < n_time_bins)
        flat_index = (
            (telescope_index[in_window] * n_pixels + pixel_index[in_window])
            * n_time_bins
            + time_bin[in_window]
        )
        signal = torch.bincount(
            flat_index,
            weights=charge[in_window],
            minlength=n_telescopes * n_pixels * n_time_bins,
        ).reshape(n_telescopes, n_pixels, n_time_bins)
        traces += signal

    if nsb_rate > 0:
        traces += torch.poisson(
            torch.full_like(traces, nsb_rate / n_time_bins), generator=generator
        )
    if pedestal_std > 0:
        traces += torch.randn(
            traces.shape, dtype=traces.dtype, device=device, generator=generator
        ) * (pedestal_std / np.sqrt(n_time_bins))

    saturated = torch.any(traces > saturation_limit, dim=2)
    gain_flags = saturated.to(torch.float32)
    traces = torch.where(
        saturated[:, :, None], traces / low_gain_factor, traces
    )
    return traces.to(torch.float32), gain_flags


def _ray_pair_chunk_limit(device):
    if device.type == 'cuda':
        free_bytes, _ = torch.cuda.mem_get_info(device)
        return max(1, min(5_000_000, int(free_bytes * 0.25) // 64))
    return 1_000_000

import numpy as np
import pytest
import torch

from sim import backend
from sim.atmosphere import (
    atmospheric_transmission_np,
    optical_path_delay_m,
    refractive_index_excess_np,
)
from sim.camera import Camera


def test_cherenkov_generation_runs_on_cpu(monkeypatch):
    monkeypatch.setattr(backend, 'get_device', lambda: torch.device('cpu'))

    photons = backend.compute_cherenkov_pool_gpu(
        np.array([0.0]), np.array([0.0]), np.array([100.0]),
        np.array([0.0]), np.array([0.0]), np.array([0.0]),
        np.array([0.0]), np.array([0.0]), np.array([-1.0]),
        np.array([1.0]), photon_yield_factor=0.01,
    )

    assert set(photons) == {
        'x_emit', 'y_emit', 'z_emit', 'x_ground', 'y_ground', 'weight'
    }
    lengths = {len(values) for values in photons.values()}
    assert len(lengths) == 1
    assert lengths.pop() > 0
    assert all(np.isfinite(values).all() for values in photons.values())
    assert np.all(photons['weight'] > 0)
    assert np.sum(photons['weight']) > 0


def test_weighted_packets_preserve_event_identity(monkeypatch):
    monkeypatch.setattr(backend, 'get_device', lambda requested='auto': torch.device('cpu'))

    packets = backend.compute_cherenkov_pool_gpu(
        np.array([0.0, 10.0]), np.array([0.0, 0.0]), np.array([100.0, 100.0]),
        np.array([0.0, 10.0]), np.array([0.0, 0.0]), np.array([0.0, 0.0]),
        np.array([0.0, 0.0]), np.array([0.0, 0.0]), np.array([-1.0, -1.0]),
        np.array([1.0, 1.0]), photon_yield_factor=0.1,
        seg_event_id=np.array([3, 7]),
        target_photons_per_packet=32,
        max_packets_per_segment=64,
        device='cpu',
        generator=torch.Generator(device='cpu').manual_seed(8),
    )

    assert set(packets['event_id']) == {3, 7}
    for event_id in (3, 7):
        mask = packets['event_id'] == event_id
        assert np.any(mask)
        assert np.sum(packets['weight'][mask]) > 0


def test_packet_budget_preserves_total_light(monkeypatch):
    monkeypatch.setattr(
        backend, 'get_device', lambda requested='auto': torch.device('cpu')
    )
    arguments = (
        np.array([0.0]), np.array([0.0]), np.array([1000.0]),
        np.array([0.0]), np.array([0.0]), np.array([0.0]),
        np.array([0.0]), np.array([0.0]), np.array([-1.0]),
        np.array([1.0]), 1.0,
    )
    exact = backend.compute_cherenkov_pool_gpu(
        *arguments,
        target_photons_per_packet=1.0,
        max_packets_per_segment=50_000,
        device='cpu',
        generator=torch.Generator(device='cpu').manual_seed(14),
    )
    thinned = backend.compute_cherenkov_pool_gpu(
        *arguments,
        target_photons_per_packet=500.0,
        max_packets_per_segment=64,
        device='cpu',
        generator=torch.Generator(device='cpu').manual_seed(14),
    )

    assert len(thinned['weight']) < len(exact['weight'])
    assert np.sum(thinned['weight']) == pytest.approx(
        np.sum(exact['weight']), rel=0.02
    )


def test_device_selection_is_explicit():
    assert backend.get_device('cpu') == torch.device('cpu')
    with pytest.raises(ValueError):
        backend.get_device('tpu')


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA is unavailable')
def test_cpu_cuda_detector_geometry_parity():
    camera = Camera(n_rings=2, pixel_size=0.1)
    packets = {
        'x_emit': np.array([0.0, 0.0], dtype=np.float32),
        'y_emit': np.array([0.0, 0.0], dtype=np.float32),
        'z_emit': np.array([100.0, 100.0], dtype=np.float32),
        'x_ground': np.array([0.0, 0.1], dtype=np.float32),
        'y_ground': np.array([0.0, 0.0], dtype=np.float32),
        'weight': np.array([10.0, 20.0], dtype=np.float32),
        'shower_start_altitude': np.array([100.0, 100.0], dtype=np.float32),
    }
    kwargs = dict(
        n_time_bins=16,
        nsb_rate=0.0,
        pedestal_std=0.0,
        saturation_limit=10_000.0,
    )
    positional = (
        packets, camera.pixel_x, camera.pixel_y, camera.pixel_size,
        0.0, 0.0, 0.0, 2.0, 1.0, 1.0,
    )
    cpu_trace, cpu_gain = backend.ray_trace_gpu(
        *positional,
        device='cpu',
        generator=torch.Generator(device='cpu').manual_seed(5),
        **kwargs,
    )
    cuda_trace, cuda_gain = backend.ray_trace_gpu(
        *positional,
        device='cuda',
        generator=torch.Generator(device='cuda').manual_seed(5),
        **kwargs,
    )

    np.testing.assert_allclose(cuda_trace, cpu_trace, rtol=1e-5, atol=1e-5)
    np.testing.assert_array_equal(cuda_gain, cpu_gain)


def test_camera_axial_lookup_uses_actual_geometry():
    camera = Camera(n_rings=12, pixel_size=0.1)
    axial_q, axial_r, n_rings = backend._camera_axial_coordinates(
        camera.pixel_x, camera.pixel_y, camera.pixel_size
    )

    assert camera.n_pixels == 469
    assert n_rings == 12
    assert len(set(zip(axial_q, axial_r))) == camera.n_pixels

    center_indices = np.flatnonzero((axial_q == 0) & (axial_r == 0))
    assert center_indices.tolist() == [np.argmin(camera.pixel_x**2 + camera.pixel_y**2)]


def test_frank_tamm_yield_and_z_obs_transmission():
    args = (
        np.array([0.0]), np.array([0.0]), np.array([8050.0]),
        np.array([0.0]), np.array([0.0]), np.array([7950.0]),
        np.array([0.0]), np.array([0.0]), np.array([-1.0]),
        np.array([100.0]), 1.0,
    )
    sea_level = backend.compute_cherenkov_pool_gpu(
        *args,
        z_obs=0.0,
        target_photons_per_packet=2.0,
        max_packets_per_segment=10_000,
        device='cpu',
        generator=torch.Generator(device='cpu').manual_seed(21),
    )
    mountain = backend.compute_cherenkov_pool_gpu(
        *args,
        z_obs=2200.0,
        target_photons_per_packet=2.0,
        max_packets_per_segment=10_000,
        device='cpu',
        generator=torch.Generator(device='cpu').manual_seed(21),
    )

    sea_yield = float(np.sum(sea_level['weight']))
    mountain_yield = float(np.sum(mountain['weight']))
    assert mountain_yield > sea_yield > 0.0

    # Analytical Frank--Tamm 300--600 nm yield check at z = 8 km (interior of Layer 2), z_obs = 2.2 km
    n_mid = 1.0 + float(refractive_index_excess_np(8000.0))
    sin2_mid = 1.0 - 1.0 / (n_mid * n_mid)
    cos_theta_c = 1.0 / n_mid
    trans_mid = float(atmospheric_transmission_np(8000.0, 2200.0, cos_theta_c))
    expected_mountain = 100.0 * backend.FRANK_TAMM_300_600_NM * sin2_mid * trans_mid
    assert mountain_yield == pytest.approx(expected_mountain, rel=0.01)

    # NumPy fallback parity
    numpy_mountain = backend._cherenkov_packets_numpy(
        *args,
        10_000, None, 2.0,
        np.random.default_rng(21),
        2200.0, None,
    )
    assert float(np.sum(numpy_mountain['weight'])) == pytest.approx(
        mountain_yield, rel=0.01
    )

    # Segments entirely below z_obs produce zero packets; straddling segments clip to z_emit > z_obs
    below = backend.compute_cherenkov_pool_gpu(
        np.array([0.0]), np.array([0.0]), np.array([2000.0]),
        np.array([0.0]), np.array([0.0]), np.array([1000.0]),
        np.array([0.0]), np.array([0.0]), np.array([-1.0]),
        np.array([100.0]), 1.0,
        z_obs=2200.0,
        device='cpu',
    )
    assert len(below['weight']) == 0

    straddling = backend.compute_cherenkov_pool_gpu(
        np.array([0.0]), np.array([0.0]), np.array([2500.0]),
        np.array([0.0]), np.array([0.0]), np.array([1500.0]),
        np.array([0.0]), np.array([0.0]), np.array([-1.0]),
        np.array([100.0]), 1.0,
        z_obs=2200.0,
        device='cpu',
        generator=torch.Generator(device='cpu').manual_seed(22),
    )
    assert len(straddling['weight']) > 0
    assert np.all(straddling['z_emit'] > 2200.0)


def test_muon_mass_threshold_and_validation():
    muon_mass_gev = 0.105658
    geom = (
        np.array([0.0]), np.array([0.0]), np.array([1000.0]),
        np.array([0.0]), np.array([0.0]), np.array([0.0]),
        np.array([0.0]), np.array([0.0]), np.array([-1.0]),
    )

    # 2 GeV muon is below the ~4.37 GeV sea-level Cherenkov threshold
    muon_2gev = backend.compute_cherenkov_pool_gpu(
        *geom, np.array([2.0]), 1.0,
        z_obs=0.0, seg_mass_gev=muon_mass_gev, device='cpu',
    )
    assert len(muon_2gev['weight']) == 0

    # 10 GeV muon is well above threshold and emits Cherenkov photons
    muon_10gev = backend.compute_cherenkov_pool_gpu(
        *geom, np.array([10.0]), 1.0,
        z_obs=0.0, seg_mass_gev=muon_mass_gev, device='cpu',
        generator=torch.Generator(device='cpu').manual_seed(31),
    )
    assert len(muon_10gev['weight']) > 0
    assert float(np.sum(muon_10gev['weight'])) > 0.0

    # Per-segment mass array mixes electrons and muons cleanly
    mixed = backend.compute_cherenkov_pool_gpu(
        np.zeros(3), np.zeros(3), np.full(3, 1000.0),
        np.zeros(3), np.zeros(3), np.zeros(3),
        np.zeros(3), np.zeros(3), np.full(3, -1.0),
        np.array([2.0, 2.0, 10.0]), 0.1,
        seg_event_id=np.array([1, 2, 3]),
        seg_mass_gev=np.array([0.000511, muon_mass_gev, muon_mass_gev]),
        z_obs=0.0, device='cpu',
        generator=torch.Generator(device='cpu').manual_seed(32),
    )
    assert set(mixed['event_id']) == {1, 3}

    with pytest.raises(ValueError):
        backend.compute_cherenkov_pool_gpu(*geom, np.array([10.0]), 1.0, z_obs=-10.0, device='cpu')
    with pytest.raises(ValueError):
        backend.compute_cherenkov_pool_gpu(*geom, np.array([10.0]), 1.0, z_obs=np.nan, device='cpu')
    with pytest.raises(ValueError):
        backend.compute_cherenkov_pool_gpu(*geom, np.array([10.0]), 1.0, seg_mass_gev=0.0, device='cpu')
    with pytest.raises(ValueError):
        backend.compute_cherenkov_pool_gpu(*geom, np.array([10.0]), 1.0, seg_mass_gev=-0.1, device='cpu')
    with pytest.raises(ValueError):
        backend.compute_cherenkov_pool_gpu(
            *geom, np.array([10.0]), 1.0,
            seg_mass_gev=np.array([0.000511, 0.105658]), device='cpu',
        )


def test_threshold_crossing_segment_is_not_dropped():
    # A 10 GeV muon crosses Cherenkov threshold around ~13.7 km.
    # For a segment from 22 km down to 10 km, the midpoint (16 km) is below
    # threshold (sin2_mid == 0), while the lower half emits Cherenkov light.
    packets = backend.compute_cherenkov_pool_gpu(
        np.array([0.0]), np.array([0.0]), np.array([22000.0]),
        np.array([0.0]), np.array([0.0]), np.array([10000.0]),
        np.array([0.0]), np.array([0.0]), np.array([-1.0]),
        np.array([10.0]), 0.1,
        seg_mass_gev=0.105658,
        z_obs=0.0,
        device='cpu',
        generator=torch.Generator(device='cpu').manual_seed(41),
    )
    assert len(packets['weight']) > 0
    assert float(np.sum(packets['weight'])) > 0.0
    assert np.all(packets['z_emit'] < 16000.0)


def test_tilted_mirror_aperture_preserves_collecting_area_at_30_deg():
    camera = Camera(n_rings=2, pixel_size=0.15)
    zenith_deg = 30.0
    z_emit = 1000.0
    dx_zen = z_emit * np.tan(np.radians(zenith_deg))

    def _beam(x_ground, y_ground):
        return {
            'x_emit': np.array([x_ground + dx_zen], dtype=np.float32),
            'y_emit': np.array([y_ground], dtype=np.float32),
            'z_emit': np.array([z_emit], dtype=np.float32),
            'x_ground': np.array([x_ground], dtype=np.float32),
            'y_ground': np.array([y_ground], dtype=np.float32),
            'weight': np.array([25.0], dtype=np.float32),
            'shower_start_altitude': np.array([z_emit], dtype=np.float32),
        }

    common_kwargs = dict(
        pointing_zenith=zenith_deg,
        pointing_azimuth=0.0,
        n_time_bins=16,
        nsb_rate=0.0,
        pedestal_std=0.0,
        saturation_limit=10_000.0,
        device='cpu',
    )

    # x = 6.5 m exceeds the 6.0 m horizontal circle but lies inside the 6.0 / cos(30°) = 6.93 m tilted ellipse
    trace_hit, _ = backend.ray_trace_gpu(
        _beam(6.5, 0.0), camera.pixel_x, camera.pixel_y, camera.pixel_size,
        0.0, 0.0, 0.0, 6.0, 1.0, 1.0,
        generator=torch.Generator(device='cpu').manual_seed(51),
        **common_kwargs,
    )
    assert float(np.sum(trace_hit)) == pytest.approx(25.0, abs=0.01)
    center = int(np.argmin(camera.pixel_x**2 + camera.pixel_y**2))
    assert float(np.sum(trace_hit[center])) == pytest.approx(25.0, abs=0.01)

    # x = 7.1 m exceeds 6.0 / cos(30°) = 6.93 m -> misses the dish
    trace_miss_x, _ = backend.ray_trace_gpu(
        _beam(7.1, 0.0), camera.pixel_x, camera.pixel_y, camera.pixel_size,
        0.0, 0.0, 0.0, 6.0, 1.0, 1.0,
        generator=torch.Generator(device='cpu').manual_seed(52),
        **common_kwargs,
    )
    assert float(np.sum(trace_miss_x)) == pytest.approx(0.0)

    # y = 6.5 m is perpendicular to the tilt axis (semi-minor axis remains 6.0 m) -> misses
    trace_miss_y, _ = backend.ray_trace_gpu(
        _beam(0.0, 6.5), camera.pixel_x, camera.pixel_y, camera.pixel_size,
        0.0, 0.0, 0.0, 6.0, 1.0, 1.0,
        generator=torch.Generator(device='cpu').manual_seed(53),
        **common_kwargs,
    )
    assert float(np.sum(trace_miss_y)) == pytest.approx(0.0)


def test_charge_weighted_fadc_anchor_ignores_stray_early_photon():
    camera = Camera(n_rings=2, pixel_size=0.15)
    packets = {
        'x_emit': np.array([0.0, 0.0], dtype=np.float32),
        'y_emit': np.array([0.0, 0.0], dtype=np.float32),
        'z_emit': np.array([100.0, 100.0], dtype=np.float32),
        'x_ground': np.array([0.0, 0.0], dtype=np.float32),
        'y_ground': np.array([0.0, 0.0], dtype=np.float32),
        'weight': np.array([1.0, 100.0], dtype=np.float32),
        'emission_time_ns': np.array([0.0, 40.0], dtype=np.float32),
    }
    trace, gain = backend.ray_trace_gpu(
        packets, camera.pixel_x, camera.pixel_y, camera.pixel_size,
        0.0, 0.0, 0.0, 2.0, 1.0, 1.0,
        n_time_bins=16,
        bin_width_ns=2.0,
        nsb_rate=0.0,
        pedestal_std=0.0,
        saturation_limit=10_000.0,
        device='cpu',
        generator=torch.Generator(device='cpu').manual_seed(61),
    )
    center = int(np.argmin(camera.pixel_x**2 + camera.pixel_y**2))
    assert float(np.sum(trace)) == pytest.approx(100.0, abs=0.05)
    assert int(np.argmax(trace[center])) == 2
    assert np.count_nonzero(gain) == 0


def test_integrated_refractive_delay_and_optics_validation():
    camera = Camera(n_rings=2, pixel_size=0.15)
    c_m_per_ns = 0.299792458
    # Two pulses with identical vacuum arrival times from z = 1 km and z = 20 km.
    # Their arrival-time separation equals the integrated atmospheric optical path
    # delay difference (~6.86 ns), rather than constant n=1.0003 (~19.0 ns).
    packets = {
        'x_emit': np.array([0.0, 0.0], dtype=np.float32),
        'y_emit': np.array([0.0, 0.0], dtype=np.float32),
        'z_emit': np.array([1000.0, 20000.0], dtype=np.float32),
        'x_ground': np.array([0.0, 0.0], dtype=np.float32),
        'y_ground': np.array([0.0, 0.0], dtype=np.float32),
        'weight': np.array([50.0, 50.0], dtype=np.float32),
        'emission_time_ns': np.array(
            [0.0, -(20000.0 - 1000.0) / c_m_per_ns], dtype=np.float32
        ),
    }
    trace, _ = backend.ray_trace_gpu(
        packets, camera.pixel_x, camera.pixel_y, camera.pixel_size,
        0.0, 0.0, 0.0, 2.0, 1.0, 1.0,
        n_time_bins=32,
        bin_width_ns=1.0,
        nsb_rate=0.0,
        pedestal_std=0.0,
        saturation_limit=10_000.0,
        device='cpu',
        generator=torch.Generator(device='cpu').manual_seed(71),
    )
    center = int(np.argmin(camera.pixel_x**2 + camera.pixel_y**2))
    waveform = trace[center]
    expected_dt_ns = float(
        (
            optical_path_delay_m(20000.0, 0.0, 1.0)
            - optical_path_delay_m(1000.0, 0.0, 1.0)
        ).item()
        / c_m_per_ns
    )
    # First pulse lands at bin 2; second pulse lands near bin 2 + expected_dt_ns (~9), not ~21
    second_peak_bin = 5 + int(np.argmax(waveform[5:]))
    assert abs((second_peak_bin - 2) - expected_dt_ns) <= 1.0

    # Negative optical parameters are rejected
    with pytest.raises(ValueError):
        backend.ray_trace_gpu(
            packets, camera.pixel_x, camera.pixel_y, camera.pixel_size,
            0.0, 0.0, 0.0, 2.0, 1.0, 1.0,
            optical_psf_deg=-0.01, device='cpu',
        )
    with pytest.raises(ValueError):
        backend.ray_trace_gpu(
            packets, camera.pixel_x, camera.pixel_y, camera.pixel_size,
            0.0, 0.0, 0.0, 2.0, 1.0, 1.0,
            optical_time_spread_ns=-0.5, device='cpu',
        )

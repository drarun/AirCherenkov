import numpy as np
import pytest
import torch

from sim import backend
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

    np.testing.assert_allclose(cuda_trace, cpu_trace, rtol=0, atol=0)
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

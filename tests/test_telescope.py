import numpy as np
import pytest
import torch
from sim.telescope import Telescope, VeritasTelescope, TelescopeArray
from sim.camera import Camera
from sim.fadc import FADCConfig


def _vertical_packets(weight=25.0):
    return {
        'x_emit': np.array([0.0], dtype=np.float32),
        'y_emit': np.array([0.0], dtype=np.float32),
        'z_emit': np.array([100.0], dtype=np.float32),
        'x_ground': np.array([0.0], dtype=np.float32),
        'y_ground': np.array([0.0], dtype=np.float32),
        'weight': np.array([weight], dtype=np.float32),
        'shower_start_altitude': np.array([100.0], dtype=np.float32),
    }

def test_telescope_initialization():
    tel = Telescope(mirror_radius=5.0)
    assert tel.mirror_radius == 5.0
    assert tel.camera.n_rings == 15

def test_ray_trace_empty():
    tel = Telescope()
    cherenkov_photons = {
        'x_ground': np.array([]), 'y_ground': np.array([])
    }
    trace, gain = tel.ray_trace(cherenkov_photons)
    assert trace.shape == (tel.camera.n_pixels, 16)
    assert gain.shape == (tel.camera.n_pixels,)

def test_veritas_telescope():
    tel = VeritasTelescope(x_tel=10.0, y_tel=20.0)
    assert tel.x_tel == 10.0
    assert tel.y_tel == 20.0
    assert tel.mirror_radius == 6.0
    assert tel.focal_length == 12.0
    assert tel.camera.n_rings == 12

def test_telescope_array():
    array = TelescopeArray.veritas_array()
    assert len(array.telescopes) == 4
    
    cherenkov_photons = {
        'x_ground': np.array([]), 'y_ground': np.array([])
    }
    traces_and_gains = array.ray_trace(cherenkov_photons)
    assert len(traces_and_gains) == 4
    for i, (trace, gain) in enumerate(traces_and_gains):
        assert trace.shape == (array.telescopes[i].camera.n_pixels, 16)
        assert gain.shape == (array.telescopes[i].camera.n_pixels,)


def test_fadc_configuration_controls_empty_detector_response():
    config = FADCConfig(
        n_time_bins=8,
        bin_width_ns=4.0,
        nsb_rate=0.0,
        pedestal_std=0.0,
    )
    telescope = Telescope(fadc_config=config, device='cpu')
    trace, gain = telescope.ray_trace({'x_ground': np.array([])})

    assert trace.shape == (telescope.camera.n_pixels, 8)
    assert gain.shape == (telescope.camera.n_pixels,)
    assert np.count_nonzero(trace) == 0
    assert np.count_nonzero(gain) == 0


@pytest.mark.parametrize(
    'kwargs',
    [
        {'n_time_bins': 0},
        {'bin_width_ns': 0.0},
        {'nsb_rate': -1.0},
        {'pedestal_std': -1.0},
        {'low_gain_factor': 0.0},
    ],
)
def test_fadc_configuration_rejects_invalid_values(kwargs):
    with pytest.raises((TypeError, ValueError)):
        FADCConfig(**kwargs)


def test_weighted_packet_charge_is_preserved_on_cpu():
    config = FADCConfig(
        nsb_rate=0.0,
        pedestal_std=0.0,
        saturation_limit=10_000.0,
    )
    telescope = Telescope(
        mirror_radius=2.0,
        mirror_reflectivity=1.0,
        quantum_efficiency=1.0,
        fadc_config=config,
        shower_start_altitude=100.0,
        device='cpu',
    )
    generator = torch.Generator(device='cpu').manual_seed(9)
    trace, gain = telescope.ray_trace(_vertical_packets(), generator=generator)

    assert np.sum(trace) == pytest.approx(25.0)
    assert np.count_nonzero(gain) == 0
    center = np.argmin(telescope.camera.pixel_x**2 + telescope.camera.pixel_y**2)
    assert np.sum(trace[center]) == pytest.approx(25.0)


def test_telescope_array_batches_geometry_without_changing_order():
    config = FADCConfig(
        nsb_rate=0.0,
        pedestal_std=0.0,
        saturation_limit=10_000.0,
    )
    array = TelescopeArray([
        Telescope(
            x_tel=0.0, mirror_radius=2.0,
            mirror_reflectivity=1.0, quantum_efficiency=1.0,
            fadc_config=config, device='cpu', shower_start_altitude=100.0,
        ),
        Telescope(
            x_tel=100.0, mirror_radius=2.0,
            mirror_reflectivity=1.0, quantum_efficiency=1.0,
            fadc_config=config, device='cpu', shower_start_altitude=100.0,
        ),
    ])
    generator = torch.Generator(device='cpu').manual_seed(4)
    results = array.ray_trace(_vertical_packets(), generator=generator)

    assert len(results) == 2
    assert np.sum(results[0][0]) == pytest.approx(25.0)
    assert np.sum(results[1][0]) == pytest.approx(0.0)


def test_telescope_array_preserves_relative_arrival_time():
    config = FADCConfig(
        n_time_bins=8,
        bin_width_ns=1.0,
        nsb_rate=0.0,
        pedestal_std=0.0,
        saturation_limit=10_000.0,
    )
    common = dict(
        mirror_radius=2.0,
        mirror_reflectivity=1.0,
        quantum_efficiency=1.0,
        fadc_config=config,
        device='cpu',
        shower_start_altitude=100.0,
    )
    array = TelescopeArray([
        Telescope(z_tel=0.0, **common),
        Telescope(z_tel=1.0, **common),
    ])
    results = array.ray_trace(
        _vertical_packets(),
        generator=torch.Generator(device='cpu').manual_seed(12),
    )
    center = np.argmin(
        array.telescopes[0].camera.pixel_x**2
        + array.telescopes[0].camera.pixel_y**2
    )
    peak_bins = [int(np.argmax(trace[center])) for trace, _ in results]

    assert peak_bins[0] > peak_bins[1]

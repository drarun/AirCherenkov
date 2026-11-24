import numpy as np
import pytest
import torch
from sim import backend
from sim.telescope import (
    CtaLST,
    CtaMST,
    CtaSST,
    HessTelescope,
    MagicTelescope,
    Telescope,
    TelescopeArray,
    VeritasTelescope,
)
from sim.camera import Camera
from sim.fadc import FADCConfig
from gui.array_models import DetectorArrayModel, get_preset_model


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

    assert np.sum(trace) == pytest.approx(25.0, abs=0.01)
    assert np.count_nonzero(gain) == 0
    center = np.argmin(telescope.camera.pixel_x**2 + telescope.camera.pixel_y**2)
    assert np.sum(trace[center]) == pytest.approx(25.0, abs=0.01)


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
    assert np.sum(results[0][0]) == pytest.approx(25.0, abs=0.01)
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

    # With per-telescope CFD window alignment, each telescope centers its readout
    # window on its own arrival time (2 bins before earliest photon).
    assert peak_bins[0] == peak_bins[1] == 2


def test_optical_psf_spreads_veritas_vs_pinhole_base_telescope():
    n_packets = 400
    packets = {
        'x_emit': np.zeros(n_packets, dtype=np.float32),
        'y_emit': np.zeros(n_packets, dtype=np.float32),
        'z_emit': np.full(n_packets, 5000.0, dtype=np.float32),
        'x_ground': np.zeros(n_packets, dtype=np.float32),
        'y_ground': np.zeros(n_packets, dtype=np.float32),
        'weight': np.ones(n_packets, dtype=np.float32),
        'shower_start_altitude': np.full(n_packets, 5000.0, dtype=np.float32),
    }
    config = FADCConfig(
        nsb_rate=0.0,
        pedestal_std=0.0,
        saturation_limit=10_000.0,
    )
    veritas = VeritasTelescope(
        z_tel=1268.0,
        mirror_reflectivity=1.0,
        quantum_efficiency=1.0,
        fadc_config=config,
        device='cpu',
    )
    pinhole = Telescope(
        z_tel=1268.0,
        mirror_radius=6.0,
        focal_length=12.0,
        n_rings=12,
        pixel_size=0.15,
        mirror_reflectivity=1.0,
        quantum_efficiency=1.0,
        fadc_config=config,
        device='cpu',
    )
    assert veritas.optical_psf_deg == pytest.approx(0.04)
    assert veritas.optical_time_spread_ns == pytest.approx(1.2)
    assert pinhole.optical_psf_deg == 0.0
    assert pinhole.optical_time_spread_ns == 0.0

    veritas_trace, _ = veritas.ray_trace(
        packets, generator=torch.Generator(device='cpu').manual_seed(101)
    )
    pinhole_trace, _ = pinhole.ray_trace(
        packets, generator=torch.Generator(device='cpu').manual_seed(101)
    )

    veritas_pix = np.sum(veritas_trace, axis=1)
    pinhole_pix = np.sum(pinhole_trace, axis=1)
    center = int(np.argmin(pinhole.camera.pixel_x**2 + pinhole.camera.pixel_y**2))

    # Base Telescope acts as a deterministic pinhole (only center pixel illuminated)
    assert int(np.count_nonzero(pinhole_pix > 0)) == 1
    assert float(pinhole_pix[center]) == pytest.approx(float(n_packets), abs=0.05)

    # VeritasTelescope optical PSF spreads the point beam into neighboring pixels
    assert int(np.count_nonzero(veritas_pix > 0)) > 1
    assert int(np.argmax(veritas_pix)) == center
    assert float(veritas_pix[center]) < float(n_packets)
    assert float(np.sum(veritas_pix)) == pytest.approx(float(n_packets), rel=0.01)

    # Davies--Cotton time spread (1.2 ns) broadens the summed waveform relative to pinhole
    time_bins = np.arange(config.n_time_bins, dtype=np.float64) * config.bin_width_ns
    veritas_wave = np.sum(veritas_trace, axis=0)
    pinhole_wave = np.sum(pinhole_trace, axis=0)
    veritas_mean = float(np.sum(time_bins * veritas_wave) / np.sum(veritas_wave))
    pinhole_mean = float(np.sum(time_bins * pinhole_wave) / np.sum(pinhole_wave))
    veritas_std = float(np.sqrt(np.sum(((time_bins - veritas_mean) ** 2) * veritas_wave) / np.sum(veritas_wave)))
    pinhole_std = float(np.sqrt(np.sum(((time_bins - pinhole_mean) ** 2) * pinhole_wave) / np.sum(pinhole_wave)))
    assert veritas_std > pinhole_std


def test_telescope_ray_trace_accepts_pointing_parameters():
    config = FADCConfig(
        nsb_rate=0.0,
        pedestal_std=0.0,
        saturation_limit=10_000.0,
    )
    tel = Telescope(
        mirror_radius=6.0,
        mirror_reflectivity=1.0,
        quantum_efficiency=1.0,
        fadc_config=config,
        device='cpu',
    )
    # Beam arriving along telescope pointing (zenith=30 deg, azimuth=180 deg):
    # sky normal is (-sin(30), 0, cos(30)), so emission at z=1000 m has x_emit = -1000*tan(30).
    z_emit = 1000.0
    x_emit = -z_emit * np.tan(np.radians(30.0))
    packets = {
        'x_emit': np.array([x_emit], dtype=np.float32),
        'y_emit': np.array([0.0], dtype=np.float32),
        'z_emit': np.array([z_emit], dtype=np.float32),
        'x_ground': np.array([0.0], dtype=np.float32),
        'y_ground': np.array([0.0], dtype=np.float32),
        'weight': np.array([20.0], dtype=np.float32),
        'shower_start_altitude': np.array([z_emit], dtype=np.float32),
    }

    trace, gain = tel.ray_trace(
        packets,
        pointing_zenith=30.0,
        pointing_azimuth=180.0,
        generator=torch.Generator(device='cpu').manual_seed(111),
    )
    center = int(np.argmin(tel.camera.pixel_x**2 + tel.camera.pixel_y**2))
    assert trace.shape == (tel.camera.n_pixels, config.n_time_bins)
    assert gain.shape == (tel.camera.n_pixels,)
    assert float(np.sum(trace)) == pytest.approx(20.0, abs=0.01)
    assert float(np.sum(trace[center])) == pytest.approx(20.0, abs=0.01)


def test_preset_optical_defaults_overrides_and_serialization():
    expected = [
        (VeritasTelescope(), 0.04, 1.2),
        (HessTelescope(), 0.035, 1.2),
        (MagicTelescope(), 0.035, 0.0),
        (CtaLST(), 0.025, 0.0),
        (CtaMST(), 0.030, 1.2),
        (CtaSST(), 0.050, 0.5),
    ]
    for tel, exp_psf, exp_spread in expected:
        assert tel.optical_psf_deg == pytest.approx(exp_psf)
        assert tel.optical_time_spread_ns == pytest.approx(exp_spread)

    # Preset defaults are overridable via kwargs
    override = VeritasTelescope(optical_psf_deg=0.0, optical_time_spread_ns=0.0)
    assert override.optical_psf_deg == 0.0
    assert override.optical_time_spread_ns == 0.0

    with pytest.raises(ValueError):
        Telescope(optical_psf_deg=-0.01)
    with pytest.raises(ValueError):
        Telescope(optical_time_spread_ns=-0.5)
    with pytest.raises(ValueError):
        Telescope(optical_psf_deg=float('nan'))

    # TelescopeArray round-trip preserves optical params
    veritas_arr = TelescopeArray.veritas_array()
    arr_dict = veritas_arr.to_dict("VERITAS")
    rebuilt_arr = TelescopeArray.from_dict(arr_dict)
    assert rebuilt_arr.telescopes[0].optical_psf_deg == pytest.approx(0.04)
    assert rebuilt_arr.telescopes[0].optical_time_spread_ns == pytest.approx(1.2)

    # Missing keys in legacy JSON default to 0.0
    legacy_dict = {
        "preset_name": "Legacy",
        "observatory_altitude_m": 1268.0,
        "telescopes": [
            {
                "x_tel": 0.0,
                "y_tel": 0.0,
                "z_tel": 1268.0,
                "mirror_radius": 6.0,
                "focal_length": 12.0,
                "mirror_reflectivity": 0.82,
                "quantum_efficiency": 0.20,
                "n_rings": 12,
                "pixel_size": 0.15,
            }
        ],
    }
    legacy_arr = TelescopeArray.from_dict(legacy_dict)
    assert legacy_arr.telescopes[0].optical_psf_deg == 0.0
    assert legacy_arr.telescopes[0].optical_time_spread_ns == 0.0

    legacy_model = DetectorArrayModel.from_dict(legacy_dict)
    assert legacy_model.telescopes_spec[0]["optical_psf_deg"] == 0.0
    assert legacy_model.telescopes_spec[0]["optical_time_spread_ns"] == 0.0
    assert legacy_model.build_telescope_array().telescopes[0].optical_psf_deg == 0.0

    # DetectorArrayModel preset round-trip preserves optical params
    preset_model = get_preset_model("VERITAS")
    rt_model = DetectorArrayModel.from_dict(preset_model.to_dict())
    rt_tel = rt_model.build_telescope_array().telescopes[0]
    assert rt_tel.optical_psf_deg == pytest.approx(0.04)
    assert rt_tel.optical_time_spread_ns == pytest.approx(1.2)


def test_telescope_array_fallback_forwards_pointing(monkeypatch):
    real_ray_trace_array = backend.ray_trace_array
    captured = {}

    def _legacy_ray_trace_gpu(
        cherenkov_photons, pixel_x, pixel_y, pixel_size,
        x_tel, y_tel, z_tel, mirror_radius,
        mirror_reflectivity, quantum_efficiency, **kwargs
    ):
        captured["pointing_zenith"] = kwargs.get("pointing_zenith")
        captured["pointing_azimuth"] = kwargs.get("pointing_azimuth")
        captured["optical_psf_deg"] = kwargs.get("optical_psf_deg")
        captured["optical_time_spread_ns"] = kwargs.get("optical_time_spread_ns")
        traces, gains = real_ray_trace_array(
            cherenkov_photons, pixel_x, pixel_y, pixel_size,
            [x_tel], [y_tel], [z_tel], [mirror_radius],
            [mirror_reflectivity], [quantum_efficiency], **kwargs
        )
        return traces[0], gains[0]

    monkeypatch.setattr(backend, 'ray_trace_array', None)
    monkeypatch.setattr(backend, 'ray_trace_gpu', _legacy_ray_trace_gpu)
    config = FADCConfig(
        nsb_rate=0.0,
        pedestal_std=0.0,
        saturation_limit=10_000.0,
    )
    array = TelescopeArray([
        Telescope(
            mirror_radius=6.0,
            mirror_reflectivity=1.0,
            quantum_efficiency=1.0,
            fadc_config=config,
            device='cpu',
        ),
    ])
    z_emit = 1000.0
    x_emit = -z_emit * np.tan(np.radians(30.0))
    packets = {
        'x_emit': np.array([x_emit], dtype=np.float32),
        'y_emit': np.array([0.0], dtype=np.float32),
        'z_emit': np.array([z_emit], dtype=np.float32),
        'x_ground': np.array([0.0], dtype=np.float32),
        'y_ground': np.array([0.0], dtype=np.float32),
        'weight': np.array([15.0], dtype=np.float32),
        'shower_start_altitude': np.array([z_emit], dtype=np.float32),
    }
    results = array.ray_trace(
        packets,
        pointing_zenith=30.0,
        pointing_azimuth=180.0,
        generator=torch.Generator(device='cpu').manual_seed(121),
    )
    center = int(np.argmin(
        array.telescopes[0].camera.pixel_x**2 + array.telescopes[0].camera.pixel_y**2
    ))
    assert captured["pointing_zenith"] == pytest.approx(30.0)
    assert captured["pointing_azimuth"] == pytest.approx(180.0)
    assert captured["optical_psf_deg"] == 0.0
    assert captured["optical_time_spread_ns"] == 0.0
    assert float(np.sum(results[0][0][center])) == pytest.approx(15.0, abs=0.01)

from pathlib import Path
from types import SimpleNamespace

import generate_visualizations
import numpy as np
import pytest
import regenerate_all
import run_full_sim
import run_gamma_camera_pipeline


def test_visualization_cli_delegates_without_running_simulation(monkeypatch, tmp_path):
    captured = {}

    def fake_generate_visualizations(**kwargs):
        captured.update(kwargs)
        return [tmp_path / 'gamma_shower.html']

    monkeypatch.setattr(
        generate_visualizations,
        'generate_visualizations',
        fake_generate_visualizations,
    )
    result = generate_visualizations.main([
        '--energy-gev', '750',
        '--start-altitude-m', '18000',
        '--max-generations', '9',
        '--max-tracks', '321',
        '--output-dir', str(tmp_path),
        '--seed', '7',
        '--device', 'cpu',
        '--plotlyjs', 'directory',
        '--skip-camera',
    ])

    assert result == [tmp_path / 'gamma_shower.html']
    assert captured == {
        'energy_gev': 750.0,
        'start_altitude_m': 18000.0,
        'max_generations': 9,
        'max_tracks': 321,
        'output_dir': Path(tmp_path),
        'seed': 7,
        'device': 'cpu',
        'plotlyjs': 'directory',
        'include_camera': False,
    }


def test_regenerate_all_remains_a_compatible_alias():
    assert regenerate_all.generate_visualizations is generate_visualizations.generate_visualizations
    assert callable(regenerate_all.main)
    assert regenerate_all.plot_shower_3d is regenerate_all.plot_shower
    assert callable(regenerate_all.plot_cherenkov_pool)


@pytest.mark.parametrize(
    'kwargs',
    [
        {'energy_gev': 0},
        {'start_altitude_m': -1},
        {'max_generations': -1},
        {'max_tracks': 0},
        {'device': 'tpu'},
        {'plotlyjs': 'embedded'},
    ],
)
def test_visualization_runner_rejects_invalid_configuration(kwargs):
    with pytest.raises(ValueError):
        generate_visualizations.generate_visualizations(**kwargs)


def test_run_full_sim_executes(monkeypatch, tmp_path):
    import os
    class FakeSimulation:
        def __init__(self, **kwargs):
            self.cherenkov_photons_by_event = {0: {
                'x_emit': np.array([0.0], dtype=np.float32),
                'y_emit': np.array([0.0], dtype=np.float32),
                'z_emit': np.array([10000.0], dtype=np.float32),
                'x_ground': np.array([10.0], dtype=np.float32),
                'y_ground': np.array([20.0], dtype=np.float32),
                'weight': np.array([1.0], dtype=np.float32),
                'shower_start_altitude': np.array([20000.0], dtype=np.float32)
            }}
            self.photon_yield_factor = 1.0
        def run(self, **kwargs):
            pass

    monkeypatch.setattr(run_full_sim, 'ShowerSimulation', FakeSimulation)
    cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        run_full_sim.main()
        assert os.path.exists("camera_cleaning_comparison.png")
    finally:
        os.chdir(cwd)


def test_directory_plotly_mode_shares_offline_bundle(tmp_path):
    import plotly.graph_objects as go

    generate_visualizations._write_html(
        go.Figure(go.Scatter(x=[0, 1], y=[0, 1])),
        tmp_path / 'one.html',
        plotlyjs='directory',
    )
    generate_visualizations._write_html(
        go.Figure(go.Scatter(x=[0, 1], y=[1, 0])),
        tmp_path / 'two.html',
        plotlyjs='directory',
    )

    assert (tmp_path / 'plotly.min.js').is_file()
    assert (tmp_path / 'one.html').stat().st_size < 100_000
    assert (tmp_path / 'two.html').stat().st_size < 100_000


def test_camera_pipeline_has_a_nonexecuting_cli_parser():
    args = run_gamma_camera_pipeline.build_parser().parse_args([
        '--energy-gev', '250', '--device', 'cpu', '--seed', '9'
    ])
    assert args.energy_gev == 250.0
    assert args.device == 'cpu'
    assert args.seed == 9


class _FakeCamera:
    n_pixels = 2

    def plot_image(self, image, *, ax, **_kwargs):
        ax.plot(np.arange(len(image)), image)


class _FakeTelescope:
    captured_cleaning_images = []

    def __init__(self, **_kwargs):
        self.camera = _FakeCamera()
        self.fadc_config = SimpleNamespace(low_gain_factor=10.0)

    def ray_trace(self, _photons, **_kwargs):
        return (
            np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
            np.array([0.0, 1.0], dtype=np.float32),
        )


def _recording_cleaner(_camera, image, **_kwargs):
    _FakeTelescope.captured_cleaning_images.append(np.asarray(image).copy())
    return np.ones(len(image), dtype=bool)


def test_camera_pipeline_restores_low_gain_before_cleaning(monkeypatch, tmp_path):
    class FakeSimulation:
        def __init__(self, **_kwargs):
            self.cherenkov_photons_by_event = [{"x_ground": np.array([0.0])}]
            self.generator = None

        def run(self, **_kwargs):
            return None

    _FakeTelescope.captured_cleaning_images.clear()
    monkeypatch.setattr(run_gamma_camera_pipeline, 'ShowerSimulation', FakeSimulation)
    monkeypatch.setattr(run_gamma_camera_pipeline, 'Telescope', _FakeTelescope)
    monkeypatch.setattr(run_gamma_camera_pipeline, 'tail_cut_clean', _recording_cleaner)
    monkeypatch.setattr(run_gamma_camera_pipeline, 'double_pass_clean', _recording_cleaner)
    monkeypatch.setattr(run_gamma_camera_pipeline, 'compute_hillas', lambda *_args: None)
    monkeypatch.setattr(run_gamma_camera_pipeline, 'print_hillas', lambda *_args: None)

    output = run_gamma_camera_pipeline.main([
        '--energy-gev', '1',
        '--max-generations', '0',
        '--device', 'cpu',
        '--output', str(tmp_path / 'camera.png'),
    ])

    assert output.is_file()
    assert len(_FakeTelescope.captured_cleaning_images) == 2
    for image in _FakeTelescope.captured_cleaning_images:
        np.testing.assert_allclose(image, [3.0, 70.0])


def test_visualization_camera_restores_low_gain_before_cleaning(
    monkeypatch, tmp_path
):
    import recon.cleaning
    import recon.hillas
    import sim.telescope

    _FakeTelescope.captured_cleaning_images.clear()
    monkeypatch.setattr(sim.telescope, 'Telescope', _FakeTelescope)
    monkeypatch.setattr(recon.cleaning, 'tail_cut_clean', _recording_cleaner)
    monkeypatch.setattr(recon.cleaning, 'double_pass_clean', _recording_cleaner)
    monkeypatch.setattr(recon.hillas, 'compute_hillas', lambda *_args: None)
    monkeypatch.setattr(recon.hillas, 'print_hillas', lambda *_args: None)

    output = tmp_path / 'camera.png'
    generate_visualizations._write_camera_analysis(
        {"x_ground": np.array([0.0])},
        '1 GeV',
        output,
        device='cpu',
    )

    assert output.is_file()
    assert len(_FakeTelescope.captured_cleaning_images) == 2
    for image in _FakeTelescope.captured_cleaning_images:
        np.testing.assert_allclose(image, [3.0, 70.0])

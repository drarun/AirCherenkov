"""Unit and integration tests for the AirCherenkov native Desktop UI."""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
import torch

h5py = pytest.importorskip("h5py")
pytest.importorskip("PySide6")
from PySide6.QtWidgets import QApplication

from analysis.io import HESSDataReader, MAGICRootReader, load_events_universal
from gui.array_models import (
    DetectorArrayModel,
    extract_stereo_features_any_array,
    get_preset_model,
    load_detector_model_json,
    save_detector_model_json,
)
from gui.main_window import AirCherenkovMainWindow
from gui.skymap import compute_skymap_and_significance, li_and_ma_significance, reconstruct_sky_offsets_for_events
from gui.telemetry import HardwareTelemetry, SystemAndPipelineStatusBar
from gui.workers import AnalysisIRFSpectralWorker, SimulationWorker, TrainingWorker
from sim.camera import Camera
from sim.telescope import MagicTelescope, TelescopeArray


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def _make_synthetic_stereo_events(
    n_events: int = 24,
    n_tels: int = 4,
    n_rings: int = 12,
    pixel_size: float = 0.15,
    seed: int = 7,
):
    """Create realistic synthetic IACT events with elliptical Hillas tracks across telescopes."""
    rng = np.random.default_rng(seed)
    cam = Camera(n_rings=n_rings, pixel_size=pixel_size)
    px = cam.pixel_x
    py = cam.pixel_y
    events = []

    for i in range(n_events):
        is_gamma = i % 2 == 0
        energy = float(10.0 ** rng.uniform(2.2, 3.8))
        amp_scale = 80.0 * ((energy / 300.0) ** 0.85)
        traces = rng.normal(loc=0.1, scale=0.15, size=(n_tels, cam.n_pixels, 16)).astype(np.float32)
        traces = np.maximum(traces, 0.0)
        gains = np.zeros((n_tels, cam.n_pixels), dtype=np.float32)

        # Target sky offset so SDP intersects cleanly
        sx = rng.uniform(-0.3, 0.3)
        sy = rng.uniform(-0.3, 0.3)
        for t in range(n_tels):
            # Place centroid radially displaced from (sx, sy) along different telescope angles
            ang = 0.4 + t * (np.pi / max(n_tels, 2)) + rng.uniform(-0.08, 0.08)
            r_cent = 0.45 + 0.08 * t
            cx = sx + r_cent * np.cos(ang)
            cy = sy + r_cent * np.sin(ang)

            dx = px - cx
            dy = py - cy
            u = dx * np.cos(ang) + dy * np.sin(ang)
            v = -dx * np.sin(ang) + dy * np.cos(ang)
            sig_l = 0.22 if is_gamma else 0.30
            sig_w = 0.09 if is_gamma else 0.18
            spatial = amp_scale * np.exp(-0.5 * ((u / sig_l) ** 2 + (v / sig_w) ** 2))
            # Put Cherenkov pulse in bins 5..9
            for b, w_b in zip((5, 6, 7, 8), (0.2, 0.45, 0.25, 0.1)):
                traces[t, :, b] += (spatial * w_b).astype(np.float32)

        events.append(
            {
                "fadc_traces": traces,
                "gain_flags": gains,
                "energy": energy,
                "label": 1 if is_gamma else 0,
                "impact_x": float(rng.uniform(-120.0, 120.0)),
                "impact_y": float(rng.uniform(-120.0, 120.0)),
                "zenith_deg": 30.0,
                "azimuth_deg": 0.0,
                "true_zenith_deg": 30.0 + float(sy),
                "true_azimuth_deg": float(sx),
                "wobble_dir": "N",
            }
        )
    return events


def test_array_presets_and_serialization(tmp_path):
    veritas = TelescopeArray.from_preset("VERITAS")
    hess = TelescopeArray.from_preset("HESS")
    magic = TelescopeArray.from_preset("MAGIC")

    assert len(veritas.telescopes) == 4
    assert len(hess.telescopes) == 4
    assert len(magic.telescopes) == 2
    assert isinstance(magic.telescopes[0], MagicTelescope)
    assert magic.telescopes[0].mirror_radius == pytest.approx(8.5)
    assert magic.telescopes[0].camera.n_pixels == 1027

    roundtrip = TelescopeArray.from_dict(magic.to_dict("MAGIC"))
    assert len(roundtrip.telescopes) == 2
    assert roundtrip.telescopes[1].x_tel == pytest.approx(42.5)
    assert roundtrip.telescopes[0].z_tel == pytest.approx(2200.0)

    det_model = get_preset_model("HESS")
    json_path = tmp_path / "hess_model.json"
    save_detector_model_json(det_model, str(json_path))
    loaded_model = load_detector_model_json(str(json_path))
    assert loaded_model.preset_name == "HESS"
    assert len(loaded_model.telescopes_spec) == 4
    assert loaded_model.observatory_altitude_m == pytest.approx(1800.0)


def test_extract_stereo_features_for_magic_and_veritas():
    events_magic = _make_synthetic_stereo_events(n_events=2, n_tels=2, n_rings=18, pixel_size=0.1)
    magic_model = get_preset_model("MAGIC")
    magic_arr = magic_model.build_telescope_array()
    feat = extract_stereo_features_any_array(
        events_magic[0]["fadc_traces"],
        magic_arr.telescopes,
        magic_model.primary_camera(),
        z_obs=magic_model.observatory_altitude_m,
        gain_flags=events_magic[0]["gain_flags"],
        include_extended=True,
    )
    assert feat.shape == (85,)
    assert np.all(np.isfinite(feat))


def test_universal_io_loaders(tmp_path):
    events = _make_synthetic_stereo_events(n_events=3, n_tels=4)

    # 1. PyTorch directory loader
    pt_dir = tmp_path / "pt_ds"
    pt_dir.mkdir()
    torch.save(events, pt_dir / "sim_batch_00000.pt")
    loaded_pt = load_events_universal(str(pt_dir))
    assert len(loaded_pt) == 3
    assert loaded_pt[0]["fadc_traces"].shape == (4, 469, 16)

    # 2. H.E.S.S. / CTA HDF5 loader
    h5_path = tmp_path / "hess_sample.h5"
    with h5py.File(h5_path, "w") as fh:
        fh.create_dataset("fadc_traces", data=np.stack([e["fadc_traces"] for e in events]))
        fh.create_dataset("gain_flags", data=np.stack([e["gain_flags"] for e in events]))
        fh.create_dataset("energy", data=np.array([e["energy"] for e in events]))
        fh.create_dataset("label", data=np.array([e["label"] for e in events]))

    reader = HESSDataReader(str(h5_path))
    assert len(list(reader.read_event())) == 3
    reader.close()

    loaded_h5 = load_events_universal(str(h5_path), array_preset="HESS")
    assert len(loaded_h5) == 3
    assert loaded_h5[1]["label"] == events[1]["label"]

    # 3. NPZ loader
    npz_path = tmp_path / "events.npz"
    np.savez(
        npz_path,
        fadc_traces=np.stack([e["fadc_traces"] for e in events]),
        energy=np.array([e["energy"] for e in events]),
        label=np.array([e["label"] for e in events]),
    )
    loaded_npz = load_events_universal(str(npz_path))
    assert len(loaded_npz) == 3

    # 4. MAGICRootReader handles missing file gracefully
    root_reader = MAGICRootReader(str(tmp_path / "nonexistent.root"))
    assert list(root_reader.read_event()) == []


def test_telemetry_and_status_bar(qapp):
    telem = HardwareTelemetry()
    stats = telem.poll()
    assert "gpu_util_pct" in stats
    assert "vram_pct" in stats
    assert "cpu_pct" in stats
    telem.close()

    bar = SystemAndPipelineStatusBar(poll_interval_ms=60000)
    bar.set_sim_status(42, "Saved 42/100")
    bar.set_train_status(80, "Epoch 8/10")
    bar.set_analysis_status(100, "Done")
    assert bar.sim_bar.value() == 42
    assert bar.train_bar.value() == 80
    assert bar.analysis_bar.value() == 100
    bar.shutdown()


def test_main_window_presets_and_event_viewer(qapp, tmp_path):
    win = AirCherenkovMainWindow(initial_preset="VERITAS", preload_events=False)
    assert len(win.viewer_tab.camera_views) == 4

    # Switch to MAGIC (2 telescopes, 1027 pixels)
    win.array_tab.preset_combo.setCurrentText("MAGIC")
    qapp.processEvents()
    assert win.array_model.preset_name == "MAGIC"
    assert len(win.viewer_tab.camera_views) == 2

    # Switch back to VERITAS and load events into Event Viewer
    win.array_tab.preset_combo.setCurrentText("VERITAS")
    events = _make_synthetic_stereo_events(n_events=4, n_tels=4)
    pt_file = tmp_path / "sample.pt"
    torch.save(events, pt_file)
    n_loaded = win.viewer_tab.load_path(str(pt_file))
    assert n_loaded == 4

    # Test cleaning modes, time slider, and clicking a pixel
    win.viewer_tab.mode_combo.setCurrentText("Tailcut Cleaned")
    win.viewer_tab.mode_combo.setCurrentText("Time-Slice Frame")
    win.viewer_tab.time_slider.setValue(6)
    win.viewer_tab.on_pixel_clicked(0, 5)
    assert win.viewer_tab.selected_pix == 5
    assert "Pixel #5" in win.viewer_tab.trace_ax.get_title()

    win.close()


def test_end_to_end_simulation_training_and_irf_workers(qapp, tmp_path):
    # 1. Small CPU simulation worker run on MAGIC (2 telescopes)
    sim_out = tmp_path / "sim_magic"
    magic_model = get_preset_model("MAGIC")
    magic_model.trigger_threshold_pe = 0.5
    magic_model.trigger_min_pixels = 1
    magic_model.trigger_min_telescopes = 1
    sim_params = {
        "num_gammas": 2,
        "num_hadrons": 0,
        "batch_size": 4,
        "save_every": 10,
        "output_dir": str(sim_out),
        "zenith_deg": 20.0,
        "azimuth_deg": 0.0,
        "wobble_offset": 0.0,
        "diffuse_fov": 0.0,
        "e_min": 500.0,
        "e_max": 1500.0,
        "impact_radius": 80.0,
        "device": "cpu",
        "seed": 11,
    }
    sim_worker = SimulationWorker(magic_model, sim_params)
    sim_done = []
    sim_worker.finished_ok.connect(sim_done.append)
    sim_worker.run()
    assert len(sim_done) == 1
    assert (sim_out / "sim_batch_00000.pt").exists()
    assert (sim_out / "sim_state.json").exists()
    assert (sim_out / "detector_model.json").exists()

    # 2. Stereo ML training worker + IRF/Spectral worker on synthetic VERITAS dataset
    ds_dir = tmp_path / "veritas_ds"
    ds_dir.mkdir()
    events = _make_synthetic_stereo_events(n_events=30, n_tels=4)
    torch.save(events, ds_dir / "sim_batch_00000.pt")
    with open(ds_dir / "sim_state.json", "w", encoding="utf-8") as fh:
        fh.write('{"gammas_thrown": 60, "hadrons_thrown": 60}')

    veritas_model = get_preset_model("VERITAS")
    model_path = tmp_path / "stereo_bundle.pt"
    train_worker = TrainingWorker(
        veritas_model,
        {
            "backend": "stereo_ml",
            "data_root": str(ds_dir),
            "model_path": str(model_path),
            "rf_estimators": 20,
            "rf_max_depth": 8,
            "size_2nd_cut": 0.0,
            "seed": 42,
        },
    )
    train_done = []
    train_worker.finished_ok.connect(train_done.append)
    train_worker.run()
    assert len(train_done) == 1
    assert model_path.exists()
    assert np.isfinite(train_done[0]["metrics"]["overall_res68"])

    # 3. IRF & Spectral Analysis worker using the trained model
    irf_worker = AnalysisIRFSpectralWorker(
        veritas_model,
        {
            "test_data": str(ds_dir),
            "model_path": str(model_path),
            "n_thrown_gammas": 0,  # auto-read sim_state.json
            "impact_radius_m": 250.0,
            "e_min_gev": 100.0,
            "e_max_gev": 10000.0,
            "gen_index": 2.0,
            "gamma_cut": 0.3,
            "obs_time_s": 3600.0,
            "spectrum_mode": "crab_reweighted",
        },
        in_memory_bundle=train_done[0]["bundle"],
    )
    irf_done = []
    irf_worker.finished_ok.connect(irf_done.append)
    irf_worker.run()
    assert len(irf_done) == 1
    out = irf_done[0]
    assert np.max(out["aeff"]["aeff"]) > 0.0
    assert out["migration"]["matrix"].shape == (20, 20)
    assert np.isfinite(out["fit"]["alpha"])
    assert np.any(out["valid_unfold"])
    assert "skymap" in out
    assert np.isfinite(out["skymap"]["source_significance_lima"])
    assert out["skymap"]["significance_map"].shape == out["skymap"]["n_on_map"].shape


def test_li_and_ma_significance_and_skymaps(qapp, tmp_path):
    # 1. Exact analytical checks of Li & Ma (1983) Eq. 17
    assert li_and_ma_significance(15, 5, 1.0) == pytest.approx(2.2874618, rel=1e-5)
    assert li_and_ma_significance(100, 200, 0.25) == pytest.approx(5.4070815, rel=1e-5)
    assert li_and_ma_significance(25, 100, 0.25) == pytest.approx(0.0, abs=1e-12)
    assert li_and_ma_significance(5, 20, 1.0) < 0.0
    assert li_and_ma_significance(0, 0, 0.5) == pytest.approx(0.0, abs=1e-12)
    assert li_and_ma_significance(0, 10, 1.0) == pytest.approx(-np.sqrt(20.0 * np.log(2.0)), rel=1e-5)
    assert li_and_ma_significance(10, 0, 1.0) == pytest.approx(+np.sqrt(20.0 * np.log(2.0)), rel=1e-5)

    # Vectorized 2D array evaluation produces matching finite results
    arr_sig = li_and_ma_significance(
        np.array([[100.0, 0.0], [25.0, 5.0]]),
        np.array([[200.0, 0.0], [100.0, 20.0]]),
        np.array([[0.25, 1.0], [0.25, 1.0]]),
    )
    assert arr_sig.shape == (2, 2)
    assert arr_sig[0, 0] == pytest.approx(5.4070815, rel=1e-5)
    assert arr_sig[0, 1] == pytest.approx(0.0, abs=1e-12)
    assert arr_sig[1, 0] == pytest.approx(0.0, abs=1e-12)
    assert arr_sig[1, 1] < 0.0

    # 2. Sky map generation across all 3 background methods on synthetic wobble dataset
    rng = np.random.default_rng(123)
    n_g, n_h = 120, 600
    cardinal = [("N", 0.0), ("E", 0.5 * np.pi), ("S", np.pi), ("W", 1.5 * np.pi)]
    sin_zen = np.sin(np.radians(30.0))

    events = []
    reco_zen = []
    reco_azi = []
    scores = []
    for i in range(n_g):
        w_name, phi = cardinal[i % 4]
        true_z = 30.0 + 0.5 * np.cos(phi)
        true_a = 0.5 * np.sin(phi) / sin_zen
        events.append(
            {
                "zenith_deg": 30.0,
                "azimuth_deg": 0.0,
                "true_zenith_deg": true_z,
                "true_azimuth_deg": true_a,
                "label": 1,
                "wobble_dir": w_name,
            }
        )
        reco_zen.append(true_z + float(rng.normal(0.0, 0.07)))
        reco_azi.append(true_a + float(rng.normal(0.0, 0.07)) / sin_zen)
        scores.append(float(rng.uniform(0.65, 0.99)))

    for j in range(n_h):
        r = np.sqrt(rng.uniform(0.0, 1.0)) * 2.2
        ang = rng.uniform(0.0, 2.0 * np.pi)
        events.append(
            {
                "zenith_deg": 30.0,
                "azimuth_deg": 0.0,
                "true_zenith_deg": 30.0,
                "true_azimuth_deg": 0.0,
                "label": 0,
                "wobble_dir": "diffuse",
            }
        )
        reco_zen.append(30.0 + r * np.cos(ang))
        reco_azi.append(r * np.sin(ang) / sin_zen)
        # Residual surviving hadrons + rejected background
        scores.append(0.75 if j % 4 == 0 else 0.15)

    coords = reconstruct_sky_offsets_for_events(events, np.array(reco_zen), np.array(reco_azi))
    cls_arr = np.array([e["label"] for e in events], dtype=np.float64)
    score_arr = np.array(scores, dtype=np.float64)

    for method in ("ring", "reflected", "mc_hadron"):
        sm = compute_skymap_and_significance(
            sky_dx_deg=coords["sky_dx_deg"],
            sky_dy_deg=coords["sky_dy_deg"],
            true_class=cls_arr,
            pred_class_score=score_arr,
            gamma_cut=0.5,
            theta_cut_deg=0.16,
            fov_half_width_deg=1.6,
            bin_size_deg=0.05,
            bg_method=method,
            wob_x_deg=coords["wob_x_deg"],
            wob_y_deg=coords["wob_y_deg"],
            wobble_offset_deg=coords["wobble_offset_deg"],
        )
        assert sm["source_significance_lima"] > 5.0
        assert sm["peak_significance_lima"] > 5.0
        assert sm["source_n_excess"] > 50.0
        assert sm["n_on_map"].shape == sm["significance_map"].shape
        assert sm["excess_map"].shape == sm["significance_map"].shape
        # Peak of Li & Ma significance map should lie close to target source (0, 0)
        py_idx, px_idx = np.unravel_index(np.argmax(sm["significance_map"]), sm["significance_map"].shape)
        peak_r = np.hypot(sm["bin_centres_deg"][px_idx], sm["bin_centres_deg"][py_idx])
        assert peak_r < 0.20

    # 3. Verify SkyMapSignificanceTab integration inside AirCherenkovMainWindow
    win = AirCherenkovMainWindow(initial_preset="VERITAS", preload_events=False)
    assert win.tabs.count() == 6
    cached = {
        "true_energy_gev": np.full(n_g + n_h, 500.0),
        "reco_energy_gev": np.full(n_g + n_h, 510.0),
        "true_class": cls_arr,
        "pred_class_score": score_arr,
        "sky_dx_deg": coords["sky_dx_deg"],
        "sky_dy_deg": coords["sky_dy_deg"],
        "wob_x_deg": coords["wob_x_deg"],
        "wob_y_deg": coords["wob_y_deg"],
        "wobble_offset_deg": coords["wobble_offset_deg"],
    }
    win.skymap_tab.set_analysis_output({"cached_inference": cached, "gamma_cut": 0.5})
    assert win.skymap_tab.last_skymap is not None
    assert "Li & Ma Significance" in win.skymap_tab.lbl_summary.text()

    # Interactive mode & parameter switches update without errors
    win.skymap_tab.combo_bg.setCurrentIndex(1)  # Reflected wobble
    assert win.skymap_tab.last_skymap["bg_method"] == "reflected"
    win.skymap_tab.combo_bg.setCurrentIndex(2)  # MC Hadron template
    assert win.skymap_tab.last_skymap["bg_method"] == "mc_hadron"
    win.skymap_tab.combo_count_mode.setCurrentIndex(1)  # Raw counts
    win.skymap_tab.spin_theta_cut.setValue(0.20)
    assert win.skymap_tab.last_skymap["theta_cut_deg"] == pytest.approx(0.20)

    win.close()


def test_upgraded_gnn_and_stage2_calibrator_integration(qapp) -> None:
    """Verify 27k-event SpatiotemporalGNN + Stage-2 EnergyCalibrator benchmark & UI wiring."""
    import os
    import numpy as np
    from PySide6.QtWidgets import QApplication

    from gui.main_window import AirCherenkovMainWindow
    from gui.workers import AnalysisIRFSpectralWorker, load_upgraded_gnn_benchmark_evaluation

    assert os.path.exists("data/eval_predictions_corsika_upgraded.npz")

    # 1. Stage-2 debiaser ON + fiducial_s80 benchmark curve
    res = load_upgraded_gnn_benchmark_evaluation(
        npz_path="data/eval_predictions_corsika_upgraded.npz",
        model_path="data/spatiotemporal_gnn_corsika.pt",
        tier="fiducial_s80",
        debias_endpoints=True,
    )
    bundle = res["bundle"]
    m = res["metrics"]
    assert bundle["backend"] == "gnn_cached"
    assert bundle["calibrator"] is not None
    assert bundle["calibrator"].stage2_model is not None
    assert m["stage2_debiased"] is True
    assert m["n_total"] == 27007
    assert m["n_train"] == 21605
    assert m["n_val"] == 27007
    assert m["cv_folds"] == 5
    from gui.workers import VERITAS_BASELINE_E_TEV, compare_with_baseline

    assert len(m["binned"]["bin_centers_tev"]) == len(VERITAS_BASELINE_E_TEV) == 7
    assert np.allclose(m["binned"]["bin_centers_tev"], VERITAS_BASELINE_E_TEV)
    assert len(m["binned"]["fine"]["bin_centers_tev"]) == 14
    cmp_fid = compare_with_baseline(m["binned"])
    assert cmp_fid["n_res_better"] == 5
    assert cmp_fid["n_res_worse"] == 0
    assert cmp_fid["n_bias_better"] == 3
    assert cmp_fid["n_bias_worse"] == 0
    assert np.allclose(
        m["binned"]["resolution_68"] * 100.0,
        [23.21, 23.00, 20.35, 17.59, 13.34, 11.25, 7.73],
        atol=0.15,
    )
    assert np.allclose(
        m["binned"]["bias"] * 100.0,
        [8.92, -1.11, -0.77, -0.04, -0.19, 0.69, -0.49],
        atol=0.15,
    )
    assert np.isclose(m["overall_res68"] * 100.0, 20.18, atol=0.15)
    assert np.isclose(m["overall_bias"] * 100.0, 0.46, atol=0.15)

    # 2. Stage-2 debiaser OFF falls back to live Stage-1 PCHIP evaluation
    res_no_s2 = load_upgraded_gnn_benchmark_evaluation(
        npz_path="data/eval_predictions_corsika_upgraded.npz",
        model_path="data/spatiotemporal_gnn_corsika.pt",
        tier="fiducial_s80",
        debias_endpoints=False,
    )
    assert res_no_s2["metrics"]["stage2_debiased"] is False
    assert res_no_s2["bundle"]["calibrator"].stage2_model is None
    assert not np.allclose(
        res_no_s2["metrics"]["binned"]["resolution_68"],
        m["binned"]["resolution_68"],
    )

    # 3. Live 5-fold CV evaluation on fiducial_all
    res_all = load_upgraded_gnn_benchmark_evaluation(
        npz_path="data/eval_predictions_corsika_upgraded.npz",
        model_path="data/spatiotemporal_gnn_corsika.pt",
        tier="fiducial_all",
        debias_endpoints=True,
    )
    assert np.allclose(
        res_all["metrics"]["binned"]["resolution_68"] * 100.0,
        [24.29, 25.91, 20.58, 17.59, 13.34, 11.25, 7.73],
        atol=0.15,
    )

    # 4. Main window preload populates TrainingTab, IRFSpectralTab, and SkyMapSignificanceTab
    win = AirCherenkovMainWindow(initial_preset="VERITAS", preload_events=True)
    try:
        assert win.train_tab.viewing_benchmark is True
        assert win.train_tab.latest_bundle is not None
        assert win.train_tab.latest_bundle["backend"] == "gnn_cached"
        assert win.irf_tab.in_memory_bundle is win.train_tab.latest_bundle
        assert win.skymap_tab.in_memory_bundle is win.train_tab.latest_bundle
        assert win.train_tab.build_params()["debias_endpoints"] is True

        # Switching VERITAS cut tier updates plots and active bundle
        win.train_tab.combo_cut_tier.setCurrentIndex(3)  # core_contained
        assert win.train_tab.latest_bundle["metrics"]["tier"] == "core_contained"
        assert np.isclose(win.train_tab.latest_bundle["metrics"]["overall_res68"] * 100.0, 20.11, atol=0.15)
        win.train_tab.combo_cut_tier.setCurrentIndex(0)  # fiducial_s80

        # 5. End-to-end IRF + Spectral + Li & Ma Skymap execution on cached gnn_cached bundle
        irf_params = win.irf_tab.build_params()
        irf_params["test_data"] = "data/corsika_crab_30deg/raw"
        worker_irf = AnalysisIRFSpectralWorker(
            win.array_model,
            irf_params,
            in_memory_bundle=win.train_tab.latest_bundle,
        )
        irf_results = []
        irf_errors = []
        worker_irf.finished_ok.connect(irf_results.append)
        worker_irf.failed.connect(irf_errors.append)
        worker_irf.run()
        QApplication.processEvents()
        assert not irf_errors, f"IRF worker failed: {irf_errors}"
        assert len(irf_results) == 1
        out = irf_results[0]
        assert float(np.max(out["aeff"]["aeff"])) > 0.0
        assert np.isfinite(out["fit"]["alpha"])
        assert float(out["skymap"]["source_significance_lima"]) > 5.0
        win.irf_tab._on_finished(out)
        win.skymap_tab._on_finished(out)

        # Verify unmistakable model & dataset provenance across top bar and Tabs 3, 4, 5, 6
        assert "SpatiotemporalGNN" in win.lbl_active_context.text()
        assert "spatiotemporal_gnn_corsika.pt" in win.lbl_active_context.text()
        assert "corsika_crab_30deg" in win.lbl_active_context.text()
        assert "Model:" in win.train_tab.lbl_kpi.text() and "Dataset:" in win.train_tab.lbl_kpi.text()
        assert "spatiotemporal_gnn_corsika.pt" in win.train_tab.lbl_kpi.text()
        assert "Model:" in win.irf_tab.lbl_summary.text() and "Dataset:" in win.irf_tab.lbl_summary.text()
        assert "spatiotemporal_gnn_corsika.pt" in win.irf_tab.lbl_summary.text()
        assert "Model:" in win.skymap_tab.lbl_summary.text() and "Dataset:" in win.skymap_tab.lbl_summary.text()
        assert "spatiotemporal_gnn_corsika.pt" in win.skymap_tab.lbl_summary.text()
        assert "spatiotemporal_gnn_corsika.pt" in win.viewer_tab.lbl_provenance.text()
        if win.viewer_tab.events:
            assert "Dataset:" in win.viewer_tab.lbl_meta.text()
            assert "spatiotemporal_gnn_corsika.pt" in win.viewer_tab.lbl_meta.text()
    finally:
        win.close()


def test_evaluate_existing_dataset_and_model_workflow(qapp, tmp_path) -> None:
    """Verify seamless loading & evaluation of any existing dataset + trained .pt checkpoint across all tabs."""
    from PySide6.QtWidgets import QApplication

    from gui.main_window import AirCherenkovMainWindow
    from gui.workers import (
        EvaluateModelWorker,
        TrainingWorker,
        evaluate_existing_model_on_dataset,
        load_gnn_state_dict_universal,
        resolve_pyg_dataset_root,
    )
    from recon.gnn import SpatiotemporalGNN

    # Build a flat simulation directory (without a pre-existing raw/ subfolder)
    ds_dir = tmp_path / "flat_veritas_sim"
    ds_dir.mkdir()
    events = _make_synthetic_stereo_events(n_events=24, n_tels=4, seed=77)
    torch.save(events, ds_dir / "sim_batch_00000.pt")
    with open(ds_dir / "sim_state.json", "w", encoding="utf-8") as fh:
        fh.write('{"gammas_thrown": 48, "hadrons_thrown": 48}')

    model_dir = tmp_path / "models"
    model_dir.mkdir()
    stereo_model_path = model_dir / "custom_stereo_bundle.pt"

    # Train a real Stereo ML bundle to save on disk
    veritas_model = get_preset_model("VERITAS")
    t_worker = TrainingWorker(
        veritas_model,
        {
            "backend": "stereo_ml",
            "data_root": str(ds_dir),
            "model_path": str(stereo_model_path),
            "rf_estimators": 20,
            "rf_max_depth": 8,
            "size_2nd_cut": 0.0,
            "seed": 42,
        },
    )
    t_done = []
    t_worker.finished_ok.connect(t_done.append)
    t_worker.run()
    assert len(t_done) == 1
    assert stereo_model_path.exists()

    # 1. MainWindow global top-bar synchronization across Tabs 3, 4, 5, and 6
    win = AirCherenkovMainWindow(initial_preset="VERITAS", preload_events=False)
    try:
        win.set_active_dataset_path(str(ds_dir))
        assert win.edit_global_data.text() == str(ds_dir)
        assert win.train_tab.edit_data.text() == str(ds_dir)
        assert win.irf_tab.edit_test.text() == str(ds_dir)
        assert win.skymap_tab.edit_test.text() == str(ds_dir)
        assert len(win.viewer_tab.events) == 24

        # Set a stale bundle first, then switch model path and verify stale cache is cleared
        win.irf_tab.in_memory_bundle = {"model_path": str(model_dir / "other.pt")}
        win.set_active_model_path(str(stereo_model_path))
        assert win.edit_global_model.text() == str(stereo_model_path)
        assert win.train_tab.edit_model.text() == str(stereo_model_path)
        assert win.irf_tab.edit_model.text() == str(stereo_model_path)
        assert win.skymap_tab.edit_model.text() == str(stereo_model_path)
        assert win.irf_tab.in_memory_bundle is None

        # 2. Evaluate existing Stereo ML .pt bundle via EvaluateModelWorker & push into Tabs 3, 4, 5
        eval_params = win.train_tab.build_params()
        eval_params.update(win._collect_eval_params())
        eval_params["gamma_cut"] = 0.25
        e_worker = EvaluateModelWorker(win.array_model, eval_params)
        e_done = []
        e_errs = []
        e_worker.finished_ok.connect(e_done.append)
        e_worker.failed.connect(e_errs.append)
        e_worker.run()
        QApplication.processEvents()
        assert not e_errs, f"EvaluateModelWorker failed: {e_errs}"
        assert len(e_done) == 1

        win.train_tab._on_eval_finished(e_done[0])
        QApplication.processEvents()
        assert "Eval=" in win.train_tab.lbl_kpi.text()
        assert "-Fold CV" in win.train_tab.lbl_kpi.text()
        assert "sigma_68=" in win.train_tab.lbl_kpi.text()
        assert "Model:" in win.train_tab.lbl_kpi.text()
        assert "custom_stereo_bundle.pt" in win.train_tab.lbl_kpi.text()
        assert "flat_veritas_sim" in win.train_tab.lbl_kpi.text()
        assert "custom_stereo_bundle.pt" in win.lbl_active_context.text()
        assert "flat_veritas_sim" in win.lbl_active_context.text()
        assert "custom_stereo_bundle.pt" in win.irf_tab.lbl_summary.text()
        assert "flat_veritas_sim" in win.irf_tab.lbl_summary.text()
        assert "custom_stereo_bundle.pt" in win.skymap_tab.lbl_summary.text()
        assert "flat_veritas_sim" in win.skymap_tab.lbl_summary.text()
        assert "custom_stereo_bundle.pt" in win.viewer_tab.lbl_provenance.text()
        assert "custom_stereo_bundle.pt" in win.viewer_tab.lbl_meta.text()
        assert "flat_veritas_sim" in win.viewer_tab.lbl_meta.text()
        assert win.train_tab.latest_bundle is not None
        assert win.train_tab.latest_bundle["backend"] == "eval_cached"
        assert win.irf_tab.last_output is not None
        assert win.skymap_tab.last_skymap is not None
        assert "source_significance_lima" in win.skymap_tab.last_skymap

        # 3. Verify train_gnn.py dict checkpoint wrapper + flat dir resolution + live GNN eval
        raw_gnn = SpatiotemporalGNN()
        gnn_ckpt_path = model_dir / "wrapped_gnn_ckpt.pt"
        torch.save(
            {
                "epoch": 5,
                "model_state_dict": {f"module.{k}": v for k, v in raw_gnn.state_dict().items()},
            },
            gnn_ckpt_path,
        )
        sd = load_gnn_state_dict_universal(str(gnn_ckpt_path))
        SpatiotemporalGNN().load_state_dict(sd)

        resolved_root = resolve_pyg_dataset_root(str(ds_dir))
        assert resolved_root == str(ds_dir.resolve())
        assert (ds_dir / "raw" / "sim_batch_00000.pt").exists()

        gnn_res = evaluate_existing_model_on_dataset(
            array_model=win.array_model,
            data_path=str(ds_dir),
            model_path=str(gnn_ckpt_path),
            size_2nd_cut=0.0,
        )
        assert gnn_res["bundle"]["backend"] == "eval_cached"
        assert gnn_res["metrics"]["backend"] == "gnn_eval"
        assert gnn_res["metrics"]["cv_folds"] >= 2
        assert gnn_res["metrics"]["n_total"] >= 10
        assert len(gnn_res["bundle"]["cached_results"]["true_energy_gev"]) == gnn_res["metrics"]["n_total"]

        # 4. Verify Evaluate Existing Model on CORSIKA Crab 27k returns full 5-fold CV
        if os.path.exists("data/spatiotemporal_gnn_corsika.pt") and os.path.exists("data/eval_predictions_corsika_upgraded.npz"):
            corsika_res = evaluate_existing_model_on_dataset(
                array_model=win.array_model,
                data_path="data/corsika_crab_30deg/raw",
                model_path="data/spatiotemporal_gnn_corsika.pt",
                tier="fiducial_s80",
                debias_endpoints=True,
            )
            assert corsika_res["metrics"]["cv_folds"] == 5
            assert corsika_res["metrics"]["n_val"] == 27007
            assert corsika_res["bundle"]["cached_results"]["val_fraction"] == pytest.approx(1.0)
    finally:
        win.close()


def test_simulation_append_mode_concatenates_and_invalidates_processed(qapp, tmp_path):
    """SimulationTab + SimulationWorker append_mode concatenates new batches & clears stale processed/data.pt."""
    import json
    import torch
    from gui.array_models import get_preset_model
    from gui.tabs import SimulationTab
    from gui.workers import SimulationWorker

    model = get_preset_model("VERITAS")
    sim_tab = SimulationTab(model)
    try:
        assert sim_tab.build_params()["append_mode"] is False
        sim_tab.chk_append.setChecked(True)
        assert sim_tab.build_params()["append_mode"] is True
    finally:
        sim_tab.close()

    ds_dir = tmp_path / "sim_ds"
    raw_dir = ds_dir / "raw"
    proc_dir = ds_dir / "processed"
    raw_dir.mkdir(parents=True)
    proc_dir.mkdir(parents=True)

    dummy_ev = {
        "fadc_traces": np.zeros((4, 499, 12), dtype=np.float32),
        "gain_flags": np.zeros((4, 499), dtype=np.float32),
        "energy": 1000.0,
        "label": 1,
        "impact_x": 0.0,
        "impact_y": 0.0,
        "zenith": 30.0,
        "azimuth": 0.0,
        "wobble_dir": "north",
        "telescope_positions": [[float(t["x_tel"]), float(t["y_tel"]), 0.0] for t in model.telescopes_spec],
        "trigger_threshold": 5.0,
    }
    torch.save([dummy_ev], raw_dir / "sim_batch_00000.pt")
    with open(raw_dir / "sim_state.json", "w", encoding="utf-8") as fh:
        json.dump({"gammas_thrown": 1, "hadrons_thrown": 0}, fh)
    stale_proc = proc_dir / "data.pt"
    stale_proc.write_bytes(b"stale")

    worker = SimulationWorker(
        model,
        {
            "num_gammas": 1,
            "num_hadrons": 0,
            "e_min": 1000.0,
            "e_max": 1000.0,
            "spectral_index": 2.0,
            "zenith_deg": 20.0,
            "azimuth_deg": 0.0,
            "wobble_offset": 0.0,
            "wobble_mode": "cardinal",
            "diffuse_fov": 0.0,
            "impact_radius": 40.0,
            "batch_size": 2,
            "save_every": 1,
            "device": "cpu",
            "output_dir": str(raw_dir),
            "append_mode": True,
        },
    )
    worker._run_simulation()

    assert (raw_dir / "sim_batch_00000.pt").exists()
    assert (raw_dir / "sim_batch_00001.pt").exists()
    assert not stale_proc.exists()
    with open(raw_dir / "sim_state.json", "r", encoding="utf-8") as fh:
        state = json.load(fh)
    assert state["gammas_thrown"] >= 2


def test_training_worker_resume_sgd_and_stale_processed_rebuild(qapp, tmp_path):
    """TrainingTab + TrainingWorker resume_training continues SGD from checkpoint & rebuilds stale processed/data.pt."""
    import time
    import torch
    from gui.array_models import get_preset_model
    from gui.tabs import TrainingTab
    from gui.workers import TrainingWorker
    from recon.gnn import SpatiotemporalGNN

    model = get_preset_model("VERITAS")
    train_tab = TrainingTab(model)
    try:
        assert train_tab.build_params()["resume_training"] is False
        train_tab.chk_resume_sgd.setChecked(True)
        assert train_tab.build_params()["resume_training"] is True
    finally:
        train_tab.close()

    ds_dir = tmp_path / "gnn_ds"
    raw_dir = ds_dir / "raw"
    proc_dir = ds_dir / "processed"
    raw_dir.mkdir(parents=True)
    proc_dir.mkdir(parents=True)

    # Create stale processed/data.pt older than raw/sim_batch_00000.pt
    stale_proc = proc_dir / "data.pt"
    stale_proc.write_bytes(b"corrupt_stale")
    old_t = time.time() - 120.0
    os.utime(stale_proc, (old_t, old_t))

    events = _make_synthetic_stereo_events(n_events=24, n_tels=4, seed=77)
    torch.save(events, raw_dir / "sim_batch_00000.pt")

    ckpt_path = tmp_path / "gnn_resume.pt"
    init_gnn = SpatiotemporalGNN()
    init_w = next(init_gnn.parameters()).detach().clone()
    torch.save(init_gnn.state_dict(), ckpt_path)

    worker = TrainingWorker(
        model,
        {
            "backend": "gnn",
            "data_root": str(raw_dir),
            "model_path": str(ckpt_path),
            "epochs": 2,
            "batch_size": 4,
            "learning_rate": 1e-3,
            "resume_training": True,
        },
    )
    worker._train_gnn()

    assert (proc_dir / "data.pt").exists()
    reloaded = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    first_key = next(iter(reloaded))
    assert not torch.allclose(reloaded[first_key], init_w.cpu())


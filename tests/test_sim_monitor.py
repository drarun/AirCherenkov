"""Unit tests for gui.sim_monitor progress & active-time timeseries tracking."""

import json
import os
from datetime import datetime, timedelta, timezone

import pytest
import torch

from gui.sim_monitor import resolve_sim_raw_dir, scan_simulation_progress


def _write_dummy_chunk(path, n_gamma: int, n_hadron: int, mtime: float) -> None:
    events = [{"label": 1, "energy": 500.0} for _ in range(n_gamma)] + [
        {"label": 0, "energy": 800.0} for _ in range(n_hadron)
    ]
    torch.save(events, path)
    os.utime(path, (mtime, mtime))


def test_multi_session_pause_gap_subtraction_and_cohort_grouping(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()

    base_t = datetime(2026, 11, 20, 12, 0, 0, tzinfo=timezone.utc)
    # Session 0: older v1.0 cohort (should NOT be merged into active v2.0 cohort)
    _write_dummy_chunk(str(raw / "sim_batch_00000.pt"), 10, 10, base_t.timestamp())

    # Session 1 (v2.0): runs for 10 minutes (600s), saves 2 chunks
    s1_start = base_t + timedelta(days=1)
    s1_c1 = s1_start + timedelta(minutes=5)
    s1_c2 = s1_start + timedelta(minutes=10)
    _write_dummy_chunk(str(raw / "sim_batch_00001.pt"), 30, 10, s1_c1.timestamp())
    _write_dummy_chunk(str(raw / "sim_batch_00002.pt"), 35, 15, s1_c2.timestamp())

    # 48-hour pause before Session 2 (same v2.0 params) runs for 15 minutes (900s)
    s2_start = s1_c2 + timedelta(hours=48)
    s2_c3 = s2_start + timedelta(minutes=7)
    s2_c4 = s2_start + timedelta(minutes=15)
    _write_dummy_chunk(str(raw / "sim_batch_00003.pt"), 40, 20, s2_c3.timestamp())
    _write_dummy_chunk(str(raw / "sim_batch_00004.pt"), 50, 25, s2_c4.timestamp())

    params_v2 = {
        "e_min": 200.0,
        "e_max": 5000.0,
        "spectral_index": 2.0,
        "hadron_e_min": 400.0,
        "hadron_e_max": 10000.0,
        "hadron_spectral_index": 2.7,
        "impact_radius": 500.0,
        "zenith_deg": 20.0,
        "azimuth_deg": 0.0,
    }
    manifest = {
        "schema_version": "v2.0",
        "sessions": [
            {
                "session_index": 0,
                "sim_version": "v1.0",
                "status": "completed",
                "chunk_start": 0,
                "chunk_end": 0,
                "gammas_saved_session": 10,
                "hadrons_saved_session": 10,
                "params": {"e_min": 100.0, "e_max": 30000.0, "impact_radius": 750.0, "zenith_deg": 30.0},
            },
            {
                "session_index": 1,
                "sim_version": "v2.0",
                "status": "paused",
                "started_at": s1_start.isoformat(),
                "updated_at": s1_c2.isoformat(),
                "chunk_start": 1,
                "chunk_end": 2,
                "target_gammas_session": 200,
                "target_hadrons_session": 100,
                "gammas_saved_session": 65,
                "hadrons_saved_session": 25,
                "params": params_v2,
            },
            {
                "session_index": 2,
                "sim_version": "v2.0",
                "status": "in_progress",
                "started_at": s2_start.isoformat(),
                "updated_at": s2_c4.isoformat(),
                "chunk_start": 3,
                "chunk_end": 4,
                "target_gammas_session": 135,
                "target_hadrons_session": 75,
                "gammas_saved_session": 90,
                "hadrons_saved_session": 45,
                "params": params_v2,
            },
        ],
    }
    with open(raw / "sim_manifest.json", "w", encoding="utf-8") as mf:
        json.dump(manifest, mf)

    assert resolve_sim_raw_dir(str(tmp_path)) == str(raw)

    snap = scan_simulation_progress(str(tmp_path), now_ts=s2_c4.timestamp() + 10.0)
    assert snap.status == "RUNNING"
    assert len(snap.cohort_sessions) == 2
    assert snap.cohort_gammas_saved == 155
    assert snap.cohort_hadrons_saved == 70
    assert snap.cohort_target_gammas == 335
    assert snap.cohort_target_hadrons == 175
    assert snap.total_dataset_events == 20 + 155 + 70
    # 10 min + 15 min + 10s since last update = 1510s active; ~48h excluded
    assert snap.active_elapsed_s == pytest.approx(1510.0, abs=2.0)
    assert snap.paused_time_excluded_s == pytest.approx(48.0 * 3600.0, abs=2.0)
    assert snap.pause_markers_s == [pytest.approx(600.0, abs=1.0)]
    assert snap.timeseries[-1].total_events == 225
    assert snap.timeseries[-1].active_elapsed_s == pytest.approx(1500.0, abs=1.0)


def test_manifestless_fallback(tmp_path):
    t0 = datetime(2026, 11, 25, 10, 0, 0, tzinfo=timezone.utc).timestamp()
    _write_dummy_chunk(str(tmp_path / "sim_batch_00000.pt"), 12, 8, t0)
    _write_dummy_chunk(str(tmp_path / "sim_batch_00001.pt"), 15, 5, t0 + 120.0)

    snap = scan_simulation_progress(
        str(tmp_path),
        target_gammas_override=50,
        target_hadrons_override=50,
        now_ts=t0 + 150.0,
        auto_tag=False,
    )
    assert snap.status == "RUNNING"
    assert snap.cohort_gammas_saved == 27
    assert snap.cohort_hadrons_saved == 13
    assert snap.cohort_target_gammas == 50
    assert snap.active_elapsed_s == pytest.approx(150.0, abs=1.0)
    assert len(snap.timeseries) == 3


def test_extract_stereo_features_return_intermediates_matches():
    import numpy as np
    from recon.ml_stereo import extract_stereo_features
    from sim.camera import Camera
    from sim.telescope import TelescopeArray

    cam = Camera(n_rings=12, pixel_size=0.15)
    tels = TelescopeArray.veritas_array().telescopes
    rng = np.random.default_rng(7)
    traces = rng.uniform(0.0, 0.5, size=(4, cam.n_pixels, 16)).astype(np.float32)
    traces[0, :6, 4:9] += 25.0
    traces[1, :6, 4:9] += 22.0
    gains = np.zeros((4, cam.n_pixels), dtype=np.float32)

    vec_only = extract_stereo_features(
        traces, tels, cam, gain_flags=gains, include_extended=True
    )
    vec_with_int, restored, dps, hillas = extract_stereo_features(
        traces, tels, cam, gain_flags=gains, include_extended=True, return_intermediates=True
    )
    assert np.array_equal(vec_only, vec_with_int)
    assert len(restored) == 4 and len(dps) == 4 and len(hillas) == 4


def test_cherenkov_dataset_parallel_and_per_chunk_cache(tmp_path):
    import numpy as np
    from analysis.dataset import CherenkovDataset
    from sim.camera import Camera

    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    cam = Camera(n_rings=12, pixel_size=0.15)
    # Construct two elongated orthogonal camera images on tel 0 and tel 1 so geometric stereo succeeds
    t0_pix = np.where((np.abs(cam.pixel_y) < 0.05) & (cam.pixel_x >= -0.05) & (cam.pixel_x <= 0.65))[0]
    t1_pix = np.where((np.abs(cam.pixel_x - 0.15) < 0.12) & (cam.pixel_y >= 0.10) & (cam.pixel_y <= 0.80))[0]
    for c_idx in range(2):
        evts = []
        for _ in range(2):
            traces = np.zeros((4, cam.n_pixels, 16), dtype=np.float32)
            traces[0, t0_pix, 5:10] = 20.0
            traces[1, t1_pix, 5:10] = 20.0
            evts.append({
                "fadc_traces": traces,
                "gain_flags": np.zeros((4, cam.n_pixels), dtype=np.float32),
                "energy": 500.0,
                "label": 1,
                "zenith_deg": 30.0,
                "azimuth_deg": 0.0,
            })
        torch.save(evts, raw_dir / f"sim_batch_{c_idx:05d}.pt")

    ds1 = CherenkovDataset(root=str(tmp_path))
    cache_dir = tmp_path / "processed" / "chunks"
    assert (cache_dir / "sim_batch_00000.pt.cache.pt").is_file()
    assert (cache_dir / "sim_batch_00001.pt.cache.pt").is_file()

    # Delete monolithic data.pt and rebuild from per-chunk caches
    (tmp_path / "processed" / "data.pt").unlink()
    ds2 = CherenkovDataset(root=str(tmp_path))
    assert len(ds2) == len(ds1)


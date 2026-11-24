import numpy as np
import pytest
import torch

from sim.atmosphere import vertical_grammage, vertical_grammage_np
from sim.interactions import M_ELECTRON_GEV, M_MUON_GEV
from sim.shower import ShowerSimulation, arclength_from_grammage, atmospheric_density


def test_shower_initialization():
    sim = ShowerSimulation(primary_types=['gamma'], energies=[100.0])
    assert sim.batch_size == 1
    assert sim.active.shape == (1, 10)
    assert sim.active[0, 0].item() == sim.PID_MAP['gamma']
    assert sim.active[0, 1].item() == 100.0
    assert not hasattr(sim, 'entropy_pool')


def test_batched_shower_initialization():
    sim = ShowerSimulation(
        primary_types=['gamma', 'proton'],
        energies=[100.0, 500.0],
        z_starts=[20000.0, 25000.0],
    )

    assert sim.batch_size == 2
    assert sim.active.shape == (2, 10)
    # labels are at column 9 (event_id)
    np.testing.assert_array_equal(sim.active[:, 9].cpu().numpy(), [0.0, 1.0])
    assert set(sim.cherenkov_photons_by_event) == {0, 1}
    assert sim.active[1, 4].item() == 25000.0


def test_tensor_step():
    sim = ShowerSimulation(primary_types=['gamma'], energies=[100.0])
    sim.step()
    assert isinstance(sim.active, torch.Tensor)
    assert sim.active.shape[1] == 10


def test_legacy_scalar_api_and_broadcasting():
    # Handles string primary_types, float/int energies and z_starts by converting them to lists
    scalar = ShowerSimulation('gamma', energies=25.0, z_starts=12000.0)
    assert scalar.batch_size == 1
    assert scalar.active[0, 1].item() == 25.0
    assert scalar.active[0, 4].item() == 12000.0

    batch = ShowerSimulation(['gamma', 'proton'], energies=50.0, z_starts=15000.0)
    assert batch.active.shape == (2, 10)
    np.testing.assert_allclose(batch.active[:, 1].cpu().numpy(), [50.0, 50.0])


def test_track_recording_returns_empty_dataframe():
    sim = ShowerSimulation('gamma', energies=10.0)
    sim.step()
    # get_tracks_dataframe returns empty dataframe since full tracks are omitted to save VRAM
    assert sim.get_tracks_dataframe().empty


def test_batched_photon_packets_keep_event_identity_without_duplicate_slices():
    sim = ShowerSimulation(
        ['gamma', 'gamma'], energies=[1.0, 1.0], z_starts=[100.0, 100.0], z_obs=0.0
    )
    sim.c_segs_start = [torch.tensor(
        [[0.0, 0.0, 100.0], [10.0, 0.0, 100.0]], dtype=torch.float32
    )]
    sim.c_segs_end = [torch.tensor(
        [[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]], dtype=torch.float32
    )]
    sim.c_segs_p = [torch.tensor(
        [[0.0, 0.0, -1.0], [0.0, 0.0, -1.0]], dtype=torch.float32
    )]
    sim.c_segs_E = [torch.tensor([1.0, 1.0], dtype=torch.float32)]
    sim.c_segs_event_id = [torch.tensor([0.0, 1.0], dtype=torch.float32)]

    sim.calculate_cherenkov_pool()

    assert set(sim.cherenkov_photons_by_event.keys()) == {0, 1}
    for event_id in (0, 1):
        packets = sim.cherenkov_photons_by_event[event_id]
        assert 'x_ground' in packets
        assert 'y_ground' in packets


def test_explicit_seed_reproduces_shower_and_packet_sampling():
    outputs = []
    for _ in range(2):
        torch.manual_seed(22)
        simulation = ShowerSimulation(
            'gamma', energies=10.0, z_starts=10000.0
        )
        simulation.run(max_generations=4, verbose=False)
        outputs.append(simulation.cherenkov_photons_by_event[0])

    assert outputs[0].keys() == outputs[1].keys()
    for key in outputs[0]:
        np.testing.assert_allclose(outputs[0][key], outputs[1][key], rtol=1e-4, atol=1e-4)


def test_arclength_from_grammage_conserves_slant_depth():
    z1 = torch.tensor([50000.0, 25000.0, 12000.0], dtype=torch.float32)
    pz = torch.tensor([-1.0, -0.8660254, -0.5], dtype=torch.float32)
    dx = torch.tensor([47.0, 120.0, 250.0], dtype=torch.float32)

    s = arclength_from_grammage(z1, pz, dx)
    z2 = z1 + pz * s

    xv1 = vertical_grammage(z1)
    xv2 = vertical_grammage(z2)
    recovered_dx = (xv2 - xv1) / (-pz)
    np.testing.assert_allclose(recovered_dx.numpy(), dx.numpy(), rtol=1e-4, atol=1e-3)


def test_site_z_obs_and_b_field_z_up_conventions():
    sim_veritas = ShowerSimulation('gamma', energies=100.0, site='VERITAS')
    sim_hess = ShowerSimulation('gamma', energies=100.0, site='HESS')
    sim_magic = ShowerSimulation('gamma', energies=100.0, site='MAGIC')
    sim_custom = ShowerSimulation('gamma', energies=100.0, site='VERITAS', z_obs=1500.0)

    assert sim_veritas.z_obs == pytest.approx(1268.0)
    assert sim_hess.z_obs == pytest.approx(1800.0)
    assert sim_magic.z_obs == pytest.approx(2200.0)
    assert sim_custom.z_obs == pytest.approx(1500.0)

    # Z-up convention: Northern hemisphere Bz < 0, Southern hemisphere (HESS) Bz > 0
    assert sim_veritas.B_field[2].item() < 0.0
    assert sim_magic.B_field[2].item() < 0.0
    assert sim_hess.B_field[2].item() > 0.0


def test_subcritical_electron_ionization_tail_tracks_down_to_20_mev():
    torch.manual_seed(101)
    sim = ShowerSimulation('e-', energies=0.060, z_starts=5000.0, site='VERITAS')

    # Step 1: 60 MeV -> ~28.5 MeV over 15 g/cm^2; survives below 85 MeV critical energy
    sim.step()
    assert sim.active.shape[0] == 1
    E_after_1 = sim.active[0, 1].item()
    assert sim.min_electron_energy <= E_after_1 < sim.critical_energy
    assert len(sim.c_segs_E) == 1

    # Continue until ranged out at 20 MeV
    for _ in range(10):
        if sim.active.shape[0] == 0:
            break
        sim.step()

    assert sim.active.shape[0] == 0
    all_E = torch.cat(sim.c_segs_E, dim=0).cpu().numpy()
    all_mass = torch.cat(sim.c_segs_mass, dim=0).cpu().numpy()
    assert len(all_E) >= 2
    assert np.all(all_E < sim.critical_energy)
    assert np.min(all_E) < 0.040
    np.testing.assert_allclose(all_mass, M_ELECTRON_GEV, rtol=1e-5)


def test_10_gev_gamma_shower_produces_cherenkov_segments_and_photons():
    torch.manual_seed(123)
    sim = ShowerSimulation('gamma', energies=10.0, z_starts=40000.0, site='VERITAS')
    sim.run(max_generations=60, verbose=False)

    assert len(sim.c_segs_E) > 0
    seg_E = torch.cat(sim.c_segs_E, dim=0).cpu().numpy()
    assert np.any(seg_E < sim.critical_energy), "Expected sub-85 MeV ionization tail segments"

    photons = sim.cherenkov_photons_by_event[0]
    assert len(photons['x_ground']) > 0
    assert np.sum(photons['weight']) > 0.0


def test_1_tev_gamma_showers_have_physical_xmax():
    torch.manual_seed(42)
    n_events = 16
    sim = ShowerSimulation(
        primary_types=['gamma'] * n_events,
        energies=[1000.0] * n_events,
        z_starts=[50000.0] * n_events,
        site='VERITAS',
    )
    sim.run(max_generations=60, verbose=False)

    event_medians = []
    for i in range(n_events):
        ph = sim.cherenkov_photons_by_event[i]
        assert len(ph.get('z_emit', [])) > 0
        depths = vertical_grammage_np(ph['z_emit'])
        weights = ph['weight'].astype(np.float64)
        order = np.argsort(depths)
        sorted_depth = depths[order]
        cum_w = np.cumsum(weights[order], dtype=np.float64)
        event_medians.append(
            float(sorted_depth[np.searchsorted(cum_w, 0.5 * cum_w[-1])])
        )

    median_xmax = float(np.median(event_medians))
    assert 280.0 <= median_xmax <= 420.0, (
        f"Expected 1 TeV gamma median Cherenkov depth in [280, 420] g/cm^2, "
        f"got {median_xmax:.1f} g/cm^2"
    )


def test_300_gev_proton_showers_produce_muon_segments():
    torch.manual_seed(7)
    n_events = 8
    sim = ShowerSimulation(
        primary_types=['proton'] * n_events,
        energies=[300.0] * n_events,
        z_starts=[50000.0] * n_events,
        site='VERITAS',
    )
    sim.run(max_generations=60, verbose=False)

    assert len(sim.c_segs_mass) > 0
    all_mass = torch.cat(sim.c_segs_mass, dim=0).cpu().numpy()
    mu_mask = np.isclose(all_mass, M_MUON_GEV, rtol=1e-4)
    e_mask = np.isclose(all_mass, M_ELECTRON_GEV, rtol=1e-4)
    assert np.any(mu_mask), "Expected muon Cherenkov segments from pi+- -> mu+- decay"
    assert np.any(e_mask), "Expected electron Cherenkov segments from EM sub-showers"


def test_save_and_load_state_roundtrip(tmp_path):
    torch.manual_seed(9)
    sim = ShowerSimulation('gamma', energies=20.0, z_starts=20000.0, site='HESS')
    sim.step()
    sim.step()

    ckpt = tmp_path / "shower_ckpt.pt"
    sim.save_state(str(ckpt))

    restored = ShowerSimulation('gamma', energies=20.0, z_starts=20000.0, site='VERITAS')
    restored.load_state(str(ckpt))
    assert restored.z_obs == pytest.approx(1800.0)
    assert len(restored.c_segs_mass) == len(sim.c_segs_mass)
    assert restored.active.shape == sim.active.shape


def test_adaptive_cherenkov_packet_granularity(monkeypatch):
    import sim.backend as backend_mod

    recorded_calls = []

    def fake_pool(*args, **kwargs):
        recorded_calls.append(kwargs)
        return {'x_ground': np.array([0.0], dtype=np.float32), 'y_ground': np.array([0.0], dtype=np.float32)}

    monkeypatch.setattr(backend_mod, 'compute_cherenkov_pool_gpu', fake_pool)

    energies = [50.0, 300.0, 2000.0, 10000.0]
    sim = ShowerSimulation(['gamma'] * 4, energies=energies, z_starts=[5000.0] * 4, site='MAGIC')
    sim.c_segs_start = [torch.tensor([[0.0, 0.0, 5000.0]] * 4, dtype=torch.float32)]
    sim.c_segs_end = [torch.tensor([[0.0, 0.0, 4900.0]] * 4, dtype=torch.float32)]
    sim.c_segs_p = [torch.tensor([[0.0, 0.0, -1.0]] * 4, dtype=torch.float32)]
    sim.c_segs_E = [torch.tensor([1.0] * 4, dtype=torch.float32)]
    sim.c_segs_mass = [torch.tensor([M_ELECTRON_GEV, M_ELECTRON_GEV, M_MUON_GEV, M_ELECTRON_GEV], dtype=torch.float32)]
    sim.c_segs_event_id = [torch.tensor([0.0, 1.0, 2.0, 3.0], dtype=torch.float32)]

    sim.calculate_cherenkov_pool()

    assert len(recorded_calls) == 4
    assert [c['target_photons_per_packet'] for c in recorded_calls] == [5.0, 10.0, 25.0, 50.0]
    assert all(c['z_obs'] == pytest.approx(2200.0) for c in recorded_calls)
    assert recorded_calls[2]['seg_mass_gev'].item() == pytest.approx(M_MUON_GEV, rel=1e-5)

import numpy as np
import pytest
import torch
from sim.shower import ShowerSimulation

def test_shower_initialization():
    sim = ShowerSimulation(primary_types=['gamma'], energies=[100.0])
    assert sim.batch_size == 1
    assert sim.active.shape == (1, 10)
    assert sim.active[0, 0].item() == sim.PID_MAP['gamma']
    assert sim.active[0, 1].item() == 100.0

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
    # master handles string primary_types, float/int energies and z_starts by converting them to lists
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
    # In master, get_tracks_dataframe returns empty dataframe since it is omitted to save VRAM
    assert sim.get_tracks_dataframe().empty

def test_batched_photon_packets_keep_event_identity_without_duplicate_slices():
    sim = ShowerSimulation(
        ['gamma', 'gamma'], energies=[1.0, 1.0], z_starts=[100.0, 100.0]
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
        # In master, compute_cherenkov_pool_gpu returns dictionary with keys: 'x_ground', 'y_ground'
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

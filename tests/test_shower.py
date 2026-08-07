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
    np.testing.assert_array_equal(sim.active[:, 9].cpu().numpy(), [0.0, 1.0])
    assert set(sim.cherenkov_photons_by_event) == {0, 1}
    assert sim.active[1, 4].item() == 25000.0


def test_tensor_step():
    sim = ShowerSimulation(primary_types=['gamma'], energies=[100.0])
    sim.step()
    # After one generation, the gamma (if it survived) may have pair produced.
    # We just want to ensure step runs without error and active particles tensor is updated.
    assert isinstance(sim.active, torch.Tensor)
    assert sim.active.shape[1] == 10


def test_legacy_scalar_api_and_broadcasting():
    scalar = ShowerSimulation('gamma', energy=25.0, z_start=12000.0)
    assert scalar.batch_size == 1
    assert scalar.active[0, 1].item() == 25.0
    assert scalar.active[0, 4].item() == 12000.0

    batch = ShowerSimulation(['gamma', 'proton'], energies=50.0, z_starts=15000.0)
    assert batch.active.shape == (2, 10)
    np.testing.assert_allclose(batch.active[:, 1].cpu().numpy(), [50.0, 50.0])


@pytest.mark.parametrize(
    'kwargs',
    [
        {'primary_types': ['gamma', 'proton'], 'energies': [10.0, 20.0, 30.0]},
        {'primary_types': ['unknown']},
        {'primary_types': ['gamma'], 'energies': [0.0]},
        {'primary_types': []},
    ],
)
def test_invalid_initialization_is_rejected(kwargs):
    with pytest.raises(ValueError):
        ShowerSimulation(**kwargs)


def test_track_recording_is_opt_in_and_event_aware():
    no_tracks = ShowerSimulation('gamma', energy=10.0)
    no_tracks.step()
    assert no_tracks.get_tracks_dataframe().empty

    sim = ShowerSimulation(
        ['gamma', 'proton'], energies=[10.0, 20.0],
        z_starts=10000.0, record_tracks=True,
    )
    sim.step()

    tracks = sim.get_tracks_dataframe()
    assert list(tracks.columns) == ShowerSimulation.TRACK_COLUMNS
    assert tracks['particle_id'].nunique() == 2
    assert (tracks.groupby('particle_id').size() == 2).all()
    assert np.isfinite(tracks[['energy', 'x', 'y', 'z']]).all().all()
    assert set(tracks['event_id']) == {0, 1}

    event_one = sim.get_tracks_dataframe(event_idx=1)
    assert set(event_one['event_id']) == {1}
    assert set(event_one['pid']) == {'proton'}


def test_track_downsampling_is_deterministic():
    torch.manual_seed(0)
    sim = ShowerSimulation('gamma', energy=100.0, record_tracks=True)
    for _ in range(4):
        sim.step()

    first = sim.get_tracks_dataframe(max_tracks=3)
    second = sim.get_tracks_dataframe(max_tracks=3)
    assert first.equals(second)
    assert first['particle_id'].nunique() == 3


def test_ground_crossing_segments_are_clipped():
    start = torch.tensor([1.0])
    end_x, end_y, end_z = ShowerSimulation._clip_segments_to_ground(
        torch.tensor([0.0]), torch.tensor([0.0]), start,
        torch.tensor([2.0]), torch.tensor([4.0]), torch.tensor([-1.0]),
    )
    torch.testing.assert_close(end_x, torch.tensor([1.0]))
    torch.testing.assert_close(end_y, torch.tensor([2.0]))
    torch.testing.assert_close(end_z, torch.tensor([0.0]))


def test_batched_photon_packets_keep_event_identity_without_duplicate_slices():
    sim = ShowerSimulation(
        ['gamma', 'gamma'], energies=1.0, z_starts=100.0,
        seed=6, device='cpu', target_photons_per_packet=32,
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

    assert set(sim.cherenkov_packets['event_id']) == {0, 1}
    for event_id in (0, 1):
        packets = sim.cherenkov_photons_by_event[event_id]
        assert len(packets['weight']) > 0
        assert np.all(packets['shower_start_altitude'] == 100.0)
        assert np.shares_memory(
            packets['x_ground'], sim.cherenkov_packets['x_ground']
        )


def test_explicit_seed_reproduces_shower_and_packet_sampling():
    outputs = []
    for _ in range(2):
        simulation = ShowerSimulation(
            'gamma', energy=10.0, z_start=10_000.0,
            seed=22, device='cpu', target_photons_per_packet=32,
        )
        simulation.run(max_generations=4, verbose=False)
        outputs.append(simulation.cherenkov_photons_by_event[0])

    assert outputs[0].keys() == outputs[1].keys()
    for key in outputs[0]:
        np.testing.assert_array_equal(outputs[0][key], outputs[1][key])

import numpy as np
import pytest
import torch

from analysis.dataset import (
    _integrated_image_trace,
    _legacy_event_value,
    _trace_features,
    extract_simulation_metadata,
    restore_low_gain_traces,
)
from train_gnn import event_grouped_split_indices


def test_low_gain_is_restored_before_waveform_features_and_charge():
    traces = np.zeros((2, 16), dtype=np.float32)
    traces[0, 2] = 4.0
    traces[1, 5] = 7.0
    gains = np.array([0.0, 1.0], dtype=np.float32)

    restored = restore_low_gain_traces(traces, gains, low_gain_factor=10.0)
    features, charge, timing = _trace_features(
        traces,
        gains,
        pixel_x=np.array([0.0, 0.1]),
        pixel_y=np.array([0.0, 0.0]),
        low_gain_factor=10.0,
    )

    np.testing.assert_allclose(restored[0], traces[0])
    np.testing.assert_allclose(restored[1], traces[1] * 10.0)
    np.testing.assert_allclose(charge, [4.0, 70.0])
    np.testing.assert_allclose(timing, [2.0, 5.0])
    torch.testing.assert_close(features[:, :16], torch.from_numpy(restored))
    torch.testing.assert_close(features[:, 16], torch.tensor([0.0, 1.0]))
    assert features.shape == (2, 22)


def test_simulation_metadata_preserves_stereo_provenance():
    event = {
        "event_id": 42,
        "telescope_ids": [11, 12, 13, 14],
        "telescope_positions_m": [
            [0.0, 0.0, 1.0],
            [100.0, 0.0, 2.0],
            [0.0, 100.0, 3.0],
            [100.0, 100.0, 4.0],
        ],
        "telescope_multiplicity": 3,
        "impact_x": 70.0,
        "impact_y": 40.0,
        "sampling_weight": 2.5,
        "trigger": {
            "camera_triggered": [True, False, True, True],
            "camera_trigger_times_ns": [4.0, None, 7.0, 9.0],
        },
    }

    metadata = extract_simulation_metadata(event, 2, 4, event_id=42)

    assert metadata["event_id"].item() == 42
    assert metadata["telescope_id"].item() == 13
    torch.testing.assert_close(
        metadata["telescope_position_m"], torch.tensor([[0.0, 100.0, 3.0]])
    )
    assert metadata["telescope_multiplicity"].item() == 3
    torch.testing.assert_close(
        metadata["impact_xy_m"], torch.tensor([[70.0, 40.0]])
    )
    assert metadata["impact_distance_m"].item() == pytest.approx(
        np.hypot(70.0, 60.0)
    )
    assert metadata["sampling_weight"].item() == pytest.approx(2.5)
    assert metadata["camera_triggered"].item() is True
    assert metadata["camera_trigger_time_ns"].item() == pytest.approx(7.0)


def test_legacy_integrated_images_keep_current_model_shape():
    image = np.array([0.0, 3.0, -2.0], dtype=np.float32)
    trace = _integrated_image_trace(image)

    assert trace.shape == (3, 16)
    np.testing.assert_allclose(trace[:, 0], [0.0, 3.0, 0.0])
    assert np.count_nonzero(trace[:, 1:]) == 0


def test_legacy_batched_metadata_distinguishes_event_and_array_fields():
    payload = {
        "impact_x": np.array([5.0, 8.0]),
        "telescope_positions_m": np.array(
            [[0.0, 0.0, 0.0], [100.0, 0.0, 0.0]], dtype=np.float32
        ),
    }

    assert _legacy_event_value(payload, "impact_x", 1, 2) == pytest.approx(8.0)
    np.testing.assert_allclose(
        _legacy_event_value(payload, "telescope_positions_m", 1, 2),
        payload["telescope_positions_m"],
    )


def test_event_grouped_split_is_deterministic_and_has_no_view_leakage():
    event_ids = [10, 10, 10, 20, 20, 30, 30, 30, 40]

    first = event_grouped_split_indices(event_ids, train_fraction=0.75, seed=17)
    second = event_grouped_split_indices(event_ids, train_fraction=0.75, seed=17)
    train_indices, validation_indices = first

    assert first == second
    assert sorted(train_indices + validation_indices) == list(range(len(event_ids)))
    train_events = {event_ids[index] for index in train_indices}
    validation_events = {event_ids[index] for index in validation_indices}
    assert train_events.isdisjoint(validation_events)
    assert train_events | validation_events == set(event_ids)


def test_event_grouped_split_rejects_an_unsplittable_dataset():
    with pytest.raises(ValueError, match="at least two showers"):
        event_grouped_split_indices([5, 5, 5], seed=1)

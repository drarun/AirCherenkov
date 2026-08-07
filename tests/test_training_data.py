import json

import numpy as np
import pytest

import generate_training_data as training


def test_event_stage_seeds_and_throws_are_reproducible():
    config = training._build_config(
        5, 5, 2, 3, 4, 250.0, 5.0, 5.0, 3, 2, 50.0,
        10.0, 2.0, 'cpu', 'cpu',
    )
    first = training._sample_throw(17, 3, 'gamma', config)
    second = training._sample_throw(17, 3, 'gamma', config)
    different = training._sample_throw(17, 4, 'gamma', config)

    assert first == second
    assert first['throw_seed'] != different['throw_seed']
    assert 100.0 <= first['energy_gev'] <= 10_000.0
    assert first['impact_radius_m'] <= 250.0
    assert first['weights']['joint_sampling_density'] > 0


def test_array_coincidence_requires_times_within_window():
    assert training._evaluate_array_coincidence(
        [True, True, False], [10.0, 14.0, None], 2, 5.0
    ) == (True, 14.0, 2)
    assert training._evaluate_array_coincidence(
        [True, True], [10.0, 20.0], 2, 5.0
    ) == (False, None, 1)


def test_low_gain_trace_is_restored_before_triggering():
    traces = np.zeros((2, 4), dtype=np.float32)
    traces[0, 1] = 2.0
    traces[1, 2] = 3.0
    charge, timing = training._camera_trigger_inputs(
        traces, np.array([1.0, 0.0]), 2.0, 10.0
    )

    np.testing.assert_allclose(charge, [20.0, 3.0])
    np.testing.assert_allclose(timing, [2.0, 4.0])


def test_manifest_resume_uses_recorded_attempt_counts(tmp_path):
    config = training._build_config(
        2, 2, 2, 2, 4, 250.0, 5.0, 5.0, 3, 2, 50.0,
        10.0, 0.0, 'cpu', 'cpu',
    )
    manifest, pending, metadata_dir, manifest_path = training._load_or_create_state(
        tmp_path, 11, config
    )
    assert pending == []
    manifest['counts']['attempted'].update({'gamma': 1, 'proton': 0, 'total': 1})
    manifest['next_event_id'] = 1
    training._checkpoint_state(manifest, [], metadata_dir, manifest_path)

    resumed, resumed_pending, *_ = training._load_or_create_state(tmp_path, 11, config)
    assert resumed_pending == []
    assert resumed['next_event_id'] == 1
    assert resumed['counts']['attempted']['gamma'] == 1
    assert json.loads(manifest_path.read_text())['schema'] == training.MANIFEST_SCHEMA


def test_manifest_refuses_configuration_drift(tmp_path):
    config = training._build_config(
        1, 1, 1, 1, 4, 250.0, 5.0, 5.0, 3, 2, 50.0,
        10.0, 0.0, 'cpu', 'cpu',
    )
    training._load_or_create_state(tmp_path, 3, config)
    changed = dict(config)
    changed['batch_size'] = 99
    with pytest.raises(ValueError, match='configuration differs'):
        training._load_or_create_state(tmp_path, 3, changed)


def test_throw_ledger_keeps_rejected_trials(tmp_path):
    manifest = {'throw_ledger_files': []}
    records = [{
        'event_id': 0,
        'primary_type': 'gamma',
        'accepted': False,
        'rejection_reason': 'array_trigger',
    }]
    training._commit_throw_ledger(records, 0, 1, manifest, tmp_path)

    assert manifest['throw_ledger_files'] == ['throws_000000000000_000000000000.json']
    payload = json.loads((tmp_path / manifest['throw_ledger_files'][0]).read_text())
    assert payload['schema'] == training.THROW_LEDGER_SCHEMA
    assert payload['events'] == records


def test_instrument_metadata_has_geometry_hash():
    array = training.TelescopeArray.veritas_array(device='cpu')
    metadata = training._instrument_metadata(array)

    assert metadata['name'] == 'veritas-like'
    assert len(metadata['telescopes']) == 4
    assert len(metadata['geometry_hash']) == 64

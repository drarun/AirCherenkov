import numpy as np
import pytest

from sim.fadc import restore_low_gain_traces


def test_restore_low_gain_traces_calibrates_flagged_pixels_without_mutation():
    traces = np.array(
        [[1.0, 2.0, -0.5], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]],
        dtype=np.float32,
    )
    original = traces.copy()
    gain_flags = np.array([0.0, 1.0, 0.6], dtype=np.float32)

    restored = restore_low_gain_traces(traces, gain_flags, low_gain_factor=10.0)

    np.testing.assert_array_equal(traces, original)
    np.testing.assert_allclose(restored[0], original[0])
    np.testing.assert_allclose(restored[1:], original[1:] * 10.0)
    assert restored.dtype == np.float32
    assert not np.shares_memory(restored, traces)


@pytest.mark.parametrize(
    ("traces", "gain_flags", "factor", "message"),
    [
        (np.zeros(3), np.zeros(3), 10.0, "traces must have shape"),
        (np.zeros((2, 3)), np.zeros(3), 10.0, "one value per pixel"),
        (
            np.array([[np.nan, 0.0]]),
            np.zeros(1),
            10.0,
            "traces must contain only finite",
        ),
        (np.zeros((1, 2)), np.array([np.inf]), 10.0, "gain_flags must contain only finite"),
        (np.zeros((1, 2)), np.zeros(1), 0.0, "greater than zero"),
    ],
)
def test_restore_low_gain_traces_rejects_invalid_readout(
    traces, gain_flags, factor, message
):
    with pytest.raises(ValueError, match=message):
        restore_low_gain_traces(traces, gain_flags, factor)

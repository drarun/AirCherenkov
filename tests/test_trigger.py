import numpy as np
import pytest

from sim.camera import Camera
from sim.trigger import CameraTrigger, evaluate_array_trigger


def test_trigger_adjacency_scales_with_camera_geometry():
    camera = Camera(n_rings=2, pixel_size=0.25)
    trigger = CameraTrigger(
        camera.pixel_x,
        camera.pixel_y,
        pixel_size=camera.pixel_size,
    )

    assert trigger.pixel_size == pytest.approx(0.25)
    assert len(trigger.clusters) > 0

    cluster = trigger.clusters[0]
    image = np.zeros(camera.n_pixels)
    timing = np.zeros(camera.n_pixels)
    image[cluster] = 6.0
    timing[cluster] = [1.0, 2.0, 3.0]
    assert trigger.evaluate(image, timing)[0]


def test_trigger_requires_spatial_and_time_coincidence():
    camera = Camera(n_rings=2, pixel_size=0.1)
    trigger = CameraTrigger(
        camera.pixel_x,
        camera.pixel_y,
        threshold_pe=5.0,
        window_ns=5.0,
        pixel_size=camera.pixel_size,
    )
    cluster = trigger.clusters[0]
    image = np.zeros(camera.n_pixels)
    timing = np.zeros(camera.n_pixels)
    image[cluster] = 6.0
    timing[cluster] = [0.0, 2.0, 9.0]

    assert trigger.evaluate(image, timing) == (False, None)


def test_unsupported_trigger_multiplicity_is_explicit():
    camera = Camera(n_rings=2)
    with pytest.raises(ValueError, match='three-pixel'):
        CameraTrigger(camera.pixel_x, camera.pixel_y, min_pixels=4)


def test_waveform_trigger_uses_first_discriminator_crossing():
    camera = Camera(n_rings=2, pixel_size=0.1)
    trigger = CameraTrigger(
        camera.pixel_x,
        camera.pixel_y,
        threshold_pe=5.0,
        window_ns=4.0,
        pixel_size=camera.pixel_size,
    )
    cluster = trigger.clusters[0]
    traces = np.zeros((camera.n_pixels, 8), dtype=np.float32)
    traces[cluster[0], 1] = 6.0
    traces[cluster[1], 2] = 6.0
    traces[cluster[2], 3] = 6.0

    triggered, time_ns = trigger.evaluate_traces(traces, bin_width_ns=2.0)
    assert triggered
    assert time_ns == pytest.approx(4.0)

    traces[cluster[2], :] = 0.0
    traces[cluster[2], 5] = 6.0
    assert trigger.evaluate_traces(traces, bin_width_ns=2.0) == (False, None)


def _find_non_adjacent_triple(trigger: CameraTrigger) -> tuple[int, int, int]:
    adj = trigger.adj_matrix.numpy()
    n_pix = adj.shape[0]
    for i in range(n_pix):
        for j in range(i + 1, n_pix):
            if adj[i, j]:
                continue
            for k in range(j + 1, n_pix):
                if not adj[i, k] and not adj[j, k]:
                    return i, j, k
    raise AssertionError("Expected at least three mutually non-adjacent pixels")


def test_sliding_sum_triggers_on_compact_triangle_and_rejects_scattered_or_late():
    camera = Camera(n_rings=2, pixel_size=0.1)
    trigger = CameraTrigger(
        camera.pixel_x,
        camera.pixel_y,
        threshold_pe=8.0,
        window_ns=5.0,
        pixel_size=camera.pixel_size,
    )
    cluster = trigger.clusters[0]

    # Compact 3-pixel triangle with 9 p.e. spread over 3 bins ([3, 3, 3]) within 5 ns triggers
    traces = np.zeros((camera.n_pixels, 16), dtype=np.float32)
    traces[cluster[0], 2:5] = 3.0  # first crossing at window 2 -> 4.0 ns
    traces[cluster[1], 3:6] = 3.0  # first crossing at window 3 -> 6.0 ns
    traces[cluster[2], 4:7] = 3.0  # first crossing at window 4 -> 8.0 ns

    triggered, t0 = trigger.evaluate_traces_sliding_sum(
        traces, bin_width_ns=2.0, integration_bins=3
    )
    assert triggered is True
    assert t0 == pytest.approx(6.0)

    # 3 non-adjacent pixels with 10 p.e. each do NOT trigger
    p0, p1, p2 = _find_non_adjacent_triple(trigger)
    scattered = np.zeros((camera.n_pixels, 16), dtype=np.float32)
    scattered[[p0, p1, p2], 4] = 10.0
    assert trigger.evaluate_traces_sliding_sum(
        scattered, bin_width_ns=2.0, integration_bins=3
    ) == (False, None)

    # 3 adjacent pixels separated by > 5 ns do NOT trigger
    late = np.zeros((camera.n_pixels, 16), dtype=np.float32)
    late[cluster[0], 1:4] = 3.0  # window 1 -> 2.0 ns
    late[cluster[1], 2:5] = 3.0  # window 2 -> 4.0 ns
    late[cluster[2], 5:8] = 3.0  # window 5 -> 10.0 ns (spread = 8.0 ns > 5.0 ns)
    assert trigger.evaluate_traces_sliding_sum(
        late, bin_width_ns=2.0, integration_bins=3
    ) == (False, None)

    # Short trace with n_time_bins <= integration_bins uses full-trace sum
    short_traces = np.zeros((camera.n_pixels, 3), dtype=np.float32)
    short_traces[cluster, :] = 3.0
    short_trig, short_t0 = trigger.evaluate_traces_sliding_sum(
        short_traces, bin_width_ns=2.0, integration_bins=3
    )
    assert short_trig is True
    assert short_t0 == pytest.approx(0.0)


def test_evaluate_array_trigger_compact_scattered_and_low_gain():
    camera = Camera(n_rings=2, pixel_size=0.1)
    cam_trig = CameraTrigger(
        camera.pixel_x,
        camera.pixel_y,
        threshold_pe=8.0,
        window_ns=5.0,
        pixel_size=camera.pixel_size,
    )
    cluster = cam_trig.clusters[0]
    p0, p1, p2 = _find_non_adjacent_triple(cam_trig)

    # 1. Two telescopes with compact 3-pixel clusters trigger -> (True, 2)
    compact_trace = np.zeros((camera.n_pixels, 16), dtype=np.float32)
    compact_trace[cluster[0], 3:6] = 3.0
    compact_trace[cluster[1], 3:6] = 3.0
    compact_trace[cluster[2], 4:7] = 3.0
    high_gain = np.zeros(camera.n_pixels, dtype=np.float32)

    img_outputs_compact = [
        (compact_trace.copy(), high_gain.copy()),
        (compact_trace.copy(), high_gain.copy()),
    ]
    assert evaluate_array_trigger(
        img_outputs_compact,
        camera,
        threshold_pe=8.0,
        min_pixels=3,
        min_telescopes=2,
    ) == (True, 2)
    assert evaluate_array_trigger(
        img_outputs_compact,
        camera,
        threshold_pe=8.0,
        min_pixels=3,
        min_telescopes=2,
        camera_trigger=cam_trig,
    ) == (True, 2)

    # 2. Two telescopes with only scattered non-adjacent noise pixels do NOT trigger -> (False, 0)
    scattered_trace = np.zeros((camera.n_pixels, 16), dtype=np.float32)
    scattered_trace[[p0, p1, p2], 5] = 10.0
    img_outputs_scattered = [
        (scattered_trace.copy(), high_gain.copy()),
        (scattered_trace.copy(), high_gain.copy()),
    ]
    assert evaluate_array_trigger(
        img_outputs_scattered,
        camera,
        threshold_pe=8.0,
        min_pixels=3,
        min_telescopes=2,
        camera_trigger=cam_trig,
    ) == (False, 0)

    # 3. Low-gain saturated traces (gain_flags=1.0, trace=2.0 -> restored=20.0) trigger properly
    low_gain_trace = np.zeros((camera.n_pixels, 16), dtype=np.float32)
    low_gain_trace[cluster, 5] = 2.0
    low_gain_flags = np.zeros(camera.n_pixels, dtype=np.float32)
    low_gain_flags[cluster] = 1.0

    img_outputs_low_gain = [
        (low_gain_trace.copy(), low_gain_flags.copy()),
        (low_gain_trace.copy(), low_gain_flags.copy()),
    ]
    assert evaluate_array_trigger(
        img_outputs_low_gain,
        camera,
        threshold_pe=8.0,
        min_pixels=3,
        min_telescopes=2,
        low_gain_factor=10.0,
        camera_trigger=cam_trig,
    ) == (True, 2)
    # Without gain restoration flags, the raw 2.0 p.e. cluster stays below 8.0 p.e.
    assert evaluate_array_trigger(
        [(low_gain_trace.copy(), high_gain.copy()), (low_gain_trace.copy(), high_gain.copy())],
        camera,
        threshold_pe=8.0,
        min_pixels=3,
        min_telescopes=2,
        low_gain_factor=10.0,
        camera_trigger=cam_trig,
    ) == (False, 0)

    # 4. Custom min_pixels != 3 falls back gracefully to pixel-count threshold
    assert evaluate_array_trigger(
        img_outputs_scattered,
        camera,
        threshold_pe=8.0,
        min_pixels=2,
        min_telescopes=2,
    ) == (True, 2)

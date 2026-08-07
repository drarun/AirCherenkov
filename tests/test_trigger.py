import numpy as np
import pytest

from sim.camera import Camera
from sim.trigger import CameraTrigger


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

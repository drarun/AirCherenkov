import numpy as np
import pytest

from recon.cleaning import double_pass_clean, tail_cut_clean
from recon.hillas import compute_hillas


class GraphCamera:
    def __init__(self, pixel_x, pixel_y, edges, pixel_size=0.1):
        self.pixel_x = np.asarray(pixel_x, dtype=float)
        self.pixel_y = np.asarray(pixel_y, dtype=float)
        self.pixel_size = pixel_size
        self._neighbors = np.zeros(
            (self.pixel_x.size, self.pixel_x.size), dtype=bool
        )
        for left, right in edges:
            self._neighbors[left, right] = True
            self._neighbors[right, left] = True

    def get_neighbor_matrix(self):
        return self._neighbors.copy()


def test_tail_cut_keeps_picture_and_adjacent_boundary_pixels():
    camera = GraphCamera(
        pixel_x=[0.0, 0.1, 0.2, 0.3],
        pixel_y=[0.0, 0.0, 0.0, 0.0],
        edges=[(0, 1), (1, 2), (2, 3)],
    )
    image = np.array([6.0, 7.0, 3.0, 3.0])

    mask = tail_cut_clean(camera, image)

    # Pixel 2 is a boundary neighbor of the core; pixel 3 is not adjacent to
    # a picture pixel and is therefore excluded.
    np.testing.assert_array_equal(mask, [True, True, True, False])


def test_double_pass_requires_core_connectivity_and_finite_axis_extent():
    camera = GraphCamera(
        pixel_x=[-0.1, 0.0, 0.1, 0.2, 0.3, -0.05, 0.05, 0.4, 0.5],
        pixel_y=[0.0, 0.0, 0.0, 0.0, 0.0, 0.05, 0.05, 0.0, 0.0],
        edges=[
            (0, 1),
            (1, 2),
            (2, 3),
            (3, 4),
            (5, 6),  # Aligned with the core, but spatially disconnected.
            (4, 7),
            (7, 8),  # Connected chain extending beyond the finite envelope.
        ],
    )
    image = np.array([10.0, 10.0, 10.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0])

    mask = double_pass_clean(
        camera,
        image,
        dist_tolerance=0.15,
        longitudinal_tolerance=0.2,
    )

    np.testing.assert_array_equal(
        mask,
        [True, True, True, True, True, False, False, False, False],
    )


@pytest.mark.parametrize(
    ("image", "match"),
    [
        (np.array([6.0, np.nan, 6.0]), "finite"),
        (np.array([6.0, np.inf, 6.0]), "finite"),
    ],
)
def test_cleaning_rejects_invalid_images(image, match):
    camera = GraphCamera(
        pixel_x=[0.0, 0.1, 0.2],
        pixel_y=[0.0, 0.0, 0.0],
        edges=[(0, 1), (1, 2)],
    )
    with pytest.raises(ValueError, match=match):
        tail_cut_clean(camera, image)
    with pytest.raises(ValueError, match=match):
        double_pass_clean(camera, image)


def test_cleaning_accepts_negative_pedestal_fluctuations():
    camera = GraphCamera(
        pixel_x=[0.0, 0.1, 0.2, 0.3],
        pixel_y=[0.0, 0.0, 0.0, 0.0],
        edges=[(0, 1), (1, 2), (2, 3)],
    )
    pedestal_subtracted = np.array([6.0, 7.0, 3.0, -1.25])
    zero_background = np.array([6.0, 7.0, 3.0, 0.0])

    np.testing.assert_array_equal(
        tail_cut_clean(camera, pedestal_subtracted),
        tail_cut_clean(camera, zero_background),
    )
    np.testing.assert_array_equal(
        double_pass_clean(camera, pedestal_subtracted),
        double_pass_clean(camera, zero_background),
    )


def test_signed_pedestal_image_flows_from_cleaning_into_hillas():
    camera = GraphCamera(
        pixel_x=[-0.2, -0.1, 0.0, 0.1, 0.2],
        pixel_y=[0.0, 0.0, 0.0, 0.0, 0.0],
        edges=[(0, 1), (1, 2), (2, 3), (3, 4)],
    )
    image = np.array([-0.5, 6.0, 7.0, 6.0, -0.25])

    mask = tail_cut_clean(camera, image)
    parameters = compute_hillas(camera, image, mask)

    np.testing.assert_array_equal(mask, [False, True, True, True, False])
    assert parameters is not None
    assert parameters.size == pytest.approx(19.0)


def test_cleaning_rejects_invalid_parameters_and_geometry():
    camera = GraphCamera(
        pixel_x=[0.0, 0.1, 0.2],
        pixel_y=[0.0, 0.0, 0.0],
        edges=[(0, 1), (1, 2)],
    )
    with pytest.raises(ValueError, match="picture_thresh"):
        tail_cut_clean(camera, np.ones(3), picture_thresh=np.nan)
    with pytest.raises(ValueError, match="min_neighbors"):
        tail_cut_clean(camera, np.ones(3), min_neighbors=1.5)
    with pytest.raises(ValueError, match="longitudinal_tolerance"):
        double_pass_clean(camera, np.full(3, 10.0), longitudinal_tolerance=np.inf)

    camera.pixel_x[1] = np.nan
    with pytest.raises(ValueError, match="coordinates"):
        double_pass_clean(camera, np.full(3, 10.0))


def test_double_pass_fadc_clean_recovers_time_gradient_and_faint_tail():
    from recon.cleaning import double_pass_fadc_clean

    camera = GraphCamera(
        pixel_x=[0.1, 0.25, 0.4, 0.55, 0.7],
        pixel_y=[0.0, 0.0, 0.0, 0.0, 0.0],
        edges=[(0, 1), (1, 2), (2, 3), (3, 4)],
        pixel_size=0.15,
    )
    # 5 pixels along +x axis with a linear arrival time gradient of +4 bins per step
    # (bin_width = 2.0 ns -> +8 ns per 0.15 deg = +53.3 ns/deg -> clamped to +40 ns/deg,
    # or let's use +1 bin = +2 ns per 0.15 deg = +13.33 ns/deg)
    traces = np.zeros((5, 16), dtype=np.float32)
    traces[0, 4] = 12.0
    traces[1, 5] = 15.0
    traces[2, 6] = 12.0
    # Faint tail pixels at bins 7 and 8 (above Pass-2 threshold)
    traces[3, 7] = 2.6
    traces[4, 8] = 1.8

    dp = double_pass_fadc_clean(traces, camera, nsb_rate=0.0)
    assert np.all(dp.mask[:4])
    assert dp.time_gradient == pytest.approx(2.0 / 0.15, rel=0.15)
    assert dp.time_rms < 0.5
    assert 0.0 <= dp.loss1 <= 1.0
    assert dp.loss2 >= dp.loss1

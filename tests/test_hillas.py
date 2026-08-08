import numpy as np
import pytest

from recon.hillas import compute_hillas, HillasParameters
from sim.camera import Camera


class ArrayCamera:
    def __init__(self, pixel_x, pixel_y):
        self.pixel_x = np.asarray(pixel_x, dtype=float)
        self.pixel_y = np.asarray(pixel_y, dtype=float)


def test_compute_hillas():
    cam = Camera(n_rings=5, pixel_size=0.1)
    image = np.zeros(cam.n_pixels)
    mask = np.zeros(cam.n_pixels, dtype=bool)

    # Empty mask should return None
    assert compute_hillas(cam, image, mask) is None

    # Activate a few pixels in a row
    mask[0:3] = True
    image[0:3] = 100.0

    params = compute_hillas(cam, image, mask)
    assert params is not None
    assert isinstance(params, HillasParameters)
    assert params.size == 300.0


@pytest.mark.parametrize("radial_sign", [1.0, -1.0])
def test_hillas_matches_analytic_rotated_ellipse(radial_sign):
    """The reported directed axis and moments match a separable ellipse."""
    theta = 0.61
    direction = np.array([np.cos(theta), np.sin(theta)])
    transverse = np.array([-np.sin(theta), np.cos(theta)])

    raw_longitudinal = np.array([-2.0, -1.0, 0.0, 1.0, 3.0])
    longitudinal_weights = np.array([1.0, 2.0, 4.0, 2.0, 1.0])
    raw_longitudinal -= np.average(
        raw_longitudinal, weights=longitudinal_weights
    )
    transverse_positions = np.array([-0.5, 0.0, 0.5])
    transverse_weights = np.array([1.0, 2.0, 1.0])

    longitudinal, crosswise = np.meshgrid(
        raw_longitudinal, transverse_positions, indexing="ij"
    )
    image = (
        longitudinal_weights[:, None] * transverse_weights[None, :]
    ).ravel()
    centroid = radial_sign * 1.4 * direction
    coordinates = (
        centroid[:, None]
        + direction[:, None] * longitudinal.ravel()
        + transverse[:, None] * crosswise.ravel()
    )
    camera = ArrayCamera(coordinates[0], coordinates[1])

    params = compute_hillas(camera, image, np.ones(image.size, dtype=bool))

    expected_direction = radial_sign * direction
    reported_direction = np.array([np.cos(params.psi), np.sin(params.psi)])
    projected = (
        (coordinates - centroid[:, None]).T @ expected_direction
    )
    expected_length = np.sqrt(np.average(projected**2, weights=image))
    expected_third_moment = np.average(projected**3, weights=image)
    expected_skewness = expected_third_moment / expected_length**3

    np.testing.assert_allclose(reported_direction, expected_direction, atol=1e-12)
    assert params.length == pytest.approx(expected_length)
    assert params.third_moment == pytest.approx(expected_third_moment)
    assert params.asymmetry == pytest.approx(expected_skewness)
    assert params.skewness == pytest.approx(expected_skewness)
    assert params.miss == pytest.approx(0.0, abs=1e-12)
    assert params.alpha == pytest.approx(0.0, abs=1e-12)


@pytest.mark.parametrize(
    ("image", "match"),
    [
        (np.array([1.0, np.nan, 2.0]), "finite"),
        (np.array([1.0, np.inf, 2.0]), "finite"),
    ],
)
def test_hillas_rejects_invalid_images(image, match):
    camera = ArrayCamera([0.0, 1.0, 2.0], [0.0, 0.0, 0.0])
    with pytest.raises(ValueError, match=match):
        compute_hillas(camera, image, np.ones(3, dtype=bool))


@pytest.mark.parametrize(
    "mask",
    [
        np.array([1.0, np.nan, 0.0]),
        np.array([1, 2, 0]),
        np.array([True, False]),
    ],
)
def test_hillas_rejects_invalid_masks(mask):
    camera = ArrayCamera([0.0, 1.0, 2.0], [0.0, 0.0, 0.0])
    with pytest.raises(ValueError, match="mask"):
        compute_hillas(camera, np.ones(3), mask)


def test_hillas_requires_three_positive_signal_pixels():
    camera = ArrayCamera([0.0, 1.0, 2.0], [0.0, 0.0, 0.0])
    assert compute_hillas(camera, np.array([1.0, 1.0, 0.0]), np.ones(3)) is None


def test_hillas_accepts_negative_pedestal_fluctuations_outside_mask():
    camera = ArrayCamera(
        [-1.0, 0.0, 1.0, 2.0],
        [0.0, 0.0, 0.0, 1.0],
    )
    mask = np.array([True, True, True, False])

    with_negative_background = compute_hillas(
        camera, np.array([2.0, 4.0, 2.0, -0.75]), mask
    )
    with_zero_background = compute_hillas(
        camera, np.array([2.0, 4.0, 2.0, 0.0]), mask
    )

    assert with_negative_background is not None
    assert with_zero_background is not None
    assert with_negative_background == with_zero_background


def test_hillas_rejects_negative_charge_inside_cleaned_mask():
    camera = ArrayCamera([0.0, 1.0, 2.0], [0.0, 0.0, 0.0])

    with pytest.raises(ValueError, match="selected.*nonnegative"):
        compute_hillas(camera, np.array([1.0, -0.1, 2.0]), np.ones(3, dtype=bool))

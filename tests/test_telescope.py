import numpy as np
from sim.telescope import Telescope
from sim.camera import Camera

def test_telescope_initialization():
    tel = Telescope(mirror_radius=5.0)
    assert tel.mirror_radius == 5.0
    assert tel.camera.n_rings == 15

def test_ray_trace_empty():
    tel = Telescope()
    cherenkov_photons = {
        'x_ground': np.array([]), 'y_ground': np.array([])
    }
    image = tel.ray_trace(cherenkov_photons)
    assert image.shape == (tel.camera.n_pixels,)

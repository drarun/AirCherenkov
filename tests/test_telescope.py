import numpy as np
from sim.telescope import Telescope, VeritasTelescope, TelescopeArray
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
    image, timing = tel.ray_trace(cherenkov_photons)
    assert image.shape == (tel.camera.n_pixels,)
    assert timing.shape == (tel.camera.n_pixels,)

def test_veritas_telescope():
    tel = VeritasTelescope(x_tel=10.0, y_tel=20.0)
    assert tel.x_tel == 10.0
    assert tel.y_tel == 20.0
    assert tel.mirror_radius == 6.0
    assert tel.focal_length == 12.0
    assert tel.camera.n_rings == 12

def test_telescope_array():
    array = TelescopeArray.veritas_array()
    assert len(array.telescopes) == 4
    
    cherenkov_photons = {
        'x_ground': np.array([]), 'y_ground': np.array([])
    }
    images_and_timings = array.ray_trace(cherenkov_photons)
    assert len(images_and_timings) == 4
    for i, (img, timing) in enumerate(images_and_timings):
        assert img.shape == (array.telescopes[i].camera.n_pixels,)
        assert timing.shape == (array.telescopes[i].camera.n_pixels,)

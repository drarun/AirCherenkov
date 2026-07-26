import numpy as np
from recon.hillas import compute_hillas, HillasParameters
from sim.camera import Camera

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

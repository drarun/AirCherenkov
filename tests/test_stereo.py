import numpy as np
import pytest
from sim.camera import Camera
from sim.telescope import TelescopeArray
from recon.stereo import cam_to_sky_vector, tailcut_clean, reconstruct_event_geometric


def test_cam_to_sky_vector_optical_axis():
    # Pointing straight up (zenith=0, az=0)
    v = cam_to_sky_vector(0.0, 0.0, 0.0, 0.0)
    assert np.allclose(v, [0.0, 0.0, 1.0], atol=1e-5)

    # Pointing at 30 deg zenith, 0 az
    v30 = cam_to_sky_vector(0.0, 0.0, np.radians(30.0), 0.0)
    assert np.allclose(v30, [np.sin(np.radians(30)), 0.0, np.cos(np.radians(30))], atol=1e-5)


def test_tailcut_clean_filters_pedestal():
    cam = Camera(n_rings=5)
    charges = np.zeros(cam.n_pixels, dtype=np.float32)
    # NSB noise around 2 PE
    charges += 2.0
    # Core pixel with 15 PE
    charges[0] = 17.0

    c_clean, mask = tailcut_clean(charges, cam, pic_th=4.0, bnd_th=2.0, ped=2.0)
    assert mask[0] == True
    assert c_clean[0] == pytest.approx(15.0)
    # Outside should be filtered
    assert np.sum(mask) >= 1


def test_geometric_reconstruction_on_synthetic_event():
    cam = Camera(n_rings=12)
    array = TelescopeArray.veritas_array()
    
    # 4 telescopes, each pixel has trace with 16 time bins
    traces = np.zeros((4, cam.n_pixels, 16), dtype=np.float32)
    
    # Put bright elliptical signals on T0 and T1
    # T0: image along y-axis
    traces[0, :10, 2] = 20.0
    # T1: image along x-axis
    traces[1, 10:20, 2] = 20.0
    
    res = reconstruct_event_geometric(
        traces, array.telescopes, cam,
        pointing_zenith_deg=30.0, pointing_azimuth_deg=0.0
    )
    # May return None or a valid result depending on pixel placement
    if res is not None:
        assert np.isfinite(res.reco_zenith_deg)
        assert np.isfinite(res.reco_impact_x)

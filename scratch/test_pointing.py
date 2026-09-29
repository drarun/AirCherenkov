import torch
import numpy as np
from sim.shower import ShowerSimulation
from sim.telescope import TelescopeArray
from sim.backend import get_device

def test():
    zenith_deg = 30.0
    azimuth_deg = 45.0
    wobble = 0.5
    
    zen_rad = np.radians(zenith_deg)
    azi_rad = np.radians(azimuth_deg)
    
    # Wobble offset
    theta_offset = np.radians(wobble)
    phi_offset = np.pi / 2 # Test another point on the wobble ring
    evt_zen = zen_rad + theta_offset * np.cos(phi_offset)
    evt_azi = azi_rad + theta_offset * np.sin(phi_offset)
    
    px = np.sin(evt_zen) * np.cos(evt_azi)
    py = np.sin(evt_zen) * np.sin(evt_azi)
    pz = -np.cos(evt_zen)
    
    z_start = 25000.0
    z_obs = 1275.0
    x_init = (z_start - z_obs) * (px / pz)
    y_init = (z_start - z_obs) * (py / pz)
    
    print(f"Shower start: {x_init:.1f}, {y_init:.1f}, {z_start}")
    print(f"Shower dir: {px:.3f}, {py:.3f}, {pz:.3f}")
    
    sim = ShowerSimulation(
        primary_types=['gamma'], energies=[1000.0], z_starts=[z_start],
        x_init=[x_init], y_init=[y_init],
        px_init=[px], py_init=[py], pz_init=[pz]
    )
    sim.run(max_generations=30, verbose=True)
    
    photons = sim.cherenkov_photons_by_event.get(0, {})
    print(f"Total Cherenkov photons: {len(photons.get('x_emit', []))}")
    
    print("Keys in cherenkov_photons_by_event:", sim.cherenkov_photons_by_event.keys())
    
    x_g = photons.get('x_ground', [])
    if len(x_g) > 0:
        x_g = np.array(x_g)
        y_g = np.array(photons['y_ground'])
        print(f"Photon ground X mean: {x_g.mean():.1f}, std: {x_g.std():.1f}")
        print(f"Photon ground Y mean: {y_g.mean():.1f}, std: {y_g.std():.1f}")
        r_g = np.sqrt((x_g - x_g.mean())**2 + (y_g - y_g.mean())**2)
        print(f"Photon ground R mean: {r_g.mean():.1f}, std: {r_g.std():.1f}, min: {r_g.min():.1f}, max: {r_g.max():.1f}")
        
        hist, bins = np.histogram(r_g, bins=20, range=(0, 1000))
        print("Radial distribution (0-1000m, bins of 50m):")
        print(hist)
    
    # Create a grid of telescopes
    telescopes = []
    import copy
    array = TelescopeArray.veritas_array()
    base_tel = array.telescopes[0]
    tel = copy.deepcopy(base_tel)
    tel.x_tel = 80.0
    tel.y_tel = 0.0
    telescopes.append(tel)
    array.telescopes = telescopes
        
    results = array.ray_trace(
        photons,
        pointing_zenith=zenith_deg,
        pointing_azimuth=azimuth_deg + 180.0,
        shower_start_altitude=z_start
    )
    
    print("Ray trace completed.")
    for i, (trace, gain) in enumerate(results):
        max_pe = trace.max()
        if max_pe > 5.0:
            print(f"Telescope {i} at ({telescopes[i].x_tel:.1f}, {telescopes[i].y_tel:.1f}) PE: {trace.sum():.1f}, Max: {max_pe:.1f}, Min: {trace.min():.1f}, Mean: {trace.mean():.1f}")
        
if __name__ == '__main__':
    test()

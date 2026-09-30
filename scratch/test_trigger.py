import torch
import numpy as np
from sim.shower import ShowerSimulation
from sim.telescope import TelescopeArray
from sim.backend import get_device

array = TelescopeArray.veritas_array()
batch_size = 10
sim = ShowerSimulation(
    primary_types=['gamma']*batch_size, energies=[1000.0]*batch_size,
    z_starts=[25000.0]*batch_size,
    x_init=[(25000.0 - 1275.0) * (0.354 / -0.866)]*batch_size,
    y_init=[(25000.0 - 1275.0) * (0.354 / -0.866)]*batch_size,
    px_init=[0.354]*batch_size, py_init=[0.354]*batch_size, pz_init=[-0.866]*batch_size
)
sim.run(max_generations=30, verbose=False)

for i in range(batch_size):
    photons = sim.cherenkov_photons_by_event.get(i, {})
    if len(photons.get('x_ground', [])) == 0: continue
    
    r = np.sqrt(np.random.rand()) * 150.0
    theta = np.random.rand() * 2 * np.pi
    ix = r * np.cos(theta)
    iy = r * np.sin(theta)
    
    for tel in array.telescopes:
        tel.x_tel -= ix
        tel.y_tel -= iy
        
    img_outputs = array.ray_trace(photons, pointing_zenith=30.0, pointing_azimuth=225.0, shower_start_altitude=25000.0)
    
    for tel in array.telescopes:
        tel.x_tel += ix
        tel.y_tel += iy
        
    trigger_count = 0
    tels_triggered = []
    for t_idx, (trace, gain) in enumerate(img_outputs):
        total_pe = np.sum(trace, axis=1)
        pixels_above_8 = np.sum(total_pe > 8.0)
        if pixels_above_8 >= 3:
            trigger_count += 1
            tels_triggered.append(t_idx)
        print(f"Ev {i} Tel {t_idx} max sum: {np.max(total_pe):.1f}, pixels>8: {pixels_above_8}")
    print(f"Ev {i} trigger_count: {trigger_count} (Tels: {tels_triggered})\n")

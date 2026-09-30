"""
Diagnose the FADC timing: check what arrival time span we get after the slant-path fix.
"""
import sys
sys.path.insert(0, 'src')
import numpy as np

zen_rad = np.radians(30.0 + 0.5)
px_d = np.sin(zen_rad); py_d = 0.0; pz_d = -np.cos(zen_rad)

z_start = 25000.0; z_obs = 1275.0
ix, iy = 70.0, 30.0
x_init = ix + (z_start - z_obs) * (px_d / pz_d)
y_init = iy

from sim.shower import ShowerSimulation
sim = ShowerSimulation(
    primary_types=['gamma'], energies=[1000.0], z_starts=[z_start],
    x_init=[x_init], y_init=[y_init],
    px_init=[px_d], py_init=[py_d], pz_init=[pz_d]
)
sim.run(max_generations=25, verbose=False)
photons = sim.cherenkov_photons_by_event[0]
print(f"Cherenkov packets: {len(photons['x_ground'])}")

ze = photons['z_emit']
pz_e = photons['pz_emit']
xg = photons['x_ground']
yg = photons['y_ground']

# Manually compute emission time as in backend.py (slant-path)
pz_abs = np.clip(np.abs(pz_e), 0.05, None)
slant_distance = (z_start - ze) / pz_abs
emission_time = slant_distance / 0.299792458  # ns

print(f"\nEmission time (ns):")
print(f"  min={emission_time.min():.1f}, max={emission_time.max():.1f}, mean={emission_time.mean():.1f}")
print(f"  span = {emission_time.max() - emission_time.min():.1f} ns")

# Now compute travel time from emission point to Tel 0 ground position
tel0_x, tel0_y, tel0_z = 0.0, 0.0, 1268.0

# Filter photons that hit Tel 0 area (rough: within 6m mirror)
dist_to_tel0 = np.sqrt((xg - tel0_x)**2 + (yg - tel0_y)**2)
near_tel0 = dist_to_tel0 < 6.0
print(f"\nPhotons near Tel 0 (within 6m mirror): {near_tel0.sum()}")

if near_tel0.sum() > 0:
    em_t = emission_time[near_tel0]
    z_em = ze[near_tel0]
    
    # Photon travel time from emission to ground level
    photon_dir_z = -1.0  # photons going downward
    photon_travel_dist = (z_em - tel0_z)  # vertical drop to tel altitude
    photon_travel_time = photon_travel_dist / 0.299792458  # ns (in vacuum)

    # Total arrival time (relative to first wavefront)
    arrival_time = em_t + photon_travel_time
    arrival_time -= arrival_time.min()  # normalize
    
    print(f"Emission time for Tel 0 photons: [{em_t.min():.1f}, {em_t.max():.1f}] ns, span={em_t.max()-em_t.min():.1f} ns")
    print(f"Arrival time span at Tel 0: {arrival_time.max() - arrival_time.min():.1f} ns")

    # What FADC bin would these land in?
    # The backend shifts by reference_time (min emission time of all packets)
    ref_time = emission_time.min()
    relative_em = em_t - ref_time
    print(f"Relative emission time for Tel 0 photons: [{relative_em.min():.1f}, {relative_em.max():.1f}] ns")
    fadc_bin = (relative_em / 2.0).astype(int)  # 2 ns bins
    print(f"FADC bins for Tel 0 photons: min={fadc_bin.min()}, max={fadc_bin.max()}")

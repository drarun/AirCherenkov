"""
Check photon weight distribution by altitude to see if Frank-Tamm yield
is concentrating light near shower max.
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
N = len(photons['x_ground'])
print(f"Total packets: {N}")

ze = photons['z_emit']
w  = photons['weight']

print(f"\nPhoton WEIGHT distribution by altitude (not just count):")
total_w = w.sum()
print(f"Total weight: {total_w:.2f}")
bins = [(z_obs, 2000), (2000, 4000), (4000, 6000), (6000, 8000), (8000, 12000), (12000, 25000)]
for lo, hi in bins:
    mask = (ze >= lo) & (ze < hi)
    w_frac = w[mask].sum() / total_w * 100
    n_frac = mask.sum() / N * 100
    print(f"  {lo:5.0f}-{hi:5.0f}m: {n_frac:5.1f}% packets, {w_frac:5.1f}% weight")

# Now compute the expected arrival time span for photons hitting Tel 3
tel3_x, tel3_y, tel3_z = -41.3, 53.9, 1268.0
xg = photons['x_ground']; yg = photons['y_ground']
dist_to_tel3 = np.sqrt((xg - tel3_x)**2 + (yg - tel3_y)**2)
near_tel3 = dist_to_tel3 < 7.0
print(f"\nPackets near Tel 3 (within 7m): {near_tel3.sum()}")
if near_tel3.sum() > 0:
    ze_tel3 = ze[near_tel3]
    w_tel3 = w[near_tel3]
    print(f"  z_emit range: [{ze_tel3.min():.0f}, {ze_tel3.max():.0f}] m")
    print(f"  z_emit mean (weighted): {np.average(ze_tel3, weights=w_tel3):.0f} m")

    # Compute arrival time using shower-axis slant timing
    pz_shower = np.cos(np.radians(30.0))  # cos(30deg) = 0.866
    emission_time = (z_start - ze_tel3) / pz_shower / 0.299792458  # ns

    # Photon travel time from emission point to Tel 3
    xe_t3 = photons['x_emit'][near_tel3]
    ye_t3 = photons['y_emit'][near_tel3]
    photon_dist = np.sqrt((xe_t3 - tel3_x)**2 + (ye_t3 - tel3_y)**2 + (ze_tel3 - tel3_z)**2)
    travel_time = photon_dist / 0.299792458

    arrival_time = emission_time + travel_time
    arrival_time -= arrival_time.min()

    print(f"\n  Arrival time at Tel 3:")
    print(f"  Total span: {arrival_time.max() - arrival_time.min():.1f} ns")
    
    # Weighted arrival time span (contains 80% of weight)
    sorted_idx = np.argsort(arrival_time)
    w_sorted = w_tel3[sorted_idx]
    cumw = np.cumsum(w_sorted)
    cumw /= cumw[-1]
    t_sorted = arrival_time[sorted_idx]
    t10 = t_sorted[np.searchsorted(cumw, 0.10)]
    t90 = t_sorted[np.searchsorted(cumw, 0.90)]
    print(f"  80% containment time window: {t10:.1f} - {t90:.1f} ns (span: {t90-t10:.1f} ns)")

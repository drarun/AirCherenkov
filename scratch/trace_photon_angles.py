import torch
import numpy as np
import matplotlib.pyplot as plt
from sim.shower import ShowerSimulation
from sim.telescope import TelescopeArray

array = TelescopeArray.veritas_array()
# 1 event, gamma, 1000 GeV, 30 deg zenith, 0 deg azimuth
zenith_deg = 30.0
azimuth_deg = 0.0
wobble = 0.5
zen_rad = np.radians(zenith_deg)
azi_rad = np.radians(azimuth_deg)

# wobble offset 0.5 deg along phi=0
wob_rad = np.radians(wobble)
evt_zen = zen_rad + wob_rad
evt_azi = azi_rad
px = np.sin(evt_zen) * np.cos(evt_azi)
py = np.sin(evt_zen) * np.sin(evt_azi)
pz = -np.cos(evt_zen)

z_start = 25000.0
z_obs = 1275.0
x_init = (z_start - z_obs) * (px / pz)
y_init = (z_start - z_obs) * (py / pz)

sim = ShowerSimulation(
    primary_types=['gamma'], energies=[1000.0], z_starts=[z_start],
    x_init=[x_init], y_init=[y_init],
    px_init=[px], py_init=[py], pz_init=[pz]
)
sim.run(max_generations=30, verbose=False)

photons = sim.cherenkov_photons_by_event[0]
print(f"Total Cherenkov packets: {len(photons['x_ground'])}")

# Check emission altitudes
ze = photons['z_emit']
weights = photons['weight']
print(f"Emission altitude ze: min={np.min(ze):.1f}m, max={np.max(ze):.1f}m, median={np.median(ze):.1f}m")

# Check where photons hit the ground
xg = photons['x_ground']
yg = photons['y_ground']
print(f"Ground x: mean={np.mean(xg):.1f}m, std={np.std(xg):.1f}m")
print(f"Ground y: mean={np.mean(yg):.1f}m, std={np.std(yg):.1f}m")

# Now check photons hitting Telescope 0 (at x=0, y=0, z=1275 or Veritas pos)
tel0 = array.telescopes[0]
print(f"Tel 0 pos: ({tel0.x_tel}, {tel0.y_tel}, {tel0.z_tel}), mirror radius: {tel0.mirror_radius}")

# Ray trace manually for Tel 0 to see u_deg, v_deg for all photons hitting mirror
c_xe = photons['x_emit']
c_ye = photons['y_emit']
c_ze = photons['z_emit']
c_xg = photons['x_ground']
c_yg = photons['y_ground']

# Project to tel_z = 1275
denom = -c_ze
proj = (tel0.z_tel - c_ze) / denom
x_hit = c_xe + proj * (c_xg - c_xe)
y_hit = c_ye + proj * (c_yg - c_ye)

dist_sq = (x_hit - tel0.x_tel)**2 + (y_hit - tel0.y_tel)**2
mirror_hit = (dist_sq <= tel0.mirror_radius**2) & (c_ze > tel0.z_tel)
n_hits = np.sum(mirror_hit)
print(f"Photons hitting Tel 0 mirror: {n_hits}")

if n_hits > 0:
    h_x = x_hit[mirror_hit]
    h_y = y_hit[mirror_hit]
    h_z = tel0.z_tel
    e_x = c_xe[mirror_hit]
    e_y = c_ye[mirror_hit]
    e_z = c_ze[mirror_hit]
    
    vx = -(h_x - e_x)
    vy = -(h_y - e_y)
    vz = -(h_z - e_z)
    
    # Telescope pointing: zenith=30, azimuth=180
    theta = np.radians(zenith_deg)
    phi = np.radians(azimuth_deg + 180.0)
    
    # Rotate around Z by -phi
    x_prime = vx * np.cos(phi) + vy * np.sin(phi)
    y_prime = -vx * np.sin(phi) + vy * np.cos(phi)
    z_prime = vz
    
    # Rotate around Y by -theta
    x_double_prime = x_prime * np.cos(theta) - z_prime * np.sin(theta)
    y_double_prime = y_prime
    z_double_prime = x_prime * np.sin(theta) + z_prime * np.cos(theta)
    
    u_deg = np.rad2deg(np.arctan2(x_double_prime, z_double_prime))
    v_deg = np.rad2deg(np.arctan2(y_double_prime, z_double_prime))
    
    print(f"u_deg: min={np.min(u_deg):.3f}, max={np.max(u_deg):.3f}, mean={np.mean(u_deg):.3f}, std={np.std(u_deg):.3f}")
    print(f"v_deg: min={np.min(v_deg):.3f}, max={np.max(v_deg):.3f}, mean={np.mean(v_deg):.3f}, std={np.std(v_deg):.3f}")
    
    # Plot scatter of u_deg, v_deg
    plt.figure(figsize=(8, 8))
    plt.scatter(u_deg, v_deg, s=10, alpha=0.5, c=e_z, cmap='plasma')
    plt.colorbar(label='Emission altitude (m)')
    plt.xlabel('u (deg)')
    plt.ylabel('v (deg)')
    plt.title(f'Photon arrival angles on Tel 0 (N={n_hits})')
    # Draw camera boundary at 1.75 deg
    circ = plt.Circle((0, 0), 1.75, color='r', fill=False, ls='--', label='Camera FOV (1.75 deg)')
    plt.gca().add_patch(circ)
    plt.xlim(-2.5, 2.5)
    plt.ylim(-2.5, 2.5)
    plt.grid(True)
    plt.legend()
    plt.savefig('scratch/photon_arrival_angles.png')
    print("Saved scratch/photon_arrival_angles.png")

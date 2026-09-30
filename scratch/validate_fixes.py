"""
Quick validation script to test the three physics fixes in production shower.py:
1. Heitler symmetric splitting (z_emit altitude distribution)
2. Observatory altitude cutoff (no photons below 1275m)
3. Slant-path emission timing (check pz_emit is populated)
"""
import sys
sys.path.insert(0, 'src')

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sim.shower import ShowerSimulation
from sim.telescope import TelescopeArray
from sim.camera import Camera

# ---- Shower setup (same as test throw params) ----
zenith_deg = 30.0
azimuth_deg = 0.0
wobble = 0.5
zen_rad = np.radians(zenith_deg)
azi_rad = np.radians(azimuth_deg)
wob_rad = np.radians(wobble)
evt_zen = zen_rad + wob_rad
evt_azi = azi_rad
px_d = np.sin(evt_zen) * np.cos(evt_azi)
py_d = np.sin(evt_zen) * np.sin(evt_azi)
pz_d = -np.cos(evt_zen)

z_start = 25000.0
z_obs = 1275.0
ix, iy = 70.0, 30.0
x_init = ix + (z_start - z_obs) * (px_d / pz_d)
y_init = iy + (z_start - z_obs) * (py_d / pz_d)

print("Running production ShowerSimulation (with all 3 fixes)...")
sim = ShowerSimulation(
    primary_types=['gamma'], energies=[1000.0], z_starts=[z_start],
    x_init=[x_init], y_init=[y_init],
    px_init=[px_d], py_init=[py_d], pz_init=[pz_d]
)
sim.run(max_generations=25, verbose=True)
photons = sim.cherenkov_photons_by_event[0]
N = len(photons['x_ground'])
print(f"\nTotal Cherenkov packets: {N}")

if N == 0:
    print("ERROR: No photons produced!")
    sys.exit(1)

# ---- Check 1: Altitude distribution ----
ze = photons['z_emit']
print(f"\n--- Fix 1+2 check: z_emit altitude distribution ---")
print(f"z_emit: min={ze.min():.0f}m, max={ze.max():.0f}m, mean={ze.mean():.0f}m")
print(f"Below z_obs (<1275m): {(ze < z_obs).sum()} / {N} packets ({100*(ze<z_obs).mean():.1f}%)")
print(f"1275-4000m:           {((ze>=z_obs)&(ze<4000)).sum()} / {N} ({100*((ze>=z_obs)&(ze<4000)).mean():.1f}%)")
print(f"4000-8000m:           {((ze>=4000)&(ze<8000)).sum()} / {N} ({100*((ze>=4000)&(ze<8000)).mean():.1f}%)")
print(f"Above 8000m:          {(ze>=8000).sum()} / {N} ({100*(ze>=8000).mean():.1f}%)")

# ---- Check 2: pz_emit (Fix 3) ----
pz_e = photons['pz_emit']
print(f"\n--- Fix 3 check: pz_emit (slant-path timing) ---")
print(f"pz_emit: min={pz_e.min():.3f}, max={pz_e.max():.3f}, mean={pz_e.mean():.3f}")
print(f"Expected ~-0.866 for 30deg zenith shower")
print(f"|pz_emit| range: [{np.abs(pz_e).min():.3f}, {np.abs(pz_e).max():.3f}]")

# ---- Camera images ----
array = TelescopeArray.veritas_array()
cam = Camera(n_rings=12)

img_outputs = array.ray_trace(
    photons,
    pointing_zenith=zenith_deg,
    pointing_azimuth=azimuth_deg + 180.0,
    shower_start_altitude=z_start
)

fig, axes = plt.subplots(2, 4, figsize=(18, 8))
for t_idx in range(4):
    ax_img = axes[0, t_idx]
    ax_fadc = axes[1, t_idx]
    trace, gain = img_outputs[t_idx]  # (469, 16)

    # Camera image
    c = np.sum(trace, axis=1)  # (469,)
    c_clean = np.maximum(0, c - 2.0)  # subtract NSB pedestal
    threshold = max(5.0, np.percentile(c_clean[c_clean > 0], 70) if (c_clean > 0).any() else 5.0)
    mask = c_clean > threshold

    sc = ax_img.scatter(cam.pixel_x, cam.pixel_y, c=c_clean, cmap='viridis', s=25, alpha=0.9)
    if mask.sum() >= 3:
        ax_img.scatter(cam.pixel_x[mask], cam.pixel_y[mask],
                       facecolors='none', edgecolors='red', s=50, linewidth=1.5)
        ax_img.set_title(f"Tel {t_idx}: Max={c.max():.1f}PE, {mask.sum()} pix")
    else:
        ax_img.set_title(f"Tel {t_idx}: Max={c.max():.1f}PE (no clean)")
    ax_img.set_xlim(-2, 2)
    ax_img.set_ylim(-2, 2)
    ax_img.set_aspect('equal')

    # FADC trace for brightest pixel
    bright_pix = int(c.argmax())
    ax_fadc.plot(np.arange(16) * 2, trace[bright_pix], 'b-o', ms=4)
    ax_fadc.set_title(f"Tel {t_idx} pix {bright_pix} FADC ({c[bright_pix]:.1f}PE)")
    ax_fadc.set_xlabel("Time (ns)")
    ax_fadc.set_ylabel("PE/bin")

print(f"\nMax PE per telescope: {[round(float(np.sum(img_outputs[t][0], axis=1).max()), 1) for t in range(4)]}")

plt.tight_layout()
plt.savefig('scratch/validate_fixes.png', dpi=150)
print("\nSaved scratch/validate_fixes.png")

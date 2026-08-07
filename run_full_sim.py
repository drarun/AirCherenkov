import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import matplotlib.pyplot as plt
import numpy as np
from sim.shower import ShowerSimulation
from sim.telescope import Telescope
from sim.backend import device_info
from recon.cleaning import tail_cut_clean, double_pass_clean
from recon.hillas import compute_hillas, print_hillas

def main():
    print("=" * 60)
    print("AirCherenkov -- Full Simulation & Analysis Pipeline")
    print("=" * 60)
    print(f"Compute backend: {device_info()}")
    
    # -- Shower Simulation -------------------------------------------
    print("\n[1/4] Simulating a 5 TeV Gamma-ray shower...")
    sim = ShowerSimulation(primary_types=['gamma'], energies=[5000.0], z_starts=[25000.0])
    sim.photon_yield_factor = 50.0
    sim.run(max_generations=18)
    
    photons = sim.cherenkov_photons_by_event[0]
    n_photons = len(photons.get('x_ground', []))
    print(f"      Generated {n_photons:,} Cherenkov photons on ground.")
    
    # -- Ray-tracing -------------------------------------------------
    print("\n[2/4] Ray-tracing through telescope optics...")
    tel = Telescope(x_tel=0.0, y_tel=0.0, mirror_radius=6.0, focal_length=15.0)
    fadc_traces, gain_flags = tel.ray_trace(photons, nsb_rate=2.0)
    
    # Integrate FADC traces to get a charge image for cleaning/Hillas
    raw_image = np.sum(fadc_traces, axis=1)
    
    n_signal = np.sum(raw_image) - tel.camera.n_pixels * 2.0
    print(f"      Camera: {tel.camera.n_pixels} pixels, ~{n_signal:.0f} signal p.e. above NSB")
    
    # -- Image Cleaning ----------------------------------------------
    print("\n[3/4] Cleaning images...")
    
    mask_std = tail_cut_clean(tel.camera, raw_image, picture_thresh=5.0, boundary_thresh=2.5)
    img_std = np.where(mask_std, raw_image, 0.0)
    
    mask_dp = double_pass_clean(tel.camera, raw_image, 
                                 pic1=5.0, bnd1=2.5, pic2=2.0, bnd2=1.0, 
                                 dist_tolerance=0.2)
    img_dp = np.where(mask_dp, raw_image, 0.0)
    
    print(f"      Tail-cut: {np.sum(mask_std)} pixels survived")
    print(f"      Double-pass: {np.sum(mask_dp)} pixels survived")
    
    # -- Hillas Parameterization -------------------------------------
    print("\n[4/4] Computing Hillas parameters...")
    
    hillas_std = compute_hillas(tel.camera, raw_image, mask_std)
    hillas_dp  = compute_hillas(tel.camera, raw_image, mask_dp)
    
    print("\n-- Standard Tail-Cut --")
    print_hillas(hillas_std)
    
    print("\n-- Double-Pass --")
    print_hillas(hillas_dp)
    
    # -- Plotting ----------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(22, 7))
    
    tel.camera.plot_image(raw_image, ax=axes[0], title="Raw Image (+ NSB)", cmap="inferno")
    tel.camera.plot_image(img_std, ax=axes[1], title="Standard Tail-Cut (5/2.5)", cmap="inferno")
    tel.camera.plot_image(img_dp, ax=axes[2], title="Double-Pass (5/2.5 > 2/1.0)", cmap="inferno")
    
    # Draw the Hillas ellipse on the Double-Pass plot if available
    if hillas_dp is not None:
        ax = axes[2]
        from matplotlib.patches import Ellipse
        ellipse = Ellipse(
            xy=(hillas_dp.centroid_x, hillas_dp.centroid_y),
            width=2 * hillas_dp.length,
            height=2 * hillas_dp.width,
            angle=np.degrees(hillas_dp.psi),
            edgecolor='lime', facecolor='none', linewidth=2, linestyle='--'
        )
        ax.add_patch(ellipse)
        ax.plot(hillas_dp.centroid_x, hillas_dp.centroid_y, 'x', color='lime', 
                markersize=12, markeredgewidth=2)
    
    plt.suptitle("AirCherenkov: Gamma-ray Shower (5 TeV)", fontsize=14, fontweight='bold')
    plt.savefig("camera_cleaning_comparison.png", bbox_inches='tight', dpi=150)
    print("\nSaved: camera_cleaning_comparison.png")

if __name__ == "__main__":
    main()

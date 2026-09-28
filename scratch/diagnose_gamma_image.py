"""Quick diagnostic: simulate a single 1 TeV gamma and visualize the camera images."""
import sys, os, torch, numpy as np
sys.path.insert(0, 'src')

from sim.shower import ShowerSimulation
from sim.telescope import TelescopeArray
from sim.camera import Camera
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Simulate a single 1 TeV gamma at zenith
print("Simulating 1 TeV gamma shower...")
sim = ShowerSimulation(primary_types=['gamma'], energies=[1000.0], z_starts=[20000.0])
sim.run(max_generations=16, verbose=True)

photons = sim.cherenkov_photons_by_event.get(0, {})
n_photons = len(photons.get('x_ground', []))
print(f"\nTotal Cherenkov photon packets: {n_photons}")

if n_photons > 0:
    xg = photons['x_ground']
    yg = photons['y_ground']
    xe = photons['x_emit']
    ye = photons['y_emit']
    ze = photons['z_emit']
    w = photons.get('weight', np.ones(n_photons))
    
    print(f"Ground pool X range: {xg.min():.1f} to {xg.max():.1f} m")
    print(f"Ground pool Y range: {yg.min():.1f} to {yg.max():.1f} m")
    print(f"Ground pool radius: {np.sqrt(xg**2 + yg**2).max():.1f} m")
    print(f"Emission Z range: {ze.min():.1f} to {ze.max():.1f} m")
    print(f"Mean emission altitude: {ze.mean():.1f} m ({ze.mean()/1000:.1f} km)")
    print(f"Total weighted photon count: {w.sum():.0f}")
    
    # Now ray trace through the array
    array = TelescopeArray.veritas_array()
    print("\nRay tracing through VERITAS array...")
    img_outputs = array.ray_trace(photons)
    
    fig, axes = plt.subplots(2, 4, figsize=(24, 12))
    cam = Camera(n_rings=12, pixel_size=0.15)  # Must match VeritasTelescope
    
    for tel_idx, (trace, gain) in enumerate(img_outputs):
        # Integrated charge image
        charge = np.sum(trace, axis=1)
        timing = np.argmax(trace, axis=1) * 2.0
        peak_pe = np.max(trace, axis=1)
        
        print(f"\nTelescope {tel_idx+1}:")
        print(f"  Total charge: {charge.sum():.1f} PE")
        print(f"  Max pixel charge: {charge.max():.1f} PE")
        print(f"  Pixels > 5 PE: {np.sum(charge > 5)}")
        print(f"  Pixels > 10 PE: {np.sum(charge > 10)}")
        print(f"  Image size (sqrt(sum(q^2))): {np.sqrt(np.sum(charge[charge > 5]**2)):.1f} PE")
        
        # Plot charge image
        cam.plot_image(charge, ax=axes[0, tel_idx], 
                      title=f'Tel {tel_idx+1}: Charge (Total={charge.sum():.0f} PE)')
        
        # Plot timing image (only for pixels with signal)
        timing_display = np.where(charge > 3, timing, 0)
        cam.plot_image(timing_display, ax=axes[1, tel_idx],
                      title=f'Tel {tel_idx+1}: Timing (ns)', cmap='plasma')
    
    fig.suptitle('1 TeV Gamma Shower — VERITAS Camera Images', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    artifact_dir = r'C:\Users\aruns\.gemini\antigravity-cli\brain\f0e1b905-b5d2-4fce-80a1-d0620d2d7de7'
    save_path = os.path.join(artifact_dir, 'gamma_event_diagnostic.png')
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved diagnostic image to: {save_path}")
    plt.close()
    
    # Also plot the ground pool
    fig2, ax2 = plt.subplots(1, 1, figsize=(8, 8))
    # Subsample if too many photons
    idx = np.random.choice(len(xg), min(10000, len(xg)), replace=False)
    ax2.scatter(xg[idx], yg[idx], s=0.5, alpha=0.3, c='blue')
    for tel in array.telescopes:
        circle = plt.Circle((tel.x_tel, tel.y_tel), tel.mirror_radius, 
                           fill=False, color='red', linewidth=2)
        ax2.add_patch(circle)
        ax2.plot(tel.x_tel, tel.y_tel, 'r+', markersize=10)
    ax2.set_aspect('equal')
    ax2.set_xlabel('X (m)')
    ax2.set_ylabel('Y (m)')
    ax2.set_title('Cherenkov Light Pool on Ground (1 TeV Gamma)')
    ax2.set_xlim(-300, 300)
    ax2.set_ylim(-300, 300)
    ax2.grid(True, alpha=0.3)
    save_path2 = os.path.join(artifact_dir, 'ground_pool_diagnostic.png')
    fig2.savefig(save_path2, dpi=150, bbox_inches='tight')
    print(f"Saved ground pool image to: {save_path2}")
    plt.close()
else:
    print("No Cherenkov photons produced!")

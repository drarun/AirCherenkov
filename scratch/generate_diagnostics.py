import glob
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import sys

# Ensure src is in path to import hillas and camera
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from sim.camera import Camera
from recon.hillas import compute_hillas

def main():
    diagnostics_dir = "AirCherenkov_Diagnostics"
    os.makedirs(diagnostics_dir, exist_ok=True)
    
    files = sorted(glob.glob('data/train_raw/sim_batch_*.pt'))
    if not files:
        print("No simulation files found in data/train_raw/")
        return

    print(f"Loading {len(files)} chunks...")
    
    gamma_r = []
    hadron_r = []
    
    gamma_size = []
    hadron_size = []
    
    gamma_width = []
    hadron_width = []
    
    gamma_length = []
    hadron_length = []

    cam = Camera(n_rings=12, pixel_size=0.15)
    
    for f in files:
        data = torch.load(f, weights_only=False)
        for ev in data:
            label = ev.get('label')
            ix = ev.get('impact_x', 0.0)
            iy = ev.get('impact_y', 0.0)
            r = np.sqrt(ix**2 + iy**2)
            
            traces = ev.get('fadc_traces') # (4, 469, 16)
            
            # Sum PE for all telescopes to get total Size, and average Width/Length
            total_size = 0.0
            widths = []
            lengths = []
            
            for tel_idx in range(traces.shape[0]):
                tel_trace = traces[tel_idx]
                pe_per_pixel = np.sum(tel_trace, axis=1) # (469,)
                
                # Image cleaning mask (simple tailcut for diagnostics)
                # Core pixels > 10 PE, boundary > 5 PE
                mask = pe_per_pixel > 10.0
                
                if np.sum(mask) >= 3:
                    hillas = compute_hillas(cam, pe_per_pixel, mask)
                    if hillas is not None:
                        total_size += hillas.size
                        widths.append(hillas.width)
                        lengths.append(hillas.length)
            
            if total_size > 0:
                if label == 1:
                    gamma_r.append(r)
                    gamma_size.append(total_size)
                    if widths: gamma_width.append(np.mean(widths))
                    if lengths: gamma_length.append(np.mean(lengths))
                else:
                    hadron_r.append(r)
                    hadron_size.append(total_size)
                    if widths: hadron_width.append(np.mean(widths))
                    if lengths: hadron_length.append(np.mean(lengths))
                    
    gamma_r = np.array(gamma_r)
    hadron_r = np.array(hadron_r)
    
    plt.style.use('dark_background')
    
    # 1. Throw Radius Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), dpi=150)
    bin_width = 15.0
    bins = np.arange(0, 800, bin_width)
    bin_centers = (bins[:-1] + bins[1:]) / 2.0
    
    n_gamma, _ = np.histogram(gamma_r, bins=bins)
    n_hadron, _ = np.histogram(hadron_r, bins=bins)
    
    ax1.plot(bin_centers, n_gamma, color='#38BDF8', lw=2.5, label=f'Gammas')
    ax1.plot(bin_centers, n_hadron, color='#F43F5E', lw=2.5, label=f'Hadrons')
    ax1.set_xlabel('Impact Radius $r$ [m]')
    ax1.set_ylabel(f'Events / {bin_width:.0f}m')
    ax1.set_title('Triggered Count vs. Radius')
    ax1.legend()
    
    area_bin = np.pi * (bins[1:]**2 - bins[:-1]**2)
    ax2.plot(bin_centers, n_gamma/area_bin, color='#38BDF8', lw=2.5, label='Gammas')
    ax2.plot(bin_centers, n_hadron/area_bin, color='#F43F5E', lw=2.5, label='Hadrons')
    ax2.set_yscale('log')
    ax2.set_xlabel('Impact Radius $r$ [m]')
    ax2.set_ylabel('Density dN/dA [events / m$^2$]')
    ax2.set_title('Triggered Surface Density vs. Radius')
    ax2.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(diagnostics_dir, "01_throw_radius.png"), bbox_inches='tight')
    plt.close()
    
    # 2. Total Size Distribution (PE)
    plt.figure(figsize=(8, 6), dpi=150)
    bins_size = np.logspace(1, 5, 50)
    plt.hist(gamma_size, bins=bins_size, alpha=0.5, color='#38BDF8', label='Gammas', density=True)
    plt.hist(hadron_size, bins=bins_size, alpha=0.5, color='#F43F5E', label='Hadrons', density=True)
    plt.xscale('log')
    plt.xlabel('Total Image Size [Photoelectrons]')
    plt.ylabel('Density')
    plt.title('Hillas Size Distribution (Gammas vs Hadrons)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(diagnostics_dir, "02_hillas_size.png"), bbox_inches='tight')
    plt.close()
    
    # 3. Hillas Width vs Length (Gammas vs Hadrons)
    plt.figure(figsize=(10, 8), dpi=150)
    plt.scatter(gamma_length, gamma_width, color='#38BDF8', alpha=0.3, s=10, label='Gammas')
    plt.scatter(hadron_length, hadron_width, color='#F43F5E', alpha=0.3, s=10, label='Hadrons')
    plt.plot([0, 0.25], [0, 0.25], 'w--', alpha=0.3)
    plt.xlim(0, 0.25)
    plt.ylim(0, 0.25)
    plt.xlabel('Hillas Length [deg]')
    plt.ylabel('Hillas Width [deg]')
    plt.title('Hillas Width vs. Length')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(diagnostics_dir, "03_hillas_width_length.png"), bbox_inches='tight')
    plt.close()

    # 4. Lateral Distribution of Size (Size vs Impact Radius)
    plt.figure(figsize=(8, 6), dpi=150)
    plt.scatter(gamma_r, gamma_size, color='#38BDF8', alpha=0.2, s=5, label='Gammas')
    plt.yscale('log')
    plt.xlim(0, 800)
    plt.xlabel('Impact Radius [m]')
    plt.ylabel('Total Size [PE]')
    plt.title('Lateral Distribution (Size vs Core Distance) for Gammas')
    plt.grid(True, alpha=0.1)
    plt.tight_layout()
    plt.savefig(os.path.join(diagnostics_dir, "04_lateral_distribution_size.png"), bbox_inches='tight')
    plt.close()
    
    print(f"Successfully generated all diagnostic plots in {diagnostics_dir}")

if __name__ == '__main__':
    main()

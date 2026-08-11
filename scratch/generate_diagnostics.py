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

import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, default="data/train_raw")
    args = parser.parse_args()
    
    diagnostics_dir = "DiagnosticPlots"
    os.makedirs(diagnostics_dir, exist_ok=True)
    
    files = sorted(glob.glob(os.path.join(args.input_dir, 'sim_batch_*.pt')))
    if not files:
        print(f"No simulation files found in {args.input_dir}")
        return

    print(f"Loading {len(files)} chunks...")
    
    gamma_r = []
    hadron_r = []
    
    gamma_size = []
    hadron_size = []
    
    gamma_true_e = []
    
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
                    gamma_true_e.append(ev.get('energy', 100.0))
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

    gamma_size = np.array(gamma_size)
    hadron_size = np.array(hadron_size)

    # 4. Lateral Distribution of Size (Size vs Impact Radius) Profile
    plt.figure(figsize=(8, 6), dpi=150)
    # Scatter plot background
    plt.scatter(gamma_r, gamma_size, color='#38BDF8', alpha=0.1, s=2, label='Gammas (All)')
    
    # Calculate profile (mean in bins)
    bins = np.arange(0, 800, 25.0)
    bin_centers = (bins[:-1] + bins[1:]) / 2.0
    
    # Compute mean size in each bin
    gamma_mean_size = np.zeros_like(bin_centers)
    hadron_mean_size = np.zeros_like(bin_centers)
    
    for i in range(len(bins)-1):
        # Gammas
        mask_g = (gamma_r >= bins[i]) & (gamma_r < bins[i+1])
        if np.any(mask_g):
            gamma_mean_size[i] = np.mean(gamma_size[mask_g])
        else:
            gamma_mean_size[i] = np.nan
            
        # Hadrons
        mask_h = (hadron_r >= bins[i]) & (hadron_r < bins[i+1])
        if np.any(mask_h):
            hadron_mean_size[i] = np.mean(hadron_size[mask_h])
        else:
            hadron_mean_size[i] = np.nan
            
    plt.plot(bin_centers, gamma_mean_size, color='#38BDF8', lw=3, label='Gammas (Mean LDF)')
    plt.plot(bin_centers, hadron_mean_size, color='#F43F5E', lw=3, label='Hadrons (Mean LDF)')

    plt.yscale('log')
    plt.xlim(0, 800)
    plt.xlabel('Impact Radius $r$ [m]')
    plt.ylabel('Total Size (Photoelectrons) [PE]')
    plt.title('Lateral Density Function (Size Profile vs Radius)')
    plt.grid(True, alpha=0.2)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(diagnostics_dir, "04_lateral_distribution_size.png"), bbox_inches='tight')
    plt.close()
    
    # 5. Hillas Length 1D Distribution
    plt.figure(figsize=(8, 6), dpi=150)
    bins_length = np.linspace(0, 0.3, 50)
    plt.hist(gamma_length, bins=bins_length, alpha=0.5, color='#38BDF8', label='Gammas', density=True)
    plt.hist(hadron_length, bins=bins_length, alpha=0.5, color='#F43F5E', label='Hadrons', density=True)
    plt.xlabel('Hillas Length [deg]')
    plt.ylabel('Density')
    plt.title('Hillas Length Distribution (Gammas vs Hadrons)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(diagnostics_dir, "05_hillas_length_dist.png"), bbox_inches='tight')
    plt.close()

    # 6. Hillas Width 1D Distribution
    plt.figure(figsize=(8, 6), dpi=150)
    bins_width = np.linspace(0, 0.15, 50)
    plt.hist(gamma_width, bins=bins_width, alpha=0.5, color='#38BDF8', label='Gammas', density=True)
    plt.hist(hadron_width, bins=bins_width, alpha=0.5, color='#F43F5E', label='Hadrons', density=True)
    plt.xlabel('Hillas Width [deg]')
    plt.ylabel('Density')
    plt.title('Hillas Width Distribution (Gammas vs Hadrons)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(diagnostics_dir, "06_hillas_width_dist.png"), bbox_inches='tight')
    plt.close()
    
    # 7. Energy Resolution & Bias (Stage 5 Proxy)
    # Reconstruct energy using a simple 2D polynomial regression: log10(E) = f(log10(Size), Radius)
    gamma_size = np.array(gamma_size)
    gamma_true_e = np.array(gamma_true_e)
    
    if len(gamma_size) > 10:
        log10_size = np.log10(gamma_size)
        log10_true_e = np.log10(gamma_true_e) # in GeV
        
        # Design matrix: [1, log10(S), r, log10(S)*r, r^2]
        A_mat = np.column_stack([
            np.ones_like(log10_size),
            log10_size,
            gamma_r,
            log10_size * gamma_r,
            gamma_r**2
        ])
        
        try:
            beta, _, _, _ = np.linalg.lstsq(A_mat, log10_true_e, rcond=None)
            log10_pred_e = A_mat @ beta
            E_pred = 10**log10_pred_e
            E_true = gamma_true_e
            
            frac_error = (E_pred - E_true) / E_true
            
            # Group into energy bins
            log_e_bins = np.linspace(np.log10(100), np.log10(30000), 10)
            bin_centers = 10**(0.5 * (log_e_bins[:-1] + log_e_bins[1:])) / 1000.0 # to TeV
            
            resolutions = []
            biases = []
            
            for i in range(len(log_e_bins)-1):
                mask = (log10_true_e >= log_e_bins[i]) & (log10_true_e < log_e_bins[i+1])
                if np.sum(mask) >= 5:
                    errors_in_bin = frac_error[mask]
                    # Resolution: 68% containment half-width of absolute fractional error
                    res = np.percentile(np.abs(errors_in_bin), 68)
                    bias = np.median(errors_in_bin)
                    resolutions.append(res)
                    biases.append(bias)
                else:
                    resolutions.append(np.nan)
                    biases.append(np.nan)
            
            # Plot Energy Performance
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), dpi=150)
            
            ax1.plot(bin_centers, np.array(resolutions) * 100.0, 'o-', color='#38BDF8', lw=2.5, label='68% containment')
            ax1.set_xscale('log')
            ax1.set_xlabel('True Energy [TeV]')
            ax1.set_ylabel('Energy Resolution [%]')
            ax1.set_title('Baseline Energy Resolution')
            ax1.axhline(y=17, color='gray', linestyle='--', alpha=0.5, label='VERITAS Goal (17%)')
            ax1.grid(True, alpha=0.2)
            ax1.legend()
            
            ax2.plot(bin_centers, np.array(biases) * 100.0, 's-', color='#F43F5E', lw=2.5, label='Median Bias')
            ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
            ax2.set_xscale('log')
            ax2.set_xlabel('True Energy [TeV]')
            ax2.set_ylabel('Energy Bias [%]')
            ax2.set_title('Baseline Energy Bias')
            ax2.grid(True, alpha=0.2)
            ax2.legend()
            
            plt.tight_layout()
            plt.savefig(os.path.join(diagnostics_dir, "07_energy_resolution_bias.png"), bbox_inches='tight')
            plt.close()
            print("Successfully generated energy resolution and bias plots.")
        except Exception as ex:
            print(f"Failed to generate energy diagnostics: {ex}")
    
    print(f"Successfully generated all diagnostic plots in {diagnostics_dir}")

if __name__ == '__main__':
    main()

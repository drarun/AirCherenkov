import os
import sys
import numpy as np
import matplotlib.pyplot as plt

def generate_plots():
    v5_cache = 'data/eval_predictions.npz'
    v6_cache = 'data/eval_predictions_spatiotemporal_gnn_v6.npz'
    v7_cache = 'data/eval_predictions_spatiotemporal_gnn_v7.npz'

    data7 = np.load(v7_cache)
    v8_cache = 'data/eval_predictions_spatiotemporal_gnn_v8.npz'
    v7_cache = 'data/eval_predictions_spatiotemporal_gnn_v7.npz'
    v5_cache = 'data/eval_predictions.npz'

    data8 = np.load(v8_cache)
    data7 = np.load(v7_cache)
    data5 = np.load(v5_cache)

    gamma_mask8 = data8['true_c'] == 1.0
    te8 = data8['true_e'][gamma_mask8]
    pe8 = data8['pred_e'][gamma_mask8]
    fe8 = (10**pe8 - 10**te8) / (10**te8)

    gamma_mask7 = data7['true_c'] == 1.0
    te7 = data7['true_e'][gamma_mask7]
    pe7 = data7['pred_e'][gamma_mask7]
    fe7 = (10**pe7 - 10**te7) / (10**te7)

    gamma_mask5 = data5['true_c'] == 1.0
    te5 = data5['true_e'][gamma_mask5]
    pe5 = data5['pred_e'][gamma_mask5]
    fe5 = (10**pe5 - 10**te5) / (10**te5)

    log_e_bins = np.linspace(np.min(te8) - 0.02, np.max(te8) + 0.02, 13)
    bin_centers = 0.5 * (log_e_bins[:-1] + log_e_bins[1:])

    res8, bias8, res_err8, bias_err8 = [], [], [], []
    res7, bias7, res_err7, bias_err7 = [], [], [], []
    res5, bias5, res_err5, bias_err5 = [], [], [], []
    valid_centers = []

    for i in range(len(log_e_bins) - 1):
        mask8 = (te8 >= log_e_bins[i]) & (te8 < log_e_bins[i+1])
        mask7 = (te7 >= log_e_bins[i]) & (te7 < log_e_bins[i+1])
        mask5 = (te5 >= log_e_bins[i]) & (te5 < log_e_bins[i+1])
        
        if mask8.sum() < 15 or mask7.sum() < 15 or mask5.sum() < 15:
            continue
        
        err8 = fe8[mask8]
        b8 = np.median(err8)
        r8 = 0.5 * (np.percentile(err8, 84) - np.percentile(err8, 16))
        
        err7 = fe7[mask7]
        b7 = np.median(err7)
        r7 = 0.5 * (np.percentile(err7, 84) - np.percentile(err7, 16))
        
        err5 = fe5[mask5]
        b5 = np.median(err5)
        r5 = 0.5 * (np.percentile(err5, 84) - np.percentile(err5, 16))

        # Bootstrap
        n_boot = 300
        rb8, bb8, rb7, bb7, rb5, bb5 = [], [], [], [], [], []
        for _ in range(n_boot):
            s8 = np.random.choice(err8, size=len(err8), replace=True)
            rb8.append(0.5 * (np.percentile(s8, 84) - np.percentile(s8, 16)))
            bb8.append(np.median(s8))
            s7 = np.random.choice(err7, size=len(err7), replace=True)
            rb7.append(0.5 * (np.percentile(s7, 84) - np.percentile(s7, 16)))
            bb7.append(np.median(s7))
            s5 = np.random.choice(err5, size=len(err5), replace=True)
            rb5.append(0.5 * (np.percentile(s5, 84) - np.percentile(s5, 16)))
            bb5.append(np.median(s5))

        res8.append(r8 * 100.0)
        bias8.append(b8 * 100.0)
        res_err8.append(np.std(rb8) * 100.0)
        bias_err8.append(np.std(bb8) * 100.0)

        res7.append(r7 * 100.0)
        bias7.append(b7 * 100.0)
        res_err7.append(np.std(rb7) * 100.0)
        bias_err7.append(np.std(bb7) * 100.0)

        res5.append(r5 * 100.0)
        bias5.append(b5 * 100.0)
        res_err5.append(np.std(rb5) * 100.0)
        bias_err5.append(np.std(bb5) * 100.0)

        valid_centers.append(10**bin_centers[i] / 1000.0)

    valid_centers = np.array(valid_centers)
    res8, bias8 = np.array(res8), np.array(bias8)
    res_err8, bias_err8 = np.array(res_err8), np.array(bias_err8)
    res7, bias7 = np.array(res7), np.array(bias7)
    res_err7, bias_err7 = np.array(res_err7), np.array(bias_err7)
    res5, bias5 = np.array(res5), np.array(bias5)
    res_err5, bias_err5 = np.array(res_err5), np.array(bias_err5)

    benchmark_energies_tev = np.logspace(np.log10(0.08), np.log10(30.0), 100)
    veritas_res_benchmark = np.sqrt(15.0**2 + (11.0 / np.sqrt(benchmark_energies_tev))**2)
    veritas_bias_benchmark = 20.0 / (1.0 + (benchmark_energies_tev / 0.12)**2.2) - 5.0 * (benchmark_energies_tev / 15.0)

    plt.rcParams.update({
        'font.size': 11,
        'font.family': 'sans-serif',
        'axes.labelsize': 12,
        'axes.titlesize': 13,
        'xtick.labelsize': 11,
        'ytick.labelsize': 11,
        'legend.fontsize': 10,
        'figure.titlesize': 15,
    })

    # =========================================================================
    # COMBINED 2-PANEL FIGURE: v8 vs v7 vs v5 vs VERITAS
    # =========================================================================
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), dpi=300)

    # --- PANEL 1: Energy Resolution ---
    ax1.plot(benchmark_energies_tev, veritas_res_benchmark, color='#333333', linestyle='--', lw=2.2,
             label='VERITAS Benchmark ($15\\% \\oplus 11\\%/\\sqrt{E}$)', zorder=2)
    ax1.axhspan(0, 20, color='green', alpha=0.08, label='Target High-Energy Regime (<20%)')

    ax1.errorbar(valid_centers, res5, yerr=res_err5, fmt='s:', color='#9467bd',
                 ecolor='#9467bd', elinewidth=1.5, capsize=3, markersize=6, lw=1.5,
                 label='v5 Baseline (Flawed Physics)', zorder=3, alpha=0.5)

    ax1.errorbar(valid_centers, res7, yerr=res_err7, fmt='^--', color='#d95f02',
                 ecolor='#d95f02', elinewidth=1.5, capsize=3, markersize=6, lw=1.5,
                 label='v7 Fine-Tuned (Flawed Physics)', zorder=4, alpha=0.6)

    ax1.errorbar(valid_centers, res8, yerr=res_err8, fmt='o-', color='#1f77b4',
                 ecolor='#1f77b4', elinewidth=2.0, capsize=4, markersize=7, lw=2.4,
                 label='v8 (Corrected Physics + Sqrt-Sampler)', zorder=5)

    ax1.set_xscale('log')
    ax1.set_xlabel('True Energy [TeV]', fontweight='bold')
    ax1.set_ylabel('Energy Resolution (0.5 * IQR) [%]', fontweight='bold')
    ax1.set_title('Energy Resolution vs. True Energy', fontweight='bold', pad=10)
    ax1.grid(True, which='both', linestyle=':', alpha=0.5)
    ax1.set_xlim([0.07, 32.0])
    ax1.set_ylim([0, 95])
    ax1.legend(loc='upper right', frameon=True, facecolor='white', framealpha=0.9)

    # --- PANEL 2: Energy Bias ---
    ax2.axhspan(-10, 10, color='#2ca02c', alpha=0.15, label='VERITAS Tolerance Window ($\\pm 10\\%$)')
    ax2.axhline(0, color='black', linestyle='-', lw=1.2, alpha=0.6, label='Zero Bias Baseline')
    ax2.plot(benchmark_energies_tev, veritas_bias_benchmark, color='#333333', linestyle='--', lw=2.0,
             label='VERITAS Typical Lookup Bias', zorder=2)

    ax2.errorbar(valid_centers, bias5, yerr=bias_err5, fmt='s:', color='#9467bd',
                 ecolor='#9467bd', elinewidth=1.5, capsize=3, markersize=6, lw=1.5,
                 label='v5 Baseline (Flawed Physics)', zorder=3, alpha=0.5)

    ax2.errorbar(valid_centers, bias7, yerr=bias_err7, fmt='^--', color='#d95f02',
                 ecolor='#d95f02', elinewidth=1.5, capsize=3, markersize=6, lw=1.5,
                 label='v7 Fine-Tuned (Flawed Physics)', zorder=4, alpha=0.6)

    ax2.errorbar(valid_centers, bias8, yerr=bias_err8, fmt='o-', color='#1f77b4',
                 ecolor='#1f77b4', elinewidth=2.0, capsize=4, markersize=7, lw=2.4,
                 label='v8 (Corrected Physics + Sqrt-Sampler)', zorder=5)

    ax2.set_xscale('log')
    ax2.set_xlabel('True Energy [TeV]', fontweight='bold')
    ax2.set_ylabel('Relative Energy Bias [(E_reco - E_true) / E_true] [%]', fontweight='bold')
    ax2.set_title('Energy Bias vs. True Energy', fontweight='bold', pad=10)
    ax2.grid(True, which='both', linestyle=':', alpha=0.5)
    ax2.set_xlim([0.07, 32.0])
    ax2.set_ylim([-95, 120])
    ax2.legend(loc='upper right', frameon=True, facecolor='white', framealpha=0.9)

    fig.suptitle('AirCherenkov SpatiotemporalGNN: Impact of Physics Fixes',
                 fontweight='bold', fontsize=15, y=0.98)
    plt.tight_layout()

    # Save to data/
    combined_png = 'data/energy_performance.png'
    fig.savefig(combined_png, dpi=300, bbox_inches='tight')
    print(f"Saved: {combined_png}")

    # Save to OneDrive Desktop as PDF
    desktop = os.path.expanduser('~/OneDrive/Desktop')
    if not os.path.exists(desktop):
        desktop = os.path.expanduser('~/Desktop')
    desktop_pdf = os.path.join(desktop, 'VERITAS_GNN_Energy_Performance_v8.pdf')
    fig.savefig(desktop_pdf, dpi=300, bbox_inches='tight')
    print(f"Saved: {desktop_pdf}")

    # Artifact dir
    artifact_dir = r'C:\Users\aruns\.gemini\antigravity-cli\brain\f0e1b905-b5d2-4fce-80a1-d0620d2d7de7'
    if os.path.exists(artifact_dir):
        artifact_png = os.path.join(artifact_dir, 'veritas_energy_performance_v8.png')
        fig.savefig(artifact_png, dpi=300, bbox_inches='tight')
        print(f"Saved: {artifact_png}")

    plt.close(fig)

if __name__ == '__main__':
    generate_plots()

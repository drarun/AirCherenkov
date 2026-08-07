import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from sklearn.metrics import roc_auc_score, roc_curve

from recon.gnn import SpatiotemporalGNN
from analysis.dataset import CherenkovDataset
from torch_geometric.loader import DataLoader

def evaluate():
    # Pre-transform
    from sim.camera import Camera
    cam = Camera(n_rings=12)
    edge_index = cam.edge_index
    
    class AddEdgeIndex(object):
        def __init__(self, edge_idx):
            self.edge_idx = edge_idx
        def __call__(self, data):
            data.edge_index = self.edge_idx
            return data

    dataset = CherenkovDataset(root='data/test', pre_transform=AddEdgeIndex(edge_index))
    loader = DataLoader(dataset, batch_size=128, shuffle=False)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = SpatiotemporalGNN().to(device)
    model.load_state_dict(torch.load('data/spatiotemporal_gnn.pt', map_location=device, weights_only=True))
    model.eval()
    
    true_e, pred_e = [], []
    true_c, pred_c = [], []
    
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            c_out, e_out = model(batch.x, batch.edge_index, batch.batch)
            
            true_e.extend(batch.y_energy.cpu().numpy())
            pred_e.extend(e_out.view(-1).cpu().numpy())
            
            true_c.extend(batch.y_class.cpu().numpy())
            pred_c.extend(torch.sigmoid(c_out).view(-1).cpu().numpy())
            
    true_e = np.array(true_e)
    pred_e = np.array(pred_e)
    true_c = np.array(true_c)
    pred_c = np.array(pred_c)
    
    # Filter to gamma-only events for energy analysis
    gamma_mask = (true_c == 1.0)
    te = true_e[gamma_mask]  # log10(E_true / GeV)
    pe = pred_e[gamma_mask]  # log10(E_reco / GeV)
    
    # Convert to linear energy for resolution/bias calculations
    E_true = 10**te  # GeV
    E_reco = 10**pe  # GeV
    
    # ========================================================================
    # Figure 1: Energy Reconstruction + ROC (existing, polished)
    # ========================================================================
    fig1, axes1 = plt.subplots(1, 2, figsize=(14, 6))
    
    # 1a. Energy Migration (log-log scatter)
    ax = axes1[0]
    h = ax.hist2d(te, pe, bins=60, cmap='inferno', norm=LogNorm(), cmin=1)
    ax.plot([1.5, 5.0], [1.5, 5.0], 'w--', lw=2, alpha=0.8, label='Perfect reconstruction')
    ax.set_xlabel(r'True $\log_{10}(E\,/\,\mathrm{GeV})$', fontsize=13)
    ax.set_ylabel(r'Reconstructed $\log_{10}(E\,/\,\mathrm{GeV})$', fontsize=13)
    ax.set_title('Energy Migration Matrix (Gamma only)', fontsize=14)
    ax.legend(fontsize=11)
    fig1.colorbar(h[3], ax=ax, label='Counts')
    
    # 1b. ROC Curve
    ax = axes1[1]
    try:
        fpr, tpr, thresholds = roc_curve(true_c, pred_c)
        auc = roc_auc_score(true_c, pred_c)
        ax.plot(fpr, tpr, 'c-', lw=2, label=f'SpatiotemporalGNN (AUC = {auc:.3f})')
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.5)
        ax.set_xlabel('False Positive Rate (Hadron Leakage)', fontsize=13)
        ax.set_ylabel('True Positive Rate (Gamma Efficiency)', fontsize=13)
        ax.set_title('Gamma/Hadron Classification ROC', fontsize=14)
        ax.legend(fontsize=12)
        ax.set_xlim([-0.02, 1.02])
        ax.set_ylim([-0.02, 1.02])
        ax.grid(True, alpha=0.3)
    except Exception as e:
        ax.set_title(f'ROC Failed: {str(e)}')
        
    fig1.tight_layout()
    fig1.savefig('data/evaluation.png', dpi=300)
    print("Saved data/evaluation.png")
    
    # ========================================================================
    # Figure 2: Energy Resolution & Bias vs True Energy
    # ========================================================================
    # Define energy bins (logarithmic)
    log_e_bins = np.linspace(np.min(te) - 0.05, np.max(te) + 0.05, 12)
    bin_centers = 0.5 * (log_e_bins[:-1] + log_e_bins[1:])
    
    # Fractional energy error: (E_reco - E_true) / E_true
    frac_error = (E_reco - E_true) / E_true
    
    resolutions = []
    biases = []
    res_errs = []
    bias_errs = []
    valid_centers = []
    
    for i in range(len(log_e_bins) - 1):
        mask = (te >= log_e_bins[i]) & (te < log_e_bins[i+1])
        if mask.sum() < 10:
            continue
        
        errors_in_bin = frac_error[mask]
        
        # Resolution: 68% containment half-width (robust to outliers)
        sorted_abs = np.sort(np.abs(errors_in_bin))
        idx_68 = int(0.68 * len(sorted_abs))
        resolution_68 = sorted_abs[idx_68] if idx_68 < len(sorted_abs) else sorted_abs[-1]
        
        # Bias: median fractional error
        bias = np.median(errors_in_bin)
        
        # Bootstrap error estimates
        n_bootstrap = 200
        res_boot = []
        bias_boot = []
        for _ in range(n_bootstrap):
            sample = np.random.choice(errors_in_bin, size=len(errors_in_bin), replace=True)
            sorted_s = np.sort(np.abs(sample))
            idx_68_s = int(0.68 * len(sorted_s))
            res_boot.append(sorted_s[idx_68_s] if idx_68_s < len(sorted_s) else sorted_s[-1])
            bias_boot.append(np.median(sample))
        
        resolutions.append(resolution_68)
        biases.append(bias)
        res_errs.append(np.std(res_boot))
        bias_errs.append(np.std(bias_boot))
        valid_centers.append(bin_centers[i])
    
    valid_centers = np.array(valid_centers)
    resolutions = np.array(resolutions)
    biases = np.array(biases)
    res_errs = np.array(res_errs)
    bias_errs = np.array(bias_errs)
    
    fig2, axes2 = plt.subplots(1, 2, figsize=(14, 6))
    
    # 2a. Energy Resolution vs True Energy
    ax = axes2[0]
    ax.errorbar(10**valid_centers / 1000.0, resolutions * 100, yerr=res_errs * 100,
                fmt='o-', color='dodgerblue', capsize=4, markersize=6, lw=2,
                label='68% containment')
    ax.set_xscale('log')
    ax.set_xlabel(r'True Energy [TeV]', fontsize=13)
    ax.set_ylabel(r'Energy Resolution [%]', fontsize=13)
    ax.set_title('Energy Resolution vs True Energy', fontsize=14)
    ax.axhline(y=17, color='gray', linestyle='--', alpha=0.5, label='VERITAS benchmark (17%)')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, max(resolutions * 100) * 1.3])
    
    # 2b. Energy Bias vs True Energy
    ax = axes2[1]
    ax.errorbar(10**valid_centers / 1000.0, biases * 100, yerr=bias_errs * 100,
                fmt='s-', color='coral', capsize=4, markersize=6, lw=2,
                label='Median bias')
    ax.set_xscale('log')
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel(r'True Energy [TeV]', fontsize=13)
    ax.set_ylabel(r'Energy Bias [%]', fontsize=13)
    ax.set_title('Energy Bias vs True Energy', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    fig2.tight_layout()
    fig2.savefig('data/energy_performance.png', dpi=300)
    print("Saved data/energy_performance.png")
    
    # ========================================================================
    # Print Summary Statistics
    # ========================================================================
    overall_rmse = np.sqrt(np.mean((te - pe)**2))
    overall_res = np.median(np.abs(frac_error)) * 100
    overall_bias = np.median(frac_error) * 100
    
    print(f"\n{'='*50}")
    print(f"  Evaluation Summary")
    print(f"{'='*50}")
    print(f"  Gamma events evaluated:  {gamma_mask.sum()}")
    print(f"  Total events evaluated:  {len(true_c)}")
    print(f"  Energy RMSE (log10):     {overall_rmse:.3f}")
    print(f"  Median Energy Resolution: {overall_res:.1f}%")
    print(f"  Median Energy Bias:       {overall_bias:+.1f}%")
    try:
        print(f"  Classification AUC:      {auc:.3f}")
    except:
        pass
    print(f"{'='*50}")

if __name__ == '__main__':
    evaluate()

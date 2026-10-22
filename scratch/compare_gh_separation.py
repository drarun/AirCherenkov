"""
Gamma/Hadron Separation Benchmark using Stereoscopic Machine Learning (Task 3).

Applies gh_feature_mask() to strictly exclude directional/camera-center features,
preventing wobble-position data leakage.
Evaluates:
- Overall Stratified 5-Fold CV ROC-AUC
- ROC-AUC in 4 bins of reconstructed array log10(size)
- Background rejection (1 / FPR) at 80% and 90% gamma efficiency
- Quality Factor Q = TPR / sqrt(FPR) (bounded to TPR >= 0.5 and >= 20 surviving background events)
- Combined analysis performance: ML cut + theta^2 < 0.04 deg^2 (theta < 0.2 deg)
Outputs diagnostic plots to scratch/gh_benchmark.png and artifacts dir.
"""
import os
import shutil
import sys
sys.path.insert(0, 'src')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold

from recon.ml_stereo import StereoClassifier, gh_feature_mask


def run_gh_benchmark():
    npz_path = 'data/stereo_features_750m.npz'
    if not os.path.exists(npz_path):
        print(f"Error: {npz_path} does not exist.")
        sys.exit(1)

    print(f"Loading cached features from {npz_path}...")
    data = np.load(npz_path)
    X_full = data['X']
    labels = data['labels']
    log10_e = data['log10_e']
    geom_ok = data['geom_ok']
    theta_deg = data['theta_deg']

    n_events = len(labels)
    n_gammas = int(np.sum(labels == 1))
    n_protons = int(np.sum(labels == 0))
    print(f"Loaded {n_events} events ({n_gammas} gammas, {n_protons} protons).")

    # Apply direction-leakage mask: drops cx, cy, dist, psi, miss, alpha, and reco zen/azi
    mask = gh_feature_mask(X_full.shape[1])
    X = X_full[:, mask]
    print(f"Applied gh_feature_mask: using {X.shape[1]} / {X_full.shape[1]} direction-agnostic features.")

    # 5-Fold Stratified Cross-Validation
    print("Training 5-fold Stratified Random Forest classifier...")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof_probs = np.zeros(n_events, dtype=np.float32)

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, labels)):
        clf = StereoClassifier(n_estimators=300, max_depth=16, random_state=42 + fold)
        clf.fit(X[train_idx], labels[train_idx])
        probs = clf.predict_proba(X[val_idx])
        oof_probs[val_idx] = probs
        fold_auc = roc_auc_score(labels[val_idx], probs)
        print(f"  Fold {fold+1}/5 ROC-AUC: {fold_auc:.4f}")

    # Overall metrics
    overall_auc = roc_auc_score(labels, oof_probs)
    fpr, tpr, thresholds = roc_curve(labels, oof_probs)

    # Rejection at 80% and 90% gamma efficiency
    idx_80 = int(np.argmin(np.abs(tpr - 0.80)))
    idx_90 = int(np.argmin(np.abs(tpr - 0.90)))
    fpr_80 = float(fpr[idx_80])
    fpr_90 = float(fpr[idx_90])
    rej_80 = 1.0 / max(fpr_80, 1e-6)
    rej_90 = 1.0 / max(fpr_90, 1e-6)

    # Robust Quality Factor calculation
    # Only evaluate where TPR >= 0.50 and surviving background events >= 20
    n_bkg_total = n_protons
    surviving_bkg = fpr * n_bkg_total
    valid_q_mask = (tpr >= 0.50) & (surviving_bkg >= 20) & (fpr > 0)

    q_factors = np.zeros_like(tpr)
    q_factors[valid_q_mask] = tpr[valid_q_mask] / np.sqrt(fpr[valid_q_mask])
    max_q_idx = int(np.argmax(q_factors))
    max_q = float(q_factors[max_q_idx])
    max_q_tpr = float(tpr[max_q_idx])
    max_q_fpr = float(fpr[max_q_idx])
    max_q_thresh = float(thresholds[max_q_idx])

    print("\n" + "=" * 70)
    print("        GAMMA / HADRON SEPARATION BENCHMARK (DIRECTION-AGNOSTIC)       ")
    print("=" * 70)
    print(f"Total Events: {n_events} ({n_gammas} gammas, {n_protons} protons)")
    print(f"Overall ROC-AUC:                  {overall_auc:.4f}")
    print(f"Hadron Leakage @ 80% Gamma Eff:   {fpr_80*100:.2f}%  (Rejection = {rej_80:.1f}x)")
    print(f"Hadron Leakage @ 90% Gamma Eff:   {fpr_90*100:.2f}%  (Rejection = {rej_90:.1f}x)")
    print(f"Optimal Robust Quality Factor Q:  {max_q:.2f} (at {max_q_tpr*100:.1f}% gamma eff, {max_q_fpr*100:.2f}% hadron leakage)")
    print("-" * 70)

    # Performance across reconstructed shower size (feature 53 = log10(1 + tot_size))
    log_size = X_full[:, 53]
    size_bins = [np.percentile(log_size, p) for p in [0, 25, 50, 75, 100]]
    print("ROC-AUC vs Reconstructed Array Size:")
    for b_idx in range(4):
        s_low, s_high = size_bins[b_idx], size_bins[b_idx+1]
        m = (log_size >= s_low) & (log_size <= s_high)
        if len(np.unique(labels[m])) == 2:
            sub_auc = roc_auc_score(labels[m], oof_probs[m])
            print(f"  Quartile {b_idx+1} [size {10**s_low-1:.0f} - {10**s_high-1:.0f} p.e.]: AUC = {sub_auc:.4f}")

    # Combined Analysis: ML Cut + Theta^2 Cut (theta < 0.2 deg)
    print("-" * 70)
    theta_cut = 0.20  # degrees
    theta_pass = (theta_deg <= theta_cut) & geom_ok
    gamma_theta_eff = np.sum(theta_pass & (labels == 1)) / max(n_gammas, 1)
    proton_theta_leak = np.sum(theta_pass & (labels == 0)) / max(n_protons, 1)
    print(f"Direction-only cut (theta < {theta_cut} deg):")
    print(f"  Gamma Efficiency: {gamma_theta_eff*100:.1f}%, Proton Leakage: {proton_theta_leak*100:.2f}% (Rejection = {1.0/max(proton_theta_leak, 1e-6):.1f}x)")

    # Combined: ML threshold at 80% gamma eff + theta cut
    ml_pass = oof_probs >= thresholds[idx_80]
    comb_pass = ml_pass & theta_pass
    comb_gamma_eff = np.sum(comb_pass & (labels == 1)) / max(n_gammas, 1)
    comb_proton_leak = np.sum(comb_pass & (labels == 0)) / max(n_protons, 1)
    comb_rej = 1.0 / max(comb_proton_leak, 1e-6)
    print(f"Combined Analysis Cut (ML @ 80% eff + theta < {theta_cut} deg):")
    print(f"  Total Gamma Efficiency: {comb_gamma_eff*100:.1f}%")
    print(f"  Total Proton Leakage:   {comb_proton_leak*100:.3f}% (Total Rejection = {comb_rej:.1f}x)")
    print("=" * 70)

    # Plotting 4-Panel Benchmark Figure
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))

    # Panel A: ROC Curve (Linear)
    ax = axes[0, 0]
    ax.plot(fpr * 100, tpr * 100, color='#1f77b4', lw=2.5, label=f'Stereo ML RF (AUC = {overall_auc:.4f})')
    ax.plot([0, 100], [0, 100], 'k--', alpha=0.5, label='Random Guess')
    ax.scatter([fpr_80 * 100], [80.0], color='red', s=60, zorder=5, label=f'80% eff (Leak = {fpr_80*100:.1f}%)')
    ax.scatter([fpr_90 * 100], [90.0], color='darkorange', s=60, zorder=5, label=f'90% eff (Leak = {fpr_90*100:.1f}%)')
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 105)
    ax.set_xlabel('Hadron Leakage [%]', fontsize=11)
    ax.set_ylabel('Gamma Efficiency [%]', fontsize=11)
    ax.set_title('A. ROC Curve (Linear Scale)', fontsize=12, fontweight='bold')
    ax.legend(loc='lower right', frameon=True, fontsize=10)
    ax.grid(True, alpha=0.3)

    # Panel B: ROC Curve (Log-X Scale / High Rejection Region)
    ax = axes[0, 1]
    ax.semilogx(np.maximum(fpr, 1e-4) * 100, tpr * 100, color='#1f77b4', lw=2.5, label='Stereo ML RF')
    ax.axhline(80.0, color='red', ls=':', alpha=0.7)
    ax.axhline(90.0, color='darkorange', ls=':', alpha=0.7)
    ax.set_xlim(0.05, 100)
    ax.set_ylim(40, 102)
    ax.set_xlabel('Hadron Leakage [%] (Log Scale)', fontsize=11)
    ax.set_ylabel('Gamma Efficiency [%]', fontsize=11)
    ax.set_title('B. Background Rejection Performance', fontsize=12, fontweight='bold')
    ax.legend(loc='lower right', frameon=True, fontsize=10)
    ax.grid(True, which='both', alpha=0.3)

    # Panel C: Quality Factor Q vs Gamma Efficiency
    ax = axes[1, 0]
    valid_plot_mask = (tpr >= 0.20) & (fpr > 0)
    ax.plot(tpr[valid_plot_mask] * 100, tpr[valid_plot_mask] / np.sqrt(fpr[valid_plot_mask]),
            color='#2ca02c', lw=2.5, label='Q-Factor')
    ax.scatter([max_q_tpr * 100], [max_q], color='red', s=70, zorder=5,
               label=f'Max Robust Q = {max_q:.2f} (@ {max_q_tpr*100:.0f}% eff)')
    ax.set_xlim(20, 100)
    ax.set_ylim(0, max_q * 1.3)
    ax.set_xlabel('Gamma Efficiency [%]', fontsize=11)
    ax.set_ylabel(r'Quality Factor $Q = \varepsilon_\gamma / \sqrt{\varepsilon_h}$', fontsize=11)
    ax.set_title('C. Quality Factor vs Gamma Efficiency', fontsize=12, fontweight='bold')
    ax.legend(loc='upper left', frameon=True, fontsize=10)
    ax.grid(True, alpha=0.3)

    # Panel D: Classifier Score Distributions
    ax = axes[1, 1]
    gamma_scores = oof_probs[labels == 1]
    proton_scores = oof_probs[labels == 0]
    bins = np.linspace(0, 1, 50)
    ax.hist(proton_scores, bins=bins, density=True, alpha=0.6, color='#d62728', label='Protons (Hadrons)')
    ax.hist(gamma_scores, bins=bins, density=True, alpha=0.6, color='#1f77b4', label='Gammas')
    ax.axvline(thresholds[idx_80], color='black', ls='--', lw=1.5, label=f'80% Gamma Cut ({thresholds[idx_80]:.2f})')
    ax.set_xlabel('Gamma Probability Score', fontsize=11)
    ax.set_ylabel('Probability Density', fontsize=11)
    ax.set_title('D. Classifier Score Distributions (Out-of-Fold)', fontsize=12, fontweight='bold')
    ax.legend(loc='upper center', frameon=True, fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.suptitle("Stereoscopic Gamma / Hadron Separation Benchmark (Direction-Agnostic Features)\n"
                 "VERITAS Array Simulation | 30 deg Zenith | 750m Impact Throw | 5-Fold Stratified CV",
                 fontsize=13, fontweight='bold', y=0.99)
    plt.tight_layout()

    os.makedirs('scratch', exist_ok=True)
    plot_file = 'scratch/gh_benchmark.png'
    plt.savefig(plot_file, dpi=150)
    print(f"\nSaved benchmark plot to {plot_file}")

    # Copy to artifacts directory
    artifact_dir = r"C:\Users\aruns\.gemini\antigravity-cli\brain\f0e1b905-b5d2-4fce-80a1-d0620d2d7de7"
    if os.path.exists(artifact_dir):
        dest_plot = os.path.join(artifact_dir, "gh_benchmark.png")
        shutil.copyfile(plot_file, dest_plot)
        print(f"Copied plot to artifact directory: {dest_plot}")


if __name__ == '__main__':
    run_gh_benchmark()

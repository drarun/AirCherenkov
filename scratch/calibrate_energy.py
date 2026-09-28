"""
Post-hoc energy calibration for SpatiotemporalGNN v5.

Fits a monotonic isotonic regression mapping E_reco -> E_true on 80% of gamma
events, evaluates on the held-out 20%, and saves the calibration function.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.isotonic import IsotonicRegression
import pickle

def main():
    # =========================================================================
    # 1. Load v5 predictions
    # =========================================================================
    v5_cache = 'data/eval_predictions.npz'
    if not os.path.exists(v5_cache):
        print(f"Error: {v5_cache} not found. Run evaluate_gnns.py with v5 model first.")
        return

    data = np.load(v5_cache)
    true_e = data['true_e']   # log10(E_true / GeV)
    pred_e = data['pred_e']   # log10(E_reco / GeV)
    true_c = data['true_c']
    pred_c = data['pred_c']

    gamma_mask = (true_c == 1.0)
    te_gamma = true_e[gamma_mask]
    pe_gamma = pred_e[gamma_mask]

    print(f"Loaded {len(true_e)} total events, {gamma_mask.sum()} gamma events")

    # =========================================================================
    # 2. Split gamma events into calibration (80%) and test (20%)
    # =========================================================================
    np.random.seed(42)
    n_gamma = len(te_gamma)
    indices = np.random.permutation(n_gamma)
    split = int(0.8 * n_gamma)
    cal_idx = indices[:split]
    test_idx = indices[split:]

    te_cal, pe_cal = te_gamma[cal_idx], pe_gamma[cal_idx]
    te_test, pe_test = te_gamma[test_idx], pe_gamma[test_idx]

    print(f"Calibration set: {len(cal_idx)} gamma events")
    print(f"Test set: {len(test_idx)} gamma events")

    # =========================================================================
    # 3. Fit isotonic regression: log10(E_reco) -> log10(E_true)
    # =========================================================================
    iso_reg = IsotonicRegression(increasing=True, out_of_bounds='clip')
    iso_reg.fit(pe_cal, te_cal)

    # Apply calibration
    pe_cal_calibrated = iso_reg.predict(pe_cal)
    pe_test_calibrated = iso_reg.predict(pe_test)

    # Also calibrate ALL gamma events for plotting
    pe_all_calibrated = iso_reg.predict(pe_gamma)

    print("\nCalibration function fitted successfully.")

    # =========================================================================
    # 4. Evaluate: Before vs After calibration on TEST set
    # =========================================================================
    def compute_metrics(true_log, pred_log, label=""):
        E_true = 10**true_log
        E_reco = 10**pred_log
        frac_err = (E_reco - E_true) / E_true

        rmse = np.sqrt(np.mean((true_log - pred_log)**2))
        pearson_r = np.corrcoef(true_log, pred_log)[0, 1]
        q84 = np.percentile(frac_err, 84)
        q16 = np.percentile(frac_err, 16)
        overall_res = 0.5 * (q84 - q16) * 100
        overall_bias = np.median(frac_err) * 100

        print(f"\n  {label}")
        print(f"  {'='*50}")
        print(f"  RMSE (log10):          {rmse:.4f}")
        print(f"  Pearson r:             {pearson_r:.4f}")
        print(f"  Resolution (IQR/2):    {overall_res:.1f}%")
        print(f"  Median Bias:           {overall_bias:+.1f}%")

        return frac_err, rmse, pearson_r, overall_res, overall_bias

    print("\n" + "="*60)
    print("  TEST SET EVALUATION (held-out 20% of gamma events)")
    print("="*60)

    fe_before, _, r_before, res_before, bias_before = compute_metrics(
        te_test, pe_test, "BEFORE Calibration (raw v5)")

    fe_after, _, r_after, res_after, bias_after = compute_metrics(
        te_test, pe_test_calibrated, "AFTER Calibration (isotonic)")

    # =========================================================================
    # 5. Binned comparison on TEST set
    # =========================================================================
    log_bins = np.linspace(1.9, 4.5, 9)

    print(f"\n{'Energy Range':<20} | {'N':>5} | {'v5 Bias':>9} | {'Cal Bias':>9} | {'v5 Res':>8} | {'Cal Res':>8}")
    print("-" * 75)

    binned_centers = []
    binned_res_before, binned_bias_before = [], []
    binned_res_after, binned_bias_after = [], []
    binned_res_err_b, binned_bias_err_b = [], []
    binned_res_err_a, binned_bias_err_a = [], []

    for i in range(len(log_bins) - 1):
        m = (te_test >= log_bins[i]) & (te_test < log_bins[i+1])
        if m.sum() < 15:
            continue

        fb = fe_before[m]
        fa = fe_after[m]

        bb = np.median(fb) * 100
        ba = np.median(fa) * 100
        rb = 0.5 * (np.percentile(fb, 84) - np.percentile(fb, 16)) * 100
        ra = 0.5 * (np.percentile(fa, 84) - np.percentile(fa, 16)) * 100

        # Bootstrap
        n_boot = 300
        rb_boot, bb_boot, ra_boot, ba_boot = [], [], [], []
        for _ in range(n_boot):
            sb = np.random.choice(fb, size=len(fb), replace=True)
            sa = np.random.choice(fa, size=len(fa), replace=True)
            rb_boot.append(0.5 * (np.percentile(sb, 84) - np.percentile(sb, 16)) * 100)
            bb_boot.append(np.median(sb) * 100)
            ra_boot.append(0.5 * (np.percentile(sa, 84) - np.percentile(sa, 16)) * 100)
            ba_boot.append(np.median(sa) * 100)

        e_str = f"{10**log_bins[i]:.0f} - {10**log_bins[i+1]:.0f} GeV"
        print(f"{e_str:<20} | {m.sum():>5} | {bb:+8.1f}% | {ba:+8.1f}% | {rb:7.1f}% | {ra:7.1f}%")

        center_tev = 10**(0.5*(log_bins[i] + log_bins[i+1])) / 1000.0
        binned_centers.append(center_tev)
        binned_res_before.append(rb)
        binned_bias_before.append(bb)
        binned_res_after.append(ra)
        binned_bias_after.append(ba)
        binned_res_err_b.append(np.std(rb_boot))
        binned_bias_err_b.append(np.std(bb_boot))
        binned_res_err_a.append(np.std(ra_boot))
        binned_bias_err_a.append(np.std(ba_boot))

    binned_centers = np.array(binned_centers)
    binned_res_before = np.array(binned_res_before)
    binned_bias_before = np.array(binned_bias_before)
    binned_res_after = np.array(binned_res_after)
    binned_bias_after = np.array(binned_bias_after)

    # =========================================================================
    # 6. Generate comparison plots
    # =========================================================================
    benchmark_energies_tev = np.logspace(np.log10(0.08), np.log10(30.0), 100)
    veritas_res = np.sqrt(15.0**2 + (11.0 / np.sqrt(benchmark_energies_tev))**2)
    veritas_bias = 20.0 / (1.0 + (benchmark_energies_tev / 0.12)**2.2) - 5.0 * (benchmark_energies_tev / 15.0)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6), dpi=300)

    # --- Resolution ---
    ax = axes[0]
    ax.plot(benchmark_energies_tev, veritas_res, 'k--', lw=2, label='VERITAS Benchmark', zorder=1)
    ax.axhspan(0, 20, color='green', alpha=0.08, label='Target (<20%)')
    ax.errorbar(binned_centers, binned_res_before, yerr=binned_res_err_b,
                fmt='o--', color='#9467bd', capsize=3, markersize=6, lw=1.8,
                label='v5 Raw (uncalibrated)', alpha=0.7, zorder=3)
    ax.errorbar(binned_centers, binned_res_after, yerr=binned_res_err_a,
                fmt='o-', color='#1f77b4', capsize=4, markersize=7, lw=2.2,
                label='v5 + Isotonic Calibration', zorder=4)
    ax.set_xscale('log')
    ax.set_xlabel('True Energy [TeV]', fontweight='bold')
    ax.set_ylabel('Energy Resolution (0.5 * IQR) [%]', fontweight='bold')
    ax.set_title('Energy Resolution vs. True Energy', fontweight='bold')
    ax.set_xlim([0.07, 32.0])
    ax.set_ylim([0, 100])
    ax.legend(loc='upper right', frameon=True, facecolor='white', framealpha=0.9)
    ax.grid(True, which='both', linestyle=':', alpha=0.5)

    # --- Bias ---
    ax = axes[1]
    ax.axhspan(-10, 10, color='#2ca02c', alpha=0.15, label='VERITAS Tolerance ($\\pm10\\%$)')
    ax.axhline(0, color='black', lw=1.2, alpha=0.6)
    ax.plot(benchmark_energies_tev, veritas_bias, 'k--', lw=2, label='VERITAS Typical Bias', zorder=1)
    ax.errorbar(binned_centers, binned_bias_before, yerr=binned_bias_err_b,
                fmt='s--', color='#9467bd', capsize=3, markersize=6, lw=1.8,
                label='v5 Raw (uncalibrated)', alpha=0.7, zorder=3)
    ax.errorbar(binned_centers, binned_bias_after, yerr=binned_bias_err_a,
                fmt='s-', color='#d95f02', capsize=4, markersize=7, lw=2.2,
                label='v5 + Isotonic Calibration', zorder=4)
    ax.set_xscale('log')
    ax.set_xlabel('True Energy [TeV]', fontweight='bold')
    ax.set_ylabel('Relative Energy Bias [%]', fontweight='bold')
    ax.set_title('Energy Bias vs. True Energy', fontweight='bold')
    ax.set_xlim([0.07, 32.0])
    ax.set_ylim([-100, 70])
    ax.legend(loc='lower left', frameon=True, facecolor='white', framealpha=0.9)
    ax.grid(True, which='both', linestyle=':', alpha=0.5)

    fig.suptitle('SpatiotemporalGNN v5: Post-Hoc Isotonic Energy Calibration vs. VERITAS',
                 fontweight='bold', fontsize=15, y=0.98)
    plt.tight_layout()

    fig.savefig('data/energy_calibration_comparison.png', dpi=300, bbox_inches='tight')
    print(f"\nSaved: data/energy_calibration_comparison.png")

    desktop = os.path.expanduser('~/OneDrive/Desktop')
    if not os.path.exists(desktop):
        desktop = os.path.expanduser('~/Desktop')
    pdf_path = os.path.join(desktop, 'VERITAS_GNN_Energy_Calibration.pdf')
    fig.savefig(pdf_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {pdf_path}")

    artifact_dir = r'C:\Users\aruns\.gemini\antigravity-cli\brain\f0e1b905-b5d2-4fce-80a1-d0620d2d7de7'
    if os.path.exists(artifact_dir):
        art_path = os.path.join(artifact_dir, 'energy_calibration_comparison.png')
        fig.savefig(art_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {art_path}")

    plt.close(fig)

    # =========================================================================
    # 7. Plot the calibration function itself
    # =========================================================================
    fig2, ax2 = plt.subplots(figsize=(7, 7), dpi=200)
    e_range = np.linspace(pe_gamma.min(), pe_gamma.max(), 500)
    e_calibrated = iso_reg.predict(e_range)
    ax2.plot([1.8, 4.5], [1.8, 4.5], 'k--', lw=1.5, alpha=0.5, label='Identity (no correction)')
    ax2.plot(e_range, e_calibrated, 'r-', lw=2.5, label='Isotonic calibration curve')
    ax2.set_xlabel('log10(E_reco / GeV)', fontweight='bold')
    ax2.set_ylabel('log10(E_calibrated / GeV)', fontweight='bold')
    ax2.set_title('Energy Calibration Transfer Function', fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_aspect('equal')
    fig2.tight_layout()
    fig2.savefig('data/calibration_curve.png', dpi=200, bbox_inches='tight')
    print(f"Saved: data/calibration_curve.png")
    plt.close(fig2)

    # =========================================================================
    # 8. Save calibration model for inference
    # =========================================================================
    cal_path = 'data/energy_calibration_isotonic.pkl'
    with open(cal_path, 'wb') as f:
        pickle.dump(iso_reg, f)
    print(f"Saved calibration model: {cal_path}")

    # Also save calibrated predictions
    cal_pred_path = 'data/eval_predictions_v5_calibrated.npz'
    # Calibrate ALL predictions (not just gamma)
    all_calibrated = iso_reg.predict(pred_e)
    np.savez(cal_pred_path,
             true_e=true_e, pred_e=all_calibrated,
             true_c=true_c, pred_c=pred_c,
             pred_e_raw=pred_e)
    print(f"Saved calibrated predictions: {cal_pred_path}")

    print("\n" + "="*60)
    print("  CALIBRATION COMPLETE")
    print("="*60)

if __name__ == '__main__':
    main()

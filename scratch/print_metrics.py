import numpy as np

data = np.load('data/eval_predictions.npz')
true_e, pred_e = data['true_e'], data['pred_e']
is_gamma = data['true_c'] == 1.0
te = true_e[is_gamma]
pe = pred_e[is_gamma]
te_gev = 10.0 ** te
pe_gev = 10.0 ** pe

print(f"Prediction range: [{pe.min():.2f}, {pe.max():.2f}] log10(GeV) vs True: [{te.min():.2f}, {te.max():.2f}]")
print(f"Prediction energy range: [{pe_gev.min():.1f} GeV, {pe_gev.max():.1f} GeV]")
print("=" * 68)
print(f"{'Energy Range':<22} | {'N events':<8} | {'Median Bias (%)':<16} | {'Resolution (%)':<16}")
print("-" * 68)

bins = [
    (1.7, 2.0, '50 - 100 GeV'),
    (2.0, 2.3, '100 - 200 GeV'),
    (2.3, 2.7, '200 - 500 GeV'),
    (2.7, 3.0, '500 GeV - 1 TeV'),
    (3.0, 3.3, '1 - 2 TeV (Benchmark)'),
    (3.3, 3.7, '2 - 5 TeV'),
    (3.7, 4.3, '5 - 20 TeV'),
]

for lo, hi, label in bins:
    mask = (te >= lo) & (te < hi)
    if mask.sum() == 0:
        continue
    frac = (pe_gev[mask] - te_gev[mask]) / te_gev[mask]
    bias = np.median(frac) * 100.0
    res = np.median(np.abs(frac)) * 100.0
    print(f"{label:<22} | {mask.sum():<8} | {bias:+16.1f}% | {res:16.1f}%")

tev_mask = (te_gev >= 800) & (te_gev <= 1200)
tev_frac = (pe_gev[tev_mask] - te_gev[tev_mask]) / te_gev[tev_mask]
print("=" * 68)
print(f"Exact 1 TeV window (800-1200 GeV): N={tev_mask.sum()}, Median Bias={np.median(tev_frac)*100.0:+.1f}%, Resolution={np.median(np.abs(tev_frac))*100.0:.1f}%")

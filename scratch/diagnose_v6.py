import os
import numpy as np

v6_cache = 'data/eval_predictions_spatiotemporal_gnn_v6.npz'
v5_cache = 'data/eval_predictions.npz'

data6 = np.load(v6_cache)
true_e = data6['true_e']
pred_e6 = data6['pred_e']
true_c = data6['true_c']
pred_c6 = data6['pred_c']

gamma_mask = (true_c == 1.0)
te = true_e[gamma_mask]
pe6 = pred_e6[gamma_mask]

has_v5 = os.path.exists(v5_cache)
if has_v5:
    data5 = np.load(v5_cache)
    pe5 = data5['pred_e'][gamma_mask]
    pred_c5 = data5['pred_c']

E_true = 10**te
fe6 = (10**pe6 - E_true) / E_true
if has_v5:
    fe5 = (10**pe5 - E_true) / E_true

log_bins = np.linspace(1.9, 4.5, 9)

print('========================================================================================')
print('                 HEAD-TO-HEAD BINNED COMPARISON: v5 vs v6')
print('========================================================================================')
print(f'{"Energy Range":<18} | {"Count":<6} | {"v5 Bias":<9} | {"v6 Bias":<9} | {"v5 Res (IQR)":<12} | {"v6 Res (IQR)":<12}')
print('----------------------------------------------------------------------------------------')

for i in range(len(log_bins)-1):
    m = (te >= log_bins[i]) & (te < log_bins[i+1])
    if m.sum() == 0: continue
    e_str = f'{10**log_bins[i]:.0f} - {10**log_bins[i+1]:.0f} GeV'
    
    # v6
    b6 = np.median(fe6[m]) * 100
    r6 = 0.5 * (np.percentile(fe6[m], 84) - np.percentile(fe6[m], 16)) * 100
    
    # v5
    if has_v5:
        b5 = np.median(fe5[m]) * 100
        r5 = 0.5 * (np.percentile(fe5[m], 84) - np.percentile(fe5[m], 16)) * 100
        print(f'{e_str:<18} | {m.sum():<6} | {b5:+8.1f}% | {b6:+8.1f}% | {r5:10.1f}%  | {r6:10.1f}%')
    else:
        print(f'{e_str:<18} | {m.sum():<6} |    N/A    | {b6:+8.1f}% |     N/A     | {r6:10.1f}%')

print('========================================================================================')
r_v6 = np.corrcoef(te, pe6)[0, 1]
rmse_v6 = np.sqrt(np.mean((te - pe6)**2))
print(f'v6: RMSE = {rmse_v6:.3f}, Pearson r = {r_v6:.4f}')

if has_v5:
    r_v5 = np.corrcoef(te, pe5)[0, 1]
    rmse_v5 = np.sqrt(np.mean((te - pe5)**2))
    print(f'v5: RMSE = {rmse_v5:.3f}, Pearson r = {r_v5:.4f}')
print('========================================================================================')

import os
import numpy as np

v5_cache = 'data/eval_predictions.npz'
v6_cache = 'data/eval_predictions_spatiotemporal_gnn_v6.npz'
v7_cache = 'data/eval_predictions_spatiotemporal_gnn_v7.npz'

data7 = np.load(v7_cache)
true_e = data7['true_e']
pe7 = data7['pred_e']
true_c = data7['true_c']
gamma_mask = (true_c == 1.0)
te = true_e[gamma_mask]
pe7 = pe7[gamma_mask]

data5 = np.load(v5_cache)
pe5 = data5['pred_e'][gamma_mask]

data6 = np.load(v6_cache)
pe6 = data6['pred_e'][gamma_mask]

E_true = 10**te
fe7 = (10**pe7 - E_true) / E_true
fe5 = (10**pe5 - E_true) / E_true
fe6 = (10**pe6 - E_true) / E_true

log_bins = np.linspace(1.9, 4.5, 9)

print('===================================================================================================')
print('                 HEAD-TO-HEAD BINNED COMPARISON: v5 vs v6 vs v7')
print('===================================================================================================')
print(f'{"Energy Range":<18} | {"Count":<6} | {"v5 Bias":<8} | {"v6 Bias":<8} | {"v7 Bias":<8} | {"v5 Res":<7} | {"v6 Res":<7} | {"v7 Res":<7}')
print('---------------------------------------------------------------------------------------------------')

for i in range(len(log_bins)-1):
    m = (te >= log_bins[i]) & (te < log_bins[i+1])
    if m.sum() == 0: continue
    e_str = f'{10**log_bins[i]:.0f} - {10**log_bins[i+1]:.0f} GeV'
    
    b5 = np.median(fe5[m]) * 100
    r5 = 0.5 * (np.percentile(fe5[m], 84) - np.percentile(fe5[m], 16)) * 100

    b6 = np.median(fe6[m]) * 100
    r6 = 0.5 * (np.percentile(fe6[m], 84) - np.percentile(fe6[m], 16)) * 100

    b7 = np.median(fe7[m]) * 100
    r7 = 0.5 * (np.percentile(fe7[m], 84) - np.percentile(fe7[m], 16)) * 100
    
    print(f'{e_str:<18} | {m.sum():<6} | {b5:+7.1f}% | {b6:+7.1f}% | {b7:+7.1f}% | {r5:6.1f}% | {r6:6.1f}% | {r7:6.1f}%')

print('===================================================================================================')
rmse5 = np.sqrt(np.mean((te - pe5)**2))
rmse6 = np.sqrt(np.mean((te - pe6)**2))
rmse7 = np.sqrt(np.mean((te - pe7)**2))
r5 = np.corrcoef(te, pe5)[0, 1]
r6 = np.corrcoef(te, pe6)[0, 1]
r7 = np.corrcoef(te, pe7)[0, 1]
print(f'v5: RMSE = {rmse5:.3f}, Pearson r = {r5:.4f}')
print(f'v6: RMSE = {rmse6:.3f}, Pearson r = {r6:.4f}')
print(f'v7: RMSE = {rmse7:.3f}, Pearson r = {r7:.4f}')
print('===================================================================================================')

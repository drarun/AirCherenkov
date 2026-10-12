"""Can any monotonic post-hoc calibration meet VERITAS bias/resolution targets with v5 GNN outputs?"""
import numpy as np

d = np.load('data/eval_predictions_v5_calibrated.npz')
g = d['true_c'] == 1.0
t = d['true_e'][g]
p = d['pred_e_raw'][g]

edges = np.linspace(2.0, 4.4, 9)
ctr = 0.5 * (edges[:-1] + edges[1:])

# Response curve: median log10(E_reco) per TRUE-energy bin
med = np.array([np.median(p[(t >= a) & (t < b)]) for a, b in zip(edges[:-1], edges[1:])])
slope = np.gradient(med, ctr)
print("Response curve (median log E_reco vs log E_true):")
for c, m, s in zip(ctr, med, slope):
    print(f"  log E_true={c:.2f}  median log E_reco={m:.3f}  slope d(reco)/d(true)={s:.2f}")

def stats(e_cal, label):
    print(f"\n{label}")
    print("  E_true bin [GeV]   median bias   68% |dE/E|   sigma(log E) (16-84%/2)")
    for a, b in zip(edges[:-1], edges[1:]):
        m = (t >= a) & (t < b)
        r = 10 ** (e_cal[m] - t[m]) - 1
        dl = e_cal[m] - t[m]
        sig = 0.5 * (np.percentile(dl, 84) - np.percentile(dl, 16))
        print(f"  {10**a:6.0f}-{10**b:6.0f}     {100*np.median(r):+7.1f}%     {100*np.percentile(np.abs(r), 68):6.1f}%     {sig:.3f} dex")

stats(p, "RAW v5")

# Isotonic fit of E[true | reco] (what Option 1 as written does)
from sklearn.isotonic import IsotonicRegression
iso = IsotonicRegression(out_of_bounds='clip').fit(p, t)
stats(iso.predict(p), "ISOTONIC E[true|reco]  (regression-to-prior; bias in true bins NOT removed)")

# Inverse median response (bias-in-true-bins correction), requires strictly increasing response
mono = np.maximum.accumulate(med + 1e-6 * np.arange(len(med)))
inv = np.interp(p, mono, ctr)
stats(inv, "INVERSE MEDIAN RESPONSE  (removes median bias in true bins, inflates spread by 1/slope)")

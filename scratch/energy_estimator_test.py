"""Quick test: is the high-E energy collapse an estimator artifact (steep training prior)?"""
import sys; sys.path.insert(0, 'src')
import numpy as np
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.model_selection import KFold
sys.path.insert(0, 'scratch')
from energy_benchmark import true_impact_distances, EDGES

d = np.load('data/stereo_features_750m.npz')
idx = np.where((d['labels'] == 1) & d['geom_ok'])[0]
X = d['X'][idx]; t = d['log10_e'][idx]
Xo = X.copy(); Xo[:, 61:65] = true_impact_distances(d, idx, X)
# log-transform impact distances and add size*impact interaction proxies
def aug(A):
    imp = np.log10(1 + A[:, 61:65]); sz = A[:, [1, 14, 27, 40]]
    return np.hstack([A, imp, sz + 2.0 * imp])  # size corrected for ~1/r^2 falloff proxy
w_flat = 10 ** (1.0 * (t - 2))  # reweight E^-2 -> E^-1 (milder than flat, per v6 lesson)

def oof(model_fn, A, w=None):
    p = np.zeros_like(t)
    for tr, va in KFold(3, shuffle=True, random_state=0).split(A):
        m = model_fn(); m.fit(A[tr], t[tr], sample_weight=None if w is None else w[tr]); p[va] = m.predict(A[va])
    return p

def summ(name, p):
    out = []
    for a, b in zip(EDGES[:-1], EDGES[1:]):
        m = (t >= a) & (t < b); dl = p[m] - t[m]
        out.append(f"{100*(10**np.median(dl)-1):+5.0f}%/{0.5*(np.percentile(dl,84)-np.percentile(dl,16)):.2f}")
    print(f"{name:34s} " + "  ".join(out))

print("bias/sigma(logE) per bin: " + "  ".join(f"{10**a/1000:.1f}-{10**b/1000:.1f}TeV" for a, b in zip(EDGES[:-1], EDGES[1:])))
rf = lambda: RandomForestRegressor(300, max_depth=18, min_samples_leaf=3, n_jobs=-1, random_state=0)
gb = lambda: HistGradientBoostingRegressor(max_iter=600, learning_rate=0.05, max_leaf_nodes=31, random_state=0)
for fname, A in [('reco', aug(X)), ('oracle', aug(Xo))]:
    summ(f"RF   {fname} unweighted", oof(rf, A))
    summ(f"RF   {fname} E^-1 weighted", oof(rf, A, w_flat))
    summ(f"HGB  {fname} E^-1 weighted", oof(gb, A, w_flat))

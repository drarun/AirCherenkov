"""
Task 4: Stereo RF energy reconstruction benchmark (gammas, 750 m throw, 30 deg zenith).

- 5-fold out-of-fold RandomForest on the 65 stereo features, target log10(E_true/GeV).
- Bias correction: inverse MEDIAN RESPONSE fitted on the training fold's OOB predictions
  (median reco per TRUE-energy bin, inverted). This removes bias in true-energy bins;
  isotonic E[true|reco] does not (see scratch/calibration_ceiling.py).
- Variants:
    reco   : features as reconstructed (deployable)
    oracle : impact-distance features 61-64 replaced by TRUE shower-frame distances
             (diagnostic ceiling: how much the poor geometric core costs)
- Selections: 'all' (geom_ok) and 'quality' (n_trig>=3, size>=300 p.e., reco core < 250 m).
"""
import os, shutil, sys
sys.path.insert(0, 'src')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from sim.telescope import TelescopeArray

Z_OBS = 1275.0
EDGES = np.log10([100, 200, 400, 800, 1600, 3200, 6400, 12800, 30000])
CTR = 0.5 * (EDGES[:-1] + EDGES[1:])


def true_impact_distances(d, idx, X):
    tels = TelescopeArray.veritas_array().telescopes
    zen = np.radians(d['true_zenith'][idx]); azi = np.radians(d['true_azimuth'][idx] + 180.0)
    axis = np.stack([np.sin(zen) * np.cos(azi), np.sin(zen) * np.sin(azi), np.cos(zen)], 1)
    core = np.stack([d['impact_x'][idx], d['impact_y'][idx], np.full(len(idx), Z_OBS)], 1)
    out = np.zeros((len(idx), 4), np.float32)
    for t, tel in enumerate(tels):
        r = np.array([tel.x_tel, tel.y_tel, tel.z_tel]) - core
        perp = r - (r * axis).sum(1, keepdims=True) * axis
        out[:, t] = np.linalg.norm(perp, axis=1) * (X[:, t * 13] > 0)
    return out


def inverse_median_response(pred_tr, true_tr):
    med, ctr = [], []
    for a, b in zip(EDGES[:-1], EDGES[1:]):
        m = (true_tr >= a) & (true_tr < b)
        if m.sum() >= 15:
            med.append(np.median(pred_tr[m])); ctr.append(0.5 * (a + b))
    med = np.maximum.accumulate(np.array(med) + 1e-6 * np.arange(len(med)))
    ctr = np.array(ctr)
    slope_lo = (ctr[1] - ctr[0]) / (med[1] - med[0])
    slope_hi = (ctr[-1] - ctr[-2]) / (med[-1] - med[-2])

    def f(p):
        out = np.interp(p, med, ctr)
        out = np.where(p < med[0], ctr[0] + (p - med[0]) * slope_lo, out)
        return np.where(p > med[-1], ctr[-1] + (p - med[-1]) * slope_hi, out)
    return f


def run(X, t):
    raw = np.zeros_like(t); cal = np.zeros_like(t)
    for k, (tr, va) in enumerate(KFold(5, shuffle=True, random_state=42).split(X)):
        rf = RandomForestRegressor(n_estimators=300, max_depth=18, min_samples_leaf=3,
                                   oob_score=True, n_jobs=-1, random_state=42 + k)
        rf.fit(X[tr], t[tr])
        f = inverse_median_response(rf.oob_prediction_, t[tr])
        raw[va] = rf.predict(X[va]); cal[va] = f(raw[va])
    return raw, cal


def per_bin(t, p):
    rows = []
    for a, b in zip(EDGES[:-1], EDGES[1:]):
        m = (t >= a) & (t < b)
        if m.sum() < 15:
            rows.append((np.nan,) * 4); continue
        r = 10 ** (p[m] - t[m]) - 1
        dl = p[m] - t[m]
        rows.append((100 * np.median(r), 100 * np.percentile(np.abs(r - np.median(r)), 68),
                     0.5 * (np.percentile(dl, 84) - np.percentile(dl, 16)), m.sum()))
    return np.array(rows)


def main():
    d = np.load('data/stereo_features_750m.npz')
    X_all = d['X']; y = d['labels']; ok = d['geom_ok']
    idx = np.where((y == 1) & ok)[0]
    X = X_all[idx]; t = d['log10_e'][idx]
    X_or = X.copy(); X_or[:, 61:65] = true_impact_distances(d, idx, X)
    quality = (X[:, 52] >= 3) & (X[:, 53] >= np.log10(301)) & (np.hypot(X[:, 57], X[:, 58]) < 250)
    print(f"Gammas with geometry: {len(idx)}; passing quality cuts: {quality.sum()}")

    res = {}
    for name, XX in [('reco', X), ('oracle', X_or)]:
        raw, cal = run(XX, t)
        res[name] = (raw, cal)

    lines = []
    def report(title, t_, p_):
        rows = per_bin(t_, p_)
        lines.append(f"\n{title}\n  E_true [GeV]      N   bias    res68(|dE/E - med|)  sigma(logE)")
        for (a, b), r in zip(zip(EDGES[:-1], EDGES[1:]), rows):
            lines.append(f"  {10**a:6.0f}-{10**b:6.0f} {int(r[3]) if np.isfinite(r[3]) else 0:5d} {r[0]:+6.1f}%   {r[1]:6.1f}%             {r[2]:.3f}")
        return rows

    tabs = {}
    for name in ['reco', 'oracle']:
        raw, cal = res[name]
        tabs[(name, 'all')] = report(f"[{name}] ALL geom_ok, calibrated", t, cal)
        tabs[(name, 'quality')] = report(f"[{name}] QUALITY CUTS, calibrated", t[quality], cal[quality])
    report("[reco] ALL geom_ok, RAW (uncalibrated RF)", t, res['reco'][0])
    txt = "\n".join(lines); print(txt)
    with open('scratch/energy_benchmark.txt', 'w') as fh:
        fh.write(txt)

    # Plot
    fig, ax = plt.subplots(2, 2, figsize=(14, 11))
    E = 10 ** CTR / 1000
    sty = {('reco', 'all'): ('#1f77b4', '-', 'Reco features, all'),
           ('reco', 'quality'): ('#1f77b4', '--', 'Reco features, quality cuts'),
           ('oracle', 'all'): ('#2ca02c', '-', 'Oracle core, all'),
           ('oracle', 'quality'): ('#2ca02c', '--', 'Oracle core, quality cuts')}
    for key, (c, ls, lab) in sty.items():
        ax[0, 0].plot(E, tabs[key][:, 0], color=c, ls=ls, marker='o', label=lab)
        ax[0, 1].plot(E, tabs[key][:, 1], color=c, ls=ls, marker='o', label=lab)
    ax[0, 0].axhspan(-10, 10, color='gold', alpha=0.25, label='VERITAS target |bias| < 10%')
    ax[0, 1].axhspan(15, 25, color='gold', alpha=0.25, label='VERITAS typical 15-25%')
    for a, ttl, yl in [(ax[0, 0], 'A. Energy bias (calibrated)', 'Median dE/E [%]'),
                       (ax[0, 1], 'B. Energy resolution (calibrated)', '68% containment |dE/E - median| [%]')]:
        a.set_xscale('log'); a.set_xlabel('True energy [TeV]'); a.set_ylabel(yl)
        a.set_title(ttl, fontweight='bold'); a.grid(True, which='both', alpha=0.3); a.legend(fontsize=9)
    ax[0, 0].set_ylim(-60, 60); ax[0, 1].set_ylim(0, 100)
    for a, name in [(ax[1, 0], 'reco'), (ax[1, 1], 'oracle')]:
        h = a.hist2d(t, res[name][1], bins=60, range=[[2, 4.5], [1.5, 5]], cmin=1, cmap='viridis',
                     norm=matplotlib.colors.LogNorm())
        a.plot([2, 4.5], [2, 4.5], 'w--')
        a.set_xlabel('log10(E_true / GeV)'); a.set_ylabel('log10(E_reco / GeV), calibrated')
        a.set_title(f'{"C" if name == "reco" else "D"}. Migration ({name} features)', fontweight='bold')
        fig.colorbar(h[3], ax=a)
    plt.suptitle('Stereo RF Energy Reconstruction | 30 deg zenith, 750 m throw, E^-2 gammas | 5-fold OOF',
                 fontweight='bold')
    plt.tight_layout()
    plt.savefig('scratch/energy_benchmark.png', dpi=150)
    art = r"C:\Users\aruns\.gemini\antigravity-cli\brain\f0e1b905-b5d2-4fce-80a1-d0620d2d7de7"
    shutil.copyfile('scratch/energy_benchmark.png', os.path.join(art, 'energy_benchmark.png'))
    print("Saved scratch/energy_benchmark.png")


if __name__ == '__main__':
    main()

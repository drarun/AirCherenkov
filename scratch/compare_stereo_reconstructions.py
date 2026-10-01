"""
Compare Geometric Stereoscopic Reconstruction vs Machine Learning Stereoscopic Reconstruction
against Monte Carlo Truth for 1 TeV Gamma-ray events.

Improvements over v1:
- ML features now include asymmetry, miss, alpha, and geometric azimuth seed (61-dim)
- Native multi-output RandomForestRegressor (shared trees, captures X/Y correlations)
- Fixed wobble ring geometry (circular on the sky)
- Larger evaluation dataset (~880 events)
- Proper empirical CDF computation
"""
import os
import sys
sys.path.insert(0, 'src')

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold

from sim.camera import Camera
from sim.telescope import TelescopeArray
from recon.stereo import reconstruct_event_geometric
from recon.ml_stereo import extract_stereo_features, MLStereoReconstructor


def compute_angular_separation(zen1_deg, azi1_deg, zen2_deg, azi2_deg):
    """Compute spherical angular separation in degrees."""
    z1, a1 = np.radians(zen1_deg), np.radians(azi1_deg)
    z2, a2 = np.radians(zen2_deg), np.radians(azi2_deg)

    v1 = np.column_stack([np.sin(z1) * np.cos(a1), np.sin(z1) * np.sin(a1), np.cos(z1)])
    v2 = np.column_stack([np.sin(z2) * np.cos(a2), np.sin(z2) * np.sin(a2), np.cos(z2)])

    cos_sep = np.sum(v1 * v2, axis=1)
    cos_sep = np.clip(cos_sep, -1.0, 1.0)
    return np.degrees(np.arccos(cos_sep))


def main():
    # Use new v2 data (fixed wobble geometry, larger dataset)
    data_path = 'data/stereo_eval_v2/sim_batch_00000.pt'
    if not os.path.exists(data_path):
        # Fall back to original data
        data_path = 'data/stereo_eval/sim_batch_00000.pt'
    if not os.path.exists(data_path):
        print(f"Error: no evaluation data found.")
        sys.exit(1)

    print(f"Loading events from {data_path}...")
    events = torch.load(data_path, weights_only=False)
    n_events = len(events)
    print(f"Loaded {n_events} events.")

    cam = Camera(n_rings=12)
    array = TelescopeArray.veritas_array()
    tels = array.telescopes

    # 1. Feature Extraction & Geometric Reconstruction
    print("Running Geometric Reconstruction and extracting features...")
    X_list = []
    y_impact_list = []
    y_direction_list = []

    geom_impact_x = []
    geom_impact_y = []
    geom_zenith = []
    geom_azimuth = []
    geom_valid = []

    true_impact_x = []
    true_impact_y = []
    true_zenith = []
    true_azimuth = []

    for idx, ev in enumerate(events):
        traces = ev['fadc_traces']
        p_zen = ev.get('zenith_deg', 30.0)
        p_azi = ev.get('azimuth_deg', 0.0)

        t_ix = float(ev['impact_x'])
        t_iy = float(ev['impact_y'])
        t_zen = float(ev.get('true_zenith_deg', p_zen))
        t_azi = float(ev.get('true_azimuth_deg', p_azi))

        true_impact_x.append(t_ix)
        true_impact_y.append(t_iy)
        true_zenith.append(t_zen)
        true_azimuth.append(t_azi)

        # Feature vector for ML
        feat = extract_stereo_features(traces, tels, cam, pointing_zenith_deg=p_zen, pointing_azimuth_deg=p_azi)
        X_list.append(feat)
        y_impact_list.append([t_ix, t_iy])
        y_direction_list.append([t_zen, t_azi])

        # Geometric Reconstruction
        res = reconstruct_event_geometric(traces, tels, cam, pointing_zenith_deg=p_zen, pointing_azimuth_deg=p_azi)
        if res is not None:
            geom_valid.append(True)
            geom_impact_x.append(res.reco_impact_x)
            geom_impact_y.append(res.reco_impact_y)
            geom_zenith.append(res.reco_zenith_deg)
            geom_azimuth.append(res.reco_azimuth_deg)
        else:
            geom_valid.append(False)
            geom_impact_x.append(np.nan)
            geom_impact_y.append(np.nan)
            geom_zenith.append(np.nan)
            geom_azimuth.append(np.nan)

    X = np.array(X_list, dtype=np.float32)
    y_impact = np.array(y_impact_list, dtype=np.float32)
    y_direction = np.array(y_direction_list, dtype=np.float32)

    geom_valid = np.array(geom_valid)
    geom_impact_x = np.array(geom_impact_x)
    geom_impact_y = np.array(geom_impact_y)
    geom_zenith = np.array(geom_zenith)
    geom_azimuth = np.array(geom_azimuth)

    true_impact_x = np.array(true_impact_x)
    true_impact_y = np.array(true_impact_y)
    true_zenith = np.array(true_zenith)
    true_azimuth = np.array(true_azimuth)

    print(f"Geometric reconstruction succeeded for {np.sum(geom_valid)} / {n_events} events ({100*np.mean(geom_valid):.1f}%).")

    # 2. ML 5-Fold Cross Validation
    print("Training ML Stereoscopic Reconstructor with 5-fold cross-validation...")
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    ml_pred_impact = np.zeros_like(y_impact)
    ml_pred_direction = np.zeros_like(y_direction)

    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        model = MLStereoReconstructor(n_estimators=200, max_depth=14, random_state=42 + fold)
        model.fit(X[train_idx], y_impact[train_idx], y_direction[train_idx])
        p_imp, p_dir = model.predict(X[val_idx])
        ml_pred_impact[val_idx] = p_imp
        ml_pred_direction[val_idx] = p_dir

    # 3. Compute Metrics on common valid events
    valid_idx = np.where(geom_valid)[0]

    # Geometric metrics
    geom_imp_err = np.hypot(geom_impact_x[valid_idx] - true_impact_x[valid_idx],
                            geom_impact_y[valid_idx] - true_impact_y[valid_idx])
    geom_ang_err = compute_angular_separation(
        geom_zenith[valid_idx], geom_azimuth[valid_idx],
        true_zenith[valid_idx], true_azimuth[valid_idx]
    )

    # ML metrics (out-of-fold)
    ml_imp_err = np.hypot(ml_pred_impact[valid_idx, 0] - true_impact_x[valid_idx],
                          ml_pred_impact[valid_idx, 1] - true_impact_y[valid_idx])
    ml_ang_err = compute_angular_separation(
        ml_pred_direction[valid_idx, 0], ml_pred_direction[valid_idx, 1],
        true_zenith[valid_idx], true_azimuth[valid_idx]
    )

    # Print summary statistics
    print("\n" + "=" * 70)
    print("           STEREOSCOPIC RECONSTRUCTION BENCHMARK COMPARISON (v2)      ")
    print("=" * 70)
    print(f"Events Evaluated: {len(valid_idx)} (1 TeV Gamma, 30 deg Zen, 0.5 deg Wobble, 100m Imp)")
    print("-" * 70)
    print("METRIC                           | GEOMETRIC (SDP) | ML (RANDOM FOREST)")
    print("-" * 70)
    print(f"Direction 68% Containment (t68)  |    {np.percentile(geom_ang_err, 68):.3f} deg     |    {np.percentile(ml_ang_err, 68):.3f} deg")
    print(f"Direction Median Error           |    {np.median(geom_ang_err):.3f} deg     |    {np.median(ml_ang_err):.3f} deg")
    print(f"Direction Mean Error             |    {np.mean(geom_ang_err):.3f} deg     |    {np.mean(ml_ang_err):.3f} deg")
    print(f"Direction 95% Containment (t95)  |    {np.percentile(geom_ang_err, 95):.3f} deg     |    {np.percentile(ml_ang_err, 95):.3f} deg")
    print("-" * 70)
    print(f"Impact Core 68% Containment (R68)|    {np.percentile(geom_imp_err, 68):.1f} m         |    {np.percentile(ml_imp_err, 68):.1f} m")
    print(f"Impact Core Median Error         |    {np.median(geom_imp_err):.1f} m         |    {np.median(ml_imp_err):.1f} m")
    print(f"Impact Core Mean Error           |    {np.mean(geom_imp_err):.1f} m         |    {np.mean(ml_imp_err):.1f} m")
    print(f"Impact Core 95% Containment (R95)|    {np.percentile(geom_imp_err, 95):.1f} m         |    {np.percentile(ml_imp_err, 95):.1f} m")
    print("=" * 70)

    # 4. Generate Publication-Quality Comparison Plot
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # Panel A: Angular Resolution Cumulative Distribution
    ax = axes[0, 0]
    sorted_geom_ang = np.sort(geom_ang_err)
    sorted_ml_ang = np.sort(ml_ang_err)
    # Proper empirical CDF: F(x_i) = i / N
    ecdf_frac = np.arange(1, len(sorted_geom_ang) + 1) / len(sorted_geom_ang) * 100

    g68_ang = np.percentile(geom_ang_err, 68)
    m68_ang = np.percentile(ml_ang_err, 68)

    ax.plot(sorted_geom_ang, ecdf_frac, label=f"Geometric SDP (θ68: {g68_ang:.3f}°)", color='#1f77b4', lw=2.5)
    ax.plot(sorted_ml_ang, ecdf_frac, label=f"ML Random Forest (θ68: {m68_ang:.3f}°)", color='#ff7f0e', lw=2.5, ls='--')
    ax.axhline(68.0, color='gray', ls=':', alpha=0.7, label='68% Containment')
    ax.axvline(g68_ang, color='#1f77b4', ls=':', alpha=0.5)
    ax.axvline(m68_ang, color='#ff7f0e', ls=':', alpha=0.5)
    ax.set_xlim(0, max(1.2, float(np.percentile(geom_ang_err, 95))))
    ax.set_ylim(0, 100)
    ax.set_xlabel("Angular Error $\\Delta\\theta$ [degrees]", fontsize=11)
    ax.set_ylabel("Cumulative Fraction [%]", fontsize=11)
    ax.set_title("A. Direction Reconstruction Resolution (Cumulative)", fontsize=12, fontweight='bold')
    ax.legend(loc='lower right', frameon=True, fontsize=10)
    ax.grid(True, alpha=0.3)

    # Panel B: Core Impact Resolution Cumulative Distribution
    ax = axes[0, 1]
    sorted_geom_imp = np.sort(geom_imp_err)
    sorted_ml_imp = np.sort(ml_imp_err)
    ecdf_frac_imp = np.arange(1, len(sorted_geom_imp) + 1) / len(sorted_geom_imp) * 100

    g68_imp = np.percentile(geom_imp_err, 68)
    m68_imp = np.percentile(ml_imp_err, 68)

    ax.plot(sorted_geom_imp, ecdf_frac_imp, label=f"Geometric SDP (R68: {g68_imp:.1f} m)", color='#1f77b4', lw=2.5)
    ax.plot(sorted_ml_imp, ecdf_frac_imp, label=f"ML Random Forest (R68: {m68_imp:.1f} m)", color='#ff7f0e', lw=2.5, ls='--')
    ax.axhline(68.0, color='gray', ls=':', alpha=0.7, label='68% Containment')
    ax.axvline(g68_imp, color='#1f77b4', ls=':', alpha=0.5)
    ax.axvline(m68_imp, color='#ff7f0e', ls=':', alpha=0.5)
    ax.set_xlim(0, max(100.0, float(np.percentile(geom_imp_err, 95))))
    ax.set_ylim(0, 100)
    ax.set_xlabel("Core Position Error $\\Delta R$ [meters]", fontsize=11)
    ax.set_ylabel("Cumulative Fraction [%]", fontsize=11)
    ax.set_title("B. Core Impact Reconstruction Resolution (Cumulative)", fontsize=12, fontweight='bold')
    ax.legend(loc='lower right', frameon=True, fontsize=10)
    ax.grid(True, alpha=0.3)

    # Panel C: Ground Impact Parameter Scatter (True vs Reconstructed)
    ax = axes[1, 0]
    ax.scatter(true_impact_x[valid_idx], true_impact_y[valid_idx], c='black', marker='o', s=30, alpha=0.6, label='MC Truth Core')
    ax.scatter(geom_impact_x[valid_idx], geom_impact_y[valid_idx], c='#1f77b4', marker='^', s=35, alpha=0.7, label='Geometric Reco')
    ax.scatter(ml_pred_impact[valid_idx, 0], ml_pred_impact[valid_idx, 1], c='#ff7f0e', marker='x', s=35, alpha=0.7, label='ML Reco')

    # Draw telescopes
    for t_idx, tel in enumerate(tels):
        ax.plot(tel.x_tel, tel.y_tel, 's', color='darkred', markersize=9, zorder=5)
        ax.text(tel.x_tel + 4, tel.y_tel + 4, f"T{t_idx}", color='darkred', fontweight='bold', fontsize=9)

    circ = plt.Circle((0, 0), 100.0, color='gray', fill=False, ls='--', lw=1.5, label='100 m Throw Radius')
    ax.add_patch(circ)
    ax.set_xlim(-150, 150)
    ax.set_ylim(-150, 150)
    ax.set_aspect('equal')
    ax.set_xlabel("Ground X [meters]", fontsize=11)
    ax.set_ylabel("Ground Y [meters]", fontsize=11)
    ax.set_title("C. Core Position (Ground Array Frame)", fontsize=12, fontweight='bold')
    ax.legend(loc='upper right', frameon=True, fontsize=9)
    ax.grid(True, alpha=0.3)

    # Panel D: Arrival Direction Distribution (Sky Frame)
    ax = axes[1, 1]
    # Projected offset relative to pointing
    p_zen, p_azi = 30.0, 0.0
    true_du = (true_azimuth[valid_idx] - p_azi) * np.sin(np.radians(true_zenith[valid_idx]))
    true_dv = true_zenith[valid_idx] - p_zen

    geom_du = (geom_azimuth[valid_idx] - p_azi) * np.sin(np.radians(geom_zenith[valid_idx]))
    geom_dv = geom_zenith[valid_idx] - p_zen

    ml_du = (ml_pred_direction[valid_idx, 1] - p_azi) * np.sin(np.radians(ml_pred_direction[valid_idx, 0]))
    ml_dv = ml_pred_direction[valid_idx, 0] - p_zen

    ax.scatter(true_du, true_dv, c='black', marker='o', s=30, alpha=0.6, label='MC Truth (0.5° Wobble)')
    ax.scatter(geom_du, geom_dv, c='#1f77b4', marker='^', s=35, alpha=0.7, label='Geometric Reco')
    ax.scatter(ml_du, ml_dv, c='#ff7f0e', marker='x', s=35, alpha=0.7, label='ML Reco')

    # Draw wobble circle
    wobble_circ = plt.Circle((0, 0), 0.5, color='gray', fill=False, ls='--', lw=1.5, label='0.5° Wobble Ring')
    ax.add_patch(wobble_circ)
    ax.plot(0, 0, '+', color='red', markersize=10, mew=2, label='Telescope Pointing')
    ax.set_xlim(-0.9, 0.9)
    ax.set_ylim(-0.9, 0.9)
    ax.set_aspect('equal')
    ax.set_xlabel("$\\Delta$Azimuth $\\cdot \\sin(\\theta)$ [degrees]", fontsize=11)
    ax.set_ylabel("$\\Delta$Zenith [degrees]", fontsize=11)
    ax.set_title("D. Arrival Direction (Camera / Sky Field-of-View)", fontsize=12, fontweight='bold')
    ax.legend(loc='upper right', frameon=True, fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.suptitle("Stereoscopic Direction and Impact Reconstruction: Geometric vs ML vs MC Truth\n"
                 "VERITAS Array Simulation | 1 TeV Gamma-Rays | 30° Zenith, 0.5° Wobble Offset, 100m Impact Radius",
                 fontsize=13, fontweight='bold', y=0.99)
    plt.tight_layout()
    plot_file = 'scratch/stereo_reconstruction_comparison_v2.png'
    plt.savefig(plot_file, dpi=150)
    print(f"\nSaved comparison plot to {plot_file}")

    # Copy to artifacts directory
    artifact_dir = r"C:\Users\aruns\.gemini\antigravity-cli\brain\f0e1b905-b5d2-4fce-80a1-d0620d2d7de7"
    if os.path.exists(artifact_dir):
        dest_plot = os.path.join(artifact_dir, "stereo_reconstruction_comparison_v2.png")
        import shutil
        shutil.copyfile(plot_file, dest_plot)
        print(f"Copied plot to artifact directory: {dest_plot}")


if __name__ == '__main__':
    main()

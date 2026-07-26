"""
Regenerate all AirCherenkov visualizations after physics fixes.
Produces:
  - gamma_shower.html         (3D particle tracks)
  - proton_shower.html        (3D particle tracks)
  - gamma_cherenkov_pool.html (Cherenkov ground footprint)
  - proton_cherenkov_pool.html
  - camera_cleaning_comparison.png (camera image + cleaning + Hillas)
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import numpy as np
import time
import plotly.graph_objects as go
import plotly.express as px

from sim.shower import ShowerSimulation
from sim.telescope import Telescope
from sim.backend import device_info
from recon.cleaning import tail_cut_clean, double_pass_clean
from recon.hillas import compute_hillas, print_hillas

# ── Style settings (matching proton_shower.html dark theme) ──────────
COLOR_MAP = {
    'gamma': '#FFD700',       # Gold
    'e-': '#4DA6FF',          # Sky blue
    'e+': '#FF4D4D',          # Coral red
    'proton': '#FFFFFF',      # White
    'pi0': '#00FF88',         # Neon green
    'pi_charged': '#FF8C00',  # Dark orange
    'mu': '#DA70D6',          # Orchid purple
}

LAYOUT = dict(
    template='plotly_dark',
    paper_bgcolor='#0a0a0a',
    plot_bgcolor='#0a0a0a',
    font=dict(family='Courier New', color='#e0e0e0'),
    width=1100,
    height=900,
)

def plot_shower_3d(df, title):
    """3D particle track plot with proton_shower.html styling."""
    fig = go.Figure()
    
    # Downsample if too many particles (keep first 5000 for viz)
    unique_ids = df['particle_id'].unique()
    if len(unique_ids) > 5000:
        keep_ids = np.random.choice(unique_ids, 5000, replace=False)
        df = df[df['particle_id'].isin(keep_ids)]
    
    grouped = df.groupby('particle_id')
    for pid, group in grouped:
        ptype = group['pid'].iloc[0]
        fig.add_trace(go.Scatter3d(
            x=group['x'], y=group['y'], z=group['z'],
            mode='lines',
            line=dict(color=COLOR_MAP.get(ptype, 'gray'), width=1.5),
            name=ptype, showlegend=False,
            hoverinfo='text',
            hovertext=f"Type: {ptype}<br>Energy: {group['energy'].iloc[0]:.2f} GeV"
        ))
    
    # Legend entries
    for ptype, color in COLOR_MAP.items():
        if ptype in df['pid'].unique():
            fig.add_trace(go.Scatter3d(
                x=[None], y=[None], z=[None],
                mode='lines', line=dict(color=color, width=5),
                name=ptype
            ))
    
    fig.update_layout(
        title=dict(text=title, font=dict(size=20)),
        scene=dict(
            xaxis_title='X (meters)',
            yaxis_title='Y (meters)',
            zaxis_title='Altitude (meters)',
            aspectmode='data',
            xaxis=dict(backgroundcolor='#0a0a0a', gridcolor='#333'),
            yaxis=dict(backgroundcolor='#0a0a0a', gridcolor='#333'),
            zaxis=dict(backgroundcolor='#0a0a0a', gridcolor='#333'),
        ),
        legend_title="Particle Type",
        **LAYOUT
    )
    return fig


def plot_cherenkov_pool(cherenkov_df, title):
    """Cherenkov ground footprint with dark styling."""
    fig = px.scatter(cherenkov_df, x='x', y='y', opacity=0.05,
                     title=title, width=900, height=900,
                     range_x=[-1500, 1500], range_y=[-1500, 1500],
                     template='plotly_dark')
    fig.update_traces(marker=dict(size=2, color='#FFD700'))
    fig.update_layout(
        paper_bgcolor='#0a0a0a', plot_bgcolor='#0a0a0a',
        xaxis_title="Ground X (m)", yaxis_title="Ground Y (m)",
        font=dict(family='Courier New', color='#e0e0e0'),
    )
    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    return fig


def main():
    print("=" * 60)
    print("AirCherenkov - Regenerate All Visualizations")
    print("=" * 60)
    print(f"Compute backend: {device_info()}")
    
    ENERGY = 5000.0  # 5 TeV
    MAX_GEN = 18
    
    # ── 1. Gamma-ray shower ──────────────────────────────────────────
    print(f"\n[1/5] Simulating {ENERGY/1000:.0f} TeV Gamma-ray shower...")
    t0 = time.time()
    gamma_sim = ShowerSimulation('gamma', energy=ENERGY)
    gamma_sim.run(max_generations=MAX_GEN)
    gamma_df = gamma_sim.get_tracks_dataframe()
    gamma_cherenkov = gamma_sim.get_cherenkov_dataframe()
    n_phot = len(gamma_sim.cherenkov_photons.get('x_ground', []))
    print(f"      {len(gamma_sim.all_particles)} particles, {n_phot:,} Cherenkov photons ({time.time()-t0:.1f}s)")
    
    # ── 2. Proton shower ─────────────────────────────────────────────
    print(f"\n[2/5] Simulating {ENERGY/1000:.0f} TeV Proton shower...")
    t0 = time.time()
    proton_sim = ShowerSimulation('proton', energy=ENERGY)
    proton_sim.run(max_generations=MAX_GEN)
    proton_df = proton_sim.get_tracks_dataframe()
    proton_cherenkov = proton_sim.get_cherenkov_dataframe()
    n_phot_p = len(proton_sim.cherenkov_photons.get('x_ground', []))
    print(f"      {len(proton_sim.all_particles)} particles, {n_phot_p:,} Cherenkov photons ({time.time()-t0:.1f}s)")
    
    # ── 3. Generate 3D HTML plots ────────────────────────────────────
    print("\n[3/5] Generating 3D interactive HTML plots...")
    
    fig = plot_shower_3d(gamma_df, f"Gamma-ray Shower - {ENERGY/1000:.0f} TeV (Electromagnetic Cascade)")
    fig.write_html("gamma_shower.html")
    print("      [OK] gamma_shower.html")
    
    fig = plot_shower_3d(proton_df, f"Proton Shower - {ENERGY/1000:.0f} TeV (Hadronic Cascade)")
    fig.write_html("proton_shower.html")
    print("      [OK] proton_shower.html")
    
    # ── 4. Cherenkov pool plots ──────────────────────────────────────
    print("\n[4/5] Generating Cherenkov ground pool plots...")
    
    if not gamma_cherenkov.empty:
        fig = plot_cherenkov_pool(gamma_cherenkov, 
                                  f"Gamma-ray Cherenkov Pool - {ENERGY/1000:.0f} TeV")
        fig.write_html("gamma_cherenkov_pool.html")
        print("      [OK] gamma_cherenkov_pool.html")
    
    if not proton_cherenkov.empty:
        fig = plot_cherenkov_pool(proton_cherenkov,
                                  f"Proton Cherenkov Pool - {ENERGY/1000:.0f} TeV")
        fig.write_html("proton_cherenkov_pool.html")
        print("      [OK] proton_cherenkov_pool.html")
    
    # ── 5. Camera image + cleaning + Hillas ──────────────────────────
    print("\n[5/5] Ray-tracing & camera analysis (gamma shower)...")
    # Move telescope to a typical impact distance of 120m to see an elliptical image
    tel = Telescope(x_tel=120.0, y_tel=0.0)
    raw_image = tel.ray_trace(gamma_sim.cherenkov_photons)
    
    mask_tc = tail_cut_clean(tel.camera, raw_image)
    mask_dp = double_pass_clean(tel.camera, raw_image)
    
    img_tc = np.where(mask_tc, raw_image, 0.0)
    img_dp = np.where(mask_dp, raw_image, 0.0)
    
    hillas_tc = compute_hillas(tel.camera, raw_image, mask_tc)
    hillas_dp = compute_hillas(tel.camera, raw_image, mask_dp)
    
    signal_pe = np.sum(raw_image[mask_tc]) if np.any(mask_tc) else 0
    print(f"      Camera: {tel.camera.n_pixels} pixels, ~{signal_pe:.0f} signal p.e. above NSB")
    print(f"      Tail-cut: {np.sum(mask_tc)} pixels survived")
    print(f"      Double-pass: {np.sum(mask_dp)} pixels survived")
    
    if hillas_tc:
        print("\n-- Standard Tail-Cut --")
        print_hillas(hillas_tc)
    if hillas_dp:
        print("\n-- Double-Pass --")
        print_hillas(hillas_dp)
    
    # Generate camera comparison PNG
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Ellipse
    
    fig_cam, axes = plt.subplots(1, 3, figsize=(22, 7))
    fig_cam.suptitle(f'AirCherenkov: Gamma-ray Shower ({ENERGY/1000:.0f} TeV)', 
                      fontsize=16, fontweight='bold')
    
    tel.camera.plot_image(raw_image, ax=axes[0], title="Raw Image (+ NSB)", cmap="inferno")
    tel.camera.plot_image(img_tc, ax=axes[1], title="Standard Tail-Cut (5/2.5)", cmap="inferno")
    tel.camera.plot_image(img_dp, ax=axes[2], title="Double-Pass (5/2.5 > 2/1.0)", cmap="inferno")
    
    # Draw Hillas ellipse on double-pass panel
    if hillas_dp:
        h = hillas_dp
        ell = Ellipse(xy=(h.centroid_x, h.centroid_y),
                      width=2*h.length, height=2*h.width,
                      angle=np.degrees(h.psi),
                      edgecolor='lime', facecolor='none', linewidth=2, linestyle='--')
        axes[2].add_patch(ell)
        axes[2].plot(h.centroid_x, h.centroid_y, 'x', color='lime', markersize=12, markeredgewidth=2)
    
    plt.tight_layout()
    fig_cam.savefig('camera_cleaning_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("\n      [OK] camera_cleaning_comparison.png")
    
    print("\n" + "=" * 60)
    print("All images regenerated successfully!")
    print("=" * 60)


if __name__ == '__main__':
    main()

"""
Generate camera images for the first 4 fixed events from the batch file.
Shows 4 events × 4 telescopes in a grid.
"""
import sys
sys.path.insert(0, 'src')
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sim.camera import Camera
from recon.hillas import compute_hillas

cam = Camera(n_rings=12)
batch = torch.load('data/viewer_test/raw_fixed/sim_batch_00000.pt', weights_only=False)
print(f"Loaded {len(batch)} events")

# Show 4 events
n_show = min(4, len(batch))
fig, axes = plt.subplots(n_show, 4, figsize=(18, 4*n_show))
if n_show == 1:
    axes = axes[None, :]

for ev_idx in range(n_show):
    ev = batch[ev_idx]
    traces = ev['fadc_traces']  # (4, 469, 16)
    label = ev['label']
    energy = ev['energy']
    impact_x = ev.get('impact_x', 'N/A')
    impact_y = ev.get('impact_y', 'N/A')
    
    for t_idx in range(4):
        ax = axes[ev_idx, t_idx]
        c = np.sum(traces[t_idx], axis=-1)  # (469,)
        c_clean = np.maximum(0, c - 2.0)   # subtract NSB pedestal
        
        # Use 2-level cleaning: tailcut
        thresh_high = max(7.0, np.max(c_clean) * 0.3)
        thresh_low = thresh_high * 0.5
        core_mask = c_clean > thresh_high
        neighbor_mask = core_mask.copy()
        # simple neighbor check via camera geometry (approx with distance)
        for i in range(len(cam.pixel_x)):
            if core_mask[i]:
                dists = np.sqrt((cam.pixel_x - cam.pixel_x[i])**2 + (cam.pixel_y - cam.pixel_y[i])**2)
                neighbor_mask |= (dists < 0.3) & (c_clean > thresh_low)
        mask = neighbor_mask
        
        sc = ax.scatter(cam.pixel_x, cam.pixel_y, c=c_clean, cmap='viridis', s=18, alpha=0.9,
                        vmin=0, vmax=np.percentile(c_clean, 99)+1)
        
        if mask.sum() >= 3:
            ax.scatter(cam.pixel_x[mask], cam.pixel_y[mask],
                       facecolors='none', edgecolors='red', s=45, linewidth=1.5)
            h = compute_hillas(cam, c_clean, mask)
            if h:
                L = h.length
                x1 = h.centroid_x - L * np.cos(h.psi)
                y1 = h.centroid_y - L * np.sin(h.psi)
                x2 = h.centroid_x + L * np.cos(h.psi)
                y2 = h.centroid_y + L * np.sin(h.psi)
                ax.plot([x1, x2], [y1, y2], 'r-', lw=2)
                ax.plot(h.centroid_x, h.centroid_y, 'r^', ms=6)
                ax.set_title(f"Ev{ev_idx} T{t_idx}: L={h.length:.2f} W={h.width:.2f}")
            else:
                ax.set_title(f"Ev{ev_idx} T{t_idx}: max={c.max():.1f}PE")
        else:
            ax.set_title(f"Ev{ev_idx} T{t_idx}: max={c.max():.1f}PE ({mask.sum()} pix)")
        
        if ev_idx == 0 and t_idx == 0:
            ax.set_ylabel(f"Ev {ev_idx}: E={energy:.0f}GeV\nimp=({impact_x:.0f},{impact_y:.0f})m", 
                         fontsize=8)
        elif t_idx == 0:
            ax.set_ylabel(f"Ev {ev_idx}: imp=({impact_x:.0f},{impact_y:.0f})m", fontsize=8)
        
        ax.set_xlim(-2.1, 2.1)
        ax.set_ylim(-2.1, 2.1)
        ax.set_aspect('equal')
        circ = plt.Circle((0, 0), 1.75, color='gray', fill=False, ls=':')
        ax.add_patch(circ)

plt.suptitle("Fixed Events: 1 TeV Gamma, 30° zenith, Heitler+Obs cutoff+Slant timing", fontsize=11)
plt.tight_layout()
plt.savefig('scratch/camera_fixed_events.png', dpi=120)
print("Saved scratch/camera_fixed_events.png")

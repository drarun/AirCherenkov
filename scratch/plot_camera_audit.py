import torch
import numpy as np
import matplotlib.pyplot as plt
from sim.camera import Camera
from recon.hillas import compute_hillas

cam = Camera(n_rings=12)
events = torch.load('data/viewer_test/raw/sim_batch_00000.pt', weights_only=False)

fig, axes = plt.subplots(4, 4, figsize=(16, 16))

for ev_idx in range(4):
    ev = events[ev_idx]
    traces = ev['fadc_traces'] # (4, 469, 16)
    charges = np.sum(traces, axis=2) # (4, 469)
    for tel_idx in range(4):
        ax = axes[ev_idx, tel_idx]
        c = charges[tel_idx]
        
        # Two-level image cleaning (picture threshold 10 PE, boundary threshold 5 PE)
        # Or simple threshold > 8 PE
        mask = c > 8.0
        
        # Scatter plot of all pixels
        sc = ax.scatter(cam.pixel_x, cam.pixel_y, c=c, cmap='viridis', s=25, alpha=0.8)
        
        # Highlight cleaned pixels
        if np.sum(mask) > 0:
            ax.scatter(cam.pixel_x[mask], cam.pixel_y[mask], facecolors='none', edgecolors='red', s=45, linewidth=1.5)
            
        h = compute_hillas(cam, c, mask) if np.sum(mask) >= 3 else None
        if h:
            # Plot Hillas ellipse major axis
            L = h.length
            x1 = h.centroid_x - L * np.cos(h.psi)
            y1 = h.centroid_y - L * np.sin(h.psi)
            x2 = h.centroid_x + L * np.cos(h.psi)
            y2 = h.centroid_y + L * np.sin(h.psi)
            ax.plot([x1, x2], [y1, y2], 'r-', lw=2)
            ax.set_title(f"Ev {ev_idx} T{tel_idx}: L={h.length:.2f}, W={h.width:.2f}, N={np.sum(mask)}")
        else:
            ax.set_title(f"Ev {ev_idx} T{tel_idx}: Max={np.max(c):.1f}, N_clean={np.sum(mask)}")
        ax.set_xlim(-2.0, 2.0)
        ax.set_ylim(-2.0, 2.0)
        ax.set_aspect('equal')

plt.tight_layout()
plt.savefig('scratch/camera_images_audit.png', dpi=150)
print("Saved scratch/camera_images_audit.png")

import torch
import numpy as np
from sim.camera import Camera
from recon.hillas import compute_hillas

cam = Camera(n_rings=12)
events = torch.load('data/viewer_test/raw/sim_batch_00000.pt', weights_only=False)
print(f"Total events: {len(events)}")

for ev_idx in range(min(15, len(events))):
    ev = events[ev_idx]
    traces = ev['fadc_traces'] # shape (4, 469, 16)
    charges = np.sum(traces, axis=2) # shape (4, 469)
    print(f"--- Event {ev_idx}: Energy={ev['energy']} GeV, Impact=({ev['impact_x']:.1f}, {ev['impact_y']:.1f}) m ---")
    for tel_idx in range(4):
        c = charges[tel_idx]
        max_c = np.max(c)
        tot_c = np.sum(c)
        p5 = np.sum(c > 5)
        p10 = np.sum(c > 10)
        p20 = np.sum(c > 20)
        mask = c > 10.0
        h = compute_hillas(cam, c, mask) if np.sum(mask) >= 3 else None
        h_str = f"L={h.length:.4f}, W={h.width:.4f}, size={h.size:.1f}, (x,y)=({h.centroid_x:.2f},{h.centroid_y:.2f})" if h else "No Hillas (<3 pix > 10PE)"
        print(f"  Tel {tel_idx}: Max={max_c:.1f}, Tot={tot_c:.1f}, >5PE={p5}, >10PE={p10}, >20PE={p20} | {h_str}")

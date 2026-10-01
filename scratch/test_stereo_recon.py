import sys
sys.path.insert(0, 'src')
import torch
import numpy as np
from sim.camera import Camera
from recon.hillas import compute_hillas

cam = Camera(n_rings=12)
batch = torch.load('data/viewer_test/raw_fixed/sim_batch_00000.pt', weights_only=False)

def cam_to_sky(u_deg, v_deg, theta, phi):
    u = np.radians(u_deg)
    v = np.radians(v_deg)
    vec_cam = np.array([np.tan(u), np.tan(v), 1.0])
    vec_cam /= np.linalg.norm(vec_cam)
    
    Ry = np.array([
        [np.cos(theta), 0, np.sin(theta)],
        [0, 1, 0],
        [-np.sin(theta), 0, np.cos(theta)]
    ])
    Rz = np.array([
        [np.cos(phi), -np.sin(phi), 0],
        [np.sin(phi), np.cos(phi), 0],
        [0, 0, 1]
    ])
    return Rz @ (Ry @ vec_cam)

tels = [
    np.array([0.0, 0.0, 1268.0]),
    np.array([85.0, 32.0, 1268.0]),
    np.array([34.0, 107.0, 1268.0]),
    np.array([-41.0, 54.0, 1268.0])
]

z_obs = 1275.0

print(f"Testing stereoscopic reconstruction across {len(batch)} events...\n")
dir_errors_deg = []
impact_errors_m = []

for ev_idx, ev in enumerate(batch):
    traces = ev['fadc_traces']
    theta_p = np.radians(ev['zenith_deg'])
    phi_p = np.radians(ev['azimuth_deg'] + 180.0)
    
    planes = []
    for t_idx in range(4):
        c = np.maximum(0, np.sum(traces[t_idx], axis=-1) - 2.0)
        mask = c > 7.0
        for i in range(len(cam.pixel_x)):
            if mask[i]:
                d = np.hypot(cam.pixel_x - cam.pixel_x[i], cam.pixel_y - cam.pixel_y[i])
                mask |= (d < 0.25) & (c > 3.0)
        h = compute_hillas(cam, c, mask)
        if h and h.size > 100:
            cx, cy = h.centroid_x, h.centroid_y
            psi = h.psi
            delta = 0.5 * h.length
            p1 = cam_to_sky(cx - delta * np.cos(psi), cy - delta * np.sin(psi), theta_p, phi_p)
            p2 = cam_to_sky(cx + delta * np.cos(psi), cy + delta * np.sin(psi), theta_p, phi_p)
            n = np.cross(p1, p2)
            n /= np.linalg.norm(n)
            planes.append((t_idx, tels[t_idx], n, h.size, h.width / max(h.length, 1e-4)))
            
    if len(planes) < 2:
        continue
        
    dirs = []
    cores = []
    
    for i in range(len(planes)):
        for j in range(i + 1, len(planes)):
            t1, R1, n1, S1, wl1 = planes[i]
            t2, R2, n2, S2, wl2 = planes[j]
            d = np.cross(n1, n2)
            sin_ang = np.linalg.norm(d)
            if sin_ang < 0.05:
                continue
            d /= sin_ang
            if d[2] < 0:
                d = -d
            w = (sin_ang**2) / (1.0/S1 + 1.0/S2) * max(0.01, 1 - wl1) * max(0.01, 1 - wl2)
            dirs.append(d * w)
            
            b1 = np.dot(n1, R1) - n1[2] * z_obs
            b2 = np.dot(n2, R2) - n2[2] * z_obs
            A = np.array([[n1[0], n1[1]], [n2[0], n2[1]]])
            det = np.linalg.det(A)
            if abs(det) > 1e-4:
                core_xy = np.linalg.solve(A, np.array([b1, b2]))
                cores.append((core_xy, w * abs(det)))
                
    if not dirs or not cores:
        continue
        
    reco_dir = np.sum(dirs, axis=0)
    reco_dir /= np.linalg.norm(reco_dir)
    
    tot_cw = sum(cw for _, cw in cores)
    reco_core = sum(c * cw for c, cw in cores) / tot_cw
    
    # True direction vector in ground frame (direction towards the source in the sky)
    true_zen = np.radians(ev['zenith_deg'])
    true_azi = np.radians(ev['azimuth_deg'] + 180.0)
    true_dir = np.array([
        np.sin(true_zen) * np.cos(true_azi),
        np.sin(true_zen) * np.sin(true_azi),
        np.cos(true_zen)
    ])
    
    # Angular distance between reco_dir and true_dir in degrees
    cos_sep = np.clip(np.dot(reco_dir, true_dir), -1.0, 1.0)
    ang_err_deg = np.degrees(np.arccos(cos_sep))
    dir_errors_deg.append(ang_err_deg)
    
    true_ix = ev['impact_x']
    true_iy = ev['impact_y']
    core_err_m = np.hypot(reco_core[0] - true_ix, reco_core[1] - true_iy)
    impact_errors_m.append(core_err_m)
    
    if ev_idx < 5:
        print(f"Event {ev_idx:2d}: AngErr = {ang_err_deg:.3f} deg | True Imp = ({true_ix:5.1f}, {true_iy:5.1f}) | Reco Imp = ({reco_core[0]:5.1f}, {reco_core[1]:5.1f}) | CoreErr = {core_err_m:5.1f} m")

print(f"\n--- Summary over {len(dir_errors_deg)} events ---")
print(f"Direction Error (deg): Median = {np.median(dir_errors_deg):.3f}°, Mean = {np.mean(dir_errors_deg):.3f}°, 68% Containment = {np.percentile(dir_errors_deg, 68):.3f}°")
print(f"Impact Error (m):      Median = {np.median(impact_errors_m):.1f} m, Mean = {np.mean(impact_errors_m):.1f} m, 68% Containment = {np.percentile(impact_errors_m, 68):.1f} m")

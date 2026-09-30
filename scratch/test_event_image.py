import torch
import numpy as np
import matplotlib.pyplot as plt
import sys
from sim.shower import ShowerSimulation, atmospheric_density, mean_free_path, norm_dir
from sim.telescope import TelescopeArray
from sim.camera import Camera
from recon.hillas import compute_hillas

class HeitlerShowerSimulation(ShowerSimulation):
    def step(self):
        if self.active.shape[0] == 0:
            return
            
        p = self.active
        z_obs = 1275.0
        alive_mask = (p[:, 1] >= self.critical_energy) & (p[:, 4] > z_obs)
        p = p[alive_mask]
        if p.shape[0] == 0:
            self.active = p
            return

        pid = p[:, 0].to(torch.int32)
        E = p[:, 1]
        x, y, z = p[:, 2], p[:, 3], p[:, 4]
        px, py, pz = p[:, 5], p[:, 6], p[:, 7]
        gen = p[:, 8]
        evt = p[:, 9]
        N = p.shape[0]
        dist = torch.zeros(N, device=self.device, dtype=self.dtype)
        
        mask_gamma = (pid == 0)
        mask_e = (pid == 1) | (pid == 2)
        mask_had = (pid == 3) | (pid == 4)
        mask_pi0 = (pid == 5)
        mask_mu = (pid == 6)
        
        X0_air = 36.62
        if mask_gamma.any():
            dist[mask_gamma] = torch.empty(mask_gamma.sum(), device=self.device, dtype=self.dtype).exponential_() * mean_free_path(z[mask_gamma], X0_air * 9.0 / 7.0)
        if mask_e.any():
            dist[mask_e] = torch.empty(mask_e.sum(), device=self.device, dtype=self.dtype).exponential_() * mean_free_path(z[mask_e], X0_air)
        if mask_had.any():
            dist[mask_had] = torch.empty(mask_had.sum(), device=self.device, dtype=self.dtype).exponential_() * mean_free_path(z[mask_had], 90.0)
        if mask_pi0.any():
            dist[mask_pi0] = 10.0
        if mask_mu.any():
            dist[mask_mu] = 5000.0

        px_eff = px.clone()
        py_eff = py.clone()
        pz_eff = pz.clone()

        # Highland Scattering
        mask_e = (pid == 1) | (pid == 2)
        if mask_e.any():
            z_e = z[mask_e]
            dist_e = dist[mask_e]
            E_e = E[mask_e]
            x_gcm2 = dist_e * atmospheric_density(z_e) * 100.0
            
            valid_scatter = x_gcm2 > 0
            if valid_scatter.any():
                idx_scatter = mask_e.nonzero(as_tuple=True)[0][valid_scatter]
                x_g = x_gcm2[valid_scatter]
                Ee = E_e[valid_scatter]
                
                theta_rms = (0.0136 / Ee) * torch.sqrt(x_g / X0_air) * (1 + 0.038 * torch.log(x_g / X0_air))
                theta_rms = torch.clamp(theta_rms, min=0.0, max=0.3)
                
                theta1 = torch.normal(mean=0.0, std=theta_rms)
                theta2 = torch.normal(mean=0.0, std=theta_rms)
                
                v_x = px[idx_scatter]
                v_y = py[idx_scatter]
                v_z = pz[idx_scatter]
                
                ref_z_mask = torch.abs(v_z) < 0.99
                ref_x = torch.where(ref_z_mask, 0.0, 1.0).to(self.device)
                ref_y = torch.zeros_like(v_x)
                ref_z = torch.where(ref_z_mask, 1.0, 0.0).to(self.device)
                
                e1_x = v_y * ref_z - v_z * ref_y
                e1_y = v_z * ref_x - v_x * ref_z
                e1_z = v_x * ref_y - v_y * ref_x
                
                norm_e1 = torch.sqrt(e1_x**2 + e1_y**2 + e1_z**2)
                safe = norm_e1 > 0
                e1_x = torch.where(safe, e1_x / norm_e1, 1.0).to(self.device)
                e1_y = torch.where(safe, e1_y / norm_e1, 0.0).to(self.device)
                e1_z = torch.where(safe, e1_z / norm_e1, 0.0).to(self.device)
                
                e2_x = v_y * e1_z - v_z * e1_y
                e2_y = v_z * e1_x - v_x * e1_z
                e2_z = v_x * e1_y - v_y * e1_x
                
                norm_e2 = torch.sqrt(e2_x**2 + e2_y**2 + e2_z**2)
                safe = norm_e2 > 0
                e2_x = torch.where(safe, e2_x / norm_e2, 0.0).to(self.device)
                e2_y = torch.where(safe, e2_y / norm_e2, 1.0).to(self.device)
                e2_z = torch.where(safe, e2_z / norm_e2, 0.0).to(self.device)
                
                inv_sqrt3 = 0.577350269
                p_eff_x = v_x + theta1 * inv_sqrt3 * e1_x + theta2 * inv_sqrt3 * e2_x
                p_eff_y = v_y + theta1 * inv_sqrt3 * e1_y + theta2 * inv_sqrt3 * e2_y
                p_eff_z = v_z + theta1 * inv_sqrt3 * e1_z + theta2 * inv_sqrt3 * e2_z
                p_eff_x, p_eff_y, p_eff_z = norm_dir(p_eff_x, p_eff_y, p_eff_z)
                px_eff[idx_scatter] = p_eff_x.to(self.dtype)
                py_eff[idx_scatter] = p_eff_y.to(self.dtype)
                pz_eff[idx_scatter] = p_eff_z.to(self.dtype)
                
                p_new_x = v_x + theta1 * e1_x + theta2 * e2_x
                p_new_y = v_y + theta1 * e1_y + theta2 * e2_y
                p_new_z = v_z + theta1 * e1_z + theta2 * e2_z
                p_new_x, p_new_y, p_new_z = norm_dir(p_new_x, p_new_y, p_new_z)
                px[idx_scatter] = p_new_x.to(self.dtype)
                py[idx_scatter] = p_new_y.to(self.dtype)
                pz[idx_scatter] = p_new_z.to(self.dtype)

        x_new = x + px_eff * dist
        y_new = y + py_eff * dist
        z_new = z + pz_eff * dist
        
        # Ray Trace Cherenkov Segments ONLY ABOVE OBSERVATORY
        valid_z = (z_new > z_obs) & (z > z_obs)
        c_mask = mask_e & valid_z
        if c_mask.any():
            self.c_segs_start.append(torch.stack([x[c_mask], y[c_mask], z[c_mask]], dim=1))
            self.c_segs_end.append(torch.stack([x_new[c_mask], y_new[c_mask], z_new[c_mask]], dim=1))
            self.c_segs_p.append(torch.stack([px_eff[c_mask], py_eff[c_mask], pz_eff[c_mask]], dim=1))
            self.c_segs_E.append(E[c_mask])
            self.c_segs_event_id.append(evt[c_mask])
            
        z_new = torch.clamp(z_new, min=z_obs)
        
        # Ionization
        mask_ion = mask_e | (pid == 3) | (pid == 4)
        if mask_ion.any():
            x_gcm2 = dist[mask_ion] * atmospheric_density(z_new[mask_ion]) * 100.0
            E[mask_ion] -= 0.0021 * x_gcm2
            
        alive = (E >= self.critical_energy) & (z_new > z_obs)
        idx_alive = alive.nonzero(as_tuple=True)[0]
        new_particles = []
        
        # 1. Gamma -> e+ e- (symmetric Heitler splitting u in [0.4, 0.6])
        mask_g = (pid[idx_alive] == 0)
        if mask_g.any():
            idx_g = idx_alive[mask_g]
            N_g = idx_g.shape[0]
            u = self._get_rand(N_g) * 0.2 + 0.4
            e1_E = u * E[idx_g]
            e2_E = (1 - u) * E[idx_g]
            
            phi = self._get_rand(N_g) * 2 * np.pi
            pt = 0.0005
            px1, py1, pz1 = norm_dir(px[idx_g] + pt*torch.cos(phi), py[idx_g] + pt*torch.sin(phi), pz[idx_g])
            px2, py2, pz2 = norm_dir(px[idx_g] - pt*torch.cos(phi), py[idx_g] - pt*torch.sin(phi), pz[idx_g])
            
            gen_new = gen[idx_g] + 1
            evt_new = evt[idx_g]
            x_g = x_new[idx_g]
            y_g = y_new[idx_g]
            z_g = z_new[idx_g]
            
            e1 = torch.stack([torch.full_like(e1_E, 1), e1_E, x_g, y_g, z_g, px1, py1, pz1, gen_new, evt_new], dim=1)
            e2 = torch.stack([torch.full_like(e2_E, 2), e2_E, x_g, y_g, z_g, px2, py2, pz2, gen_new, evt_new], dim=1)
            new_particles.extend([e1, e2])
            
        # 2. Bremsstrahlung (symmetric Heitler splitting u in [0.4, 0.6])
        mask_b = (pid[idx_alive] == 1) | (pid[idx_alive] == 2)
        if mask_b.any():
            idx_b = idx_alive[mask_b]
            N_b = idx_b.shape[0]
            E_b = E[idx_b]
            
            u = self._get_rand(N_b) * 0.2 + 0.4
            e_E = (1 - u) * E_b
            g_E = u * E_b
            
            phi = self._get_rand(N_b) * 2 * np.pi
            pt = 0.0005
            px1, py1, pz1 = norm_dir(px[idx_b] - pt*torch.cos(phi), py[idx_b] - pt*torch.sin(phi), pz[idx_b])
            px2, py2, pz2 = norm_dir(px[idx_b] + pt*torch.cos(phi), py[idx_b] + pt*torch.sin(phi), pz[idx_b])
            
            gen_new = gen[idx_b] + 1
            evt_new = evt[idx_b]
            x_b = x_new[idx_b]
            y_b = y_new[idx_b]
            z_b = z_new[idx_b]
            
            e_out = torch.stack([pid[idx_b].float(), e_E, x_b, y_b, z_b, px1, py1, pz1, gen_new, evt_new], dim=1)
            g_out = torch.stack([torch.full_like(g_E, 0), g_E, x_b, y_b, z_b, px2, py2, pz2, gen_new, evt_new], dim=1)
            new_particles.extend([e_out, g_out])

        if new_particles:
            self.active = torch.cat(new_particles, dim=0)
        else:
            self.active = torch.empty((0, 10), device=self.device, dtype=self.dtype)

array = TelescopeArray.veritas_array()
cam = Camera(n_rings=12)

zenith_deg = 30.0
azimuth_deg = 0.0
wobble = 0.5
zen_rad = np.radians(zenith_deg)
azi_rad = np.radians(azimuth_deg)
wob_rad = np.radians(wobble)
evt_zen = zen_rad + wob_rad
evt_azi = azi_rad
px = np.sin(evt_zen) * np.cos(evt_azi)
py = np.sin(evt_zen) * np.sin(evt_azi)
pz = -np.cos(evt_zen)

z_start = 25000.0
z_obs = 1275.0

# Shower core at ground level: throw at (ix, iy) = (70m, 30m)
ix, iy = 70.0, 30.0

x_init = ix + (z_start - z_obs) * (px / pz)
y_init = iy + (z_start - z_obs) * (py / pz)

sim = HeitlerShowerSimulation(
    primary_types=['gamma'], energies=[1000.0], z_starts=[z_start],
    x_init=[x_init], y_init=[y_init],
    px_init=[px], py_init=[py], pz_init=[pz]
)
sim.run(max_generations=25, verbose=False)
photons = sim.cherenkov_photons_by_event[0]
print(f"Total Cherenkov packets: {len(photons['x_ground'])}")

# Ray trace
img_outputs = array.ray_trace(
    photons,
    pointing_zenith=zenith_deg,
    pointing_azimuth=azimuth_deg + 180.0,
    shower_start_altitude=z_start
)

fig, axes = plt.subplots(1, 4, figsize=(16, 4))
for t_idx in range(4):
    ax = axes[t_idx]
    trace, gain = img_outputs[t_idx]
    c = np.sum(trace, axis=1) # (469,)
    
    c_clean = np.maximum(0, c - 2.0)
    mask = c_clean > 5.0
    
    sc = ax.scatter(cam.pixel_x, cam.pixel_y, c=c_clean, cmap='viridis', s=25, alpha=0.8)
    if np.sum(mask) >= 3:
        ax.scatter(cam.pixel_x[mask], cam.pixel_y[mask], facecolors='none', edgecolors='red', s=45, linewidth=1.5)
        h = compute_hillas(cam, c_clean, mask)
        if h:
            L = h.length
            x1 = h.centroid_x - L * np.cos(h.psi)
            y1 = h.centroid_y - L * np.sin(h.psi)
            x2 = h.centroid_x + L * np.cos(h.psi)
            y2 = h.centroid_y + L * np.sin(h.psi)
            ax.plot([x1, x2], [y1, y2], 'r-', lw=2)
            ax.plot(h.centroid_x, h.centroid_y, 'ro', ms=5)
            ax.set_title(f"Tel {t_idx}: L={h.length:.2f}, W={h.width:.2f}, Size={h.size:.0f}")
    else:
        ax.set_title(f"Tel {t_idx}: MaxPE={np.max(c):.1f}")
        
    ax.set_xlim(-2.0, 2.0)
    ax.set_ylim(-2.0, 2.0)
    ax.set_aspect('equal')
    circ = plt.Circle((0, 0), 1.75, color='gray', fill=False, ls=':')
    ax.add_patch(circ)

plt.tight_layout()
plt.savefig('scratch/test_event_image.png', dpi=150)
print("Saved scratch/test_event_image.png")

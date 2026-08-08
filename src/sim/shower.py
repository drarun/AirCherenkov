import numpy as np
import pandas as pd
import torch
import time

# --- JIT Compiled Physics Utilities ---
@torch.jit.script
def atmospheric_density(z: torch.Tensor) -> torch.Tensor:
    return 1.225e-3 * torch.exp(-z / 7640.0)

@torch.jit.script
def mean_free_path(z: torch.Tensor, X_length: float) -> torch.Tensor:
    rho = atmospheric_density(z)
    return X_length / (rho * 100.0)

@torch.jit.script
def norm_dir(px: torch.Tensor, py: torch.Tensor, pz: torch.Tensor):
    norm = torch.sqrt(px**2 + py**2 + pz**2)
    safe = norm > 0
    return (torch.where(safe, px/norm, px),
            torch.where(safe, py/norm, py),
            torch.where(safe, pz/norm, pz))

class ShowerSimulation:
    def __init__(self, primary_types=['gamma'], energies=[1000.0], z_starts=[20000.0],
                 site="VERITAS", px_init=0.0, py_init=0.0, pz_init=-1.0):
        """
        A 3D toy Monte Carlo using fully tensorized PyTorch operations.
        Supports batch processing of multiple independent showers.
        
        Parameters
        ----------
        px_init, py_init, pz_init : float
            Initial direction vector components. Default (0, 0, -1) = straight down.
            For a shower arriving at zenith angle θ and azimuth φ:
                px_init = sin(θ) * cos(φ)
                py_init = sin(θ) * sin(φ)
                pz_init = -cos(θ)
        """
        self.critical_energy = 0.085 # 85 MeV
        self.photon_yield_factor = 1.0 
        
        from sim.backend import get_device
        self.device = get_device()
        if self.device is None:
            self.device = torch.device('cpu')
            
        self.PID_MAP = {'gamma': 0, 'e+': 1, 'e-': 2, 'proton': 3, 'pi_charged': 4, 'pi0': 5, 'mu': 6}
        self.INV_PID_MAP = {v: k for k, v in self.PID_MAP.items()}
        
        # Ensure inputs are lists and broadcast to match primary_types length
        if isinstance(primary_types, str):
            primary_types = [primary_types]
        batch_size = len(primary_types)

        if isinstance(energies, (float, int)):
            energies = [float(energies)] * batch_size
        elif len(energies) == 1:
            energies = [float(energies[0])] * batch_size
        elif len(energies) != batch_size:
            raise ValueError("Length of energies must match length of primary_types")

        if isinstance(z_starts, (float, int)):
            z_starts = [float(z_starts)] * batch_size
        elif len(z_starts) == 1:
            z_starts = [float(z_starts[0])] * batch_size
        elif len(z_starts) != batch_size:
            raise ValueError("Length of z_starts must match length of primary_types")
        self.dtype = torch.float32
            
        # Magnetic field vectors (X: North, Y: East, Z: Down) in Tesla
        if site == "VERITAS":
            self.B_field = torch.tensor([23.9e-6, 4.1e-6, 40.8e-6], device=self.device, dtype=self.dtype)
        elif site == "HESS":
            # H.E.S.S. Namibia (Southern hemisphere, Z is upwards relative to field lines)
            self.B_field = torch.tensor([10.5e-6, -3.2e-6, -25.8e-6], device=self.device, dtype=self.dtype)
        elif site == "CTA_NORTH":
            # CTA La Palma
            self.B_field = torch.tensor([24.2e-6, 1.2e-6, 30.8e-6], device=self.device, dtype=self.dtype)
        else:
            self.B_field = torch.tensor([0.0, 0.0, 0.0], device=self.device, dtype=self.dtype)
            
        batch_size = len(primary_types)
        
        # Pre-allocate Entropy Pool (50 Million random numbers in bfloat16) to avoid random overhead in loop
        self.entropy_pool = torch.rand(50_000_000, device=self.device, dtype=self.dtype)
        self.entropy_idx = 0
        
        # State tensor [PID, E, x, y, z, px, py, pz, generation, event_id]
        init_state = []
        for i in range(batch_size):
            init_state.append([self.PID_MAP[primary_types[i]], energies[i], 0.0, 0.0, z_starts[i], px_init, py_init, pz_init, 0.0, float(i)])
            
        self.active = torch.tensor(init_state, dtype=self.dtype, device=self.device)
        self.batch_size = batch_size
        
        # Cherenkov segment accumulators
        self.c_segs_start = []
        self.c_segs_end = []
        self.c_segs_p = []
        self.c_segs_E = []
        self.c_segs_event_id = []
        
        # Output dictionary mapping event_id to its Cherenkov photons
        self.cherenkov_photons_by_event = {i: {'x_ground': np.array([]), 'y_ground': np.array([])} for i in range(batch_size)}
    def _get_rand(self, size):
        """Consume random numbers from the pre-allocated entropy pool."""
        if self.entropy_idx + size > self.entropy_pool.numel():
            # Refill pool if exhausted (rare)
            self.entropy_pool = torch.rand(50_000_000, device=self.device, dtype=self.dtype)
            self.entropy_idx = 0
        out = self.entropy_pool[self.entropy_idx:self.entropy_idx + size]
        self.entropy_idx += size
        return out
    def step(self):
        """Advance the simulation by one generation using batched tensor ops."""
        if self.active.shape[0] == 0:
            return
            
        p = self.active
        # Filter dead immediately
        alive_mask = (p[:, 1] >= self.critical_energy) & (p[:, 4] > 0)
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
        
        # Mean free paths
        mask_em = (pid == 0) | (pid == 1) | (pid == 2)
        mask_had = (pid == 3) | (pid == 4)
        mask_pi0 = (pid == 5)
        mask_mu = (pid == 6)
        
        if mask_em.any():
            dist[mask_em] = torch.empty(mask_em.sum(), device=self.device, dtype=self.dtype).exponential_() * mean_free_path(z[mask_em], 36.62)
        if mask_had.any():
            dist[mask_had] = torch.empty(mask_had.sum(), device=self.device, dtype=self.dtype).exponential_() * mean_free_path(z[mask_had], 90.0)
        if mask_pi0.any():
            dist[mask_pi0] = 10.0
        if mask_mu.any():
            dist[mask_mu] = 5000.0

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
                
                log_term = torch.log(x_g / 36.62)
                theta_rms = (0.0136 / Ee) * torch.sqrt(x_g / 36.62) * (1 + 0.038 * log_term)
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
                
                p_new_x = v_x + theta1 * e1_x + theta2 * e2_x
                p_new_y = v_y + theta1 * e1_y + theta2 * e2_y
                p_new_z = v_z + theta1 * e1_z + theta2 * e2_z
                
                p_new_x, p_new_y, p_new_z = norm_dir(p_new_x, p_new_y, p_new_z)
                px[idx_scatter] = p_new_x.to(self.dtype)
                py[idx_scatter] = p_new_y.to(self.dtype)
                pz[idx_scatter] = p_new_z.to(self.dtype)
                
        # Lorentz Force Geomagnetic Deflection
        if mask_e.any():
            c_idx = mask_e.nonzero(as_tuple=True)[0]
            p_vec = torch.stack([px[c_idx], py[c_idx], pz[c_idx]], dim=1)
            
            # Extract charge (+1 for e+, -1 for e-)
            q = torch.zeros_like(E[c_idx])
            q[pid[c_idx] == 1] = 1.0
            q[pid[c_idx] == 2] = -1.0
            
            # B-field expanded to match batch
            B_expanded = self.B_field.unsqueeze(0).expand(len(c_idx), -1)
            
            # v x B
            cross_prod = torch.cross(p_vec, B_expanded, dim=1)
            
            # Deflection magnitude: (0.3 * q * dist) / E
            # 0.3 handles the c conversion for E in GeV, B in Tesla, ds in meters
            deflection_mag = (0.3 * q * dist[c_idx]) / E[c_idx]
            
            dp = cross_prod * deflection_mag.unsqueeze(1)
            
            px[c_idx] = (px[c_idx] + dp[:, 0]).to(self.dtype)
            py[c_idx] = (py[c_idx] + dp[:, 1]).to(self.dtype)
            pz[c_idx] = (pz[c_idx] + dp[:, 2]).to(self.dtype)
            
            p_nx, p_ny, p_nz = norm_dir(px[c_idx], py[c_idx], pz[c_idx])
            px[c_idx] = p_nx.to(self.dtype)
            py[c_idx] = p_ny.to(self.dtype)
            pz[c_idx] = p_nz.to(self.dtype)

        x_new = x + px * dist
        y_new = y + py * dist
        z_new = z + pz * dist
        
        # Ray Trace Cherenkov Segments
        valid_z = (z_new > 0) & (z > 0)
        c_mask = mask_e & valid_z
        if c_mask.any():
            self.c_segs_start.append(torch.stack([x[c_mask], y[c_mask], z[c_mask]], dim=1))
            self.c_segs_end.append(torch.stack([x_new[c_mask], y_new[c_mask], z_new[c_mask]], dim=1))
            self.c_segs_p.append(torch.stack([px[c_mask], py[c_mask], pz[c_mask]], dim=1))
            self.c_segs_E.append(E[c_mask])
            self.c_segs_event_id.append(evt[c_mask])
            
        z_new = torch.clamp(z_new, min=0.0)
        
        # Ionization
        mask_ion = mask_e | (pid == 3) | (pid == 4)
        if mask_ion.any():
            x_gcm2 = dist[mask_ion] * atmospheric_density(z_new[mask_ion]) * 100.0
            E[mask_ion] -= 0.0021 * x_gcm2
            
        # Re-filter alive
        alive = (E >= self.critical_energy) & (z_new > 0)
        idx_alive = alive.nonzero(as_tuple=True)[0]
        new_particles = []
        
        # 1. Gamma -> e+ e-
        mask_g = (pid[idx_alive] == 0)
        if mask_g.any():
            idx_g = idx_alive[mask_g]
            N_g = idx_g.shape[0]
            u = self._get_rand(N_g) * 0.8 + 0.1
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
            
        # 2. Bremsstrahlung
        mask_b = (pid[idx_alive] == 1) | (pid[idx_alive] == 2)
        if mask_b.any():
            idx_b = idx_alive[mask_b]
            N_b = idx_b.shape[0]
            E_b = E[idx_b]
            min_u = self.critical_energy / E_b
            rand_val = self._get_rand(N_b)
            u = torch.pow(min_u, rand_val)
            u = torch.max(min_u, u)
            
            g_E = u * E_b
            e_E = (1 - u) * E_b
            
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

        # 3. Hadronic
        mask_h = (pid[idx_alive] == 3) | (pid[idx_alive] == 4)
        if mask_h.any():
            idx_h = idx_alive[mask_h]
            N_h = idx_h.shape[0]
            E_h = E[idx_h]
            pt = 0.5 / E_h
            e_split = E_h / 5.0
            
            gen_new = gen[idx_h] + 1
            evt_new = evt[idx_h]
            x_h = x_new[idx_h]
            y_h = y_new[idx_h]
            z_h = z_new[idx_h]
            
            for i in range(5):
                phi = self._get_rand(N_h) * 2 * np.pi
                px_c, py_c, pz_c = norm_dir(px[idx_h] + pt*torch.cos(phi), py[idx_h] + pt*torch.sin(phi), pz[idx_h])
                c_pid = 5 if i < 2 else 4
                child = torch.stack([torch.full_like(e_split, c_pid), e_split, x_h, y_h, z_h, px_c, py_c, pz_c, gen_new, evt_new], dim=1)
                new_particles.append(child)
                
        # 4. Pi0 -> 2 gamma
        mask_p0 = (pid[idx_alive] == 5)
        if mask_p0.any():
            idx_p0 = idx_alive[mask_p0]
            N_p0 = idx_p0.shape[0]
            E_p0 = E[idx_p0]
            pt = 0.135 / E_p0
            phi = self._get_rand(N_p0) * 2 * np.pi
            
            px1, py1, pz1 = norm_dir(px[idx_p0] + pt*torch.cos(phi), py[idx_p0] + pt*torch.sin(phi), pz[idx_p0])
            px2, py2, pz2 = norm_dir(px[idx_p0] - pt*torch.cos(phi), py[idx_p0] - pt*torch.sin(phi), pz[idx_p0])
            
            gen_new = gen[idx_p0] + 1
            evt_new = evt[idx_p0]
            x_p0 = x_new[idx_p0]
            y_p0 = y_new[idx_p0]
            z_p0 = z_new[idx_p0]
            e_split = E_p0 / 2.0
            
            g1 = torch.stack([torch.full_like(e_split, 0), e_split, x_p0, y_p0, z_p0, px1, py1, pz1, gen_new, evt_new], dim=1)
            g2 = torch.stack([torch.full_like(e_split, 0), e_split, x_p0, y_p0, z_p0, px2, py2, pz2, gen_new, evt_new], dim=1)
            new_particles.extend([g1, g2])

        if new_particles:
            self.active = torch.cat(new_particles, dim=0)
        else:
            self.active = torch.empty((0, 10), device=self.device)
        
    def run(self, max_generations=12, verbose=True):
        t0 = time.time()
        for gen in range(max_generations):
            if self.active.shape[0] == 0:
                break
            n_active = self.active.shape[0]
            self.step()
            if verbose:
                elapsed = time.time() - t0
                print(f"      Gen {gen:2d}: {n_active:6d} active particles ({elapsed:.1f}s)")
        
        if verbose:
            print(f"      Cascade complete ({time.time()-t0:.1f}s)")
            print(f"      Computing Cherenkov pool on GPU...")
        t1 = time.time()
        self.calculate_cherenkov_pool()
        if verbose:
            n_phot = sum(len(d.get('x_ground', [])) for d in self.cherenkov_photons_by_event.values())
            print(f"      Cherenkov pool: {n_phot:,} photons ({time.time()-t1:.1f}s)")

    def calculate_cherenkov_pool(self):
        from sim.backend import compute_cherenkov_pool_gpu
        if not self.c_segs_start:
            return
            
        # Concat all segments on GPU directly
        starts = torch.cat(self.c_segs_start, dim=0)
        ends = torch.cat(self.c_segs_end, dim=0)
        ps = torch.cat(self.c_segs_p, dim=0)
        Es = torch.cat(self.c_segs_E, dim=0)
        evts = torch.cat(self.c_segs_event_id, dim=0)
        
        # Ray trace per event to avoid mixing photons
        for i in range(self.batch_size):
            mask = (evts == i)
            if not mask.any():
                continue
                
            e_starts = starts[mask]
            e_ends = ends[mask]
            e_ps = ps[mask]
            e_Es = Es[mask]
            
            self.cherenkov_photons_by_event[i] = compute_cherenkov_pool_gpu(
                e_starts[:, 0], e_starts[:, 1], e_starts[:, 2],
                e_ends[:, 0], e_ends[:, 1], e_ends[:, 2],
                e_ps[:, 0], e_ps[:, 1], e_ps[:, 2],
                e_Es, self.photon_yield_factor
            )

    def get_cherenkov_dataframe(self, event_idx=0):
        photons = self.cherenkov_photons_by_event.get(event_idx, {'x_ground': []})
        if len(photons.get('x_ground', [])) == 0:
            return pd.DataFrame(columns=['x', 'y'])
            
        n_tot = len(photons['x_ground'])
        max_plot = 10000
        if n_tot > max_plot:
            indices = np.random.choice(n_tot, max_plot, replace=False)
        else:
            indices = np.arange(n_tot)
            
        data = {
            'x': photons['x_ground'][indices],
            'y': photons['y_ground'][indices]
        }
        return pd.DataFrame(data)

    def get_tracks_dataframe(self):
        # We omitted storing full tracks to save VRAM. Return empty.
        return pd.DataFrame(columns=['particle_id', 'pid', 'energy', 'generation', 'x', 'y', 'z', 'event_id'])

if __name__ == "__main__":
    sim = ShowerSimulation(['gamma', 'proton'], energies=[100.0, 500.0])
    sim.run(max_generations=12)

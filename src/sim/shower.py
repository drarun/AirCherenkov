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
def arclength_from_grammage(z: torch.Tensor, pz: torch.Tensor, delta_x_gcm2: torch.Tensor) -> torch.Tensor:
    """Convert slant grammage delta_x (g/cm^2) to geometric arclength (m)
    in an exponential atmosphere rho(z) = rho0 * exp(-z / H), matching
    CORSIKA 8's ExponentialAtmosphere::getArclengthFromGrammage.
    """
    H = 7640.0
    rho_m = atmospheric_density(z) * 100.0  # g/cm^2 per meter
    neg_pz = -pz
    horiz = torch.abs(neg_pz) < 1e-5
    safe_neg_pz = torch.where(horiz, torch.ones_like(neg_pz), neg_pz)
    arg = torch.clamp(1.0 + (safe_neg_pz * delta_x_gcm2) / torch.clamp(H * rho_m, min=1e-12), min=1e-6)
    dist_slanted = (H / safe_neg_pz) * torch.log(arg)
    dist_horiz = delta_x_gcm2 / torch.clamp(rho_m, min=1e-12)
    return torch.clamp(torch.where(horiz, dist_horiz, dist_slanted), min=0.0)

@torch.jit.script
def norm_dir(px: torch.Tensor, py: torch.Tensor, pz: torch.Tensor):
    norm = torch.sqrt(px**2 + py**2 + pz**2)
    safe = norm > 0
    return (torch.where(safe, px/norm, px),
            torch.where(safe, py/norm, py),
            torch.where(safe, pz/norm, pz))

class ShowerSimulation:
    def __init__(self, primary_types=['gamma'], energies=[1000.0], z_starts=[50000.0],
                 site="VERITAS", x_init=0.0, y_init=0.0, px_init=0.0, py_init=0.0, pz_init=-1.0, dtype=torch.float32):
        """
        A 3D Monte Carlo air-shower simulation using batched PyTorch tensor operations
        with CORSIKA-aligned exponential-atmosphere slant-grammage transport.
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
        self.critical_energy = 0.085
        self.photon_yield_factor = 1.0 
        
        from sim.backend import get_device
        self.device = get_device()
        if self.device is None:
            self.device = torch.device('cpu')
            
        self.dtype = dtype
            
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
            
        # Magnetic field vectors (X: North, Y: East, Z: Down) in Tesla
        if site == "VERITAS":
            self.B_field = torch.tensor([23.9e-6, 4.1e-6, 40.8e-6], device=self.device, dtype=self.dtype)
        elif site == "MAGIC":
            self.B_field = torch.tensor([10.5e-6, -3.2e-6, -25.8e-6], device=self.device, dtype=self.dtype)
        elif site == "HESS":
            # H.E.S.S. Namibia (Southern hemisphere, Z is upwards relative to field lines)
            self.B_field = torch.tensor([24.2e-6, 1.2e-6, 30.8e-6], device=self.device, dtype=self.dtype)
        elif site == "CTA_NORTH":
            # CTA La Palma
            self.B_field = torch.tensor([24.2e-6, 1.2e-6, 30.8e-6], device=self.device, dtype=self.dtype)
        else:
            self.B_field = torch.tensor([0.0, 0.0, 0.0], device=self.device, dtype=self.dtype)
            
        batch_size = len(primary_types)
        
        # Pre-allocate Entropy Pool (50 Million random numbers) to avoid random overhead in loop
        self.entropy_pool = torch.rand(50_000_000, device=self.device, dtype=self.dtype)
        self.entropy_idx = 0
        
        # Handle array-like direction inputs for diffuse backgrounds
        if not isinstance(px_init, (list, tuple, np.ndarray, torch.Tensor)):
            px_init = [float(px_init)] * batch_size
        if not isinstance(py_init, (list, tuple, np.ndarray, torch.Tensor)):
            py_init = [float(py_init)] * batch_size
        if not isinstance(pz_init, (list, tuple, np.ndarray, torch.Tensor)):
            pz_init = [float(pz_init)] * batch_size
        if not isinstance(x_init, (list, tuple, np.ndarray, torch.Tensor)):
            x_init = [float(x_init)] * batch_size
        if not isinstance(y_init, (list, tuple, np.ndarray, torch.Tensor)):
            y_init = [float(y_init)] * batch_size

        # State tensor [PID, E, x, y, z, px, py, pz, generation, event_id]
        init_state = []
        for i in range(batch_size):
            init_state.append([self.PID_MAP[primary_types[i]], energies[i], float(x_init[i]), float(y_init[i]), z_starts[i], float(px_init[i]), float(py_init[i]), float(pz_init[i]), 0.0, float(i)])
            
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
        z_obs = 1275.0  # VERITAS observatory altitude (m)
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
        delta_x = torch.zeros(N, device=self.device, dtype=self.dtype)
        
        # Particle type masks
        mask_gamma = (pid == 0)
        mask_e = (pid == 1) | (pid == 2)
        mask_had = (pid == 3) | (pid == 4)
        mask_pi0 = (pid == 5)
        mask_mu = (pid == 6)
        mask_ion = mask_e | mask_had
        
        # Sample slant interaction grammage delta_x (g/cm^2)
        # Pair production MFP = (9/7) * X0; Bremsstrahlung half-energy step = X0 * ln(2)
        X0_air = 36.62  # radiation length in air (g/cm^2)
        X_brems = X0_air * 0.69314718  # X0 * ln(2) = 25.383 g/cm^2
        if mask_gamma.any():
            delta_x[mask_gamma] = torch.empty(mask_gamma.sum(), device=self.device, dtype=self.dtype).exponential_() * (X0_air * 9.0 / 7.0)
        if mask_e.any():
            delta_x[mask_e] = torch.empty(mask_e.sum(), device=self.device, dtype=self.dtype).exponential_() * X_brems
        if mask_had.any():
            delta_x[mask_had] = torch.empty(mask_had.sum(), device=self.device, dtype=self.dtype).exponential_() * 90.0

        # Cap charged-particle step grammage at ionization range to critical energy
        # (matches CORSIKA 8 ContinuousProcess::getMaxStepLength)
        alpha_ion = 0.0021  # GeV / (g/cm^2)
        if mask_ion.any():
            max_ion_x = torch.clamp((E[mask_ion] - self.critical_energy) / alpha_ion, min=0.0) + 1e-4
            delta_x[mask_ion] = torch.minimum(delta_x[mask_ion], max_ion_x)

        # Convert slant grammage to geometric arclength integrating exponential atmosphere
        mask_grammage = mask_gamma | mask_e | mask_had
        if mask_grammage.any():
            dist[mask_grammage] = arclength_from_grammage(
                z[mask_grammage], pz[mask_grammage], delta_x[mask_grammage]
            )
        if mask_pi0.any():
            dist[mask_pi0] = 10.0
        if mask_mu.any():
            dist[mask_mu] = 5000.0

        px_eff = px.clone()
        py_eff = py.clone()
        pz_eff = pz.clone()

        # Highland Multiple Coulomb Scattering using exact traversed grammage
        if mask_e.any():
            E_e = E[mask_e]
            x_gcm2 = delta_x[mask_e]
            
            valid_scatter = x_gcm2 > 1e-6
            if valid_scatter.any():
                idx_scatter = mask_e.nonzero(as_tuple=True)[0][valid_scatter]
                x_g = x_gcm2[valid_scatter]
                Ee = E_e[valid_scatter]
                
                # theta_0 = (13.6 MeV / p c) * sqrt(x/X0) * (1 + 0.038 * ln(x/X0))
                ratio_x0 = torch.clamp(x_g / X0_air, min=1e-4)
                theta_rms = (0.0136 / Ee) * torch.sqrt(ratio_x0) * torch.clamp(1.0 + 0.038 * torch.log(ratio_x0), min=0.5)
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
                
                # Position advances by 1/sqrt(3) of the angular kick
                inv_sqrt3 = 0.577350269
                p_eff_x = v_x + theta1 * inv_sqrt3 * e1_x + theta2 * inv_sqrt3 * e2_x
                p_eff_y = v_y + theta1 * inv_sqrt3 * e1_y + theta2 * inv_sqrt3 * e2_y
                p_eff_z = v_z + theta1 * inv_sqrt3 * e1_z + theta2 * inv_sqrt3 * e2_z
                p_eff_x, p_eff_y, p_eff_z = norm_dir(p_eff_x, p_eff_y, p_eff_z)
                px_eff[idx_scatter] = p_eff_x.to(self.dtype)
                py_eff[idx_scatter] = p_eff_y.to(self.dtype)
                pz_eff[idx_scatter] = p_eff_z.to(self.dtype)
                
                # Momentum advances by full angular kick
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
            
            # Lorentz force is constant, so mean deflection is half the total
            px_eff[c_idx] = (px_eff[c_idx] + dp[:, 0] * 0.5).to(self.dtype)
            py_eff[c_idx] = (py_eff[c_idx] + dp[:, 1] * 0.5).to(self.dtype)
            pz_eff[c_idx] = (pz_eff[c_idx] + dp[:, 2] * 0.5).to(self.dtype)
            px_eff_nx, py_eff_ny, pz_eff_nz = norm_dir(px_eff[c_idx], py_eff[c_idx], pz_eff[c_idx])
            px_eff[c_idx] = px_eff_nx.to(self.dtype)
            py_eff[c_idx] = py_eff_ny.to(self.dtype)
            pz_eff[c_idx] = pz_eff_nz.to(self.dtype)

        z_raw = z + pz_eff * dist

        # Clip steps crossing the observation plane z_obs instead of discarding them
        z_obs = 1275.0
        f_clip = torch.where(
            z_raw < z_obs,
            (z - z_obs) / torch.clamp(z - z_raw, min=1e-6),
            torch.ones_like(dist)
        )
        f_clip = torch.clamp(f_clip, min=0.0, max=1.0)
        dist_clipped = dist * f_clip
        delta_x_clipped = delta_x * f_clip

        x_new = x + px_eff * dist_clipped
        y_new = y + py_eff * dist_clipped
        z_new = z + pz_eff * dist_clipped
        
        # Ray Trace Cherenkov Segments (all electron segments above observatory altitude)
        c_mask = mask_e & (z > z_obs) & (dist_clipped > 1e-3)
        if c_mask.any():
            E_mid = torch.clamp(E[c_mask] - 0.5 * alpha_ion * delta_x_clipped[c_mask], min=self.critical_energy)
            self.c_segs_start.append(torch.stack([x[c_mask], y[c_mask], z[c_mask]], dim=1))
            self.c_segs_end.append(torch.stack([x_new[c_mask], y_new[c_mask], z_new[c_mask]], dim=1))
            self.c_segs_p.append(torch.stack([px_eff[c_mask], py_eff[c_mask], pz_eff[c_mask]], dim=1))
            self.c_segs_E.append(E_mid)
            self.c_segs_event_id.append(evt[c_mask])
            
        z_new = torch.clamp(z_new, min=z_obs)
        
        # Continuous ionization energy loss using exact traversed grammage
        if mask_ion.any():
            E[mask_ion] -= alpha_ion * delta_x_clipped[mask_ion]
            
        # Re-filter alive particles that did not hit ground or range out below critical energy
        alive = (E >= self.critical_energy) & (z_raw > z_obs)
        idx_alive = alive.nonzero(as_tuple=True)[0]
        new_particles = []
        
        # 1. Gamma -> e+ e-
        mask_g = (pid[idx_alive] == 0)
        if mask_g.any():
            idx_g = idx_alive[mask_g]
            N_g = idx_g.shape[0]
            u = self._get_rand(N_g) * 0.2 + 0.4  # symmetric Heitler: u in [0.4, 0.6]
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
            
            # Bremsstrahlung: e -> e + gamma, symmetric Heitler split u in [0.4, 0.6]
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
            
    def save_state(self, filepath="shower_checkpoint.pt"):
        state = {
            'active': self.active,
            'entropy_pool': self.entropy_pool,
            'entropy_idx': self.entropy_idx,
            'c_segs_start': self.c_segs_start,
            'c_segs_end': self.c_segs_end,
            'c_segs_p': self.c_segs_p,
            'c_segs_E': self.c_segs_E,
            'c_segs_event_id': self.c_segs_event_id,
            'cherenkov_photons_by_event': self.cherenkov_photons_by_event,
            'batch_size': self.batch_size,
            'critical_energy': self.critical_energy,
            'photon_yield_factor': self.photon_yield_factor
        }
        torch.save(state, filepath)
        
    def load_state(self, filepath="shower_checkpoint.pt"):
        state = torch.load(filepath, map_location=self.device)
        self.active = state['active']
        self.entropy_pool = state['entropy_pool']
        self.entropy_idx = state['entropy_idx']
        self.c_segs_start = state['c_segs_start']
        self.c_segs_end = state['c_segs_end']
        self.c_segs_p = state['c_segs_p']
        self.c_segs_E = state['c_segs_E']
        self.c_segs_event_id = state['c_segs_event_id']
        self.cherenkov_photons_by_event = state['cherenkov_photons_by_event']
        self.batch_size = state['batch_size']
        self.critical_energy = state['critical_energy']
        self.photon_yield_factor = state['photon_yield_factor']

        
    def run(self, max_generations=30, verbose=True):
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
                e_Es, self.photon_yield_factor,
                target_photons_per_packet=100.0
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
    sim.run(max_generations=30)

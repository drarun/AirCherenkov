import math
import time
import numpy as np
import pandas as pd
import torch

from sim.atmosphere import (
    SITE_ALTITUDES_M,
    SITE_MAGNETIC_FIELDS_T,
    arclength_from_grammage,
    atmospheric_density,
    vertical_grammage,
)
from sim.interactions import (
    BREM_LAMBDA_GCM2,
    BREM_SOFT_LOSS_PER_X0,
    C_TAU_MUON_M,
    C_TAU_PION_M,
    LAMBDA_PION_GCM2,
    LAMBDA_PROTON_GCM2,
    M_ELECTRON_GEV,
    M_MUON_GEV,
    M_PION_CHARGED_GEV,
    M_PION_ZERO_GEV,
    X0_AIR,
    apply_angular_kick,
    norm_dir,
    orthonormal_basis,
    sample_bremsstrahlung_k,
    sample_hadronic_interaction,
    sample_pair_production_u,
    sample_pion_decay_to_muon,
)


def mean_free_path(z: torch.Tensor, X_length: float) -> torch.Tensor:
    """Return local geometric mean free path (m) for interaction length ``X_length`` (g/cm^2)."""
    rho = atmospheric_density(z)
    return X_length / (rho * 100.0)


def _grammage_from_arclength(
    z: torch.Tensor, pz: torch.Tensor, dist: torch.Tensor
) -> torch.Tensor:
    """Exact slant grammage (g/cm^2) traversed along ``dist`` (m) in the 5-layer atmosphere."""
    z64 = z.to(torch.float64)
    pz64 = pz.to(torch.float64)
    d64 = torch.clamp(dist.to(torch.float64), min=0.0)
    z_end = z64 + pz64 * d64
    neg_pz = -pz64
    horiz = torch.abs(neg_pz) < 1.0e-5
    safe_neg_pz = torch.where(horiz, torch.ones_like(neg_pz), neg_pz)
    dx_slanted = (vertical_grammage(z_end) - vertical_grammage(z64)) / safe_neg_pz
    dx_horiz = atmospheric_density(z64) * 100.0 * d64
    dx = torch.where(horiz, dx_horiz, dx_slanted)
    return torch.clamp(dx, min=0.0).to(dtype=dist.dtype)


class ShowerSimulation:
    def __init__(
        self,
        primary_types=['gamma'],
        energies=[1000.0],
        z_starts=[50000.0],
        site="VERITAS",
        x_init=0.0,
        y_init=0.0,
        px_init=0.0,
        py_init=0.0,
        pz_init=-1.0,
        dtype=torch.float32,
        z_obs=None,
    ):
        """
        A 3D Monte Carlo air-shower simulation using batched PyTorch tensor operations
        with 5-layer CORSIKA US Standard Atmosphere slant-grammage transport,
        Bethe-Heitler electromagnetic splitting, hadronic cascades with pion decay
        to muons, and 20 MeV ionization/Cherenkov tracking.

        Parameters
        ----------
        px_init, py_init, pz_init : float or array-like
            Initial direction vector components. Default (0, 0, -1) = straight down.
            For a shower arriving at zenith angle theta and azimuth phi:
                px_init = sin(theta) * cos(phi)
                py_init = sin(theta) * sin(phi)
                pz_init = -cos(theta)
        site : str
            Observatory site name used for default altitude and Z-up geomagnetic field.
        z_obs : float or None
            Observation altitude in meters a.s.l. Defaults to ``SITE_ALTITUDES_M[site]``.
        """
        self.critical_energy = 0.085       # GeV, radiative bremsstrahlung threshold
        self.min_electron_energy = 0.020   # GeV, 20 MeV ionization/Cherenkov tracking threshold
        self.min_muon_energy = 1.0         # GeV, muon tracking threshold (sea-level Cherenkov ~4.36 GeV)
        self.photon_yield_factor = 1.0
        self.site = site

        from sim.backend import get_device
        self.device = get_device()
        if self.device is None:
            self.device = torch.device('cpu')

        self.dtype = dtype

        self.PID_MAP = {
            'gamma': 0,
            'e+': 1,
            'e-': 2,
            'proton': 3,
            'pi_charged': 4,
            'pi0': 5,
            'mu': 6,
        }
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
        else:
            energies = [float(e) for e in energies]

        if isinstance(z_starts, (float, int)):
            z_starts = [float(z_starts)] * batch_size
        elif len(z_starts) == 1:
            z_starts = [float(z_starts[0])] * batch_size
        elif len(z_starts) != batch_size:
            raise ValueError("Length of z_starts must match length of primary_types")
        else:
            z_starts = [float(z) for z in z_starts]

        self.primary_energies = list(energies)

        # Site-dependent observation plane altitude (m a.s.l.) and Z-up geomagnetic field (T)
        self.z_obs = float(SITE_ALTITUDES_M.get(site, 1268.0) if z_obs is None else z_obs)
        self.B_field = torch.tensor(
            SITE_MAGNETIC_FIELDS_T.get(site, (0.0, 0.0, 0.0)),
            device=self.device,
            dtype=self.dtype,
        )

        # Handle array-like position and direction inputs for diffuse backgrounds
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
            init_state.append([
                self.PID_MAP[primary_types[i]],
                energies[i],
                float(x_init[i]),
                float(y_init[i]),
                z_starts[i],
                float(px_init[i]),
                float(py_init[i]),
                float(pz_init[i]),
                0.0,
                float(i),
            ])

        self.active = torch.tensor(init_state, dtype=self.dtype, device=self.device)
        self.batch_size = batch_size

        # Cherenkov segment accumulators
        self.c_segs_start = []
        self.c_segs_end = []
        self.c_segs_p = []
        self.c_segs_E = []
        self.c_segs_mass = []
        self.c_segs_event_id = []

        # Output dictionary mapping event_id to its Cherenkov photons
        self.cherenkov_photons_by_event = {
            i: {'x_ground': np.array([]), 'y_ground': np.array([])}
            for i in range(batch_size)
        }

    def _get_rand(self, size):
        """Draw uniform random numbers in [0, 1) on the simulation device."""
        return torch.rand(size, device=self.device, dtype=self.dtype)

    def step(self):
        """Advance the simulation by one generation using batched tensor ops."""
        if self.active.shape[0] == 0:
            return

        p = self.active
        z_obs = self.z_obs
        min_gamma_E = max(0.040, 2.0 * self.min_electron_energy)
        min_had_E = 0.30
        min_pi0_E = 0.135

        # Filter dead / below-threshold particles immediately
        pid_in = p[:, 0].to(torch.int32)
        E_in = p[:, 1]
        z_in = p[:, 4]
        alive_mask = (z_in > z_obs) & (
            ((pid_in == 0) & (E_in >= min_gamma_E))
            | (((pid_in == 1) | (pid_in == 2)) & (E_in >= self.min_electron_energy))
            | (((pid_in == 3) | (pid_in == 4)) & (E_in >= min_had_E))
            | ((pid_in == 5) & (E_in >= min_pi0_E))
            | ((pid_in == 6) & (E_in >= self.min_muon_energy))
        )
        p = p[alive_mask]
        if p.shape[0] == 0:
            self.active = p
            return

        pid = p[:, 0].to(torch.int32)
        E = p[:, 1].clone()
        x, y, z = p[:, 2], p[:, 3], p[:, 4]
        px, py, pz = p[:, 5].clone(), p[:, 6].clone(), p[:, 7].clone()
        gen = p[:, 8]
        evt = p[:, 9]
        N = p.shape[0]

        dist = torch.zeros(N, device=self.device, dtype=self.dtype)
        delta_x = torch.zeros(N, device=self.device, dtype=self.dtype)

        # Particle type masks
        mask_gamma = (pid == 0)
        mask_e = (pid == 1) | (pid == 2)
        mask_e_rad = mask_e & (E >= self.critical_energy)
        mask_e_ion = mask_e & (E < self.critical_energy)
        mask_p = (pid == 3)
        mask_pi_ch = (pid == 4)
        mask_had = mask_p | mask_pi_ch
        mask_pi0 = (pid == 5)
        mask_mu = (pid == 6)

        alpha_ion = 0.0021     # GeV / (g/cm^2) for e+- and hadrons
        alpha_ion_mu = 0.0020  # GeV / (g/cm^2) for muons

        # Sample slant step grammage delta_x (g/cm^2)
        if mask_gamma.any():
            n_g = int(mask_gamma.sum().item())
            delta_x[mask_gamma] = (
                torch.empty(n_g, device=self.device, dtype=self.dtype).exponential_()
                * (X0_AIR * 9.0 / 7.0)
            )
        if mask_e_rad.any():
            n_er = int(mask_e_rad.sum().item())
            delta_x[mask_e_rad] = (
                torch.empty(n_er, device=self.device, dtype=self.dtype).exponential_()
                * BREM_LAMBDA_GCM2
            )
        if mask_e_ion.any():
            # 20--85 MeV ionization-only Cherenkov tail stepped smoothly in <= 15 g/cm^2 slices
            rem_e = torch.clamp(
                (E[mask_e_ion] - self.min_electron_energy) / alpha_ion, min=0.0
            ) + 1.0e-4
            delta_x[mask_e_ion] = torch.minimum(
                torch.full_like(rem_e, 15.0), rem_e
            )
        if mask_p.any():
            n_p = int(mask_p.sum().item())
            delta_x[mask_p] = (
                torch.empty(n_p, device=self.device, dtype=self.dtype).exponential_()
                * LAMBDA_PROTON_GCM2
            )
        if mask_pi_ch.any():
            n_pi = int(mask_pi_ch.sum().item())
            delta_x[mask_pi_ch] = (
                torch.empty(n_pi, device=self.device, dtype=self.dtype).exponential_()
                * LAMBDA_PION_GCM2
            )
        if mask_mu.any():
            rem_mu = torch.clamp(
                (E[mask_mu] - self.min_muon_energy) / alpha_ion_mu, min=0.0
            ) + 1.0e-4
            delta_x[mask_mu] = torch.minimum(
                torch.full_like(rem_mu, 50.0), rem_mu
            )

        # Cap electron and hadron steps at their continuous ionization range
        if mask_e.any():
            max_ion_x_e = (
                torch.clamp((E[mask_e] - self.min_electron_energy) / alpha_ion, min=0.0)
                + 1.0e-4
            )
            delta_x[mask_e] = torch.minimum(delta_x[mask_e], max_ion_x_e)
        if mask_had.any():
            max_ion_x_h = (
                torch.clamp((E[mask_had] - min_had_E) / alpha_ion, min=0.0)
                + 1.0e-4
            )
            delta_x[mask_had] = torch.minimum(delta_x[mask_had], max_ion_x_h)

        # Convert slant grammage to geometric arclength in the 5-layer CORSIKA atmosphere
        mask_grammage = mask_gamma | mask_e | mask_had | mask_mu
        if mask_grammage.any():
            dist[mask_grammage] = arclength_from_grammage(
                z[mask_grammage], pz[mask_grammage], delta_x[mask_grammage]
            )

        # Charged pion decay competition: pi+- -> mu+- + nu_mu
        pi_decays = torch.zeros(N, device=self.device, dtype=torch.bool)
        if mask_pi_ch.any():
            n_pi = int(mask_pi_ch.sum().item())
            d_dec_pi = (
                torch.empty(n_pi, device=self.device, dtype=self.dtype).exponential_()
                * (E[mask_pi_ch] / M_PION_CHARGED_GEV)
                * C_TAU_PION_M
            )
            dec_local = d_dec_pi < dist[mask_pi_ch]
            if dec_local.any():
                pi_idx = mask_pi_ch.nonzero(as_tuple=True)[0][dec_local]
                pi_decays[pi_idx] = True
                d_new = d_dec_pi[dec_local]
                dist[pi_idx] = d_new
                delta_x[pi_idx] = torch.minimum(
                    delta_x[pi_idx],
                    _grammage_from_arclength(z[pi_idx], pz[pi_idx], d_new),
                )

        # Muon decay competition
        mu_decays = torch.zeros(N, device=self.device, dtype=torch.bool)
        if mask_mu.any():
            n_mu = int(mask_mu.sum().item())
            d_dec_mu = (
                torch.empty(n_mu, device=self.device, dtype=self.dtype).exponential_()
                * (E[mask_mu] / M_MUON_GEV)
                * C_TAU_MUON_M
            )
            dec_local = d_dec_mu < dist[mask_mu]
            if dec_local.any():
                mu_idx = mask_mu.nonzero(as_tuple=True)[0][dec_local]
                mu_decays[mu_idx] = True
                d_new = d_dec_mu[dec_local]
                dist[mu_idx] = d_new
                delta_x[mu_idx] = torch.minimum(
                    delta_x[mu_idx],
                    _grammage_from_arclength(z[mu_idx], pz[mu_idx], d_new),
                )

        # Neutral pion: prompt decay (c*tau ~ 25 nm)
        if mask_pi0.any():
            dist[mask_pi0] = 1.0

        px_eff = px.clone()
        py_eff = py.clone()
        pz_eff = pz.clone()

        # Highland Multiple Coulomb Scattering for electrons and muons
        mask_mcs = mask_e | mask_mu
        if mask_mcs.any():
            x_gcm2 = delta_x[mask_mcs]
            valid_scatter = x_gcm2 > 1.0e-6
            if valid_scatter.any():
                idx_scatter = mask_mcs.nonzero(as_tuple=True)[0][valid_scatter]
                x_g = x_gcm2[valid_scatter]
                Ee = E[idx_scatter]

                # theta_0 = (13.6 MeV / p c) * sqrt(x/X0) * (1 + 0.038 * ln(x/X0))
                ratio_x0 = torch.clamp(x_g / X0_AIR, min=1.0e-4)
                theta_rms = (
                    (0.0136 / Ee)
                    * torch.sqrt(ratio_x0)
                    * torch.clamp(1.0 + 0.038 * torch.log(ratio_x0), min=0.5)
                )
                theta_rms = torch.clamp(theta_rms, min=0.0, max=0.3)

                theta1 = torch.normal(mean=0.0, std=theta_rms)
                theta2 = torch.normal(mean=0.0, std=theta_rms)

                v_x = px[idx_scatter]
                v_y = py[idx_scatter]
                v_z = pz[idx_scatter]
                e1_x, e1_y, e1_z, e2_x, e2_y, e2_z = orthonormal_basis(v_x, v_y, v_z)

                # Position advances by 1/sqrt(3) of the angular kick
                inv_sqrt3 = 0.577350269
                p_eff_x, p_eff_y, p_eff_z = norm_dir(
                    v_x + theta1 * inv_sqrt3 * e1_x + theta2 * inv_sqrt3 * e2_x,
                    v_y + theta1 * inv_sqrt3 * e1_y + theta2 * inv_sqrt3 * e2_y,
                    v_z + theta1 * inv_sqrt3 * e1_z + theta2 * inv_sqrt3 * e2_z,
                )
                px_eff[idx_scatter] = p_eff_x.to(self.dtype)
                py_eff[idx_scatter] = p_eff_y.to(self.dtype)
                pz_eff[idx_scatter] = p_eff_z.to(self.dtype)

                # Momentum advances by full angular kick
                p_new_x, p_new_y, p_new_z = norm_dir(
                    v_x + theta1 * e1_x + theta2 * e2_x,
                    v_y + theta1 * e1_y + theta2 * e2_y,
                    v_z + theta1 * e1_z + theta2 * e2_z,
                )
                px[idx_scatter] = p_new_x.to(self.dtype)
                py[idx_scatter] = p_new_y.to(self.dtype)
                pz[idx_scatter] = p_new_z.to(self.dtype)

        # Preliminary observation-plane clipping BEFORE Lorentz deflection so geomagnetic
        # bending uses the clipped path length and midpoint energy E_mid.
        z_raw_pre = z + pz_eff * dist
        f_clip_pre = torch.where(
            z_raw_pre < z_obs,
            (z - z_obs) / torch.clamp(z - z_raw_pre, min=1.0e-6),
            torch.ones_like(dist),
        )
        f_clip_pre = torch.clamp(f_clip_pre, min=0.0, max=1.0)
        dist_pre = dist * f_clip_pre
        delta_x_pre = delta_x * f_clip_pre

        alpha_per_p = torch.where(
            mask_mu,
            torch.full_like(E, alpha_ion_mu),
            torch.full_like(E, alpha_ion),
        )
        min_E_track = torch.where(
            mask_mu,
            torch.full_like(E, self.min_muon_energy),
            torch.full_like(E, self.min_electron_energy),
        )
        E_mid_pre = torch.clamp(
            E - 0.5 * alpha_per_p * delta_x_pre, min=min_E_track
        )

        # Lorentz Force Geomagnetic Deflection (e+, e-, mu+-)
        mask_lorentz = mask_e | mask_mu
        if mask_lorentz.any():
            c_idx = mask_lorentz.nonzero(as_tuple=True)[0]
            p_vec = torch.stack([px[c_idx], py[c_idx], pz[c_idx]], dim=1)

            # Extract charge (+1 for e+, -1 for e-, random +-1 for mu)
            q = torch.zeros_like(E[c_idx])
            q[pid[c_idx] == 1] = 1.0
            q[pid[c_idx] == 2] = -1.0
            is_mu_c = (pid[c_idx] == 6)
            if is_mu_c.any():
                n_mu_c = int(is_mu_c.sum().item())
                q[is_mu_c] = torch.where(
                    self._get_rand(n_mu_c) < 0.5,
                    torch.ones(n_mu_c, device=self.device, dtype=self.dtype),
                    -torch.ones(n_mu_c, device=self.device, dtype=self.dtype),
                )

            B_expanded = self.B_field.unsqueeze(0).expand(len(c_idx), -1)
            cross_prod = torch.linalg.cross(p_vec, B_expanded, dim=1)

            # Deflection: (0.3 * q * dist_clipped / E_mid) * (v_hat x B)
            deflection_mag = (0.3 * q * dist_pre[c_idx]) / E_mid_pre[c_idx]
            dp = cross_prod * deflection_mag.unsqueeze(1)
            dp_norm = torch.linalg.vector_norm(dp, dim=1, keepdim=True)
            dp = dp * torch.clamp(0.5 / torch.clamp(dp_norm, min=1.0e-12), max=1.0)

            p_nx, p_ny, p_nz = norm_dir(
                px[c_idx] + dp[:, 0],
                py[c_idx] + dp[:, 1],
                pz[c_idx] + dp[:, 2],
            )
            px[c_idx] = p_nx.to(self.dtype)
            py[c_idx] = p_ny.to(self.dtype)
            pz[c_idx] = p_nz.to(self.dtype)

            # Lorentz force is constant over the step, so mean deflection is half the total
            px_eff_nx, py_eff_ny, pz_eff_nz = norm_dir(
                px_eff[c_idx] + dp[:, 0] * 0.5,
                py_eff[c_idx] + dp[:, 1] * 0.5,
                pz_eff[c_idx] + dp[:, 2] * 0.5,
            )
            px_eff[c_idx] = px_eff_nx.to(self.dtype)
            py_eff[c_idx] = py_eff_ny.to(self.dtype)
            pz_eff[c_idx] = pz_eff_nz.to(self.dtype)

        # Final observation-plane clipping with updated effective direction
        z_raw = z + pz_eff * dist
        f_clip = torch.where(
            z_raw < z_obs,
            (z - z_obs) / torch.clamp(z - z_raw, min=1.0e-6),
            torch.ones_like(dist),
        )
        f_clip = torch.clamp(f_clip, min=0.0, max=1.0)
        dist_clipped = dist * f_clip
        delta_x_clipped = delta_x * f_clip

        x_new = x + px_eff * dist_clipped
        y_new = y + py_eff * dist_clipped
        z_new = torch.clamp(z + pz_eff * dist_clipped, min=z_obs)

        # Record Cherenkov segments for electrons and muons above z_obs
        c_mask = (mask_e | mask_mu) & (z > z_obs) & (dist_clipped > 1.0e-3)
        if c_mask.any():
            E_mid = torch.clamp(
                E[c_mask] - 0.5 * alpha_per_p[c_mask] * delta_x_clipped[c_mask],
                min=min_E_track[c_mask],
            )
            seg_mass = torch.where(
                mask_mu[c_mask],
                torch.full_like(E_mid, M_MUON_GEV),
                torch.full_like(E_mid, M_ELECTRON_GEV),
            )
            self.c_segs_start.append(torch.stack([x[c_mask], y[c_mask], z[c_mask]], dim=1))
            self.c_segs_end.append(torch.stack([x_new[c_mask], y_new[c_mask], z_new[c_mask]], dim=1))
            self.c_segs_p.append(torch.stack([px_eff[c_mask], py_eff[c_mask], pz_eff[c_mask]], dim=1))
            self.c_segs_E.append(E_mid)
            self.c_segs_mass.append(seg_mass)
            self.c_segs_event_id.append(evt[c_mask])

        # Continuous energy loss along the step
        mask_ion_eh = mask_e | mask_had
        if mask_ion_eh.any():
            E[mask_ion_eh] = E[mask_ion_eh] - alpha_ion * delta_x_clipped[mask_ion_eh]
        if mask_mu.any():
            E[mask_mu] = E[mask_mu] - alpha_ion_mu * delta_x_clipped[mask_mu]
        if mask_e_rad.any():
            # Continuous soft bremsstrahlung loss ensures exact Bethe-Heitler stopping power 1/X0
            E[mask_e_rad] = E[mask_e_rad] * torch.exp(
                -BREM_SOFT_LOSS_PER_X0 * delta_x_clipped[mask_e_rad] / X0_AIR
            )

        # End-of-step interactions and survival (only for particles above z_obs and inside atmosphere)
        in_atm = (z_raw > z_obs) & (z_raw < 112829.0)
        new_particles = []

        # 1. Gamma -> e+ e- (complete-screening Bethe-Heitler pair production)
        mask_g = in_atm & mask_gamma & (E >= min_gamma_E)
        if mask_g.any():
            idx_g = mask_g.nonzero(as_tuple=True)[0]
            N_g = idx_g.shape[0]
            E_g = E[idx_g]

            u = sample_pair_production_u(N_g, device=self.device, dtype=self.dtype)
            e1_E = u * E_g
            e2_E = (1.0 - u) * E_g

            theta = torch.clamp(M_ELECTRON_GEV / torch.clamp(E_g, min=0.01), max=0.2)
            phi = self._get_rand(N_g) * (2.0 * math.pi)
            px1, py1, pz1 = apply_angular_kick(px[idx_g], py[idx_g], pz[idx_g], theta, phi)
            px2, py2, pz2 = apply_angular_kick(px[idx_g], py[idx_g], pz[idx_g], theta, phi + math.pi)

            gen_new = gen[idx_g] + 1.0
            evt_new = evt[idx_g]
            x_g, y_g, z_g = x_new[idx_g], y_new[idx_g], z_new[idx_g]

            keep_e1 = e1_E >= self.min_electron_energy
            if keep_e1.any():
                e1 = torch.stack([
                    torch.full_like(e1_E[keep_e1], 1.0),
                    e1_E[keep_e1],
                    x_g[keep_e1],
                    y_g[keep_e1],
                    z_g[keep_e1],
                    px1[keep_e1],
                    py1[keep_e1],
                    pz1[keep_e1],
                    gen_new[keep_e1],
                    evt_new[keep_e1],
                ], dim=1)
                new_particles.append(e1)

            keep_e2 = e2_E >= self.min_electron_energy
            if keep_e2.any():
                e2 = torch.stack([
                    torch.full_like(e2_E[keep_e2], 2.0),
                    e2_E[keep_e2],
                    x_g[keep_e2],
                    y_g[keep_e2],
                    z_g[keep_e2],
                    px2[keep_e2],
                    py2[keep_e2],
                    pz2[keep_e2],
                    gen_new[keep_e2],
                    evt_new[keep_e2],
                ], dim=1)
                new_particles.append(e2)

        # 2. Radiative Electron -> e + gamma (Bethe-Heitler hard bremsstrahlung)
        mask_b = in_atm & mask_e_rad & (E >= self.critical_energy)
        if mask_b.any():
            idx_b = mask_b.nonzero(as_tuple=True)[0]
            N_b = idx_b.shape[0]
            E_b = E[idx_b]

            k = sample_bremsstrahlung_k(N_b, device=self.device, dtype=self.dtype)
            g_E = k * E_b
            e_E = (1.0 - k) * E_b

            theta = torch.clamp(M_ELECTRON_GEV / torch.clamp(E_b, min=0.01), max=0.2)
            phi = self._get_rand(N_b) * (2.0 * math.pi)
            px1, py1, pz1 = apply_angular_kick(px[idx_b], py[idx_b], pz[idx_b], theta, phi + math.pi)
            px2, py2, pz2 = apply_angular_kick(px[idx_b], py[idx_b], pz[idx_b], theta, phi)

            gen_new = gen[idx_b] + 1.0
            evt_new = evt[idx_b]
            x_b, y_b, z_b = x_new[idx_b], y_new[idx_b], z_new[idx_b]

            keep_e = e_E >= self.min_electron_energy
            if keep_e.any():
                e_out = torch.stack([
                    pid[idx_b][keep_e].to(self.dtype),
                    e_E[keep_e],
                    x_b[keep_e],
                    y_b[keep_e],
                    z_b[keep_e],
                    px1[keep_e],
                    py1[keep_e],
                    pz1[keep_e],
                    gen_new[keep_e],
                    evt_new[keep_e],
                ], dim=1)
                new_particles.append(e_out)

            keep_g = g_E >= min_gamma_E
            if keep_g.any():
                g_out = torch.stack([
                    torch.zeros_like(g_E[keep_g]),
                    g_E[keep_g],
                    x_b[keep_g],
                    y_b[keep_g],
                    z_b[keep_g],
                    px2[keep_g],
                    py2[keep_g],
                    pz2[keep_g],
                    gen_new[keep_g],
                    evt_new[keep_g],
                ], dim=1)
                new_particles.append(g_out)

        # 3. Sub-critical Electron (20--85 MeV): keep alive via pure ionization tracking
        mask_e_sub = (
            in_atm
            & mask_e
            & (E >= self.min_electron_energy)
            & (E < self.critical_energy)
        )
        if mask_e_sub.any():
            idx_es = mask_e_sub.nonzero(as_tuple=True)[0]
            e_surv = torch.stack([
                pid[idx_es].to(self.dtype),
                E[idx_es],
                x_new[idx_es],
                y_new[idx_es],
                z_new[idx_es],
                px[idx_es],
                py[idx_es],
                pz[idx_es],
                gen[idx_es] + 1.0,
                evt[idx_es],
            ], dim=1)
            new_particles.append(e_surv)

        # 4. Decaying pi+- -> mu+- + nu_mu
        mask_pi_dec = in_atm & pi_decays & (E >= M_PION_CHARGED_GEV)
        if mask_pi_dec.any():
            idx_pd = mask_pi_dec.nonzero(as_tuple=True)[0]
            mu_out = sample_pion_decay_to_muon(
                E[idx_pd],
                x_new[idx_pd],
                y_new[idx_pd],
                z_new[idx_pd],
                px[idx_pd],
                py[idx_pd],
                pz[idx_pd],
                gen[idx_pd],
                evt[idx_pd],
            )
            keep_mu = mu_out[:, 1] >= self.min_muon_energy
            if keep_mu.any():
                new_particles.append(mu_out[keep_mu])

        # 5. Interacting Hadron (leading hadron + secondary pi0/pi+-)
        mask_h = in_atm & mask_had & (~pi_decays) & (E >= min_had_E)
        if mask_h.any():
            idx_h = mask_h.nonzero(as_tuple=True)[0]
            had_children = sample_hadronic_interaction(
                pid[idx_h],
                E[idx_h],
                x_new[idx_h],
                y_new[idx_h],
                z_new[idx_h],
                px[idx_h],
                py[idx_h],
                pz[idx_h],
                gen[idx_h],
                evt[idx_h],
            )
            for child in had_children:
                c_pid = child[:, 0].to(torch.int32)
                c_E = child[:, 1]
                keep_c = ((c_pid == 5) & (c_E >= min_pi0_E)) | (
                    ((c_pid == 3) | (c_pid == 4)) & (c_E >= min_had_E)
                )
                if keep_c.any():
                    new_particles.append(child[keep_c])

        # 6. Pi0 -> 2 gamma (exact kinematic opening angle)
        mask_p0 = in_atm & mask_pi0 & (E >= min_pi0_E)
        if mask_p0.any():
            idx_p0 = mask_p0.nonzero(as_tuple=True)[0]
            E_p0 = E[idx_p0]
            e_split = E_p0 * 0.5
            keep_p0 = e_split >= min_gamma_E
            if keep_p0.any():
                idx_p0 = idx_p0[keep_p0]
                E_p0 = E_p0[keep_p0]
                e_split = e_split[keep_p0]
                N_p0 = idx_p0.shape[0]

                theta = torch.clamp(
                    M_PION_ZERO_GEV / torch.clamp(E_p0, min=0.14), max=0.8
                )
                phi = self._get_rand(N_p0) * (2.0 * math.pi)
                px1, py1, pz1 = apply_angular_kick(
                    px[idx_p0], py[idx_p0], pz[idx_p0], theta, phi
                )
                px2, py2, pz2 = apply_angular_kick(
                    px[idx_p0], py[idx_p0], pz[idx_p0], theta, phi + math.pi
                )

                gen_new = gen[idx_p0] + 1.0
                evt_new = evt[idx_p0]
                x_p0, y_p0, z_p0 = x_new[idx_p0], y_new[idx_p0], z_new[idx_p0]

                g1 = torch.stack([
                    torch.zeros_like(e_split),
                    e_split,
                    x_p0,
                    y_p0,
                    z_p0,
                    px1,
                    py1,
                    pz1,
                    gen_new,
                    evt_new,
                ], dim=1)
                g2 = torch.stack([
                    torch.zeros_like(e_split),
                    e_split,
                    x_p0,
                    y_p0,
                    z_p0,
                    px2,
                    py2,
                    pz2,
                    gen_new,
                    evt_new,
                ], dim=1)
                new_particles.extend([g1, g2])

        # 7. Surviving Muon: continue transport down to z_obs emitting Cherenkov rings
        mask_mu_surv = in_atm & mask_mu & (~mu_decays) & (E >= self.min_muon_energy)
        if mask_mu_surv.any():
            idx_ms = mask_mu_surv.nonzero(as_tuple=True)[0]
            mu_surv = torch.stack([
                torch.full_like(E[idx_ms], 6.0),
                E[idx_ms],
                x_new[idx_ms],
                y_new[idx_ms],
                z_new[idx_ms],
                px[idx_ms],
                py[idx_ms],
                pz[idx_ms],
                gen[idx_ms] + 1.0,
                evt[idx_ms],
            ], dim=1)
            new_particles.append(mu_surv)

        if new_particles:
            self.active = torch.cat(new_particles, dim=0)
        else:
            self.active = torch.empty((0, 10), device=self.device, dtype=self.dtype)

    def save_state(self, filepath="shower_checkpoint.pt"):
        state = {
            'active': self.active,
            'entropy_idx': 0,
            'c_segs_start': self.c_segs_start,
            'c_segs_end': self.c_segs_end,
            'c_segs_p': self.c_segs_p,
            'c_segs_E': self.c_segs_E,
            'c_segs_mass': self.c_segs_mass,
            'c_segs_event_id': self.c_segs_event_id,
            'cherenkov_photons_by_event': self.cherenkov_photons_by_event,
            'batch_size': self.batch_size,
            'critical_energy': self.critical_energy,
            'min_electron_energy': self.min_electron_energy,
            'min_muon_energy': self.min_muon_energy,
            'z_obs': self.z_obs,
            'primary_energies': self.primary_energies,
            'photon_yield_factor': self.photon_yield_factor,
        }
        torch.save(state, filepath)

    def load_state(self, filepath="shower_checkpoint.pt"):
        state = torch.load(filepath, map_location=self.device, weights_only=False)
        self.active = state['active']
        self.c_segs_start = state['c_segs_start']
        self.c_segs_end = state['c_segs_end']
        self.c_segs_p = state['c_segs_p']
        self.c_segs_E = state['c_segs_E']
        self.c_segs_mass = state.get('c_segs_mass', [])
        self.c_segs_event_id = state['c_segs_event_id']
        self.cherenkov_photons_by_event = state['cherenkov_photons_by_event']
        self.batch_size = state['batch_size']
        self.critical_energy = state['critical_energy']
        self.min_electron_energy = state.get('min_electron_energy', self.min_electron_energy)
        self.min_muon_energy = state.get('min_muon_energy', self.min_muon_energy)
        self.z_obs = state.get('z_obs', self.z_obs)
        self.primary_energies = state.get('primary_energies', self.primary_energies)
        self.photon_yield_factor = state['photon_yield_factor']

    def run(self, max_generations=60, verbose=True):
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
        if self.c_segs_mass and sum(m.numel() for m in self.c_segs_mass) == Es.numel():
            masses = torch.cat(self.c_segs_mass, dim=0)
        else:
            masses = torch.full_like(Es, M_ELECTRON_GEV)

        # Ray trace per event to avoid mixing photons
        for i in range(self.batch_size):
            mask = (evts == i)
            if not mask.any():
                continue

            e_starts = starts[mask]
            e_ends = ends[mask]
            e_ps = ps[mask]
            e_Es = Es[mask]
            e_masses = masses[mask]

            E0 = self.primary_energies[i] if i < len(self.primary_energies) else 1000.0
            if E0 <= 100.0:
                target_ppp = 5.0
            elif E0 <= 500.0:
                target_ppp = 10.0
            elif E0 <= 5000.0:
                target_ppp = 25.0
            else:
                target_ppp = 50.0

            self.cherenkov_photons_by_event[i] = compute_cherenkov_pool_gpu(
                e_starts[:, 0], e_starts[:, 1], e_starts[:, 2],
                e_ends[:, 0], e_ends[:, 1], e_ends[:, 2],
                e_ps[:, 0], e_ps[:, 1], e_ps[:, 2],
                e_Es, self.photon_yield_factor,
                target_photons_per_packet=target_ppp,
                z_obs=self.z_obs,
                seg_mass_gev=e_masses,
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
            'y': photons['y_ground'][indices],
        }
        return pd.DataFrame(data)

    def get_tracks_dataframe(self):
        # We omitted storing full tracks to save VRAM. Return empty.
        return pd.DataFrame(columns=['particle_id', 'pid', 'energy', 'generation', 'x', 'y', 'z', 'event_id'])


if __name__ == "__main__":
    sim = ShowerSimulation(['gamma', 'proton'], energies=[100.0, 500.0])
    sim.run(max_generations=60)

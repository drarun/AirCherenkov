import numpy as np
import pandas as pd
import torch
import time

class ShowerSimulation:
    TRACK_COLUMNS = [
        'particle_id', 'pid', 'energy', 'generation',
        'x', 'y', 'z', 'event_id',
    ]

    def __init__(
        self,
        primary_types=None,
        energies=None,
        z_starts=None,
        *,
        record_tracks=False,
        device='auto',
        seed=None,
        target_photons_per_packet=64,
        max_packets_per_segment=2048,
        primary_type=None,
        energy=None,
        z_start=None,
    ):
        """
        A 3D toy Monte Carlo using fully tensorized PyTorch operations.
        Supports batch processing of multiple independent showers.

        Set ``record_tracks=True`` to retain propagated particle segments on
        CPU for visualization. Track recording is opt-in so large training
        batches do not pay the transfer and host-memory cost.

        The singular ``primary_type``, ``energy``, and ``z_start`` keywords
        are accepted for compatibility with the original scalar API.
        """
        self.critical_energy = 0.085 # 85 MeV
        self.photon_yield_factor = 1.0 
        
        from sim.backend import get_device
        self.device = get_device(device)
        if self.device is None:
            self.device = torch.device('cpu')

        if seed is not None and not isinstance(seed, (int, np.integer)):
            raise TypeError("seed must be an integer or None")
        if target_photons_per_packet <= 0:
            raise ValueError("target_photons_per_packet must be greater than zero")
        if not isinstance(max_packets_per_segment, (int, np.integer)) or max_packets_per_segment <= 0:
            raise ValueError("max_packets_per_segment must be a positive integer")

        self.seed = None if seed is None else int(seed)
        self.generator = None
        if self.seed is not None:
            self.generator = torch.Generator(device=self.device)
            self.generator.manual_seed(self.seed)
        self.target_photons_per_packet = float(target_photons_per_packet)
        self.max_packets_per_segment = int(max_packets_per_segment)
            
        self.PID_MAP = {'gamma': 0, 'e+': 1, 'e-': 2, 'proton': 3, 'pi_charged': 4, 'pi0': 5, 'mu': 6}
        self.INV_PID_MAP = {v: k for k, v in self.PID_MAP.items()}
        
        primary_types = self._resolve_input(
            primary_types, primary_type, ['gamma'], 'primary_types', 'primary_type'
        )
        energies = self._resolve_input(
            energies, energy, [1000.0], 'energies', 'energy'
        )
        z_starts = self._resolve_input(
            z_starts, z_start, [20000.0], 'z_starts', 'z_start'
        )

        primary_types = self._as_list(primary_types)
        energies = self._as_list(energies)
        z_starts = self._as_list(z_starts)

        batch_size = max(len(primary_types), len(energies), len(z_starts))
        if batch_size == 0:
            raise ValueError("At least one primary shower must be provided")

        primary_types = self._broadcast(primary_types, batch_size, 'primary_types')
        energies = self._broadcast(energies, batch_size, 'energies')
        z_starts = self._broadcast(z_starts, batch_size, 'z_starts')

        unknown = sorted(set(primary_types) - set(self.PID_MAP))
        if unknown:
            raise ValueError(f"Unknown primary particle type(s): {', '.join(map(str, unknown))}")
        if not np.all(np.isfinite(energies)) or np.any(np.asarray(energies) <= 0):
            raise ValueError("All primary energies must be finite and greater than zero")
        if not np.all(np.isfinite(z_starts)) or np.any(np.asarray(z_starts) <= 0):
            raise ValueError("All starting altitudes must be finite and greater than zero")
            
        # VERITAS typical B-field (approximate Cartesian components in Tesla)
        # X (North), Y (East), Z (Down)
        self.B_field = torch.tensor([23.9e-6, 4.1e-6, 40.8e-6], device=self.device)
            
        # State tensor [PID, E, x, y, z, px, py, pz, generation, event_id]
        init_state = []
        for i in range(batch_size):
            init_state.append([self.PID_MAP[primary_types[i]], energies[i], 0.0, 0.0, z_starts[i], 0.0, 0.0, -1.0, 0.0, float(i)])
            
        self.active = torch.tensor(init_state, dtype=torch.float32, device=self.device)
        self.batch_size = batch_size
        self.start_altitudes = np.asarray(z_starts, dtype=np.float32)

        # Generic particle segments for optional 3D visualization. Chunks are
        # detached to CPU immediately so enabling this never grows VRAM usage.
        self.record_tracks = bool(record_tracks)
        self._track_chunks = []
        self._next_particle_id = 0
        
        # Cherenkov segment accumulators
        self.c_segs_start = []
        self.c_segs_end = []
        self.c_segs_p = []
        self.c_segs_E = []
        self.c_segs_event_id = []
        
        # Compatibility name retained: each row now represents a weighted photon
        # packet rather than necessarily one physical photon.
        self.cherenkov_photons_by_event = {
            i: self._empty_photon_packets() for i in range(batch_size)
        }
        self.cherenkov_packets = self._empty_photon_packets(include_event_id=True)

    @staticmethod
    def _empty_photon_packets(include_event_id=False):
        packets = {
            key: np.array([], dtype=np.float32)
            for key in (
                'x_emit', 'y_emit', 'z_emit', 'x_ground', 'y_ground',
                'weight', 'shower_start_altitude',
            )
        }
        if include_event_id:
            packets['event_id'] = np.array([], dtype=np.int32)
        return packets

    @staticmethod
    def _resolve_input(plural, singular, default, plural_name, singular_name):
        if plural is not None and singular is not None:
            raise TypeError(f"Use either {plural_name} or {singular_name}, not both")
        if singular is not None:
            return singular
        if plural is not None:
            return plural
        return default

    @staticmethod
    def _as_list(value):
        if isinstance(value, str) or np.isscalar(value):
            return [value]
        return list(value)

    @staticmethod
    def _broadcast(values, batch_size, name):
        if len(values) == batch_size:
            return values
        if len(values) == 1:
            return values * batch_size
        raise ValueError(
            f"{name} must contain either 1 or {batch_size} values; got {len(values)}"
        )

    def _atmospheric_density(self, z):
        return 1.225e-3 * torch.exp(-z / 7640.0)

    def _mean_free_path(self, z, X_length):
        rho = self._atmospheric_density(z)
        return X_length / (rho * 100.0)
        
    def _norm_dir(self, px, py, pz):
        norm = torch.sqrt(px**2 + py**2 + pz**2)
        safe = norm > 0
        return (torch.where(safe, px/norm, px),
                torch.where(safe, py/norm, py),
                torch.where(safe, pz/norm, pz))

    @staticmethod
    def _clip_segments_to_ground(x, y, z, x_end, y_end, z_end):
        """Interpolate below-ground endpoints back to the z=0 plane."""
        crosses_ground = z_end < 0
        denominator = z - z_end
        fraction = torch.where(
            crosses_ground & (denominator > 0),
            z / denominator,
            torch.ones_like(z),
        )
        fraction = torch.clamp(fraction, min=0.0, max=1.0)
        clipped_x = x + fraction * (x_end - x)
        clipped_y = y + fraction * (y_end - y)
        clipped_z = torch.where(crosses_ground, torch.zeros_like(z_end), z_end)
        return clipped_x, clipped_y, clipped_z

    def _record_track_segments(self, pid, energy, generation, event_id,
                               x, y, z, x_end, y_end, z_end):
        if not self.record_tracks or pid.numel() == 0:
            return

        n_segments = int(pid.numel())
        particle_ids = np.arange(
            self._next_particle_id,
            self._next_particle_id + n_segments,
            dtype=np.int64,
        )
        self._next_particle_id += n_segments

        starts = torch.stack([x, y, z], dim=1).detach().cpu().numpy().copy()
        ends = torch.stack([x_end, y_end, z_end], dim=1).detach().cpu().numpy().copy()
        self._track_chunks.append({
            'particle_id': particle_ids,
            'pid': pid.detach().cpu().numpy().astype(np.int16, copy=True),
            'energy': energy.detach().cpu().numpy().astype(np.float32, copy=True),
            'generation': generation.detach().cpu().numpy().astype(np.int32, copy=True),
            'event_id': event_id.detach().cpu().numpy().astype(np.int32, copy=True),
            'start': starts.astype(np.float32, copy=False),
            'end': ends.astype(np.float32, copy=False),
        })

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
        dist = torch.zeros(N, device=self.device, dtype=torch.float32)
        
        # Mean free paths
        mask_em = (pid == 0) | (pid == 1) | (pid == 2)
        mask_had = (pid == 3) | (pid == 4)
        mask_pi0 = (pid == 5)
        mask_mu = (pid == 6)
        
        if mask_em.any():
            dist[mask_em] = torch.empty(mask_em.sum(), device=self.device).exponential_(generator=self.generator) * self._mean_free_path(z[mask_em], 36.62)
        if mask_had.any():
            dist[mask_had] = torch.empty(mask_had.sum(), device=self.device).exponential_(generator=self.generator) * self._mean_free_path(z[mask_had], 90.0)
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
            x_gcm2 = dist_e * self._atmospheric_density(z_e) * 100.0
            
            valid_scatter = x_gcm2 > 0
            if valid_scatter.any():
                idx_scatter = mask_e.nonzero(as_tuple=True)[0][valid_scatter]
                x_g = x_gcm2[valid_scatter]
                Ee = E_e[valid_scatter]
                
                log_term = torch.log(x_g / 36.62)
                theta_rms = (0.0136 / Ee) * torch.sqrt(x_g / 36.62) * (1 + 0.038 * log_term)
                theta_rms = torch.clamp(theta_rms, min=0.0, max=0.3)
                
                theta1 = torch.normal(mean=0.0, std=theta_rms, generator=self.generator)
                theta2 = torch.normal(mean=0.0, std=theta_rms, generator=self.generator)
                
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
                
                p_new_x, p_new_y, p_new_z = self._norm_dir(p_new_x, p_new_y, p_new_z)
                px[idx_scatter] = p_new_x
                py[idx_scatter] = p_new_y
                pz[idx_scatter] = p_new_z
                
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
            
            px[c_idx] += dp[:, 0]
            py[c_idx] += dp[:, 1]
            pz[c_idx] += dp[:, 2]
            
            px[c_idx], py[c_idx], pz[c_idx] = self._norm_dir(px[c_idx], py[c_idx], pz[c_idx])

        x_new = x + px * dist
        y_new = y + py * dist
        z_new = z + pz * dist

        segment_x_end, segment_y_end, segment_z_end = self._clip_segments_to_ground(
            x, y, z, x_new, y_new, z_new
        )
        self._record_track_segments(
            pid, E, gen, evt, x, y, z,
            segment_x_end, segment_y_end, segment_z_end,
        )
        
        # Ray Trace Cherenkov Segments
        c_mask = mask_e & (z > 0)
        if c_mask.any():
            self.c_segs_start.append(torch.stack([x[c_mask], y[c_mask], z[c_mask]], dim=1))
            self.c_segs_end.append(torch.stack([
                segment_x_end[c_mask], segment_y_end[c_mask], segment_z_end[c_mask]
            ], dim=1))
            self.c_segs_p.append(torch.stack([px[c_mask], py[c_mask], pz[c_mask]], dim=1))
            self.c_segs_E.append(E[c_mask])
            self.c_segs_event_id.append(evt[c_mask])
            
        z_new = torch.clamp(z_new, min=0.0)
        
        # Ionization
        mask_ion = mask_e | (pid == 3) | (pid == 4)
        if mask_ion.any():
            x_gcm2 = dist[mask_ion] * self._atmospheric_density(z_new[mask_ion]) * 100.0
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
            u = torch.rand(N_g, device=self.device, generator=self.generator) * 0.8 + 0.1
            e1_E = u * E[idx_g]
            e2_E = (1 - u) * E[idx_g]
            
            phi = torch.rand(N_g, device=self.device, generator=self.generator) * 2 * np.pi
            pt = 0.0005
            px1, py1, pz1 = self._norm_dir(px[idx_g] + pt*torch.cos(phi), py[idx_g] + pt*torch.sin(phi), pz[idx_g])
            px2, py2, pz2 = self._norm_dir(px[idx_g] - pt*torch.cos(phi), py[idx_g] - pt*torch.sin(phi), pz[idx_g])
            
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
            rand_val = torch.rand(N_b, device=self.device, generator=self.generator)
            u = torch.pow(min_u, rand_val)
            u = torch.max(min_u, u)
            
            g_E = u * E_b
            e_E = (1 - u) * E_b
            
            phi = torch.rand(N_b, device=self.device, generator=self.generator) * 2 * np.pi
            pt = 0.0005
            px1, py1, pz1 = self._norm_dir(px[idx_b] - pt*torch.cos(phi), py[idx_b] - pt*torch.sin(phi), pz[idx_b])
            px2, py2, pz2 = self._norm_dir(px[idx_b] + pt*torch.cos(phi), py[idx_b] + pt*torch.sin(phi), pz[idx_b])
            
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
                phi = torch.rand(N_h, device=self.device, generator=self.generator) * 2 * np.pi
                px_c, py_c, pz_c = self._norm_dir(px[idx_h] + pt*torch.cos(phi), py[idx_h] + pt*torch.sin(phi), pz[idx_h])
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
            phi = torch.rand(N_p0, device=self.device, generator=self.generator) * 2 * np.pi
            
            px1, py1, pz1 = self._norm_dir(px[idx_p0] + pt*torch.cos(phi), py[idx_p0] + pt*torch.sin(phi), pz[idx_p0])
            px2, py2, pz2 = self._norm_dir(px[idx_p0] - pt*torch.cos(phi), py[idx_p0] - pt*torch.sin(phi), pz[idx_p0])
            
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
        if not isinstance(max_generations, (int, np.integer)) or max_generations < 0:
            raise ValueError("max_generations must be a non-negative integer")

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
            if self.active.shape[0] == 0:
                status = "Cascade complete"
            else:
                status = (
                    f"Generation limit reached with {self.active.shape[0]:,} "
                    "active particles"
                )
            print(f"      {status} ({time.time()-t0:.1f}s)")
            print(f"      Computing Cherenkov pool on {self.device.type.upper()}...")
        t1 = time.time()
        self.calculate_cherenkov_pool()
        if verbose:
            n_packets = sum(
                len(d.get('x_ground', []))
                for d in self.cherenkov_photons_by_event.values()
            )
            n_photons = sum(
                float(np.sum(d.get('weight', np.ones(len(d.get('x_ground', []))))))
                for d in self.cherenkov_photons_by_event.values()
            )
            print(
                f"      Cherenkov pool: {n_packets:,} weighted packets "
                f"representing {n_photons:,.0f} photons ({time.time()-t1:.1f}s)"
            )

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
        
        # Generate packets for the full batch in one backend call. Keeping the
        # event ID alongside each packet avoids an O(events * segments) scan.
        packets = compute_cherenkov_pool_gpu(
            starts[:, 0], starts[:, 1], starts[:, 2],
            ends[:, 0], ends[:, 1], ends[:, 2],
            ps[:, 0], ps[:, 1], ps[:, 2],
            Es, self.photon_yield_factor,
            seg_event_id=evts,
            target_photons_per_packet=self.target_photons_per_packet,
            max_packets_per_segment=self.max_packets_per_segment,
            device=self.device,
            generator=self.generator,
        )

        if 'event_id' not in packets:
            packets['event_id'] = np.zeros(
                len(packets.get('x_ground', [])), dtype=np.int32
            )
        packet_event_ids = np.asarray(packets['event_id'], dtype=np.int32)
        order = np.argsort(packet_event_ids, kind='stable')
        packets = {
            key: np.asarray(values)[order]
            for key, values in packets.items()
        }
        packet_event_ids = packets['event_id']
        packets['shower_start_altitude'] = self.start_altitudes[
            packet_event_ids
        ].astype(np.float32, copy=False)
        self.cherenkov_packets = packets

        for event_idx in range(self.batch_size):
            start = int(np.searchsorted(packet_event_ids, event_idx, side='left'))
            end = int(np.searchsorted(packet_event_ids, event_idx, side='right'))
            event_packets = {
                key: values[start:end]
                for key, values in packets.items()
                if key != 'event_id'
            }
            self.cherenkov_photons_by_event[event_idx] = event_packets

    def get_cherenkov_dataframe(self, event_idx=0):
        photons = self.cherenkov_photons_by_event.get(
            event_idx, self._empty_photon_packets()
        )
        if len(photons.get('x_ground', [])) == 0:
            return pd.DataFrame(columns=['x', 'y'])
            
        n_tot = len(photons['x_ground'])
        max_plot = 10000
        if n_tot > max_plot:
            indices = np.linspace(0, n_tot - 1, max_plot, dtype=np.int64)
        else:
            indices = np.arange(n_tot)
            
        data = {
            'x': photons['x_ground'][indices],
            'y': photons['y_ground'][indices],
            'weight': np.asarray(
                photons.get('weight', np.ones(n_tot, dtype=np.float32))
            )[indices],
        }
        return pd.DataFrame(data)

    def get_tracks_dataframe(self, event_idx=None, max_tracks=None):
        """
        Export recorded start/end particle segments for 3D visualization.

        Parameters
        ----------
        event_idx : int or None
            Select one event from a batched simulation. ``None`` returns all.
        max_tracks : int or None
            Deterministically downsample the number of segments while retaining
            coverage across the full shower evolution.
        """
        if not self._track_chunks:
            return pd.DataFrame(columns=self.TRACK_COLUMNS)

        segments = {
            key: np.concatenate([chunk[key] for chunk in self._track_chunks], axis=0)
            for key in self._track_chunks[0]
        }

        indices = np.arange(len(segments['particle_id']))
        if event_idx is not None:
            indices = indices[segments['event_id'] == int(event_idx)]

        if max_tracks is not None:
            if not isinstance(max_tracks, (int, np.integer)) or max_tracks <= 0:
                raise ValueError("max_tracks must be a positive integer or None")
            if len(indices) > max_tracks:
                sample_positions = np.linspace(
                    0, len(indices) - 1, num=max_tracks, dtype=np.int64
                )
                indices = indices[sample_positions]

        if len(indices) == 0:
            return pd.DataFrame(columns=self.TRACK_COLUMNS)

        starts = segments['start'][indices]
        ends = segments['end'][indices]
        coordinates = np.stack([starts, ends], axis=1).reshape(-1, 3)
        pid_names = np.asarray(
            [self.INV_PID_MAP[int(value)] for value in segments['pid'][indices]],
            dtype=object,
        )

        return pd.DataFrame({
            'particle_id': np.repeat(segments['particle_id'][indices], 2),
            'pid': np.repeat(pid_names, 2),
            'energy': np.repeat(segments['energy'][indices], 2),
            'generation': np.repeat(segments['generation'][indices], 2),
            'x': coordinates[:, 0],
            'y': coordinates[:, 1],
            'z': coordinates[:, 2],
            'event_id': np.repeat(segments['event_id'][indices], 2),
        }, columns=self.TRACK_COLUMNS)

if __name__ == "__main__":
    sim = ShowerSimulation(
        primary_types=['gamma', 'proton'],
        energies=[100.0, 500.0],
        z_starts=[20000.0, 20000.0],
    )
    sim.run(max_generations=12)

import numpy as np
import pandas as pd

class Particle:
    def __init__(self, pid, energy, x, y, z, px, py, pz, generation=0):
        self.pid = pid  # 'gamma', 'e', 'proton', 'pi0', 'pi_charged', 'mu'
        self.energy = energy
        self.x = x
        self.y = y
        self.z = z
        self.px = px
        self.py = py
        self.pz = pz
        self.generation = generation
        # To store the track for visualization
        self.track = [(x, y, z, px, py, pz)]

    def update_position(self, dx, dy, dz, px, py, pz):
        self.x += dx
        self.y += dy
        self.z += dz
        self.track.append((self.x, self.y, self.z, px, py, pz))

class ShowerSimulation:
    def __init__(self, primary_type='gamma', energy=1000.0, z_start=20000.0):
        """
        A simplified 3D toy Monte Carlo for visualizing air showers.
        Distances are in meters, Energy in GeV.
        """
        self.primary_type = primary_type
        self.energy = energy
        self.z_start = z_start
        
        self.critical_energy = 0.085 # 85 MeV
        self.active_particles = []
        self.dead_particles = []
        
        # Inject primary
        self.active_particles.append(
            Particle(pid=primary_type, energy=energy, 
                     x=0, y=0, z=z_start, 
                     px=0, py=0, pz=-1, generation=0)
        )
        self.all_particles = []
        self.photon_yield_factor = 1.0 # Can be increased for realistic camera images
        self.cherenkov_photons = {
            'x_emit': [], 'y_emit': [], 'z_emit': [],
            'x_ground': [], 'y_ground': []
        }

    def atmospheric_density(self, z):
        """Returns density in g/cm^3 using the US Standard Atmosphere exponential model."""
        rho_0 = 1.225e-3 # g/cm^3
        H = 7640.0 # meters
        return rho_0 * np.exp(-z / H)

    def grammage(self, z):
        """Returns the column depth X in g/cm^2."""
        rho_0 = 1.225e-3 # g/cm^3
        H = 7640.0 # meters
        H_cm = H * 100.0
        return rho_0 * H_cm * np.exp(-z / H)

    def mean_free_path(self, z, X_length):
        """Returns the mean free path in meters for a given interaction length in g/cm^2."""
        rho = self.atmospheric_density(z)
        return X_length / (rho * 100.0)

    @staticmethod
    def normalize_direction(px, py, pz):
        norm = np.sqrt(px**2 + py**2 + pz**2)
        if norm > 0:
            return px/norm, py/norm, pz/norm
        return px, py, pz

    def apply_scattering(self, p, dist):
        if p.pid not in ['e+', 'e-']:
            return
            
        x_gcm2 = dist * self.atmospheric_density(p.z) * 100.0
        if x_gcm2 <= 0:
            return
            
        # Compute theta_rms using Highland formula
        log_term = np.log(x_gcm2 / 36.62)
        theta_rms = (0.0136 / p.energy) * np.sqrt(x_gcm2 / 36.62) * (1 + 0.038 * log_term)
        
        # Clamp to a reasonable max (e.g. 0.3 radians) to avoid instabilities at low energy
        theta_rms = max(0.0, min(theta_rms, 0.3))
        if theta_rms == 0:
            return

        theta1 = np.random.normal(0, theta_rms)
        theta2 = np.random.normal(0, theta_rms)

        px, py, pz = p.px, p.py, p.pz
        v = np.array([px, py, pz])
        
        # Use cross products with (0,0,1) or (1,0,0) to get two perpendicular unit vectors
        if abs(pz) < 0.99:
            ref = np.array([0, 0, 1])
        else:
            ref = np.array([1, 0, 0])
            
        e1 = np.cross(v, ref)
        norm_e1 = np.linalg.norm(e1)
        if norm_e1 > 0:
            e1 = e1 / norm_e1
        else:
            e1 = np.array([1.0, 0.0, 0.0]) # Fallback
            
        e2 = np.cross(v, e1)
        norm_e2 = np.linalg.norm(e2)
        if norm_e2 > 0:
            e2 = e2 / norm_e2
        else:
            e2 = np.array([0.0, 1.0, 0.0]) # Fallback
            
        p_new = v + theta1 * e1 + theta2 * e2
        norm_p = np.linalg.norm(p_new)
        if norm_p > 0:
            p_new = p_new / norm_p
            
        p.px, p.py, p.pz = p_new[0], p_new[1], p_new[2]

    def step(self):
        """Advance the simulation by one generation."""
        new_particles = []
        
        for p in self.active_particles:
            if p.energy < self.critical_energy or p.z <= 0:
                self.dead_particles.append(p)
                continue
                
            # Determine distance to next interaction
            if p.pid in ['gamma', 'e+', 'e-']:
                dist = np.random.exponential(self.mean_free_path(p.z, 36.62))
            elif p.pid in ['proton', 'pi_charged']:
                dist = np.random.exponential(self.mean_free_path(p.z, 90.0))
            elif p.pid == 'pi0':
                dist = 10.0 # Decays almost instantly
            else: # muons don't interact much in this toy model
                dist = 5000.0 
                
            # Apply Multiple Coulomb Scattering
            self.apply_scattering(p, dist)
                
            # Update position
            dx = p.px * dist
            dy = p.py * dist
            dz = p.pz * dist
            p.update_position(dx, dy, dz, p.px, p.py, p.pz)
            
            # Stop if it hit the ground
            if p.z <= 0:
                p.z = 0
                self.dead_particles.append(p)
                continue
                
            # Ionization energy loss
            if p.pid in ['e+', 'e-', 'proton', 'pi_charged']:
                x_gcm2 = dist * self.atmospheric_density(p.z) * 100.0
                dE = 0.0021 * x_gcm2
                p.energy -= dE
                if p.energy < self.critical_energy:
                    self.dead_particles.append(p)
                    continue
                
            # Interactions
            if p.pid == 'gamma':
                # Pair production: gamma -> e+ + e-
                u = np.random.uniform(0.1, 0.9)
                e_plus_energy = u * p.energy
                e_minus_energy = (1 - u) * p.energy
                
                pt = 0.0005 
                phi = np.random.uniform(0, 2*np.pi)
                px1, py1, pz1 = self.normalize_direction(p.px + pt*np.cos(phi), p.py + pt*np.sin(phi), p.pz)
                px2, py2, pz2 = self.normalize_direction(p.px - pt*np.cos(phi), p.py - pt*np.sin(phi), p.pz)
                e1 = Particle('e+', e_plus_energy, p.x, p.y, p.z, px1, py1, pz1, p.generation+1)
                e2 = Particle('e-', e_minus_energy, p.x, p.y, p.z, px2, py2, pz2, p.generation+1)
                new_particles.extend([e1, e2])
                self.dead_particles.append(p)
                
            elif p.pid in ['e+', 'e-']:
                # Bremsstrahlung: e -> e + gamma
                u = (self.critical_energy / p.energy) ** np.random.uniform()
                u = max(self.critical_energy / p.energy, u)
                gamma_energy = u * p.energy
                e_energy = (1 - u) * p.energy
                
                pt = 0.0005
                phi = np.random.uniform(0, 2*np.pi)
                px1, py1, pz1 = self.normalize_direction(p.px - pt*np.cos(phi), p.py - pt*np.sin(phi), p.pz)
                px2, py2, pz2 = self.normalize_direction(p.px + pt*np.cos(phi), p.py + pt*np.sin(phi), p.pz)
                e_out = Particle(p.pid, e_energy, p.x, p.y, p.z, px1, py1, pz1, p.generation+1)
                g_out = Particle('gamma', gamma_energy, p.x, p.y, p.z, px2, py2, pz2, p.generation+1)
                new_particles.extend([e_out, g_out])
                self.dead_particles.append(p)
                
            elif p.pid in ['proton', 'pi_charged']:
                # Hadronic interaction -> 5 pions (2 pi0, 3 pi_charged) for toy model
                # Larger transverse momentum
                pt = 0.5 / p.energy 
                
                # Energy split roughly equally
                e_split = p.energy / 5.0
                
                phis = np.random.uniform(0, 2*np.pi, 5)
                
                children = []
                for i in range(5):
                    px_c, py_c, pz_c = self.normalize_direction(p.px + pt*np.cos(phis[i]), p.py + pt*np.sin(phis[i]), p.pz)
                    pid = 'pi0' if i < 2 else 'pi_charged'
                    children.append(Particle(pid, e_split, p.x, p.y, p.z, px_c, py_c, pz_c, p.generation+1))
                
                new_particles.extend(children)
                self.dead_particles.append(p)
                
            elif p.pid == 'pi0':
                # pi0 -> 2 gamma
                pt = 0.135 / p.energy # pi0 mass
                phi = np.random.uniform(0, 2*np.pi)
                px1, py1, pz1 = self.normalize_direction(p.px + pt*np.cos(phi), p.py + pt*np.sin(phi), p.pz)
                px2, py2, pz2 = self.normalize_direction(p.px - pt*np.cos(phi), p.py - pt*np.sin(phi), p.pz)
                g1 = Particle('gamma', p.energy/2, p.x, p.y, p.z, px1, py1, pz1, p.generation+1)
                g2 = Particle('gamma', p.energy/2, p.x, p.y, p.z, px2, py2, pz2, p.generation+1)
                new_particles.extend([g1, g2])
                self.dead_particles.append(p)
                
        self.active_particles = new_particles
        
    def run(self, max_generations=12, verbose=True):
        import time
        t0 = time.time()
        for gen in range(max_generations):
            if not self.active_particles:
                break
            n_active = len(self.active_particles)
            self.step()
            if verbose:
                elapsed = time.time() - t0
                print(f"      Gen {gen:2d}: {n_active:6d} active particles ({elapsed:.1f}s)")
        
        # Collect all tracks
        self.all_particles = self.dead_particles + self.active_particles
        n_particles = len(self.all_particles)
        if verbose:
            print(f"      Cascade complete: {n_particles} total particles ({time.time()-t0:.1f}s)")
            print(f"      Computing Cherenkov pool on GPU...")
        t1 = time.time()
        self.calculate_cherenkov_pool()
        if verbose:
            n_phot = len(self.cherenkov_photons.get('x_ground', []))
            print(f"      Cherenkov pool: {n_phot:,} photons ({time.time()-t1:.1f}s)")

    def calculate_cherenkov_pool(self):
        """
        Calculates the Cherenkov photons emitted by electrons and positrons
        and traces them to the ground (z=0).
        
        Delegates the heavy computation to sim.backend, which uses
        PyTorch CUDA when available, falling back to NumPy on CPU.
        """
        from sim.backend import compute_cherenkov_pool_gpu

        # Vectorized segment collection: extract all e+/e- tracks at once
        # Each particle has a track [(x,y,z,px,py,pz), ...] — we need pairs of consecutive points
        track_arrays = []
        energy_counts = []
        
        for p in self.all_particles:
            if p.pid not in ['e+', 'e-']:
                continue
            n_seg = len(p.track) - 1
            if n_seg <= 0:
                continue
            # Convert track to numpy array: shape (n_points, 6)
            arr = np.array(p.track)
            track_arrays.append(arr)
            energy_counts.append((p.energy, n_seg))
        
        if not track_arrays:
            for key in self.cherenkov_photons:
                self.cherenkov_photons[key] = np.array([])
            return

        # Build segment arrays using numpy slicing (no inner for-loop)
        all_starts = np.concatenate([arr[:-1] for arr in track_arrays])  # (N_segs, 6)
        all_ends = np.concatenate([arr[1:] for arr in track_arrays])     # (N_segs, 6)
        seg_energy = np.concatenate([np.full(n, e) for e, n in energy_counts])
        
        seg_x1 = all_starts[:, 0]
        seg_y1 = all_starts[:, 1]
        seg_z1 = all_starts[:, 2]
        seg_x2 = all_ends[:, 0]
        seg_y2 = all_ends[:, 1]
        seg_z2 = all_ends[:, 2]
        
        seg_px = all_starts[:, 3]
        seg_py = all_starts[:, 4]
        seg_pz = all_starts[:, 5]

        # Dispatch to GPU/CPU backend
        self.cherenkov_photons = compute_cherenkov_pool_gpu(
            seg_x1, seg_y1, seg_z1,
            seg_x2, seg_y2, seg_z2,
            seg_px, seg_py, seg_pz,
            seg_energy, self.photon_yield_factor
        )

    def get_cherenkov_dataframe(self):
        """Export Cherenkov ground photons for visualization (downsampled if huge)."""
        if len(self.cherenkov_photons['x_ground']) == 0:
            return pd.DataFrame(columns=['x', 'y'])
        
        # If there are millions of photons, downsample for Plotly visualization
        n_tot = len(self.cherenkov_photons['x_ground'])
        max_plot = 10000
        if n_tot > max_plot:
            indices = np.random.choice(n_tot, max_plot, replace=False)
        else:
            indices = np.arange(n_tot)
            
        data = {
            'x': self.cherenkov_photons['x_ground'][indices],
            'y': self.cherenkov_photons['y_ground'][indices]
        }
        return pd.DataFrame(data)

    def get_tracks_dataframe(self):
        """Export tracks for visualization."""
        data = []
        for i, p in enumerate(self.all_particles):
            for step_idx, (x, y, z, px_t, py_t, pz_t) in enumerate(p.track):
                data.append({
                    'particle_id': i,
                    'pid': p.pid,
                    'energy': p.energy,
                    'generation': p.generation,
                    'x': x,
                    'y': y,
                    'z': z
                })
        return pd.DataFrame(data)

if __name__ == "__main__":
    # Quick test
    sim = ShowerSimulation('gamma', energy=100.0)
    sim.run(max_generations=5)
    df = sim.get_tracks_dataframe()
    print(f"Generated {len(sim.all_particles)} particles, {len(df)} track points.")

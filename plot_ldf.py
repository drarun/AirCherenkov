import numpy as np
import matplotlib.pyplot as plt
import time
from sim.shower import ShowerSimulation

def compute_ldf(xg, yg):
    r = np.sqrt(xg**2 + yg**2)
    bins = np.linspace(0, 1500, 100)
    counts, edges = np.histogram(r, bins=bins)
    centers = (edges[:-1] + edges[1:]) / 2
    area = np.pi * (edges[1:]**2 - edges[:-1]**2)
    density = np.zeros_like(counts, dtype=float)
    mask = area > 0
    density[mask] = counts[mask] / area[mask]
    return centers, density

print("Simulating 5 TeV Gamma shower...")
gamma_sim = ShowerSimulation('gamma', energy=5000.0)
gamma_sim.run(max_generations=18, verbose=False)
gamma_sim.get_cherenkov_dataframe() # Trigger computation
gamma_xg = np.array(gamma_sim.cherenkov_photons['x_ground'])
gamma_yg = np.array(gamma_sim.cherenkov_photons['y_ground'])
r_g, dens_g = compute_ldf(gamma_xg, gamma_yg)
print(f"  -> Gamma pool size: {len(gamma_xg)} photons")

print("Simulating 5 TeV Proton shower...")
proton_sim = ShowerSimulation('proton', energy=5000.0)
proton_sim.run(max_generations=18, verbose=False)
proton_sim.get_cherenkov_dataframe() # Trigger computation
proton_xg = np.array(proton_sim.cherenkov_photons['x_ground'])
proton_yg = np.array(proton_sim.cherenkov_photons['y_ground'])
r_p, dens_p = compute_ldf(proton_xg, proton_yg)
print(f"  -> Proton pool size: {len(proton_xg)} photons")

plt.figure(figsize=(10, 6))
plt.plot(r_g, dens_g, label='Gamma-ray (EM)', color='blue', lw=2)
plt.plot(r_p, dens_p, label='Proton (Hadronic)', color='red', lw=2)
plt.xlabel('Radius from Core (m)', fontsize=12)
plt.ylabel('Photon Density ($m^{-2}$)', fontsize=12)
plt.title('Cherenkov Lateral Distribution Function (LDF)', fontsize=14)
plt.yscale('log')
plt.grid(True, alpha=0.3)
plt.legend(fontsize=12)
plt.savefig('ldf_comparison.png', dpi=150, bbox_inches='tight')
print('Generated ldf_comparison.png!')

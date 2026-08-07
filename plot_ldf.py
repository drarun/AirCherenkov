from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / 'src'
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import numpy as np
import matplotlib.pyplot as plt
from sim.shower import ShowerSimulation

def compute_ldf(xg, yg, weights=None):
    """Return annular photon density, accounting for weighted packets."""
    r = np.sqrt(xg**2 + yg**2)
    bins = np.linspace(0, 1500, 100)
    if weights is None:
        weights = np.ones_like(r, dtype=np.float64)
    counts, edges = np.histogram(r, bins=bins, weights=weights)
    centers = (edges[:-1] + edges[1:]) / 2
    area = np.pi * (edges[1:]**2 - edges[:-1]**2)
    density = np.zeros_like(counts, dtype=float)
    mask = area > 0
    density[mask] = counts[mask] / area[mask]
    return centers, density


def _simulate(primary, energy_gev, seed):
    simulation = ShowerSimulation(
        primary_type=primary,
        energy=energy_gev,
        z_start=20000.0,
        seed=seed,
    )
    simulation.run(max_generations=18, verbose=False)
    packets = simulation.cherenkov_photons_by_event[0]
    x_ground = np.asarray(packets['x_ground'])
    y_ground = np.asarray(packets['y_ground'])
    weights = np.asarray(packets.get('weight', np.ones(len(x_ground))))
    radius, density = compute_ldf(x_ground, y_ground, weights)
    print(
        f"  -> {primary.title()} pool: {len(x_ground):,} packets "
        f"representing {weights.sum():,.0f} photons"
    )
    return radius, density


def main(output_path=PROJECT_ROOT / 'ldf_comparison.png'):
    print("Simulating 5 TeV gamma and proton showers...")
    r_g, dens_g = _simulate('gamma', 5000.0, seed=1)
    r_p, dens_p = _simulate('proton', 5000.0, seed=2)

    plt.figure(figsize=(10, 6))
    plt.plot(r_g, dens_g, label='Gamma-ray (EM)', color='blue', lw=2)
    plt.plot(r_p, dens_p, label='Proton (Hadronic)', color='red', lw=2)
    plt.xlabel('Radius from Core (m)', fontsize=12)
    plt.ylabel('Photon Density ($m^{-2}$)', fontsize=12)
    plt.title('Cherenkov Lateral Distribution Function (LDF)', fontsize=14)
    plt.yscale('log')
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f'Generated {output_path}!')


if __name__ == '__main__':
    main()

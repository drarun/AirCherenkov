"""
Instrument Response Functions (IRFs) for the AirCherenkov pipeline.

Builds two key IRFs from Monte Carlo gamma-ray simulations:
  1. Effective Area  A_eff(E_true)
  2. Energy Migration Matrix  M(E_reco | E_true)

These are the bridge between the astrophysical source spectrum and the
counts measured by our GNN reconstruction chain.  They are consumed by
the forward-folding spectral fitter in `spectral_fit.py`.

Usage
-----
    python -m analysis.irf --test-data data/crab_test --model data/spatiotemporal_gnn.pt
"""

import argparse
import json
import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import torch



# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
# These MUST match the values used in `generate_training_data.py`.
DEFAULT_IMPACT_RADIUS_M = 350.0         # metres
DEFAULT_E_MIN_GEV = 80.0
DEFAULT_E_MAX_GEV = 30_000.0
DEFAULT_GEN_INDEX = 2.0                 # E^{-alpha} generation spectrum
N_ENERGY_BINS = 20                      # bins in log10(E / GeV)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _thrown_spectrum_weight(e_min, e_max, alpha, n_thrown):
    """
    Analytically compute the number of events thrown per log-energy bin.

    For a differential generation spectrum  dN/dE ∝ E^{-alpha}  sampled
    uniformly from a CDF between *e_min* and *e_max*, the expected number
    of events in a bin [E_lo, E_hi] is:

        n_bin = n_thrown * (E_hi^{1-α} - E_lo^{1-α}) / (E_max^{1-α} - E_min^{1-α})

    Returns an array of shape ``(len(bin_edges) - 1,)``.
    """
    def _primitive(E):
        if np.isclose(alpha, 1.0):
            return np.log(E)
        return E ** (1.0 - alpha) / (1.0 - alpha)

    bin_edges = np.logspace(np.log10(e_min), np.log10(e_max), N_ENERGY_BINS + 1)
    normalisation = _primitive(e_max) - _primitive(e_min)
    counts = np.array([
        n_thrown * (_primitive(bin_edges[i + 1]) - _primitive(bin_edges[i])) / normalisation
        for i in range(N_ENERGY_BINS)
    ])
    return bin_edges, counts


def _gamma_classification_cut(pred_class, threshold=0.5):
    """Apply a simple gamma/hadron classification cut."""
    return pred_class >= threshold


# ---------------------------------------------------------------------------
# Core IRF builders
# ---------------------------------------------------------------------------

def build_effective_area(
    true_energies_gev,
    passed_cut,
    n_thrown_total,
    impact_radius_m=DEFAULT_IMPACT_RADIUS_M,
    e_min=DEFAULT_E_MIN_GEV,
    e_max=DEFAULT_E_MAX_GEV,
    gen_index=DEFAULT_GEN_INDEX,
):
    """
    Compute the differential effective area A_eff(E_true).

    Parameters
    ----------
    true_energies_gev : array
        True energies (GeV) of gamma events that triggered AND passed the
        GNN classification cut.
    passed_cut : array of bool
        Which of the triggered gammas passed the classification cut.
    n_thrown_total : int
        Total number of gamma showers originally thrown (before trigger).
    impact_radius_m : float
        Radius of the thrown area (metres).
    e_min, e_max : float
        Energy boundaries of the thrown spectrum (GeV).
    gen_index : float
        Spectral index of the thrown E^{-alpha} generation spectrum.

    Returns
    -------
    dict with keys:
        bin_edges     – (N+1,) array, energy bin edges in GeV
        bin_centres   – (N,) array, geometric bin centres in GeV
        aeff          – (N,) array, effective area in m² per bin
        aeff_err      – (N,) array, Poisson error on A_eff
        n_selected    – (N,) int array, selected counts per bin
        n_simulated   – (N,) float array, analytically expected thrown counts
    """
    thrown_area = np.pi * impact_radius_m ** 2  # m²
    bin_edges, n_simulated = _thrown_spectrum_weight(e_min, e_max, gen_index, n_thrown_total)
    bin_centres = np.sqrt(bin_edges[:-1] * bin_edges[1:])  # geometric mean

    selected_energies = true_energies_gev[passed_cut]
    n_selected, _ = np.histogram(selected_energies, bins=bin_edges)

    # Avoid division by zero in empty bins
    safe_denom = np.where(n_simulated > 0, n_simulated, 1.0)
    efficiency = n_selected / safe_denom
    aeff = thrown_area * efficiency

    # Binomial/Poisson error: σ(A) = A_sim * sqrt(k) / N_sim  (Feldman-Cousins
    # would be better for low counts, but this is standard practice)
    aeff_err = thrown_area * np.sqrt(n_selected.astype(float)) / safe_denom

    return {
        'bin_edges': bin_edges,
        'bin_centres': bin_centres,
        'aeff': aeff,
        'aeff_err': aeff_err,
        'n_selected': n_selected,
        'n_simulated': n_simulated,
        'thrown_area_m2': thrown_area,
    }


def build_energy_migration_matrix(
    true_energies_gev,
    reco_energies_gev,
    e_min=DEFAULT_E_MIN_GEV,
    e_max=DEFAULT_E_MAX_GEV,
):
    """
    Build the energy migration matrix M(E_reco | E_true).

    The matrix is row-normalised so that each row (true-energy slice) sums
    to 1.0, giving the conditional probability P(E_reco | E_true).

    Parameters
    ----------
    true_energies_gev, reco_energies_gev : arrays
        Paired true and reconstructed energies of selected gamma events.

    Returns
    -------
    dict with keys:
        matrix        – (N, N) normalised migration matrix
        raw_counts    – (N, N) unnormalised 2D histogram
        bin_edges     – (N+1,) array, energy bin edges in GeV
        bin_centres   – (N,) array, geometric bin centres in GeV
    """
    bin_edges = np.logspace(np.log10(e_min), np.log10(e_max), N_ENERGY_BINS + 1)
    bin_centres = np.sqrt(bin_edges[:-1] * bin_edges[1:])

    raw_counts, _, _ = np.histogram2d(
        true_energies_gev, reco_energies_gev, bins=[bin_edges, bin_edges]
    )

    # Row-normalise (each true-energy row sums to 1)
    row_sums = raw_counts.sum(axis=1, keepdims=True)
    safe_sums = np.where(row_sums > 0, row_sums, 1.0)
    matrix = raw_counts / safe_sums

    return {
        'matrix': matrix,
        'raw_counts': raw_counts,
        'bin_edges': bin_edges,
        'bin_centres': bin_centres,
    }


# ---------------------------------------------------------------------------
# GNN inference helper
# ---------------------------------------------------------------------------

def run_gnn_inference(test_data_root, model_path, batch_size=128):
    """
    Run the trained SpatiotemporalGNN on the test dataset and return
    arrays of true/reconstructed energies and classification scores.

    Returns
    -------
    dict with keys:
        true_energy_gev  – (N,) array
        reco_energy_gev  – (N,) array
        true_class       – (N,) array  (1 = gamma, 0 = hadron)
        pred_class_score – (N,) array  (sigmoid output, 0–1)
    """
    from sim.camera import Camera
    from recon.gnn import SpatiotemporalGNN
    from analysis.dataset import CherenkovDataset
    from torch_geometric.loader import DataLoader

    cam = Camera(n_rings=12)
    edge_index = cam.edge_index

    class AddEdgeIndex:
        def __init__(self, edge_idx):
            self.edge_idx = edge_idx
        def __call__(self, data):
            data.edge_index = self.edge_idx
            return data

    dataset = CherenkovDataset(root=test_data_root, pre_transform=AddEdgeIndex(edge_index))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SpatiotemporalGNN().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()

    true_e, pred_e, true_c, pred_c = [], [], [], []

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            c_out, e_out = model(batch.x, batch.edge_index, batch.batch)
            true_e.extend(batch.y_energy.cpu().numpy())
            pred_e.extend(e_out.view(-1).cpu().numpy())
            true_c.extend(batch.y_class.cpu().numpy())
            pred_c.extend(torch.sigmoid(c_out).view(-1).cpu().numpy())

    true_e = np.array(true_e)  # log10(E / GeV)
    pred_e = np.array(pred_e)

    return {
        'true_energy_gev': 10.0 ** true_e,
        'reco_energy_gev': 10.0 ** pred_e,
        'true_class': np.array(true_c),
        'pred_class_score': np.array(pred_c),
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_irfs(aeff_result, migration_result, output_path='data/irfs.png'):
    """Generate publication-quality IRF diagnostic plots."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # --- Effective Area ---
    ax = axes[0]
    bc = aeff_result['bin_centres'] / 1000.0  # GeV -> TeV
    ax.errorbar(bc, aeff_result['aeff'], yerr=aeff_result['aeff_err'],
                fmt='o-', color='dodgerblue', capsize=4, markersize=5, lw=2,
                label='AirCherenkov GNN')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(r'True Energy [TeV]', fontsize=13)
    ax.set_ylabel(r'Effective Area [m$^2$]', fontsize=13)
    ax.set_title('Effective Area vs True Energy', fontsize=14)
    ax.axhline(y=1e5, color='gray', ls='--', alpha=0.4, label=r'VERITAS benchmark ($10^5$ m$^2$)')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, which='both')
    ax.set_ylim(bottom=1e2)

    # --- Migration Matrix ---
    ax = axes[1]
    be_tev = migration_result['bin_edges'] / 1000.0
    im = ax.pcolormesh(be_tev, be_tev, migration_result['matrix'],
                       cmap='inferno', norm=LogNorm(vmin=1e-3, vmax=1.0))
    ax.plot([be_tev[0], be_tev[-1]], [be_tev[0], be_tev[-1]], 'w--', lw=1.5, alpha=0.7)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(r'Reconstructed Energy [TeV]', fontsize=13)
    ax.set_ylabel(r'True Energy [TeV]', fontsize=13)
    ax.set_title(r'Energy Migration Matrix $P(E_{\rm reco}\,|\,E_{\rm true})$', fontsize=14)
    fig.colorbar(im, ax=ax, label='Probability')

    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    print(f"Saved IRF plots to {output_path}")
    return fig


# ---------------------------------------------------------------------------
# Serialisation
# ---------------------------------------------------------------------------

def save_irfs(aeff_result, migration_result, output_dir='data'):
    """Save IRFs to disk as .npz for later use by spectral_fit.py."""
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, 'irfs.npz')
    np.savez(
        path,
        # Effective area
        aeff_bin_edges=aeff_result['bin_edges'],
        aeff_bin_centres=aeff_result['bin_centres'],
        aeff=aeff_result['aeff'],
        aeff_err=aeff_result['aeff_err'],
        n_selected=aeff_result['n_selected'],
        n_simulated=aeff_result['n_simulated'],
        thrown_area_m2=aeff_result['thrown_area_m2'],
        # Migration matrix
        migration_matrix=migration_result['matrix'],
        migration_raw_counts=migration_result['raw_counts'],
        migration_bin_edges=migration_result['bin_edges'],
        migration_bin_centres=migration_result['bin_centres'],
    )
    print(f"Saved IRFs to {path}")
    return path


def load_irfs(path='data/irfs.npz'):
    """Load previously saved IRFs."""
    data = np.load(path)
    aeff_result = {
        'bin_edges': data['aeff_bin_edges'],
        'bin_centres': data['aeff_bin_centres'],
        'aeff': data['aeff'],
        'aeff_err': data['aeff_err'],
        'n_selected': data['n_selected'],
        'n_simulated': data['n_simulated'],
        'thrown_area_m2': float(data['thrown_area_m2']),
    }
    migration_result = {
        'matrix': data['migration_matrix'],
        'raw_counts': data['migration_raw_counts'],
        'bin_edges': data['migration_bin_edges'],
        'bin_centres': data['migration_bin_centres'],
    }
    return aeff_result, migration_result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--test-data', default='data/crab_test',
                        help='Root directory of the processed test dataset')
    parser.add_argument('--model', default='data/spatiotemporal_gnn.pt',
                        help='Path to trained SpatiotemporalGNN weights')
    parser.add_argument('--n-thrown-gammas', type=int, default=10_000,
                        help='Total number of gamma showers originally thrown')
    parser.add_argument('--impact-radius', type=float, default=DEFAULT_IMPACT_RADIUS_M)
    parser.add_argument('--e-min', type=float, default=DEFAULT_E_MIN_GEV)
    parser.add_argument('--e-max', type=float, default=DEFAULT_E_MAX_GEV)
    parser.add_argument('--gen-index', type=float, default=DEFAULT_GEN_INDEX)
    parser.add_argument('--gamma-cut', type=float, default=0.5,
                        help='Classification threshold for gamma selection')
    parser.add_argument('--output-dir', default='data')
    args = parser.parse_args()

    # 1. Run GNN inference
    print("Running GNN inference on test data...")
    results = run_gnn_inference(args.test_data, args.model)

    # 2. Apply gamma classification cut
    gamma_mask = results['true_class'] == 1.0
    gamma_true_e = results['true_energy_gev'][gamma_mask]
    gamma_reco_e = results['reco_energy_gev'][gamma_mask]
    gamma_score = results['pred_class_score'][gamma_mask]
    passed_cut = _gamma_classification_cut(gamma_score, threshold=args.gamma_cut)

    print(f"  Gamma events in test set:     {gamma_mask.sum()}")
    print(f"  Passed classification cut:    {passed_cut.sum()}")

    # 3. Build IRFs
    aeff = build_effective_area(
        gamma_true_e, passed_cut, args.n_thrown_gammas,
        impact_radius_m=args.impact_radius,
        e_min=args.e_min, e_max=args.e_max, gen_index=args.gen_index,
    )
    migration = build_energy_migration_matrix(
        gamma_true_e[passed_cut], gamma_reco_e[passed_cut],
        e_min=args.e_min, e_max=args.e_max,
    )

    # 4. Save & plot
    save_irfs(aeff, migration, output_dir=args.output_dir)
    plot_irfs(aeff, migration, output_path=os.path.join(args.output_dir, 'irfs.png'))

    # 5. Print summary
    peak_aeff = np.max(aeff['aeff'])
    peak_e = aeff['bin_centres'][np.argmax(aeff['aeff'])] / 1000.0
    print(f"\n  Peak effective area: {peak_aeff:.0f} m² at {peak_e:.2f} TeV")


if __name__ == '__main__':
    main()

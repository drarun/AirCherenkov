"""
Forward-folding Poisson-likelihood spectral reconstruction.

Given the IRFs produced by `irf.py` (effective area and energy migration
matrix), this module fits a parameterised spectral model to observed
event counts by minimising the negative Poisson log-likelihood.

The fit recovers the differential flux normalisation f₀ and spectral
index α of a power-law model:

    dN/dE = f₀ (E / 1 TeV)^{-α}

Usage
-----
    python -m analysis.spectral_fit --irfs data/irfs.npz --test-data data/crab_test \\
           --model data/spatiotemporal_gnn.pt --obs-time 3600

The observation time (--obs-time) is in seconds.  For our MC-based
closure test we treat the test dataset itself as the "observation".
"""

import argparse
import os

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

from analysis.irf import (
    load_irfs,
    run_gnn_inference,
    _gamma_classification_cut,
    N_ENERGY_BINS,
)


# ---------------------------------------------------------------------------
# Physical constants and reference spectra
# ---------------------------------------------------------------------------

# VERITAS-measured Crab Nebula power-law parameters (Meagher+ 2015)
CRAB_F0_VERITAS = 3.23e-7   # m^{-2} s^{-1} TeV^{-1}
CRAB_INDEX_VERITAS = 2.49

# MAGIC Crab reference (Aleksic+ 2015)
CRAB_F0_MAGIC = 3.23e-7
CRAB_INDEX_MAGIC = 2.47


# ---------------------------------------------------------------------------
# Spectral model
# ---------------------------------------------------------------------------

def power_law_flux(energy_tev, f0, alpha):
    """
    Differential flux dN/dE in units of [m^{-2} s^{-1} TeV^{-1}].

        dN/dE = f₀ * (E / 1 TeV)^{-α}
    """
    return f0 * energy_tev ** (-alpha)


# ---------------------------------------------------------------------------
# Forward-folding prediction
# ---------------------------------------------------------------------------

def predict_counts(f0, alpha, aeff, migration, obs_time_s):
    """
    Predict the number of reconstructed gamma counts in each E_reco bin
    by convolving a spectral model with the IRFs.

    N_pred(j) = T_obs * Σ_i [ dN/dE(E_i) * A_eff(E_i) * M(j|i) * ΔE_i ]

    Parameters
    ----------
    f0 : float
        Flux normalisation at 1 TeV  [m^{-2} s^{-1} TeV^{-1}].
    alpha : float
        Spectral index.
    aeff : dict
        Effective area result from `irf.build_effective_area`.
    migration : dict
        Energy migration result from `irf.build_energy_migration_matrix`.
    obs_time_s : float
        Observation time in seconds.

    Returns
    -------
    n_pred : (N,) array
        Predicted counts in each reconstructed energy bin.
    """
    bin_edges_tev = aeff['bin_edges'] / 1000.0   # GeV -> TeV
    bin_widths_tev = np.diff(bin_edges_tev)       # ΔE per bin
    centres_tev = aeff['bin_centres'] / 1000.0

    # Differential flux at each true-energy bin centre
    flux = power_law_flux(centres_tev, f0, alpha)    # m^{-2} s^{-1} TeV^{-1}

    # Expected detected events in true energy space (before migration)
    n_true = obs_time_s * flux * aeff['aeff'] * bin_widths_tev   # counts

    # Fold through the migration matrix:  n_reco_j = Σ_i M(j|i) * n_true_i
    M = migration['matrix']   # shape (N_true, N_reco), row-normalised
    n_pred = M.T @ n_true     # shape (N_reco,)

    return n_pred


# ---------------------------------------------------------------------------
# Likelihood
# ---------------------------------------------------------------------------

def neg_log_likelihood(params, n_obs, aeff, migration, obs_time_s, background=None):
    """
    Negative Poisson log-likelihood.

    -ln L = Σ_j [ μ_j - n_j ln(μ_j) ]     (ignoring constant ln(n_j!) term)

    where μ_j = N_pred(j) + B(j).
    """
    f0, alpha = params
    if f0 <= 0 or alpha <= 0:
        return 1e20

    n_pred = predict_counts(f0, alpha, aeff, migration, obs_time_s)

    if background is not None:
        n_pred = n_pred + background

    # Regularise: floor at a tiny value to avoid log(0)
    mu = np.clip(n_pred, 1e-10, None)

    # Poisson negative log-likelihood (up to a constant)
    nll = np.sum(mu - n_obs * np.log(mu))
    return nll


# ---------------------------------------------------------------------------
# Fitter
# ---------------------------------------------------------------------------

def fit_spectrum(n_obs, aeff, migration, obs_time_s, background=None,
                 f0_init=3.0e-7, alpha_init=2.5):
    """
    Fit the spectral parameters (f₀, α) by minimising the Poisson NLL.

    Returns
    -------
    dict with keys:
        f0, alpha            – best-fit parameters
        f0_err, alpha_err    – approximate 1σ errors from Hessian
        nll                  – negative log-likelihood at best fit
        success              – bool, whether the optimiser converged
        n_obs, n_pred        – observed and predicted count arrays
    """
    result = minimize(
        neg_log_likelihood,
        x0=[f0_init, alpha_init],
        args=(n_obs, aeff, migration, obs_time_s, background),
        method='Nelder-Mead',
        options={'xatol': 1e-12, 'fatol': 1e-8, 'maxiter': 10000},
    )

    f0_fit, alpha_fit = result.x

    # Estimate errors from numerical Hessian (finite differences)
    eps_f0 = f0_fit * 1e-4
    eps_alpha = 1e-4
    h = np.array([eps_f0, eps_alpha])

    hessian = np.zeros((2, 2))
    nll_centre = neg_log_likelihood(result.x, n_obs, aeff, migration, obs_time_s, background)

    for i in range(2):
        for j in range(2):
            x_pp = result.x.copy(); x_pp[i] += h[i]; x_pp[j] += h[j]
            x_pm = result.x.copy(); x_pm[i] += h[i]; x_pm[j] -= h[j]
            x_mp = result.x.copy(); x_mp[i] -= h[i]; x_mp[j] += h[j]
            x_mm = result.x.copy(); x_mm[i] -= h[i]; x_mm[j] -= h[j]

            hessian[i, j] = (
                neg_log_likelihood(x_pp, n_obs, aeff, migration, obs_time_s, background)
                - neg_log_likelihood(x_pm, n_obs, aeff, migration, obs_time_s, background)
                - neg_log_likelihood(x_mp, n_obs, aeff, migration, obs_time_s, background)
                + neg_log_likelihood(x_mm, n_obs, aeff, migration, obs_time_s, background)
            ) / (4 * h[i] * h[j])

    try:
        covariance = np.linalg.inv(hessian)
        errors = np.sqrt(np.abs(np.diag(covariance)))
    except np.linalg.LinAlgError:
        errors = np.array([np.nan, np.nan])

    n_pred = predict_counts(f0_fit, alpha_fit, aeff, migration, obs_time_s)
    if background is not None:
        n_pred = n_pred + background

    return {
        'f0': f0_fit,
        'alpha': alpha_fit,
        'f0_err': errors[0],
        'alpha_err': errors[1],
        'nll': result.fun,
        'success': result.success,
        'n_obs': n_obs,
        'n_pred': n_pred,
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_spectrum(fit_result, aeff, output_path='data/spectral_fit.png'):
    """
    Publication-quality plot: reconstructed SED with best-fit model
    overlaid on the VERITAS Crab reference.
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    bin_centres_tev = aeff['bin_centres'] / 1000.0
    bin_edges_tev = aeff['bin_edges'] / 1000.0

    # --- Left panel: Forward-folded counts ---
    ax = axes[0]
    n_obs = fit_result['n_obs']
    n_pred = fit_result['n_pred']
    obs_err = np.sqrt(np.clip(n_obs, 1, None))

    ax.errorbar(bin_centres_tev, n_obs, yerr=obs_err,
                fmt='ko', capsize=4, markersize=5, label='Observed (MC test data)')
    ax.step(bin_edges_tev, np.append(n_pred, n_pred[-1]),
            where='post', color='crimson', lw=2, label='Best-fit model')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(r'Reconstructed Energy [TeV]', fontsize=13)
    ax.set_ylabel(r'Counts', fontsize=13)
    ax.set_title('Forward-Folded Spectral Fit', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, which='both')

    # --- Right panel: Differential flux (SED butterfly) ---
    ax = axes[1]
    e_plot = np.logspace(np.log10(bin_centres_tev[0]),
                         np.log10(bin_centres_tev[-1]), 200)

    # Best fit
    flux_fit = power_law_flux(e_plot, fit_result['f0'], fit_result['alpha'])
    ax.plot(e_plot, e_plot ** 2 * flux_fit, 'r-', lw=2.5,
            label=(rf"Best fit: $f_0 = {fit_result['f0']:.2e}$, "
                   rf"$\alpha = {fit_result['alpha']:.2f} \pm {fit_result['alpha_err']:.2f}$"))

    # VERITAS reference
    flux_veritas = power_law_flux(e_plot, CRAB_F0_VERITAS, CRAB_INDEX_VERITAS)
    ax.plot(e_plot, e_plot ** 2 * flux_veritas, 'b--', lw=1.5, alpha=0.7,
            label=rf"VERITAS Crab ($\alpha = {CRAB_INDEX_VERITAS}$)")

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(r'Energy [TeV]', fontsize=13)
    ax.set_ylabel(r'$E^2\,dN/dE$ [TeV m$^{-2}$ s$^{-1}$]', fontsize=13)
    ax.set_title('Spectral Energy Distribution', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, which='both')

    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    print(f"Saved spectral fit plot to {output_path}")
    return fig


# ---------------------------------------------------------------------------
# MC closure test helpers
# ---------------------------------------------------------------------------

def build_observed_counts_from_mc(test_data_root, model_path, gamma_cut=0.5,
                                  e_min=80.0, e_max=30_000.0):
    """
    Construct the "observed" reconstructed-energy histogram from the
    MC test dataset.  In a real analysis this would come from actual
    telescope data; for our closure test we use MC truth labels to
    select only true gammas that pass the GNN cut.
    """
    results = run_gnn_inference(test_data_root, model_path)
    gamma_mask = results['true_class'] == 1.0
    gamma_reco_e = results['reco_energy_gev'][gamma_mask]
    gamma_score = results['pred_class_score'][gamma_mask]
    passed = _gamma_classification_cut(gamma_score, threshold=gamma_cut)

    bin_edges = np.logspace(np.log10(e_min), np.log10(e_max), N_ENERGY_BINS + 1)
    n_obs, _ = np.histogram(gamma_reco_e[passed], bins=bin_edges)
    return n_obs


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--irfs', default='data/irfs.npz',
                        help='Path to saved IRFs from irf.py')
    parser.add_argument('--test-data', default='data/crab_test',
                        help='Root directory of the processed test dataset')
    parser.add_argument('--model', default='data/spatiotemporal_gnn.pt',
                        help='Path to trained SpatiotemporalGNN weights')
    parser.add_argument('--obs-time', type=float, default=3600.0,
                        help='Observation time in seconds (default: 1 hour)')
    parser.add_argument('--gamma-cut', type=float, default=0.5)
    parser.add_argument('--output-dir', default='data')
    args = parser.parse_args()

    # 1. Load IRFs
    print("Loading IRFs...")
    aeff, migration = load_irfs(args.irfs)

    # 2. Build observed counts from the MC test set
    print("Building observed count histogram from MC test data...")
    n_obs = build_observed_counts_from_mc(
        args.test_data, args.model, gamma_cut=args.gamma_cut,
        e_min=float(aeff['bin_edges'][0]), e_max=float(aeff['bin_edges'][-1]),
    )
    total_obs = n_obs.sum()
    print(f"  Total observed gamma counts: {total_obs}")

    if total_obs == 0:
        print("ERROR: No gamma events survived the classification cut. "
              "Cannot fit spectrum.")
        return

    # 3. Fit
    print("Fitting spectrum via forward-folding...")
    fit = fit_spectrum(n_obs, aeff, migration, args.obs_time)

    # 4. Report
    print(f"\n{'=' * 60}")
    print(f"  Spectral Fit Results")
    print(f"{'=' * 60}")
    print(f"  Converged:             {fit['success']}")
    print(f"  f₀ (1 TeV):            {fit['f0']:.3e} ± {fit['f0_err']:.3e}  m⁻² s⁻¹ TeV⁻¹")
    print(f"  Spectral index α:      {fit['alpha']:.3f} ± {fit['alpha_err']:.3f}")
    print(f"  Neg. log-likelihood:   {fit['nll']:.2f}")
    print(f"{'=' * 60}")
    print(f"  VERITAS Crab reference:")
    print(f"    f₀ = {CRAB_F0_VERITAS:.3e},  α = {CRAB_INDEX_VERITAS}")
    print(f"{'=' * 60}")

    # 5. Plot
    plot_spectrum(fit, aeff, output_path=os.path.join(args.output_dir, 'spectral_fit.png'))


if __name__ == '__main__':
    main()

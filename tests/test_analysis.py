import os
import numpy as np
import pytest
from analysis.irf import (
    build_effective_area,
    build_energy_migration_matrix,
    save_irfs,
    load_irfs,
    _thrown_spectrum_weight
)
from analysis.spectral_fit import (
    power_law_flux,
    predict_counts,
    neg_log_likelihood,
    fit_spectrum
)

def test_thrown_spectrum_weight():
    """Verify analytical thrown spectrum weighting matches total thrown events."""
    e_min, e_max, alpha = 100.0, 10000.0, 2.0
    n_thrown = 5000
    bin_edges, counts = _thrown_spectrum_weight(e_min, e_max, alpha, n_thrown)
    
    assert len(bin_edges) == 21  # 20 bins means 21 edges
    assert len(counts) == 20
    assert np.isclose(np.sum(counts), n_thrown)

def test_effective_area_limits():
    """Verify effective area boundary cases (0% and 100% efficiency)."""
    n_thrown = 1000
    impact_radius = 350.0
    thrown_area = np.pi * (impact_radius ** 2)
    
    # Generate mock true energies
    e_min, e_max = 80.0, 30000.0
    bin_edges, _ = _thrown_spectrum_weight(e_min, e_max, 1.0, n_thrown)
    # Uniformly distribute true energies across bins (50 per bin)
    true_energies = np.repeat(np.sqrt(bin_edges[:-1] * bin_edges[1:]), 50)
    n_total_thrown = len(true_energies)  # should be 20 bins * 50 = 1000
    
    # Case 1: 0% efficiency (nothing passed classification)
    passed_cut_none = np.zeros_like(true_energies, dtype=bool)
    aeff_none = build_effective_area(
        true_energies, passed_cut_none, n_total_thrown,
        impact_radius_m=impact_radius, e_min=e_min, e_max=e_max, gen_index=1.0
    )
    assert np.all(aeff_none['aeff'] == 0.0)
    assert np.all(aeff_none['aeff_err'] == 0.0)
    
    # Case 2: 100% efficiency (everything passed classification)
    passed_cut_all = np.ones_like(true_energies, dtype=bool)
    aeff_all = build_effective_area(
        true_energies, passed_cut_all, n_total_thrown,
        impact_radius_m=impact_radius, e_min=e_min, e_max=e_max, gen_index=1.0
    )
    # Should equal the total thrown area in bins where simulated count > 0
    np.testing.assert_allclose(aeff_all['aeff'], thrown_area)

def test_migration_matrix_normalization():
    """Verify that rows (true energy slices) of the migration matrix sum to 1.0."""
    e_min, e_max = 80.0, 30000.0
    n_events = 1000
    # Generate some random correlated energies
    true_energies = 10 ** np.random.uniform(np.log10(e_min), np.log10(e_max), n_events)
    # Add some reconstruction error (bias + resolution)
    reco_energies = true_energies * np.random.normal(loc=1.0, scale=0.15, size=n_events)
    reco_energies = np.clip(reco_energies, e_min + 1, e_max - 1)
    
    migration = build_energy_migration_matrix(true_energies, reco_energies, e_min=e_min, e_max=e_max)
    matrix = migration['matrix']
    
    # Verify shape
    assert matrix.shape == (20, 20)
    
    # Verify that each row sums to 1.0 (if it has counts)
    row_sums = matrix.sum(axis=1)
    for s in row_sums:
        assert s == 0.0 or np.isclose(s, 1.0)

def test_irf_serialization_roundtrip(tmp_path):
    """Test saving and loading IRFs yields bitwise identical arrays."""
    # Create fake IRF data
    aeff = {
        'bin_edges': np.linspace(80, 30000, 21),
        'bin_centres': np.linspace(100, 29000, 20),
        'aeff': np.random.uniform(10, 10000, 20),
        'aeff_err': np.random.uniform(1, 100, 20),
        'n_selected': np.random.randint(0, 100, 20),
        'n_simulated': np.random.uniform(5, 200, 20),
        'thrown_area_m2': np.pi * 350.0**2
    }
    migration = {
        'matrix': np.random.uniform(0, 1, (20, 20)),
        'raw_counts': np.random.randint(0, 50, (20, 20)),
        'bin_edges': np.linspace(80, 30000, 21),
        'bin_centres': np.linspace(100, 29000, 20)
    }
    
    # Save
    save_path = save_irfs(aeff, migration, output_dir=str(tmp_path))
    assert os.path.exists(save_path)
    
    # Load
    loaded_aeff, loaded_migration = load_irfs(save_path)
    
    # Compare
    for key in ['bin_edges', 'bin_centres', 'aeff', 'aeff_err', 'n_selected', 'n_simulated']:
        np.testing.assert_array_equal(aeff[key], loaded_aeff[key])
    assert aeff['thrown_area_m2'] == loaded_aeff['thrown_area_m2']
    
    for key in ['matrix', 'raw_counts', 'bin_edges', 'bin_centres']:
        np.testing.assert_array_equal(migration[key], loaded_migration[key])

def test_poisson_nll_zero_bins():
    """Verify NLL does not return NaN or Inf when observed counts or predicted counts are zero."""
    n_obs = np.array([0, 10, 0, 5, 0], dtype=float)
    aeff = {
        'bin_edges': np.array([100, 200, 500, 1000, 2000, 5000], dtype=float),
        'bin_centres': np.array([150, 350, 750, 1500, 3500], dtype=float),
        'aeff': np.array([100, 1000, 10000, 10000, 10000], dtype=float)
    }
    migration = {
        'matrix': np.eye(5)  # Perfect diagonal migration
    }
    
    # If parameters yield non-zero predictions, NLL should be a finite real number
    nll = neg_log_likelihood(params=[1e-7, 2.5], n_obs=n_obs, aeff=aeff, migration=migration, obs_time_s=3600.0)
    assert np.isfinite(nll)
    
    # Verify NLL is large if parameters are negative
    nll_bad = neg_log_likelihood(params=[-1e-7, 2.5], n_obs=n_obs, aeff=aeff, migration=migration, obs_time_s=3600.0)
    assert nll_bad == 1e20

def test_spectral_fit_synthetic_recovery():
    """Perform a closure test: simulate counts with known power law and fit them back."""
    # Define simple IRFs
    bin_edges = np.logspace(np.log10(100.0), np.log10(10000.0), 11)
    bin_centres = np.sqrt(bin_edges[:-1] * bin_edges[1:])
    
    aeff = {
        'bin_edges': bin_edges,
        'bin_centres': bin_centres,
        'aeff': np.full(10, 1.0e5)  # constant 10^5 m^2 effective area
    }
    
    # Perfect reconstruction migration matrix (identity)
    migration = {
        'matrix': np.eye(10),
        'bin_edges': bin_edges,
        'bin_centres': bin_centres
    }
    
    # Target physical parameters
    true_f0 = 3.0e-7  # m^-2 s^-1 TeV^-1
    true_alpha = 2.4
    obs_time_s = 10000.0
    
    # Predict noise-free counts using the model
    n_obs = predict_counts(true_f0, true_alpha, aeff, migration, obs_time_s)
    
    # Fit back
    fit = fit_spectrum(n_obs, aeff, migration, obs_time_s, f0_init=1.0e-7, alpha_init=2.0)
    
    assert fit['success']
    # Verify recovery is very close to true parameters (noise-free case)
    assert np.isclose(fit['f0'], true_f0, rtol=1e-3)
    assert np.isclose(fit['alpha'], true_alpha, rtol=1e-3)
    assert fit['alpha_err'] > 0
    assert fit['f0_err'] > 0

import numpy as np
import pytest
from recon.energy_calibration import EnergyCalibrator
from recon.ml_stereo import StereoClassifier, StereoEnergyRegressor


def test_energy_calibrator_isotonic():
    rng = np.random.RandomState(42)
    # Synthetic compressed uncalibrated energies: true in [2, 4], pred compressed in [2.3, 3.5]
    true_e = np.linspace(2.0, 4.0, 200)
    pred_e = 2.3 + 0.6 * (true_e - 2.0) + rng.normal(0, 0.05, 200)

    calib = EnergyCalibrator(method="isotonic")
    calib.fit(pred_e, true_e)
    calibrated = calib.predict(pred_e)

    assert len(calibrated) == len(true_e)
    # Calibrated values should be expanded back towards true range
    assert np.min(calibrated) < 2.3
    assert np.max(calibrated) > 3.6
    # Relative error should improve
    assert np.mean((calibrated - true_e)**2) < np.mean((pred_e - true_e)**2)


def test_energy_calibrator_pchip():
    rng = np.random.RandomState(42)
    true_e = np.linspace(2.0, 4.0, 300)
    pred_e = 2.4 + 0.5 * (true_e - 2.0) + rng.normal(0, 0.04, 300)

    calib = EnergyCalibrator(method="pchip")
    calib.fit(pred_e, true_e)
    calibrated = calib.predict(pred_e)

    assert len(calibrated) == len(true_e)
    assert np.mean((calibrated - true_e)**2) < np.mean((pred_e - true_e)**2)


def test_energy_calibrator_debias_endpoints():
    rng = np.random.RandomState(42)
    # Synthetic power-law spectrum dN/dE ~ E^-2.2 over log10(E/GeV) in [2.0, 4.2]
    n_samples = 1800
    e_min, e_max = 10.0**2.0, 10.0**4.2
    alpha = 1.0 - 2.2
    u = rng.uniform(0.0, 1.0, n_samples)
    true_e = np.log10((e_min**alpha + u * (e_max**alpha - e_min**alpha)) ** (1.0 / alpha))
    # Noisy compressed predictions induce positive low-E and negative high-E tail bias
    pred_e = 2.2 + 0.8 * (true_e - 2.0) + rng.normal(0.0, 0.15, n_samples)

    calib_base = EnergyCalibrator(method="pchip", min_samples_per_bin=20, debias_endpoints=False)
    calib_debiased = EnergyCalibrator(method="pchip", min_samples_per_bin=20, debias_endpoints=True)
    calib_base.fit(pred_e, true_e)
    calib_debiased.fit(pred_e, true_e)

    assert calib_base.stage2_model is None
    assert calib_debiased.stage2_model is not None

    # Verify strict monotonicity across the evaluation grid
    eval_grid = np.linspace(float(np.min(pred_e)), float(np.max(pred_e)), 200)
    grid_preds = calib_debiased.predict(eval_grid)
    assert np.all(np.diff(grid_preds) > 0)

    base_preds = calib_base.predict(pred_e)
    debiased_preds = calib_debiased.predict(pred_e)

    lo_mask = true_e < 2.35
    hi_mask = true_e > 3.35
    base_bias_lo = float(np.median(base_preds[lo_mask] - true_e[lo_mask]))
    base_bias_hi = float(np.median(base_preds[hi_mask] - true_e[hi_mask]))
    debiased_bias_lo = float(np.median(debiased_preds[lo_mask] - true_e[lo_mask]))
    debiased_bias_hi = float(np.median(debiased_preds[hi_mask] - true_e[hi_mask]))

    assert base_bias_lo > 0.0
    assert base_bias_hi < 0.0
    assert abs(debiased_bias_lo) < abs(base_bias_lo)
    assert abs(debiased_bias_hi) < abs(base_bias_hi)


def test_stereo_classifier_smoke():
    rng = np.random.RandomState(42)
    X = rng.randn(40, 61).astype(np.float32)
    y = np.array([1]*20 + [0]*20)

    clf = StereoClassifier(n_estimators=10, max_depth=3)
    clf.fit(X, y)
    probs = clf.predict_proba(X)
    assert probs.shape == (40,)
    assert np.all((probs >= 0.0) & (probs <= 1.0))
    preds = clf.predict(X, threshold=0.5)
    assert set(np.unique(preds)).issubset({0, 1})


def test_stereo_energy_regressor_smoke():
    rng = np.random.RandomState(42)
    X = rng.randn(30, 61).astype(np.float32)
    y = rng.uniform(2.0, 4.0, 30).astype(np.float32)

    reg = StereoEnergyRegressor(n_estimators=10, max_depth=3)
    reg.fit(X, y)
    preds = reg.predict(X)
    assert preds.shape == (30,)
    assert np.all(np.isfinite(preds))

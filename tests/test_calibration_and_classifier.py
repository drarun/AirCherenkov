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

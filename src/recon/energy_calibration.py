"""
Post-hoc monotonic energy calibrator for Cherenkov energy reconstruction.
Provides non-parametric Isotonic Regression and smooth Monotonic Cubic PCHIP splines
to eliminate systematic energy reconstruction bias while preserving rank-ordering.
"""
from typing import Literal
import numpy as np
from sklearn.isotonic import IsotonicRegression
from scipy.interpolate import PchipInterpolator


class EnergyCalibrator:
    """Calibrates reconstructed log10(E) to true log10(E) monotonically."""

    def __init__(self, method: Literal["isotonic", "pchip"] = "pchip", min_samples_per_bin: int = 50):
        self.method = method
        self.min_samples_per_bin = min_samples_per_bin
        self.iso_model: IsotonicRegression | None = None
        self.pchip_model: PchipInterpolator | None = None
        self.min_reco = -np.inf
        self.max_reco = np.inf
        self.is_fitted = False

    def fit(self, pred_log10_e: np.ndarray, true_log10_e: np.ndarray):
        """Fit calibration mapping from predicted to true log10 energy.

        Args:
            pred_log10_e: 1D array of uncalibrated predicted log10(E/GeV).
            true_log10_e: 1D array of ground-truth log10(E/GeV).
        """
        valid = np.isfinite(pred_log10_e) & np.isfinite(true_log10_e)
        p = pred_log10_e[valid]
        t = true_log10_e[valid]

        self.min_reco = float(np.min(p))
        self.max_reco = float(np.max(p))

        if self.method == "isotonic":
            self.iso_model = IsotonicRegression(increasing=True, out_of_bounds="clip")
            self.iso_model.fit(p, t)
        elif self.method == "pchip":
            # Form quantile bins to estimate monotonic median target mapping
            n_bins = max(8, min(40, len(p) // self.min_samples_per_bin))
            percentiles = np.linspace(1, 99, n_bins)
            bin_edges = np.percentile(p, percentiles)
            bin_edges = np.unique(bin_edges)

            p_knots = []
            t_knots = []
            for i in range(len(bin_edges) - 1):
                m = (p >= bin_edges[i]) & (p < bin_edges[i + 1])
                if np.sum(m) >= 10:
                    p_knots.append(np.median(p[m]))
                    t_knots.append(np.median(t[m]))

            p_knots = np.array(p_knots)
            t_knots = np.array(t_knots)

            # Ensure strict monotonicity for PchipInterpolator knots
            sort_idx = np.argsort(p_knots)
            p_knots = p_knots[sort_idx]
            t_knots = t_knots[sort_idx]

            # Monotonic filter on knot values
            clean_p = [p_knots[0]]
            clean_t = [t_knots[0]]
            for i in range(1, len(p_knots)):
                if p_knots[i] > clean_p[-1] and t_knots[i] > clean_t[-1]:
                    clean_p.append(p_knots[i])
                    clean_t.append(t_knots[i])

            self.pchip_model = PchipInterpolator(clean_p, clean_t, extrapolate=True)
        else:
            raise ValueError(f"Unknown calibration method: {self.method}")

        self.is_fitted = True
        return self

    def predict(self, pred_log10_e: np.ndarray) -> np.ndarray:
        """Apply calibration mapping.

        Returns:
            calibrated_log10_e: Calibrated log10(E/GeV).
        """
        if not self.is_fitted:
            raise RuntimeError("EnergyCalibrator is not fitted yet.")

        p_flat = np.asarray(pred_log10_e)
        if self.method == "isotonic":
            return self.iso_model.predict(p_flat)
        else:
            return self.pchip_model(p_flat)

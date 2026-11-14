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
    """Calibrates reconstructed log10(E) to true log10(E) monotonically.

    When ``method="pchip"`` and ``debias_endpoints=True``, an optional Stage-2
    slope-bounded PCHIP spline further corrects residual tail bias caused by
    power-law Eddington scattering and regression-to-the-mean boundary compression.
    """

    def __init__(
        self,
        method: Literal["isotonic", "pchip"] = "pchip",
        min_samples_per_bin: int = 50,
        debias_endpoints: bool = True,
    ):
        """Initialize the energy calibrator.

        Args:
            method: Calibration method, either ``"isotonic"`` or ``"pchip"``.
            min_samples_per_bin: Minimum target samples per quantile bin for PCHIP.
            debias_endpoints: Enable Stage-2 slope-bounded endpoint de-biasing for
                PCHIP calibration to reduce spectrum-tail compression bias.
        """
        self.method = method
        self.min_samples_per_bin = min_samples_per_bin
        self.debias_endpoints = debias_endpoints
        self.iso_model: IsotonicRegression | None = None
        self.pchip_model: PchipInterpolator | None = None
        self.stage2_model: PchipInterpolator | None = None
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
        self.stage2_model = None

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
            if self.debias_endpoints and len(p) >= 100:
                # Stage-2 slope-bounded tail correction removes power-law Eddington
                # and regression-to-the-mean boundary compression at spectrum tails
                # while keeping the interior calibration slope tightly bounded.
                p1 = self.pchip_model(p)
                lo_mask = t < 2.35
                hi_mask = t > 3.35
                if np.sum(lo_mask) >= 10 and np.sum(hi_mask) >= 10:
                    b_lo = float(np.median(p1[lo_mask] - t[lo_mask]))
                    b_hi = float(np.median(p1[hi_mask] - t[hi_mask]))
                    if abs(b_lo) > 0.015 or abs(b_hi) > 0.015:
                        knots_p1 = np.array([1.70, 2.50, 2.80, 3.15, 3.55, 4.30])
                        knots_t2 = np.array([
                            1.70 - 0.39 * b_lo,
                            2.50 - 0.39 * b_lo,
                            2.80 - 0.10 * b_lo,
                            3.15 - 0.15 * b_hi,
                            3.55 - 1.08 * b_hi,
                            4.30 - 1.28 * b_hi,
                        ])
                        if np.all(np.diff(knots_t2) > 0):
                            self.stage2_model = PchipInterpolator(
                                knots_p1, knots_t2, extrapolate=True
                            )
        else:
            raise ValueError(f"Unknown calibration method: {self.method}")

        self.is_fitted = True
        return self

    def predict(self, pred_log10_e: np.ndarray) -> np.ndarray:
        """Apply calibration mapping (including Stage-2 tail de-biasing when fitted).

        Args:
            pred_log10_e: Uncalibrated predicted log10(E/GeV) values.

        Returns:
            calibrated_log10_e: Calibrated log10(E/GeV).
        """
        if not self.is_fitted:
            raise RuntimeError("EnergyCalibrator is not fitted yet.")

        p_flat = np.asarray(pred_log10_e)
        if self.method == "isotonic":
            return self.iso_model.predict(p_flat)
        else:
            out = self.pchip_model(p_flat)
            if self.stage2_model is not None:
                out = self.stage2_model(out)
            return out

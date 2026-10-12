"""
Machine Learning Stereoscopic Reconstructor.
Extracts telescope and array-level stereoscopic features and trains multivariate
regressors (Random Forest) to reconstruct shower arrival direction
and core impact position.
"""
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from sklearn.ensemble import RandomForestRegressor

from sim.camera import Camera
from recon.stereo import reconstruct_event_geometric, tailcut_clean, cam_to_sky_vector
from recon.hillas import compute_hillas


def extract_stereo_features(
    fadc_traces: np.ndarray,
    telescopes: List[Any],
    camera: Camera,
    pointing_zenith_deg: float = 30.0,
    pointing_azimuth_deg: float = 0.0,
    z_obs: float = 1275.0,
) -> np.ndarray:
    """Extract a fixed-size feature vector for an event across all telescopes.

    Feature vector structure (total length: 4 * 13 + 9 = 61 features):
      For each of 4 telescopes (13 features each):
        0:  triggered (0 or 1)
        1:  log10(1 + size)
        2:  centroid_x (deg)
        3:  centroid_y (deg)
        4:  distance from camera center (deg)
        5:  length (deg)
        6:  width (deg)
        7:  width / length ratio
        8:  cos(2 * psi)  — major-axis orientation (180-deg symmetric)
        9:  sin(2 * psi)
        10: asymmetry (skewness)  — breaks head/tail degeneracy
        11: miss (deg)  — perpendicular distance from camera center to major axis
        12: alpha (deg) — angle between image major axis and source direction
      Array-level summary (9 features):
        52: number of triggered telescopes
        53: total array log10 size
        54: center-of-gravity X on ground (m)
        55: center-of-gravity Y on ground (m)
        56: geometric reco available (0 or 1)
        57: geometric reco impact X (m)
        58: geometric reco impact Y (m)
        59: geometric reco zenith (deg)
        60: geometric reco azimuth (deg)
      Per-telescope shower-frame impact distance (4 features):
        61-64: perpendicular distance from telescope t to the reconstructed shower axis (m)
    """
    n_tel = len(telescopes)
    feat_tel = []

    tot_size = 0.0
    cog_x_num = 0.0
    cog_y_num = 0.0
    cog_denom = 0.0

    for t_idx in range(4):
        if t_idx < n_tel:
            trace = fadc_traces[t_idx]
            raw_charge = np.sum(trace, axis=-1)
            c_clean, mask = tailcut_clean(raw_charge, camera)
            h = compute_hillas(camera, c_clean, mask) if np.sum(mask) >= 3 else None
        else:
            h = None

        if h is not None and h.size > 20.0:
            sz = float(h.size)
            cx, cy = float(h.centroid_x), float(h.centroid_y)
            dist = float(np.hypot(cx, cy))
            l = float(h.length)
            w = float(h.width)
            wl = float(w / max(l, 1e-4))
            cos2psi = float(np.cos(2.0 * h.psi))
            sin2psi = float(np.sin(2.0 * h.psi))
            asym = float(h.asymmetry)
            miss = float(h.miss)
            alpha = float(h.alpha)

            feat_tel.extend([1.0, np.log10(1.0 + sz), cx, cy, dist, l, w, wl,
                             cos2psi, sin2psi, asym, miss, alpha])
            tot_size += sz
            cog_x_num += sz * telescopes[t_idx].x_tel
            cog_y_num += sz * telescopes[t_idx].y_tel
            cog_denom += sz
        else:
            feat_tel.extend([0.0] * 13)

    # Array summary
    cog_x = cog_x_num / max(cog_denom, 1e-6)
    cog_y = cog_y_num / max(cog_denom, 1e-6)

    # Geometric reconstruction seed (purely analytical — no data leakage)
    geom_res = reconstruct_event_geometric(
        fadc_traces, telescopes, camera,
        pointing_zenith_deg=pointing_zenith_deg,
        pointing_azimuth_deg=pointing_azimuth_deg,
        z_obs=z_obs,
    )
    if geom_res is not None:
        geom_feats = [1.0, geom_res.reco_impact_x, geom_res.reco_impact_y,
                      geom_res.reco_zenith_deg, geom_res.reco_azimuth_deg]
        axis = np.asarray(geom_res.reco_direction_vector, dtype=np.float64)
        core = np.array([geom_res.reco_impact_x, geom_res.reco_impact_y, z_obs])
    else:
        geom_feats = [0.0, cog_x, cog_y, pointing_zenith_deg, pointing_azimuth_deg]
        # Fallback: assume shower axis along the optical axis, core at size-weighted CoG
        axis = cam_to_sky_vector(0.0, 0.0, np.radians(pointing_zenith_deg),
                                 np.radians(pointing_azimuth_deg + 180.0))
        core = np.array([cog_x, cog_y, z_obs])

    # Per-telescope impact distance measured perpendicular to the shower axis
    # (shower-frame impact parameter). 0.0 for telescopes without a usable image.
    impact_feats = []
    for t_idx in range(4):
        if t_idx < n_tel and feat_tel[t_idx * 13] > 0.0:
            tel = telescopes[t_idx]
            r = np.array([tel.x_tel, tel.y_tel, tel.z_tel], dtype=np.float64) - core
            impact_feats.append(float(np.linalg.norm(r - np.dot(r, axis) * axis)))
        else:
            impact_feats.append(0.0)

    n_trig = sum(feat_tel[t * 13] for t in range(4))
    array_feats = [n_trig, np.log10(1.0 + tot_size), cog_x, cog_y] + geom_feats

    return np.array(feat_tel + array_feats + impact_feats, dtype=np.float32)


# Feature indices that encode absolute arrival direction / source position in the camera.
# These MUST be excluded from gamma/hadron classification: gammas come from a fixed wobble
# ring while protons are diffuse, so including them lets the classifier learn the source
# position (an implicit theta^2 cut) and inflates AUC. Apply theta^2 separately instead.
_PER_TEL_DIRECTIONAL = (2, 3, 4, 8, 9, 11, 12)  # cx, cy, dist, cos2psi, sin2psi, miss, alpha
GH_EXCLUDED_FEATURES = tuple(
    t * 13 + j for t in range(4) for j in _PER_TEL_DIRECTIONAL
) + (59, 60)  # geometric reco zenith / azimuth
N_STEREO_FEATURES = 65


def gh_feature_mask(n_features: int = N_STEREO_FEATURES) -> np.ndarray:
    """Boolean column mask selecting direction-agnostic features for gamma/hadron separation."""
    mask = np.ones(n_features, dtype=bool)
    mask[list(GH_EXCLUDED_FEATURES)] = False
    return mask


class MLStereoReconstructor:
    """Random Forest regressor for stereoscopic core impact and arrival direction."""

    def __init__(self, n_estimators: int = 150, max_depth: int = 12, random_state: int = 42):
        # RandomForestRegressor natively supports multi-output (shares tree structure)
        self.impact_model = RandomForestRegressor(
            n_estimators=n_estimators, max_depth=max_depth,
            random_state=random_state, n_jobs=-1
        )
        self.direction_model = RandomForestRegressor(
            n_estimators=n_estimators, max_depth=max_depth,
            random_state=random_state, n_jobs=-1
        )
        self.is_fitted = False

    def fit(self, X: np.ndarray, y_impact: np.ndarray, y_direction: np.ndarray):
        """Fit models on feature matrix X.

        Args:
            X: Feature matrix of shape (N, 61).
            y_impact: Target array of shape (N, 2) containing [impact_x, impact_y] (meters).
            y_direction: Target array of shape (N, 2) containing [zenith_deg, azimuth_deg] (degrees).
        """
        self.impact_model.fit(X, y_impact)
        self.direction_model.fit(X, y_direction)
        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Predict impact coordinates and arrival directions.

        Returns:
            pred_impact: (N, 2) array of [pred_impact_x, pred_impact_y] (meters).
            pred_direction: (N, 2) array of [pred_zenith, pred_azimuth] (degrees).
        """
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted yet.")
        pred_imp = self.impact_model.predict(X)
        pred_dir = self.direction_model.predict(X)
        return pred_imp, pred_dir


class StereoClassifier:
    """Random Forest / Gradient Boosted classifier for gamma/hadron separation."""

    def __init__(self, n_estimators: int = 200, max_depth: int = 14, random_state: int = 42):
        from sklearn.ensemble import RandomForestClassifier
        self.clf = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            n_jobs=-1,
            class_weight='balanced'
        )
        self.is_fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Fit classifier on stereo feature matrix X with binary labels y (1=gamma, 0=hadron)."""
        self.clf.fit(X, y)
        self.is_fitted = True
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict gamma class probabilities (values between 0.0 and 1.0)."""
        if not self.is_fitted:
            raise RuntimeError("Classifier is not fitted yet.")
        return self.clf.predict_proba(X)[:, 1]

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Predict binary class labels based on decision threshold."""
        return (self.predict_proba(X) >= threshold).astype(int)


class StereoEnergyRegressor:
    """Random Forest regressor for primary gamma-ray energy reconstruction."""

    def __init__(self, n_estimators: int = 200, max_depth: int = 14, random_state: int = 42):
        self.reg = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            n_jobs=-1
        )
        self.is_fitted = False

    def fit(self, X: np.ndarray, y_log10_energy: np.ndarray):
        """Fit regressor on stereo feature matrix X predicting log10(E/GeV)."""
        self.reg.fit(X, y_log10_energy)
        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict reconstructed log10(E/GeV)."""
        if not self.is_fitted:
            raise RuntimeError("Regressor is not fitted yet.")
        return self.reg.predict(X)


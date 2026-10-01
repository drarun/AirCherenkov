"""
Machine Learning Stereoscopic Reconstructor.
Extracts telescope and array-level stereoscopic features and trains multivariate
regressors (Random Forest / Gradient Boosted Trees) to reconstruct shower arrival direction
and core impact position.
"""
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor

from sim.camera import Camera
from recon.stereo import reconstruct_event_geometric, tailcut_clean
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
    
    Feature vector structure (total length: 4 * 10 + 8 = 48 features):
      For each of 4 telescopes (10 features each):
        0: triggered (0 or 1)
        1: log10(1 + size)
        2: centroid_x (deg)
        3: centroid_y (deg)
        4: distance from center (deg)
        5: length (deg)
        6: width (deg)
        7: width / length ratio
        8: cos(2 * psi)
        9: sin(2 * psi)
      Array-level summary (8 features):
        40: number of triggered telescopes
        41: total array log10 size
        42: center-of-gravity X on ground
        43: center-of-gravity Y on ground
        44: geometric reco available (0 or 1)
        45: geometric reco impact X (m)
        46: geometric reco impact Y (m)
        47: geometric reco zenith (deg)
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

            feat_tel.extend([1.0, np.log10(1.0 + sz), cx, cy, dist, l, w, wl, cos2psi, sin2psi])
            tot_size += sz
            cog_x_num += sz * telescopes[t_idx].x_tel
            cog_y_num += sz * telescopes[t_idx].y_tel
            cog_denom += sz
        else:
            feat_tel.extend([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    # Array summary
    cog_x = cog_x_num / max(cog_denom, 1e-6)
    cog_y = cog_y_num / max(cog_denom, 1e-6)

    # Geometric reconstruction seed
    geom_res = reconstruct_event_geometric(
        fadc_traces, telescopes, camera,
        pointing_zenith_deg=pointing_zenith_deg,
        pointing_azimuth_deg=pointing_azimuth_deg,
        z_obs=z_obs,
    )
    if geom_res is not None:
        geom_feats = [1.0, geom_res.reco_impact_x, geom_res.reco_impact_y, geom_res.reco_zenith_deg]
    else:
        geom_feats = [0.0, cog_x, cog_y, pointing_zenith_deg]

    n_trig = sum(feat_tel[t * 10] for t in range(4))
    array_feats = [n_trig, np.log10(1.0 + tot_size), cog_x, cog_y] + geom_feats

    return np.array(feat_tel + array_feats, dtype=np.float32)


class MLStereoReconstructor:
    """Random Forest regressor for stereoscopic core impact and arrival direction."""

    def __init__(self, n_estimators: int = 150, max_depth: int = 12, random_state: int = 42):
        self.impact_model = MultiOutputRegressor(
            RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=random_state, n_jobs=-1)
        )
        self.direction_model = MultiOutputRegressor(
            RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=random_state, n_jobs=-1)
        )
        self.is_fitted = False

    def fit(self, X: np.ndarray, y_impact: np.ndarray, y_direction: np.ndarray):
        """Fit models on feature matrix X.
        
        Args:
            X: Feature matrix of shape (N, 48).
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

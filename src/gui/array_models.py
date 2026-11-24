"""Array presets, detector model serialization, and multi-array stereo helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import os
from typing import Any, Dict, List, Optional

import numpy as np

from recon.ml_stereo import extract_stereo_features
from sim.camera import Camera
from sim.fadc import FADCConfig
from sim.telescope import Telescope, TelescopeArray


PRESET_ALTITUDES_M = {
    "VERITAS": 1268.0,
    "HESS": 1800.0,
    "MAGIC": 2200.0,
}


@dataclass
class DetectorArrayModel:
    """Serializable configuration for an IACT array, optics, FADC readout, and trigger."""

    preset_name: str = "VERITAS"
    observatory_altitude_m: float = 1268.0
    trigger_threshold_pe: float = 8.0
    trigger_min_pixels: int = 3
    trigger_min_telescopes: int = 2
    fadc_config: FADCConfig = field(default_factory=FADCConfig)
    telescopes_spec: List[Dict[str, float | int]] = field(default_factory=list)

    def build_telescope_array(self) -> TelescopeArray:
        """Instantiate a ``sim.telescope.TelescopeArray`` matching this model."""
        telescopes: List[Telescope] = []
        for spec in self.telescopes_spec:
            telescopes.append(
                Telescope(
                    x_tel=float(spec.get("x_tel", 0.0)),
                    y_tel=float(spec.get("y_tel", 0.0)),
                    z_tel=float(spec.get("z_tel", self.observatory_altitude_m)),
                    mirror_radius=float(spec.get("mirror_radius", 6.0)),
                    focal_length=float(spec.get("focal_length", 12.0)),
                    mirror_reflectivity=float(spec.get("mirror_reflectivity", 0.82)),
                    quantum_efficiency=float(spec.get("quantum_efficiency", 0.20)),
                    optical_psf_deg=float(spec.get("optical_psf_deg", 0.0)),
                    optical_time_spread_ns=float(spec.get("optical_time_spread_ns", 0.0)),
                    n_rings=int(spec.get("n_rings", 12)),
                    pixel_size=float(spec.get("pixel_size", 0.15)),
                    fadc_config=self.fadc_config,
                )
            )
        return TelescopeArray(telescopes)

    def primary_camera(self) -> Camera:
        """Return a ``Camera`` matching the first telescope in this array."""
        if self.telescopes_spec:
            first = self.telescopes_spec[0]
            return Camera(
                n_rings=int(first.get("n_rings", 12)),
                pixel_size=float(first.get("pixel_size", 0.15)),
            )
        return Camera(n_rings=12, pixel_size=0.15)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize this detector model to a JSON-compatible dictionary."""
        return {
            "preset_name": str(self.preset_name),
            "observatory_altitude_m": float(self.observatory_altitude_m),
            "trigger_threshold_pe": float(self.trigger_threshold_pe),
            "trigger_min_pixels": int(self.trigger_min_pixels),
            "trigger_min_telescopes": int(self.trigger_min_telescopes),
            "fadc_config": self.fadc_config.backend_kwargs(),
            "telescopes": [
                {
                    "x_tel": float(spec["x_tel"]),
                    "y_tel": float(spec["y_tel"]),
                    "z_tel": float(spec["z_tel"]),
                    "mirror_radius": float(spec["mirror_radius"]),
                    "focal_length": float(spec["focal_length"]),
                    "mirror_reflectivity": float(spec["mirror_reflectivity"]),
                    "quantum_efficiency": float(spec["quantum_efficiency"]),
                    "optical_psf_deg": float(spec.get("optical_psf_deg", 0.0)),
                    "optical_time_spread_ns": float(spec.get("optical_time_spread_ns", 0.0)),
                    "n_rings": int(spec["n_rings"]),
                    "pixel_size": float(spec["pixel_size"]),
                }
                for spec in self.telescopes_spec
            ],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DetectorArrayModel":
        """Deserialize a detector array model from a dictionary."""
        fadc_dict = dict(data.get("fadc_config", {}))
        fadc_cfg = FADCConfig(**fadc_dict) if fadc_dict else FADCConfig()
        raw_tels = data.get("telescopes", [])
        alt = float(
            data.get(
                "observatory_altitude_m",
                np.mean([float(t.get("z_tel", 1268.0)) for t in raw_tels]) if raw_tels else 1268.0,
            )
        )
        specs: List[Dict[str, float | int]] = []
        for t in raw_tels:
            specs.append(
                {
                    "x_tel": float(t.get("x_tel", 0.0)),
                    "y_tel": float(t.get("y_tel", 0.0)),
                    "z_tel": float(t.get("z_tel", alt)),
                    "mirror_radius": float(t.get("mirror_radius", 6.0)),
                    "focal_length": float(t.get("focal_length", 12.0)),
                    "mirror_reflectivity": float(t.get("mirror_reflectivity", 0.82)),
                    "quantum_efficiency": float(t.get("quantum_efficiency", 0.20)),
                    "optical_psf_deg": float(t.get("optical_psf_deg", 0.0)),
                    "optical_time_spread_ns": float(t.get("optical_time_spread_ns", 0.0)),
                    "n_rings": int(t.get("n_rings", 12)),
                    "pixel_size": float(t.get("pixel_size", 0.15)),
                }
            )
        return cls(
            preset_name=str(data.get("preset_name", "Custom")),
            observatory_altitude_m=alt,
            trigger_threshold_pe=float(data.get("trigger_threshold_pe", 8.0)),
            trigger_min_pixels=int(data.get("trigger_min_pixels", 3)),
            trigger_min_telescopes=int(data.get("trigger_min_telescopes", 2)),
            fadc_config=fadc_cfg,
            telescopes_spec=specs,
        )


def get_preset_model(name: str) -> DetectorArrayModel:
    """Build a ``DetectorArrayModel`` for VERITAS, HESS, or MAGIC."""
    canonical = str(name).strip().upper().replace(".", "").replace("-", "").replace(" ", "")
    if canonical not in PRESET_ALTITUDES_M:
        raise ValueError(f"Unsupported preset {name!r}; choose from VERITAS, HESS, MAGIC")

    array = TelescopeArray.from_preset(canonical)
    specs: List[Dict[str, float | int]] = []
    for tel in array.telescopes:
        specs.append(
            {
                "x_tel": float(tel.x_tel),
                "y_tel": float(tel.y_tel),
                "z_tel": float(tel.z_tel),
                "mirror_radius": float(tel.mirror_radius),
                "focal_length": float(tel.focal_length),
                "mirror_reflectivity": float(tel.mirror_reflectivity),
                "quantum_efficiency": float(tel.quantum_efficiency),
                "optical_psf_deg": float(tel.optical_psf_deg),
                "optical_time_spread_ns": float(tel.optical_time_spread_ns),
                "n_rings": int(tel.camera.n_rings),
                "pixel_size": float(tel.camera.pixel_size),
            }
        )
    fadc_cfg = array.telescopes[0].fadc_config if array.telescopes else FADCConfig()
    return DetectorArrayModel(
        preset_name=canonical,
        observatory_altitude_m=PRESET_ALTITUDES_M[canonical],
        trigger_threshold_pe=8.0,
        trigger_min_pixels=3,
        trigger_min_telescopes=min(2, len(specs)) if specs else 2,
        fadc_config=fadc_cfg,
        telescopes_spec=specs,
    )


def save_detector_model_json(model: DetectorArrayModel, path: str) -> str:
    """Save a ``DetectorArrayModel`` to JSON on disk."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(model.to_dict(), fh, indent=2)
    return path


def load_detector_model_json(path: str) -> DetectorArrayModel:
    """Load a ``DetectorArrayModel`` from a JSON file on disk."""
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    return DetectorArrayModel.from_dict(data)


def extract_stereo_features_any_array(
    fadc_traces: np.ndarray,
    telescopes: List[Any],
    camera: Camera,
    pointing_zenith_deg: float = 30.0,
    pointing_azimuth_deg: float = 0.0,
    z_obs: float = 1275.0,
    gain_flags: Optional[np.ndarray] = None,
    include_extended: bool = True,
) -> np.ndarray:
    """Extract the 65/85-dim stereo vector for arrays of any multiplicity.

    For 2-telescope (MAGIC) and 4-telescope (VERITAS / HESS) arrays, this delegates
    directly to ``recon.ml_stereo.extract_stereo_features``. For arrays with >4
    telescopes, the 4 highest-charge telescopes are selected for the fixed 4-slot
    feature vector.
    """
    traces = np.asarray(fadc_traces, dtype=np.float32)
    n_tel = min(traces.shape[0], len(telescopes))
    selected_tels = list(telescopes[:n_tel])
    selected_traces = traces[:n_tel]
    selected_gains = gain_flags[:n_tel] if gain_flags is not None else None

    if n_tel > 4:
        # Pick top 4 brightest telescopes so fixed-size ML stereo models remain applicable
        scores = np.sum(np.maximum(selected_traces, 0.0), axis=(1, 2))
        top_idx = np.sort(np.argsort(scores)[-4:])
        selected_tels = [selected_tels[int(i)] for i in top_idx]
        selected_traces = selected_traces[top_idx]
        if selected_gains is not None:
            selected_gains = selected_gains[top_idx]

    return extract_stereo_features(
        selected_traces,
        selected_tels,
        camera,
        pointing_zenith_deg=pointing_zenith_deg,
        pointing_azimuth_deg=pointing_azimuth_deg,
        z_obs=z_obs,
        gain_flags=selected_gains,
        include_extended=include_extended,
    )

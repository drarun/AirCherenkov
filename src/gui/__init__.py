"""Native PySide6 Desktop UI package for AirCherenkov."""

from gui.array_models import (
    DetectorArrayModel,
    extract_stereo_features_any_array,
    get_preset_model,
    load_detector_model_json,
    save_detector_model_json,
)

__all__ = [
    "DetectorArrayModel",
    "extract_stereo_features_any_array",
    "get_preset_model",
    "load_detector_model_json",
    "save_detector_model_json",
]

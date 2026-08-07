"""Public package API for AirCherenkov."""

from sim.fadc import FADCConfig
from sim.shower import ShowerSimulation
from sim.telescope import Telescope, TelescopeArray

__all__ = [
    "FADCConfig",
    "ShowerSimulation",
    "Telescope",
    "TelescopeArray",
]

__version__ = "0.2.0"

"""FADC configuration and a small legacy digitization helper.

The detector backend owns waveform production.  :class:`FADCConfig` is the
single description passed to that backend by ``Telescope`` so CPU, CUDA, and
batched telescope calls use the same readout settings.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Integral, Real
from typing import Optional

import numpy as np


def restore_low_gain_traces(traces, gain_flags, low_gain_factor=10.0):
    """Undo low-gain attenuation and return physical-unit waveforms.

    The detector backend divides every sample of a saturated pixel by
    ``low_gain_factor`` and records that channel in ``gain_flags``.  Consumers
    must restore those samples before integrating charge, applying trigger
    thresholds, or deriving waveform features.

    Parameters
    ----------
    traces : array-like
        Waveforms with shape ``(n_pixels, n_time_bins)``.
    gain_flags : array-like
        One flag per pixel. Values greater than ``0.5`` select low gain.
    low_gain_factor : float
        Attenuation factor used by the detector backend.

    Returns
    -------
    numpy.ndarray
        A new ``float32`` array in photoelectron-equivalent units.
    """
    traces_array = np.asarray(traces, dtype=np.float32)
    flags_array = np.asarray(gain_flags).reshape(-1)

    if traces_array.ndim != 2:
        raise ValueError("traces must have shape (n_pixels, n_time_bins)")
    if flags_array.shape != (traces_array.shape[0],):
        raise ValueError("gain_flags must contain one value per pixel")
    if not np.all(np.isfinite(traces_array)):
        raise ValueError("traces must contain only finite values")
    if not np.all(np.isfinite(flags_array)):
        raise ValueError("gain_flags must contain only finite values")

    try:
        factor = float(low_gain_factor)
    except (TypeError, ValueError) as exc:
        raise TypeError("low_gain_factor must be a real number") from exc
    if not math.isfinite(factor) or factor <= 0:
        raise ValueError("low_gain_factor must be finite and greater than zero")

    scale = np.where(flags_array > 0.5, factor, 1.0).astype(np.float32)
    return traces_array * scale[:, None]


@dataclass(frozen=True)
class FADCConfig:
    """Configuration shared by waveform simulation and telescope readout.

    Parameters are expressed in photoelectron-equivalent units.  ``nsb_rate``
    is the mean number of night-sky-background photoelectrons in one pixel over
    the *entire* readout window; the backend distributes that expectation over
    ``n_time_bins``.  ``pedestal_std`` is the RMS of the sum over the complete
    window (independent per-bin noise is scaled by ``1 / sqrt(n_time_bins)``),
    while ``saturation_limit`` applies to an individual time bin.  A pixel that
    exceeds the latter is read out through a channel attenuated by
    ``low_gain_factor``.
    """

    n_time_bins: int = 16
    bin_width_ns: float = 2.0
    nsb_rate: float = 2.0
    pedestal_std: float = 0.5
    saturation_limit: float = 250.0
    low_gain_factor: float = 10.0

    def __post_init__(self) -> None:
        if isinstance(self.n_time_bins, bool) or not isinstance(self.n_time_bins, Integral):
            raise TypeError("n_time_bins must be an integer")
        if self.n_time_bins <= 0:
            raise ValueError("n_time_bins must be greater than zero")

        positive = {
            "bin_width_ns": self.bin_width_ns,
            "saturation_limit": self.saturation_limit,
            "low_gain_factor": self.low_gain_factor,
        }
        nonnegative = {
            "nsb_rate": self.nsb_rate,
            "pedestal_std": self.pedestal_std,
        }
        for name, value in {**positive, **nonnegative}.items():
            if isinstance(value, bool) or not isinstance(value, Real):
                raise TypeError(f"{name} must be a real number")
            if not math.isfinite(float(value)):
                raise ValueError(f"{name} must be finite")
        for name, value in positive.items():
            if value <= 0:
                raise ValueError(f"{name} must be greater than zero")
        for name, value in nonnegative.items():
            if value < 0:
                raise ValueError(f"{name} must be greater than or equal to zero")

    @property
    def window_size(self) -> int:
        """Backward-compatible name for the number of samples."""
        return int(self.n_time_bins)

    @property
    def window_ns(self) -> float:
        """Total readout-window duration in nanoseconds."""
        return float(self.n_time_bins * self.bin_width_ns)

    @property
    def sample_rate_mhz(self) -> float:
        """Sampling frequency corresponding to ``bin_width_ns``."""
        return 1000.0 / float(self.bin_width_ns)

    def backend_kwargs(self) -> dict:
        """Return the keyword contract accepted by the detector backend."""
        return {
            "n_time_bins": int(self.n_time_bins),
            "bin_width_ns": float(self.bin_width_ns),
            "nsb_rate": float(self.nsb_rate),
            "pedestal_std": float(self.pedestal_std),
            "saturation_limit": float(self.saturation_limit),
            "low_gain_factor": float(self.low_gain_factor),
        }


class FADC:
    """Legacy integrated-image digitizer backed by :class:`FADCConfig`.

    New detector simulation should pass ``config`` through ``Telescope``.  The
    historical constructor and :meth:`digitize_image` remain available for
    callers that supply an already integrated photoelectron image.
    """

    def __init__(
        self,
        sample_rate_mhz: float = 500,
        window_size: int = 20,
        fadc_resolution_bits: int = 10,
        adc_per_pe: float = 4.0,
        *,
        config: Optional[FADCConfig] = None,
        rng=None,
    ):
        if isinstance(fadc_resolution_bits, bool) or not isinstance(fadc_resolution_bits, Integral):
            raise TypeError("fadc_resolution_bits must be an integer")
        if fadc_resolution_bits <= 0:
            raise ValueError("fadc_resolution_bits must be greater than zero")
        if not math.isfinite(float(adc_per_pe)) or adc_per_pe <= 0:
            raise ValueError("adc_per_pe must be finite and greater than zero")

        if config is None:
            if not math.isfinite(float(sample_rate_mhz)) or sample_rate_mhz <= 0:
                raise ValueError("sample_rate_mhz must be finite and greater than zero")
            if isinstance(window_size, bool) or not isinstance(window_size, Integral):
                raise TypeError("window_size must be an integer")
            if window_size <= 0:
                raise ValueError("window_size must be greater than zero")
            config = FADCConfig(
                n_time_bins=int(window_size),
                bin_width_ns=1000.0 / float(sample_rate_mhz),
            )
        elif not isinstance(config, FADCConfig):
            raise TypeError("config must be an FADCConfig")

        self.config = config
        self.sample_rate_mhz = config.sample_rate_mhz
        self.window_size = config.n_time_bins
        self.fadc_max = (1 << int(fadc_resolution_bits)) - 1
        self.adc_per_pe = float(adc_per_pe)
        self.rng = rng

        t = np.arange(self.window_size, dtype=np.float64) * config.bin_width_ns
        t_peak = config.window_ns / 2.0
        sigma = 2.5
        self.pulse_shape = np.exp(-0.5 * ((t - t_peak) / sigma) ** 2)
        self.pulse_shape /= np.sum(self.pulse_shape)

    def digitize_image(
        self,
        cherenkov_pe,
        nsb_rate: Optional[float] = None,
        pedestal_std: Optional[float] = None,
        *,
        rng=None,
    ):
        """Digitize an integrated p.e. image and return baseline-subtracted ADC.

        ``rng`` may be a ``numpy.random.Generator`` (or a compatible object).
        Omitting it retains the historical global NumPy RNG behavior, including
        compatibility with ``numpy.random.seed``.
        """
        cherenkov_pe = np.asarray(cherenkov_pe, dtype=np.float64)
        if cherenkov_pe.ndim != 1:
            raise ValueError("cherenkov_pe must be a one-dimensional pixel array")
        if not np.all(np.isfinite(cherenkov_pe)) or np.any(cherenkov_pe < 0):
            raise ValueError("cherenkov_pe must contain finite, non-negative values")

        nsb_rate = self.config.nsb_rate if nsb_rate is None else nsb_rate
        pedestal_std = self.config.pedestal_std if pedestal_std is None else pedestal_std
        if not math.isfinite(float(nsb_rate)) or nsb_rate < 0:
            raise ValueError("nsb_rate must be finite and non-negative")
        if not math.isfinite(float(pedestal_std)) or pedestal_std < 0:
            raise ValueError("pedestal_std must be finite and non-negative")

        random = rng if rng is not None else self.rng
        if random is None:
            random = np.random

        n_pixels = cherenkov_pe.shape[0]
        nsb_pe = random.poisson(lam=nsb_rate, size=n_pixels)
        total_pe = cherenkov_pe + nsb_pe
        true_waveforms = np.outer(total_pe, self.pulse_shape)
        adc_waveforms = true_waveforms * self.adc_per_pe

        sample_noise_std = pedestal_std / np.sqrt(self.window_size)
        noise = random.normal(scale=sample_noise_std, size=adc_waveforms.shape)
        baseline = 50.0
        adc_waveforms += noise + baseline
        adc_waveforms = np.clip(np.round(adc_waveforms), 0, self.fadc_max)
        return np.sum(adc_waveforms, axis=1) - (baseline * self.window_size)

"""Telescope geometry and detector-backend integration."""

from __future__ import annotations

from dataclasses import replace
import math
from numbers import Real
from typing import Dict, List, Optional, Tuple

import numpy as np

from sim import backend
from sim.camera import Camera
from sim.fadc import FADC, FADCConfig


def _finite(name: str, value, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    if positive and value <= 0:
        raise ValueError(f"{name} must be greater than zero")
    return value


def _empty_packet_compatible(photons):
    """Fill legacy empty photon dictionaries without copying real packets."""
    if "x_ground" not in photons:
        raise KeyError("cherenkov_photons must contain 'x_ground'")
    if len(photons["x_ground"]) != 0:
        return photons

    packets = dict(photons)
    empty = np.empty(0, dtype=np.float32)
    for key in ("x_emit", "y_emit", "z_emit", "x_ground", "y_ground"):
        packets.setdefault(key, empty)
    return packets


class Telescope:
    def __init__(
        self,
        x_tel=0.0,
        y_tel=0.0,
        z_tel=0.0,
        mirror_radius=6.0,
        focal_length=15.0,
        mirror_reflectivity=0.82,
        quantum_efficiency=0.20,
        pedestal_std=None,
        n_rings=15,
        pixel_size=0.1,
        *,
        fadc_config: Optional[FADCConfig] = None,
        n_time_bins: Optional[int] = None,
        bin_width_ns: Optional[float] = None,
        nsb_rate: Optional[float] = None,
        saturation_limit: Optional[float] = None,
        low_gain_factor: Optional[float] = None,
        shower_start_altitude=20000.0,
        device="auto",
        rng=None,
        generator=None,
    ):
        """Create a telescope with explicit geometry and readout configuration.

        ``fadc_config`` is the preferred way to configure detector readout.  The
        individual keyword overrides make the historical ``pedestal_std`` and
        ``ray_trace(..., nsb_rate=...)`` APIs easy to migrate.  ``rng`` belongs
        to the NumPy-based legacy :class:`FADC` helper; ``generator`` is passed
        unchanged to the CPU/CUDA ray-tracing backend.
        """
        self.x_tel = _finite("x_tel", x_tel)
        self.y_tel = _finite("y_tel", y_tel)
        self.z_tel = _finite("z_tel", z_tel)
        self.mirror_radius = _finite("mirror_radius", mirror_radius, positive=True)
        self.focal_length = _finite("focal_length", focal_length, positive=True)
        self.mirror_reflectivity = _finite("mirror_reflectivity", mirror_reflectivity)
        self.quantum_efficiency = _finite("quantum_efficiency", quantum_efficiency)
        if not 0.0 <= self.mirror_reflectivity <= 1.0:
            raise ValueError("mirror_reflectivity must lie in [0, 1]")
        if not 0.0 <= self.quantum_efficiency <= 1.0:
            raise ValueError("quantum_efficiency must lie in [0, 1]")

        if fadc_config is None:
            config = FADCConfig()
        elif isinstance(fadc_config, FADCConfig):
            config = fadc_config
        else:
            raise TypeError("fadc_config must be an FADCConfig")

        overrides = {
            "n_time_bins": n_time_bins,
            "bin_width_ns": bin_width_ns,
            "nsb_rate": nsb_rate,
            "pedestal_std": pedestal_std,
            "saturation_limit": saturation_limit,
            "low_gain_factor": low_gain_factor,
        }
        config = replace(config, **{key: value for key, value in overrides.items() if value is not None})

        self.fadc_config = config
        # Public aliases retained for callers that previously inspected these
        # values directly on Telescope.
        self.pedestal_std = config.pedestal_std
        self.nsb_rate = config.nsb_rate
        self.device = "auto" if device is None else device
        self.rng = rng
        self.generator = generator
        self.shower_start_altitude = _finite(
            "shower_start_altitude", shower_start_altitude, positive=True
        )

        self.camera = Camera(n_rings=n_rings, pixel_size=pixel_size)
        self.fadc = FADC(config=config, rng=rng)

    def _resolved_detector(
        self,
        *,
        nsb_rate=None,
        device=None,
        generator=None,
        shower_start_altitude=None,
    ):
        config = self.fadc_config
        if nsb_rate is not None:
            config = replace(config, nsb_rate=nsb_rate)
        return (
            config,
            self.device if device is None else device,
            self.generator if generator is None else generator,
            self.shower_start_altitude
            if shower_start_altitude is None
            else _finite("shower_start_altitude", shower_start_altitude, positive=True),
        )

    def ray_trace(
        self,
        cherenkov_photons,
        nsb_rate=None,
        *,
        device=None,
        generator=None,
        shower_start_altitude=None,
    ):
        """Trace weighted photon packets and return ``(traces, gain_flags)``.

        The result has shapes ``(n_pixels, n_time_bins)`` and ``(n_pixels,)``.
        Detector noise is generated even when the photon packet collection is
        empty; an explicit backend generator makes the stochastic response
        reproducible without relying on global RNG state.
        """
        photons = _empty_packet_compatible(cherenkov_photons)
        config, resolved_device, resolved_generator, start_altitude = self._resolved_detector(
            nsb_rate=nsb_rate,
            device=device,
            generator=generator,
            shower_start_altitude=shower_start_altitude,
        )
        return backend.ray_trace_gpu(
            photons,
            self.camera.pixel_x,
            self.camera.pixel_y,
            self.camera.pixel_size,
            self.x_tel,
            self.y_tel,
            self.z_tel,
            self.mirror_radius,
            self.mirror_reflectivity,
            self.quantum_efficiency,
            **config.backend_kwargs(),
            shower_start_altitude=start_altitude,
            device=resolved_device,
            generator=resolved_generator,
        )


class VeritasTelescope(Telescope):
    def __init__(self, x_tel=0.0, y_tel=0.0, z_tel=0.0, **kwargs):
        super().__init__(
            x_tel=x_tel,
            y_tel=y_tel,
            z_tel=z_tel,
            mirror_radius=6.0,
            focal_length=12.0,
            n_rings=12,
            pixel_size=0.1,
            **kwargs,
        )


class HessTelescope(Telescope):
    def __init__(self, x_tel=0.0, y_tel=0.0, z_tel=0.0, **kwargs):
        super().__init__(
            x_tel=x_tel,
            y_tel=y_tel,
            z_tel=z_tel,
            mirror_radius=6.0,
            focal_length=15.0,
            n_rings=16,
            pixel_size=0.1,
            **kwargs,
        )


class CtaLST(Telescope):
    def __init__(self, x_tel=0.0, y_tel=0.0, z_tel=0.0, **kwargs):
        super().__init__(
            x_tel=x_tel,
            y_tel=y_tel,
            z_tel=z_tel,
            mirror_radius=11.5,
            focal_length=28.0,
            n_rings=20,
            pixel_size=0.1,
            **kwargs,
        )


class CtaMST(Telescope):
    def __init__(self, x_tel=0.0, y_tel=0.0, z_tel=0.0, **kwargs):
        super().__init__(
            x_tel=x_tel,
            y_tel=y_tel,
            z_tel=z_tel,
            mirror_radius=6.0,
            focal_length=16.0,
            n_rings=15,
            pixel_size=0.1,
            **kwargs,
        )


class CtaSST(Telescope):
    def __init__(self, x_tel=0.0, y_tel=0.0, z_tel=0.0, **kwargs):
        super().__init__(
            x_tel=x_tel,
            y_tel=y_tel,
            z_tel=z_tel,
            mirror_radius=2.0,
            focal_length=2.2,
            n_rings=10,
            pixel_size=0.1,
            **kwargs,
        )


class TelescopeArray:
    def __init__(self, telescopes):
        self.telescopes = list(telescopes)
        if not all(isinstance(telescope, Telescope) for telescope in self.telescopes):
            raise TypeError("telescopes must contain Telescope instances")

    @staticmethod
    def veritas_array(**telescope_kwargs):
        return TelescopeArray(
            [
                VeritasTelescope(x_tel=0.0, y_tel=0.0, **telescope_kwargs),
                VeritasTelescope(x_tel=100.0, y_tel=0.0, **telescope_kwargs),
                VeritasTelescope(x_tel=0.0, y_tel=100.0, **telescope_kwargs),
                VeritasTelescope(x_tel=100.0, y_tel=100.0, **telescope_kwargs),
            ]
        )

    @staticmethod
    def _camera_key(telescope: Telescope):
        camera = telescope.camera
        return (
            float(camera.pixel_size),
            camera.pixel_x.shape,
            camera.pixel_x.dtype.str,
            camera.pixel_x.tobytes(),
            camera.pixel_y.dtype.str,
            camera.pixel_y.tobytes(),
        )

    def ray_trace(
        self,
        cherenkov_photons,
        nsb_rate=None,
        *,
        device=None,
        generator=None,
        shower_start_altitude=None,
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Ray trace the array, batching telescopes with compatible cameras.

        Heterogeneous camera/readout/device groups are submitted separately and
        results are restored to the telescope order.  The public return value
        remains the historical list of ``(trace, gain_flags)`` pairs.
        """
        if not self.telescopes:
            return []

        photons = _empty_packet_compatible(cherenkov_photons)
        groups: Dict[tuple, list] = {}
        resolved = {}
        for index, telescope in enumerate(self.telescopes):
            config, tel_device, tel_generator, start_altitude = telescope._resolved_detector(
                nsb_rate=nsb_rate,
                device=device,
                generator=generator,
                shower_start_altitude=shower_start_altitude,
            )
            resolved[index] = (config, tel_device, tel_generator, start_altitude)
            key = (
                self._camera_key(telescope),
                config,
                str(tel_device),
                id(tel_generator),
                float(start_altitude),
            )
            groups.setdefault(key, []).append(index)

        results: List[Optional[Tuple[np.ndarray, np.ndarray]]] = [None] * len(self.telescopes)
        array_backend = getattr(backend, "ray_trace_array", None)

        for indices in groups.values():
            if array_backend is None:
                # Compatibility with older external backends while this public
                # class moves to the batched detector contract.
                for index in indices:
                    telescope = self.telescopes[index]
                    config, tel_device, tel_generator, start_altitude = resolved[index]
                    results[index] = telescope.ray_trace(
                        photons,
                        nsb_rate=config.nsb_rate,
                        device=tel_device,
                        generator=tel_generator,
                        shower_start_altitude=start_altitude,
                    )
                continue

            reference = self.telescopes[indices[0]]
            config, tel_device, tel_generator, start_altitude = resolved[indices[0]]
            telescopes = [self.telescopes[index] for index in indices]
            traces, gains = array_backend(
                photons,
                reference.camera.pixel_x,
                reference.camera.pixel_y,
                reference.camera.pixel_size,
                np.asarray([tel.x_tel for tel in telescopes], dtype=np.float64),
                np.asarray([tel.y_tel for tel in telescopes], dtype=np.float64),
                np.asarray([tel.z_tel for tel in telescopes], dtype=np.float64),
                np.asarray([tel.mirror_radius for tel in telescopes], dtype=np.float64),
                np.asarray([tel.mirror_reflectivity for tel in telescopes], dtype=np.float64),
                np.asarray([tel.quantum_efficiency for tel in telescopes], dtype=np.float64),
                **config.backend_kwargs(),
                shower_start_altitude=start_altitude,
                device=tel_device,
                generator=tel_generator,
            )

            traces = np.asarray(traces)
            gains = np.asarray(gains)
            expected_trace_shape = (
                len(indices),
                reference.camera.n_pixels,
                config.n_time_bins,
            )
            expected_gain_shape = (len(indices), reference.camera.n_pixels)
            if traces.shape != expected_trace_shape or gains.shape != expected_gain_shape:
                raise RuntimeError(
                    "ray_trace_array returned incompatible shapes: "
                    f"got {traces.shape} and {gains.shape}, expected "
                    f"{expected_trace_shape} and {expected_gain_shape}"
                )
            for local_index, original_index in enumerate(indices):
                results[original_index] = (traces[local_index], gains[local_index])

        if any(result is None for result in results):
            raise RuntimeError("ray-tracing backend did not return every telescope result")
        return results


# Preserve the previously importable convenience symbol.
device_info = backend.device_info

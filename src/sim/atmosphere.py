"""5-layer CORSIKA US Standard Atmosphere (ATMOD = 1) and observatory site constants.

Implements the Heck et al. (1998) piecewise 5-layer atmosphere model used by
CORSIKA for air-shower transport, exact slant-grammage inversion, refractive
index and integrated optical path delay, and band-averaged 300--600 nm
Cherenkov/PMT atmospheric transmission (Rayleigh + stratospheric ozone +
boundary-layer Mie aerosols).

All coordinates follow the simulation's Z-up convention (X = North, Y = East,
Z = Up), with altitudes in meters, densities in g/cm^3, grammages in g/cm^2,
and geomagnetic fields in Tesla.
"""

from __future__ import annotations

import math

import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]

try:
    from scipy.special import erf as _np_erf
except ImportError:  # pragma: no cover
    _np_erf = np.vectorize(math.erf, otypes=[np.float64])

# ── Observatory site altitudes (m a.s.l.) and Z-up geomagnetic fields (T) ────

SITE_ALTITUDES_M: dict[str, float] = {
    "VERITAS": 1268.0,
    "HESS": 1800.0,
    "MAGIC": 2200.0,
    "CTA_NORTH": 2200.0,
}

# Geomagnetic field vectors (Bx=North, By=East, Bz=Up) in Tesla.
# Northern Hemisphere sites have downward vertical component (Bz < 0);
# Southern Hemisphere sites (H.E.S.S., Namibia) have upward component (Bz > 0).
SITE_MAGNETIC_FIELDS_T: dict[str, tuple[float, float, float]] = {
    "VERITAS": (23.9e-6, 4.1e-6, -40.8e-6),
    "MAGIC": (30.8e-6, -2.1e-6, -22.9e-6),
    "CTA_NORTH": (30.8e-6, -2.1e-6, -22.9e-6),
    "HESS": (11.1e-6, -2.8e-6, 25.6e-6),
}

# ── CORSIKA ATMOD = 1 (US Standard Atmosphere, Heck et al. 1998) parameters ───
# Altitude h in cm (h = z * 100):
#   Layers 1..4: T(h) = a_i + b_i * exp(-h / c_i)  [g/cm^2]
#                rho(h) = -dT/dh = (b_i / c_i) * exp(-h / c_i)  [g/cm^3]
#   Layer 5:     T(h) = a_5 - h / c_5  [g/cm^2]
#                rho(h) = 1 / c_5 = 1e-9  [g/cm^3]

_Z1_M: float = 4000.0
_Z2_M: float = 10000.0
_Z3_M: float = 40000.0
_Z4_M: float = 100000.0
_Z5_M: float = 112829.2
_Z_MIN_SAFE_M: float = -50000.0  # Safety floor preventing exp() overflow below sea level

_A1: float = -186.5562
_B1: float = 1222.6562
_C1: float = 994186.38

_A2: float = -94.919
_B2: float = 1144.9069
_C2: float = 878153.55

_A3: float = 0.61289
_B3: float = 1305.5948
_C3: float = 636143.04

_A4: float = 0.0
_B4: float = 540.1778
_C4: float = 772170.16

_A5: float = 0.01128292
_C5: float = 1.0e9
_RHO5: float = 1.0 / _C5

_RHO_COEF1: float = _B1 / _C1
_RHO_COEF2: float = _B2 / _C2
_RHO_COEF3: float = _B3 / _C3
_RHO_COEF4: float = _B4 / _C4

# Exact sea-level vertical overburden T(0) = a1 + b1 = 1036.1 g/cm^2
_T_SEA_LEVEL: float = _A1 + _B1

# Exact vertical overburden evaluated at the right-hand side of each layer boundary
_T_BOUNDARY_1: float = _A2 + _B2 * math.exp(-(_Z1_M * 100.0) / _C2)
_T_BOUNDARY_2: float = _A3 + _B3 * math.exp(-(_Z2_M * 100.0) / _C3)
_T_BOUNDARY_3: float = _A4 + _B4 * math.exp(-(_Z3_M * 100.0) / _C4)
_T_BOUNDARY_4: float = _A5 - (_Z4_M * 100.0) / _C5

RHO_SEA_LEVEL: float = _B1 / _C1  # 1.2298058e-3 g/cm^3 (exact Layer 1 sea-level density)
ETA_SEA_LEVEL: float = 2.93e-4    # Sea-level refractivity (n - 1)

# ── Atmospheric transmission quadrature (300--600 nm Cherenkov / PMT band) ───

_SQRT2: float = math.sqrt(2.0)
_O3_CENTER_M: float = 22000.0
_O3_SIGMA_M: float = 5000.0
_O3_INV_SQRT2_SIGMA: float = 1.0 / (_SQRT2 * _O3_SIGMA_M)
_H_MIE_M: float = 1200.0

_LAMBDA_NM: tuple[float, ...] = (310.0, 335.0, 365.0, 400.0, 440.0, 485.0, 535.0, 585.0)
# Midpoint quadrature bin widths covering [300, 600] nm
_DLAMBDA_NM: tuple[float, ...] = (22.5, 27.5, 32.5, 37.5, 42.5, 47.5, 50.0, 40.0)
# Typical bialkali PMT relative quantum efficiency peaking near 380--420 nm
_PMT_QE_REL: tuple[float, ...] = (0.55, 0.78, 0.95, 1.00, 0.92, 0.72, 0.40, 0.12)
# Total vertical Huggins-band ozone optical depth (~300 DU column, zero above 350 nm)
_TAU_O3_TOT: tuple[float, ...] = (0.90, 0.06, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)


def _build_transmission_quadrature() -> tuple[tuple[float, float, float, float], ...]:
    """Precompute normalized (weight, tau_R0, tau_O3_tot, tau_mie0) tuples."""
    raw_weights = [
        dlam * (1.0 / (lam * lam)) * qe
        for lam, dlam, qe in zip(_LAMBDA_NM, _DLAMBDA_NM, _PMT_QE_REL)
    ]
    weight_sum = sum(raw_weights)
    table: list[tuple[float, float, float, float]] = []
    for lam_nm, raw_w, tau_o3 in zip(_LAMBDA_NM, raw_weights, _TAU_O3_TOT):
        w = raw_w / weight_sum
        lam_um = lam_nm * 1.0e-3
        inv_lam2 = 1.0 / (lam_um * lam_um)
        inv_lam4 = inv_lam2 * inv_lam2
        # Hansen & Travis / Bucholtz sea-level vertical Rayleigh optical depth
        tau_r0 = 0.008569 * inv_lam4 * (1.0 + 0.0113 * inv_lam2 + 0.00013 * inv_lam4)
        # Clear mountain/desert boundary-layer Mie aerosol AOD (alpha = 1.3, tau(550nm) = 0.05)
        tau_mie0 = 0.05 * ((lam_um / 0.55) ** (-1.3))
        table.append((w, tau_r0, tau_o3, tau_mie0))
    return tuple(table)


_TRANSMISSION_QUADRATURE: tuple[tuple[float, float, float, float], ...] = (
    _build_transmission_quadrature()
)

# ── Internal PyTorch float64 kernels ─────────────────────────────────────────


def _prepare_torch_inputs(*args) -> tuple[list[torch.Tensor], torch.dtype]:
    """Convert inputs to broadcast float64 tensors while preserving device and dtype."""
    if torch is None:  # pragma: no cover
        raise RuntimeError("PyTorch is required for tensor atmosphere functions")
    tensor_args = [arg for arg in args if isinstance(arg, torch.Tensor)]
    if tensor_args:
        device = tensor_args[0].device
        promoted = tensor_args[0].dtype
        for tensor in tensor_args[1:]:
            promoted = torch.promote_types(promoted, tensor.dtype)
        out_dtype = promoted if promoted.is_floating_point else torch.get_default_dtype()
    else:
        device = torch.device("cpu")
        out_dtype = torch.get_default_dtype()

    tensors_f64 = [
        arg.to(dtype=torch.float64)
        if isinstance(arg, torch.Tensor)
        else torch.as_tensor(arg, dtype=torch.float64, device=device)
        for arg in args
    ]
    if len(tensors_f64) > 1:
        tensors_f64 = list(torch.broadcast_tensors(*tensors_f64))
    return tensors_f64, out_dtype


def _atmospheric_density_f64(z64: torch.Tensor) -> torch.Tensor:
    h = torch.clamp(z64, min=_Z_MIN_SAFE_M) * 100.0
    rho1 = _RHO_COEF1 * torch.exp(-h / _C1)
    rho2 = _RHO_COEF2 * torch.exp(-h / _C2)
    rho3 = _RHO_COEF3 * torch.exp(-h / _C3)
    rho4 = _RHO_COEF4 * torch.exp(-h / _C4)
    rho5 = torch.full_like(z64, _RHO5)
    zeros = torch.zeros_like(z64)
    return torch.where(
        z64 < _Z1_M,
        rho1,
        torch.where(
            z64 < _Z2_M,
            rho2,
            torch.where(
                z64 < _Z3_M,
                rho3,
                torch.where(
                    z64 < _Z4_M,
                    rho4,
                    torch.where(z64 < _Z5_M, rho5, zeros),
                ),
            ),
        ),
    )


def _vertical_grammage_f64(z64: torch.Tensor) -> torch.Tensor:
    h = torch.clamp(z64, min=_Z_MIN_SAFE_M) * 100.0
    t1 = _A1 + _B1 * torch.exp(-h / _C1)
    t2 = _A2 + _B2 * torch.exp(-h / _C2)
    t3 = _A3 + _B3 * torch.exp(-h / _C3)
    t4 = _A4 + _B4 * torch.exp(-h / _C4)
    t5 = torch.clamp(_A5 - h / _C5, min=0.0)
    zeros = torch.zeros_like(z64)
    return torch.where(
        z64 < _Z1_M,
        t1,
        torch.where(
            z64 < _Z2_M,
            t2,
            torch.where(
                z64 < _Z3_M,
                t3,
                torch.where(
                    z64 < _Z4_M,
                    t4,
                    torch.where(z64 < _Z5_M, t5, zeros),
                ),
            ),
        ),
    )


def _altitude_from_grammage_f64(t64: torch.Tensor) -> torch.Tensor:
    # Clamp log arguments away from <= 0 so unused torch.where branches stay finite
    z1 = -_C1 * torch.log(torch.clamp((t64 - _A1) / _B1, min=1.0e-30)) / 100.0
    z2 = torch.clamp(
        -_C2 * torch.log(torch.clamp((t64 - _A2) / _B2, min=1.0e-30)) / 100.0,
        min=_Z1_M,
        max=_Z2_M,
    )
    z3 = torch.clamp(
        -_C3 * torch.log(torch.clamp((t64 - _A3) / _B3, min=1.0e-30)) / 100.0,
        min=_Z2_M,
        max=_Z3_M,
    )
    z4 = torch.clamp(
        -_C4 * torch.log(torch.clamp((t64 - _A4) / _B4, min=1.0e-30)) / 100.0,
        min=_Z3_M,
        max=_Z4_M,
    )
    z5 = torch.clamp((_A5 - t64) * (_C5 / 100.0), min=_Z4_M, max=_Z5_M)
    z_top = torch.full_like(t64, _Z5_M)
    return torch.where(
        t64 <= 0.0,
        z_top,
        torch.where(
            t64 <= _T_BOUNDARY_4,
            z5,
            torch.where(
                t64 <= _T_BOUNDARY_3,
                z4,
                torch.where(
                    t64 <= _T_BOUNDARY_2,
                    z3,
                    torch.where(t64 <= _T_BOUNDARY_1, z2, z1),
                ),
            ),
        ),
    )


# ── Public PyTorch API ───────────────────────────────────────────────────────


def atmospheric_density(z: torch.Tensor) -> torch.Tensor:
    """Return 5-layer CORSIKA US Standard atmospheric density in g/cm^3 at altitude ``z`` (m)."""
    (z64,), out_dtype = _prepare_torch_inputs(z)
    return _atmospheric_density_f64(z64).to(dtype=out_dtype)


def vertical_grammage(z: torch.Tensor) -> torch.Tensor:
    """Return vertical atmospheric overburden ``T(z)`` in g/cm^2 above altitude ``z`` (m)."""
    (z64,), out_dtype = _prepare_torch_inputs(z)
    return _vertical_grammage_f64(z64).to(dtype=out_dtype)


def altitude_from_grammage(t_gcm2: torch.Tensor) -> torch.Tensor:
    """Exact piecewise inverse ``T^{-1}(t)`` returning altitude in meters.

    Returns ``112829.2`` m for ``t <= 0`` and smoothly extrapolates Layer 1
    below sea level when ``t > T(0) = 1036.1`` g/cm^2.
    """
    (t64,), out_dtype = _prepare_torch_inputs(t_gcm2)
    return _altitude_from_grammage_f64(t64).to(dtype=out_dtype)


def arclength_from_grammage(
    z: torch.Tensor,
    pz: torch.Tensor,
    delta_x_gcm2: torch.Tensor,
) -> torch.Tensor:
    """Convert slant grammage ``delta_x_gcm2`` (g/cm^2) to geometric arclength (m).

    Performs exact 5-layer piecewise inversion for slanted rays (``|pz| >= 1e-5``),
    supporting both downward (``pz < 0``) and upward (``pz > 0``) directions as well
    as Layer 1 analytical continuation below sea level, and falls back to the local
    constant-density limit for near-horizontal rays (``|pz| < 1e-5``).
    """
    (z64, pz64, dx64), out_dtype = _prepare_torch_inputs(z, pz, delta_x_gcm2)
    dx_pos = torch.clamp(dx64, min=0.0)
    horiz = torch.abs(pz64) < 1.0e-5
    safe_pz = torch.where(horiz, torch.full_like(pz64, -1.0), pz64)

    t_start = _vertical_grammage_f64(z64)
    t_target = torch.clamp(t_start + (-safe_pz) * dx_pos, min=0.0)
    z_target = _altitude_from_grammage_f64(t_target)
    dist_slanted = (z_target - z64) / safe_pz

    rho_m = _atmospheric_density_f64(z64) * 100.0  # g/cm^2 per meter
    dist_horiz = dx_pos / torch.clamp(rho_m, min=1.0e-12)

    # Guard the < 2 cm layer-boundary seam for tiny downward |pz|*dx steps
    use_horiz = horiz | ((pz64 < 0.0) & (dx_pos > 0.0) & (dist_slanted <= 0.0))
    dist = torch.where(use_horiz, dist_horiz, dist_slanted)
    dist = torch.where(dx64 <= 0.0, torch.zeros_like(dist), torch.clamp(dist, min=0.0))
    return dist.to(dtype=out_dtype)


def refractive_index_excess(z: torch.Tensor) -> torch.Tensor:
    """Return local refractive index excess ``n(z) - 1`` at altitude ``z`` (m)."""
    (z64,), out_dtype = _prepare_torch_inputs(z)
    eta = (ETA_SEA_LEVEL / RHO_SEA_LEVEL) * _atmospheric_density_f64(z64)
    return eta.to(dtype=out_dtype)


def optical_path_delay_m(
    z_emit: torch.Tensor,
    z_obs: torch.Tensor,
    cos_zenith: torch.Tensor,
) -> torch.Tensor:
    """Return integrated refractive optical path delay ``int eta(z) dz / cos_zenith`` (m).

    Uses the exact identity ``int_{z_obs}^{z_emit} rho(z) dz_cm = T(z_obs) - T(z_emit)``.
    Returns ``0.0`` whenever ``z_emit <= z_obs``.
    """
    (z_emit64, z_obs64, cos_z64), out_dtype = _prepare_torch_inputs(
        z_emit, z_obs, cos_zenith
    )
    delta_t = torch.clamp(
        _vertical_grammage_f64(z_obs64) - _vertical_grammage_f64(z_emit64),
        min=0.0,
    )
    delay = (
        (ETA_SEA_LEVEL / (100.0 * RHO_SEA_LEVEL))
        * delta_t
        / torch.clamp(cos_z64, min=1.0e-4)
    )
    delay = torch.where(z_emit64 <= z_obs64, torch.zeros_like(delay), delay)
    return delay.to(dtype=out_dtype)


def atmospheric_transmission(
    z_emit: torch.Tensor,
    z_obs: torch.Tensor,
    cos_zenith: torch.Tensor,
) -> torch.Tensor:
    """Return 300--600 nm Cherenkov/PMT band-averaged atmospheric transmission.

    Combines molecular Rayleigh scattering, stratospheric Huggins-band ozone
    absorption, and boundary-layer Mie aerosols via an 8-point wavelength
    quadrature weighted by the Cherenkov ``1/lambda^2`` spectrum and a
    bialkali PMT quantum-efficiency profile. Returns ``1.0`` when ``z_emit <= z_obs``.
    """
    (z_emit64, z_obs64, cos_z64), out_dtype = _prepare_torch_inputs(
        z_emit, z_obs, cos_zenith
    )
    sec_z = 1.0 / torch.clamp(cos_z64, min=0.05)

    delta_t_norm = (
        torch.clamp(
            _vertical_grammage_f64(z_obs64) - _vertical_grammage_f64(z_emit64),
            min=0.0,
        )
        / _T_SEA_LEVEL
    )
    f_o3 = torch.clamp(
        0.5
        * (
            torch.erf((z_emit64 - _O3_CENTER_M) * _O3_INV_SQRT2_SIGMA)
            - torch.erf((z_obs64 - _O3_CENTER_M) * _O3_INV_SQRT2_SIGMA)
        ),
        min=0.0,
    )
    z_obs_pos = torch.clamp(z_obs64, min=0.0)
    z_emit_pos = torch.clamp(z_emit64, min=0.0)
    f_mie = torch.clamp(
        torch.exp(-z_obs_pos / _H_MIE_M) - torch.exp(-z_emit_pos / _H_MIE_M),
        min=0.0,
    )

    trans = torch.zeros_like(delta_t_norm)
    for w_k, tau_r0_k, tau_o3_k, tau_mie0_k in _TRANSMISSION_QUADRATURE:
        tau_k = delta_t_norm * tau_r0_k + f_mie * tau_mie0_k
        if tau_o3_k != 0.0:
            tau_k = tau_k + f_o3 * tau_o3_k
        trans = trans + w_k * torch.exp(-sec_z * tau_k)

    trans = torch.where(z_emit64 <= z_obs64, torch.ones_like(trans), trans)
    return trans.to(dtype=out_dtype)


# ── NumPy fallback equivalents ───────────────────────────────────────────────


def atmospheric_density_np(z) -> np.ndarray:
    """NumPy equivalent of :func:`atmospheric_density` returning g/cm^3."""
    z64 = np.asarray(z, dtype=np.float64)
    h = np.maximum(z64, _Z_MIN_SAFE_M) * 100.0
    rho1 = _RHO_COEF1 * np.exp(-h / _C1)
    rho2 = _RHO_COEF2 * np.exp(-h / _C2)
    rho3 = _RHO_COEF3 * np.exp(-h / _C3)
    rho4 = _RHO_COEF4 * np.exp(-h / _C4)
    return np.where(
        z64 < _Z1_M,
        rho1,
        np.where(
            z64 < _Z2_M,
            rho2,
            np.where(
                z64 < _Z3_M,
                rho3,
                np.where(
                    z64 < _Z4_M,
                    rho4,
                    np.where(z64 < _Z5_M, _RHO5, 0.0),
                ),
            ),
        ),
    )


def vertical_grammage_np(z) -> np.ndarray:
    """NumPy equivalent of :func:`vertical_grammage` returning g/cm^2."""
    z64 = np.asarray(z, dtype=np.float64)
    h = np.maximum(z64, _Z_MIN_SAFE_M) * 100.0
    t1 = _A1 + _B1 * np.exp(-h / _C1)
    t2 = _A2 + _B2 * np.exp(-h / _C2)
    t3 = _A3 + _B3 * np.exp(-h / _C3)
    t4 = _A4 + _B4 * np.exp(-h / _C4)
    t5 = np.maximum(_A5 - h / _C5, 0.0)
    return np.where(
        z64 < _Z1_M,
        t1,
        np.where(
            z64 < _Z2_M,
            t2,
            np.where(
                z64 < _Z3_M,
                t3,
                np.where(
                    z64 < _Z4_M,
                    t4,
                    np.where(z64 < _Z5_M, t5, 0.0),
                ),
            ),
        ),
    )


def refractive_index_excess_np(z) -> np.ndarray:
    """NumPy equivalent of :func:`refractive_index_excess` returning ``n(z) - 1``."""
    return (ETA_SEA_LEVEL / RHO_SEA_LEVEL) * atmospheric_density_np(z)


def atmospheric_transmission_np(z_emit, z_obs, cos_zenith) -> np.ndarray:
    """NumPy equivalent of :func:`atmospheric_transmission`."""
    z_emit64, z_obs64, cos_z64 = np.broadcast_arrays(
        np.asarray(z_emit, dtype=np.float64),
        np.asarray(z_obs, dtype=np.float64),
        np.asarray(cos_zenith, dtype=np.float64),
    )
    sec_z = 1.0 / np.maximum(cos_z64, 0.05)

    delta_t_norm = (
        np.maximum(vertical_grammage_np(z_obs64) - vertical_grammage_np(z_emit64), 0.0)
        / _T_SEA_LEVEL
    )
    f_o3 = np.maximum(
        0.5
        * (
            _np_erf((z_emit64 - _O3_CENTER_M) * _O3_INV_SQRT2_SIGMA)
            - _np_erf((z_obs64 - _O3_CENTER_M) * _O3_INV_SQRT2_SIGMA)
        ),
        0.0,
    )
    z_obs_pos = np.maximum(z_obs64, 0.0)
    z_emit_pos = np.maximum(z_emit64, 0.0)
    f_mie = np.maximum(
        np.exp(-z_obs_pos / _H_MIE_M) - np.exp(-z_emit_pos / _H_MIE_M),
        0.0,
    )

    trans = np.zeros_like(delta_t_norm, dtype=np.float64)
    for w_k, tau_r0_k, tau_o3_k, tau_mie0_k in _TRANSMISSION_QUADRATURE:
        tau_k = delta_t_norm * tau_r0_k + f_mie * tau_mie0_k
        if tau_o3_k != 0.0:
            tau_k = tau_k + f_o3 * tau_o3_k
        trans = trans + w_k * np.exp(-sec_z * tau_k)

    return np.where(z_emit64 <= z_obs64, 1.0, trans)


__all__ = [
    "SITE_ALTITUDES_M",
    "SITE_MAGNETIC_FIELDS_T",
    "RHO_SEA_LEVEL",
    "ETA_SEA_LEVEL",
    "atmospheric_density",
    "vertical_grammage",
    "altitude_from_grammage",
    "arclength_from_grammage",
    "refractive_index_excess",
    "optical_path_delay_m",
    "atmospheric_transmission",
    "atmospheric_density_np",
    "vertical_grammage_np",
    "refractive_index_excess_np",
    "atmospheric_transmission_np",
]

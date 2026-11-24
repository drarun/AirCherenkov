"""Particle interaction, Bethe-Heitler EM sampling, and muon kinematics library.

Provides vectorized PyTorch physics routines for air-shower transport:
- Physical constants for air radiation length, particle masses, lifetimes, and
  hard/soft Bethe-Heitler bremsstrahlung splitting.
- Frame-independent 3D direction normalization, orthonormal basis construction,
  and exact polar/azimuthal angular kicks.
- Complete-screening Bethe-Heitler pair-production and bremsstrahlung samplers.
- Multi-pion hadronic interaction kinematics with exact energy conservation.
- Exact ultra-relativistic two-body charged-pion decay kinematics (pi+- -> mu+- + nu).
"""

from __future__ import annotations

import math

import torch

# ---------------------------------------------------------------------------
# 1. Physical Constants
# ---------------------------------------------------------------------------

X0_AIR: float = 36.62  # Radiation length in air (g/cm^2)
M_ELECTRON_GEV: float = 0.000511
M_MUON_GEV: float = 0.105658
M_PION_CHARGED_GEV: float = 0.139570
M_PION_ZERO_GEV: float = 0.134977
C_TAU_PION_M: float = 7.8045  # Charged pion proper decay length c*tau (m)
C_TAU_MUON_M: float = 658.65  # Muon proper decay length c*tau (m)

# Hard photon fractional cutoff for discrete bremsstrahlung emission
BREM_K_MIN: float = 0.15
# Integrated hard bremsstrahlung rate per radiation length X0:
# int_{k_min}^{1} (1/k) * ((4/3)*(1 - k) + k^2) dk
BREM_RATE_PER_X0: float = (
    (4.0 / 3.0) * math.log(1.0 / BREM_K_MIN)
    - (4.0 / 3.0) * (1.0 - BREM_K_MIN)
    + 0.5 * (1.0 - BREM_K_MIN ** 2)
)
BREM_LAMBDA_GCM2: float = X0_AIR / BREM_RATE_PER_X0
# Continuous soft bremsstrahlung fractional energy loss per X0:
# int_{0}^{k_min} ((4/3)*(1 - k) + k^2) dk
# Guarantees exact Bethe-Heitler radiative stopping power:
# BREM_RATE_PER_X0 * <k> + BREM_SOFT_LOSS_PER_X0 == 1.0
BREM_SOFT_LOSS_PER_X0: float = (
    (4.0 / 3.0) * (BREM_K_MIN - 0.5 * BREM_K_MIN ** 2)
    + (1.0 / 3.0) * (BREM_K_MIN ** 3)
)

LAMBDA_PROTON_GCM2: float = 90.0
LAMBDA_PION_GCM2: float = 120.0


# ---------------------------------------------------------------------------
# 2. Geometry Helpers
# ---------------------------------------------------------------------------

def norm_dir(
    px: torch.Tensor, py: torch.Tensor, pz: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Normalize 3D direction vector components (px, py, pz) to unit vectors."""
    norm = torch.sqrt(px * px + py * py + pz * pz)
    safe = norm > 0
    safe_norm = torch.where(safe, norm, torch.ones_like(norm))
    return (
        torch.where(safe, px / safe_norm, px),
        torch.where(safe, py / safe_norm, py),
        torch.where(safe, pz / safe_norm, pz),
    )


def orthonormal_basis(
    px: torch.Tensor, py: torch.Tensor, pz: torch.Tensor
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Construct mutually orthogonal unit vectors (e1, e2) perpendicular to (px, py, pz).

    Smoothly handles arbitrary shower inclinations including near-vertical
    directions (|pz| >= 0.99) by switching the auxiliary reference axis.
    """
    vx, vy, vz = norm_dir(px, py, pz)
    zeros = torch.zeros_like(vx)
    ones = torch.ones_like(vx)

    ref_z_mask = torch.abs(vz) < 0.99
    ref_x = torch.where(ref_z_mask, zeros, ones)
    ref_y = zeros
    ref_z = torch.where(ref_z_mask, ones, zeros)

    # e1 = normalize(v x ref)
    e1x = vy * ref_z - vz * ref_y
    e1y = vz * ref_x - vx * ref_z
    e1z = vx * ref_y - vy * ref_x

    norm_e1 = torch.sqrt(e1x * e1x + e1y * e1y + e1z * e1z)
    safe_e1 = norm_e1 > 0
    safe_norm_e1 = torch.where(safe_e1, norm_e1, ones)
    e1x = torch.where(safe_e1, e1x / safe_norm_e1, ones)
    e1y = torch.where(safe_e1, e1y / safe_norm_e1, zeros)
    e1z = torch.where(safe_e1, e1z / safe_norm_e1, zeros)

    # e2 = normalize(v x e1)
    e2x = vy * e1z - vz * e1y
    e2y = vz * e1x - vx * e1z
    e2z = vx * e1y - vy * e1x

    norm_e2 = torch.sqrt(e2x * e2x + e2y * e2y + e2z * e2z)
    safe_e2 = norm_e2 > 0
    safe_norm_e2 = torch.where(safe_e2, norm_e2, ones)
    e2x = torch.where(safe_e2, e2x / safe_norm_e2, zeros)
    e2y = torch.where(safe_e2, e2y / safe_norm_e2, ones)
    e2z = torch.where(safe_e2, e2z / safe_norm_e2, zeros)

    return e1x, e1y, e1z, e2x, e2y, e2z


def apply_angular_kick(
    px: torch.Tensor,
    py: torch.Tensor,
    pz: torch.Tensor,
    theta: torch.Tensor,
    phi: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Apply an exact polar deflection theta (rad) at azimuth phi in the plane
    orthogonal to (px, py, pz):

        p_new = cos(theta) * p_hat + sin(theta) * (cos(phi) * e1 + sin(phi) * e2)

    Calling with (theta1, phi) and (theta2, phi + pi) produces exact opposite
    transverse deflections for any shower inclination.
    """
    vx, vy, vz = norm_dir(px, py, pz)
    e1x, e1y, e1z, e2x, e2y, e2z = orthonormal_basis(vx, vy, vz)

    theta_t = torch.as_tensor(theta, device=vx.device, dtype=vx.dtype)
    phi_t = torch.as_tensor(phi, device=vx.device, dtype=vx.dtype)

    cos_t = torch.cos(theta_t)
    sin_t = torch.sin(theta_t)
    cos_p = torch.cos(phi_t)
    sin_p = torch.sin(phi_t)

    tx = cos_p * e1x + sin_p * e2x
    ty = cos_p * e1y + sin_p * e2y
    tz = cos_p * e1z + sin_p * e2z

    p_new_x = cos_t * vx + sin_t * tx
    p_new_y = cos_t * vy + sin_t * ty
    p_new_z = cos_t * vz + sin_t * tz
    return norm_dir(p_new_x, p_new_y, p_new_z)


# ---------------------------------------------------------------------------
# 3. Electromagnetic Bethe-Heitler Samplers
# ---------------------------------------------------------------------------

def sample_pair_production_u(
    n: int,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Sample fractional positron energy u in [0, 1] from complete-screening
    Bethe-Heitler pair production:

        p(u) proportional to u^2 + (1 - u)^2 + (2/3) * u * (1 - u)

    Clamped to [1e-4, 1 - 1e-4].
    """
    if n <= 0:
        return torch.empty(0, device=device, dtype=dtype)

    u_out = torch.empty(n, device=device, dtype=dtype)
    pending = torch.arange(n, device=device)

    while pending.numel() > 0:
        m = pending.numel()
        u_try = torch.rand(m, device=device, dtype=dtype)
        one_minus_u = 1.0 - u_try
        w = u_try * u_try + one_minus_u * one_minus_u + (2.0 / 3.0) * u_try * one_minus_u
        accept = torch.rand(m, device=device, dtype=dtype) <= w
        u_out[pending[accept]] = u_try[accept]
        pending = pending[~accept]

    return torch.clamp(u_out, min=1e-4, max=1.0 - 1e-4)


def sample_bremsstrahlung_k(
    n: int,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
    k_min: float = BREM_K_MIN,
) -> torch.Tensor:
    """Sample fractional hard bremsstrahlung photon energy k in [k_min, 1] from
    complete-screening Bethe-Heitler:

        p(k) proportional to (1 / k) * ((4/3) * (1 - k) + k^2)

    Uses proposal k = k_min ** (1 - r) with r ~ Uniform(0, 1) and envelope weight
    g(k) = ((4/3) * (1 - k) + k^2) / (4/3) <= 1.0. Clamped to [k_min, 1 - 1e-4].
    """
    if not (0.0 < k_min < 1.0):
        raise ValueError(f"k_min must be in (0, 1), got {k_min}")
    if n <= 0:
        return torch.empty(0, device=device, dtype=dtype)

    log_k_min = math.log(k_min)
    k_out = torch.empty(n, device=device, dtype=dtype)
    pending = torch.arange(n, device=device)

    while pending.numel() > 0:
        m = pending.numel()
        r = torch.rand(m, device=device, dtype=dtype)
        k_try = torch.exp((1.0 - r) * log_k_min)
        g = ((4.0 / 3.0) * (1.0 - k_try) + k_try * k_try) / (4.0 / 3.0)
        accept = torch.rand(m, device=device, dtype=dtype) <= g
        k_out[pending[accept]] = k_try[accept]
        pending = pending[~accept]

    return torch.clamp(k_out, min=k_min, max=max(k_min, 1.0 - 1e-4))


# ---------------------------------------------------------------------------
# 4. Hadronic Interaction & Pion Decay Kinematics
# ---------------------------------------------------------------------------

def sample_hadronic_interaction(
    pid_h: torch.Tensor,
    E_h: torch.Tensor,
    x_h: torch.Tensor,
    y_h: torch.Tensor,
    z_h: torch.Tensor,
    px_h: torch.Tensor,
    py_h: torch.Tensor,
    pz_h: torch.Tensor,
    gen_h: torch.Tensor,
    evt_h: torch.Tensor,
) -> list[torch.Tensor]:
    """Sample hadronic interaction secondaries (leading hadron + secondary pions).

    Returns a list of 10-column particle state tensors with layout:
        [PID, E, x, y, z, px, py, pz, gen, evt]
    """
    device = E_h.device
    dtype = E_h.dtype
    n = E_h.shape[0]
    if n == 0:
        return []

    pid_f = pid_h.to(device=device, dtype=dtype)
    x_f = x_h.to(device=device, dtype=dtype)
    y_f = y_h.to(device=device, dtype=dtype)
    z_f = z_h.to(device=device, dtype=dtype)
    px_f = px_h.to(device=device, dtype=dtype)
    py_f = py_h.to(device=device, dtype=dtype)
    pz_f = pz_h.to(device=device, dtype=dtype)
    gen_new = gen_h.to(device=device, dtype=dtype) + 1.0
    evt_f = evt_h.to(device=device, dtype=dtype)

    out: list[torch.Tensor] = []

    # 1. Leading hadron: inelasticity K ~ Uniform(0.3, 0.7)
    inelasticity = 0.3 + 0.4 * torch.rand(n, device=device, dtype=dtype)
    E_lead = (1.0 - inelasticity) * E_h
    pT_lead = torch.empty(n, device=device, dtype=dtype).exponential_() * 0.35
    theta_lead = torch.clamp(pT_lead / torch.clamp(E_lead, min=0.2), max=0.5)
    phi_lead = torch.rand(n, device=device, dtype=dtype) * (2.0 * math.pi)
    px_l, py_l, pz_l = apply_angular_kick(px_f, py_f, pz_f, theta_lead, phi_lead)

    lead = torch.stack(
        [pid_f, E_lead, x_f, y_f, z_f, px_l, py_l, pz_l, gen_new, evt_f],
        dim=1,
    )
    out.append(lead)

    # 2. Secondary pions sharing E_avail = K * E_h
    E_avail = inelasticity * E_h
    n_pi = torch.clamp(
        torch.round(2.5 + 1.8 * torch.log(torch.clamp(E_h, min=1.0))),
        min=3.0,
        max=12.0,
    ).to(torch.int64)

    max_pi = 12
    slot_idx = torch.arange(max_pi, device=device).unsqueeze(0)  # (1, 12)
    slot_active = slot_idx < n_pi.unsqueeze(1)  # (n, 12)

    weights = torch.empty((n, max_pi), device=device, dtype=dtype).exponential_()
    weights = torch.where(
        slot_active,
        torch.clamp(weights, min=1e-12),
        torch.zeros_like(weights),
    )
    weight_sum = torch.clamp(weights.sum(dim=1, keepdim=True), min=1e-12)
    E_pions = E_avail.unsqueeze(1) * (weights / weight_sum)

    for i in range(max_pi):
        active_i = slot_active[:, i]
        if not active_i.any():
            continue

        E_c = E_pions[active_i, i]
        m = E_c.shape[0]

        # Isospin: pi0 (pid=5) with prob 1/3, pi+- (pid=4) with prob 2/3
        is_pi0 = torch.rand(m, device=device, dtype=dtype) < (1.0 / 3.0)
        c_pid = torch.where(
            is_pi0,
            torch.full_like(E_c, 5.0),
            torch.full_like(E_c, 4.0),
        )

        pT_c = torch.empty(m, device=device, dtype=dtype).exponential_() * 0.35
        theta_c = torch.clamp(pT_c / torch.clamp(E_c, min=0.2), max=0.8)
        phi_c = torch.rand(m, device=device, dtype=dtype) * (2.0 * math.pi)
        px_c, py_c, pz_c = apply_angular_kick(
            px_f[active_i], py_f[active_i], pz_f[active_i], theta_c, phi_c
        )

        child = torch.stack(
            [
                c_pid,
                E_c,
                x_f[active_i],
                y_f[active_i],
                z_f[active_i],
                px_c,
                py_c,
                pz_c,
                gen_new[active_i],
                evt_f[active_i],
            ],
            dim=1,
        )
        out.append(child)

    return out


def sample_pion_decay_to_muon(
    E_pi: torch.Tensor,
    x_pi: torch.Tensor,
    y_pi: torch.Tensor,
    z_pi: torch.Tensor,
    px_pi: torch.Tensor,
    py_pi: torch.Tensor,
    pz_pi: torch.Tensor,
    gen_pi: torch.Tensor,
    evt_pi: torch.Tensor,
) -> torch.Tensor:
    """Sample exact ultra-relativistic two-body decay pi+- -> mu+- + nu_mu.

    Returns a 10-column muon state tensor with pid = 6.0 and gen = gen_pi + 1.0.
    """
    device = E_pi.device
    dtype = E_pi.dtype
    n = E_pi.shape[0]
    if n == 0:
        return torch.empty((0, 10), device=device, dtype=dtype)

    x_f = x_pi.to(device=device, dtype=dtype)
    y_f = y_pi.to(device=device, dtype=dtype)
    z_f = z_pi.to(device=device, dtype=dtype)
    px_f = px_pi.to(device=device, dtype=dtype)
    py_f = py_pi.to(device=device, dtype=dtype)
    pz_f = pz_pi.to(device=device, dtype=dtype)
    gen_new = gen_pi.to(device=device, dtype=dtype) + 1.0
    evt_f = evt_pi.to(device=device, dtype=dtype)

    r = (M_MUON_GEV / M_PION_CHARGED_GEV) ** 2
    frac_mu = r + (1.0 - r) * torch.rand(n, device=device, dtype=dtype)
    E_mu = frac_mu * E_pi

    # Exact lab transverse momentum from isotropic rest-frame decay:
    # p_t_mu = 0.5 * m_pi * (1 - r) * sin(theta*)
    #        = m_pi * sqrt((frac_mu - r) * (1 - frac_mu))
    p_t_mu = M_PION_CHARGED_GEV * torch.sqrt(
        torch.clamp((frac_mu - r) * (1.0 - frac_mu), min=0.0)
    )
    theta_mu = torch.clamp(p_t_mu / torch.clamp(E_mu, min=0.11), max=0.5)
    phi_mu = torch.rand(n, device=device, dtype=dtype) * (2.0 * math.pi)
    px_mu, py_mu, pz_mu = apply_angular_kick(px_f, py_f, pz_f, theta_mu, phi_mu)

    return torch.stack(
        [
            torch.full_like(E_mu, 6.0),
            E_mu,
            x_f,
            y_f,
            z_f,
            px_mu,
            py_mu,
            pz_mu,
            gen_new,
            evt_f,
        ],
        dim=1,
    )

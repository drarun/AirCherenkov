import math

import numpy as np
import pytest
import torch

from sim.interactions import (
    BREM_K_MIN,
    BREM_LAMBDA_GCM2,
    BREM_RATE_PER_X0,
    BREM_SOFT_LOSS_PER_X0,
    C_TAU_MUON_M,
    C_TAU_PION_M,
    LAMBDA_PION_GCM2,
    LAMBDA_PROTON_GCM2,
    M_ELECTRON_GEV,
    M_MUON_GEV,
    M_PION_CHARGED_GEV,
    M_PION_ZERO_GEV,
    X0_AIR,
    apply_angular_kick,
    norm_dir,
    orthonormal_basis,
    sample_bremsstrahlung_k,
    sample_hadronic_interaction,
    sample_pair_production_u,
    sample_pion_decay_to_muon,
)


def _available_devices():
    devices = [torch.device("cpu")]
    if torch.cuda.is_available():
        devices.append(torch.device("cuda"))
    return devices


def test_physical_constants_and_radiative_stopping_power_identity():
    assert X0_AIR == pytest.approx(36.62)
    assert M_ELECTRON_GEV == pytest.approx(0.000511)
    assert M_MUON_GEV == pytest.approx(0.105658)
    assert M_PION_CHARGED_GEV == pytest.approx(0.139570)
    assert M_PION_ZERO_GEV == pytest.approx(0.134977)
    assert C_TAU_PION_M == pytest.approx(7.8045)
    assert C_TAU_MUON_M == pytest.approx(658.65)
    assert BREM_K_MIN == pytest.approx(0.15)
    assert LAMBDA_PROTON_GCM2 == pytest.approx(90.0)
    assert LAMBDA_PION_GCM2 == pytest.approx(120.0)
    assert BREM_LAMBDA_GCM2 == pytest.approx(X0_AIR / BREM_RATE_PER_X0)

    # Analytical hard stopping power: int_{k_min}^1 ((4/3)*(1-k) + k^2) dk
    k0 = BREM_K_MIN
    hard_stopping_per_x0 = (
        (4.0 / 3.0) * (1.0 - k0)
        - (2.0 / 3.0) * (1.0 - k0 ** 2)
        + (1.0 / 3.0) * (1.0 - k0 ** 3)
    )
    assert hard_stopping_per_x0 + BREM_SOFT_LOSS_PER_X0 == pytest.approx(1.0, abs=1e-12)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_orthonormal_basis_arbitrary_inclinations(dtype):
    # Vertical down, vertical up, near-vertical (|pz| >= 0.99), 30 deg, 60 deg, horizontal
    angles_deg = [0.0, 5.0, 30.0, 60.0, 90.0, 180.0]
    azimuths_deg = [0.0, 37.0, 125.0, 240.0, 315.0, 0.0]

    px_list, py_list, pz_list = [], [], []
    for zen_deg, az_deg in zip(angles_deg, azimuths_deg):
        zen = math.radians(zen_deg)
        az = math.radians(az_deg)
        px_list.append(math.sin(zen) * math.cos(az))
        py_list.append(math.sin(zen) * math.sin(az))
        pz_list.append(-math.cos(zen))

    px = torch.tensor(px_list, dtype=dtype)
    py = torch.tensor(py_list, dtype=dtype)
    pz = torch.tensor(pz_list, dtype=dtype)

    e1x, e1y, e1z, e2x, e2y, e2z = orthonormal_basis(px, py, pz)

    assert e1x.dtype == dtype
    assert e2z.dtype == dtype

    vx, vy, vz = norm_dir(px, py, pz)
    norm_e1 = torch.sqrt(e1x**2 + e1y**2 + e1z**2)
    norm_e2 = torch.sqrt(e2x**2 + e2y**2 + e2z**2)
    dot_v_e1 = vx * e1x + vy * e1y + vz * e1z
    dot_v_e2 = vx * e2x + vy * e2y + vz * e2z
    dot_e1_e2 = e1x * e2x + e1y * e2y + e1z * e2z

    tol = 1e-6 if dtype == torch.float32 else 1e-12
    np.testing.assert_allclose(norm_e1.numpy(), np.ones_like(px_list), atol=tol)
    np.testing.assert_allclose(norm_e2.numpy(), np.ones_like(px_list), atol=tol)
    np.testing.assert_allclose(dot_v_e1.numpy(), np.zeros_like(px_list), atol=tol)
    np.testing.assert_allclose(dot_v_e2.numpy(), np.zeros_like(px_list), atol=tol)
    np.testing.assert_allclose(dot_e1_e2.numpy(), np.zeros_like(px_list), atol=tol)


def test_apply_angular_kick_exact_angle_and_back_to_back():
    # Sweep zenith and azimuth angles across the full sphere
    zeniths = [0.0, 5.0, 30.0, 45.0, 60.0, 90.0]
    azimuths = [0.0, 42.0, 110.0, 195.0, 270.0, 350.0]
    thetas = [0.001, 0.02, 0.15, 0.35, 0.6, 1.0]
    phis = [0.3, 1.1, 2.5, 3.8, 4.9, 5.7]

    px = torch.tensor(
        [math.sin(math.radians(z)) * math.cos(math.radians(a)) for z, a in zip(zeniths, azimuths)],
        dtype=torch.float64,
    )
    py = torch.tensor(
        [math.sin(math.radians(z)) * math.sin(math.radians(a)) for z, a in zip(zeniths, azimuths)],
        dtype=torch.float64,
    )
    pz = torch.tensor(
        [-math.cos(math.radians(z)) for z in zeniths],
        dtype=torch.float64,
    )
    theta = torch.tensor(thetas, dtype=torch.float64)
    phi = torch.tensor(phis, dtype=torch.float64)

    nx, ny, nz = apply_angular_kick(px, py, pz, theta, phi)

    # Unit vectors
    norms = torch.sqrt(nx**2 + ny**2 + nz**2)
    np.testing.assert_allclose(norms.numpy(), np.ones(len(thetas)), atol=1e-12)

    # Exact deflection angle arccos(p_old . p_new) == theta independent of shower orientation
    dots = torch.clamp(px * nx + py * ny + pz * nz, -1.0, 1.0)
    recovered_theta = torch.acos(dots)
    np.testing.assert_allclose(recovered_theta.numpy(), theta.numpy(), rtol=1e-10, atol=1e-12)

    # Back-to-back kicks: (theta1, phi) and (theta2, phi + pi) have opposite transverse parts
    theta2 = torch.tensor([0.002, 0.05, 0.25, 0.1, 0.4, 0.7], dtype=torch.float64)
    nx2, ny2, nz2 = apply_angular_kick(px, py, pz, theta2, phi + math.pi)

    # Remove longitudinal projection along p_old to isolate transverse vectors
    p1_dot_v = nx * px + ny * py + nz * pz
    t1x, t1y, t1z = nx - p1_dot_v * px, ny - p1_dot_v * py, nz - p1_dot_v * pz
    t1_norm = torch.sqrt(t1x**2 + t1y**2 + t1z**2)

    p2_dot_v = nx2 * px + ny2 * py + nz2 * pz
    t2x, t2y, t2z = nx2 - p2_dot_v * px, ny2 - p2_dot_v * py, nz2 - p2_dot_v * pz
    t2_norm = torch.sqrt(t2x**2 + t2y**2 + t2z**2)

    # Unit transverse directions must be exact negatives (dot == -1)
    t_cos = (t1x * t2x + t1y * t2y + t1z * t2z) / (t1_norm * t2_norm)
    np.testing.assert_allclose(t_cos.numpy(), -np.ones(len(thetas)), atol=1e-12)


def test_sample_pair_production_u_moments_and_symmetry():
    torch.manual_seed(12345)
    n = 200_000
    u = sample_pair_production_u(n, device="cpu", dtype=torch.float32)

    assert u.shape == (n,)
    assert u.dtype == torch.float32
    assert u.min().item() >= 1e-4 - 1e-7
    assert u.max().item() <= 1.0 - 1e-4 + 1e-7

    # Analytical Bethe-Heitler moments for p(u) ~ u^2 + (1-u)^2 + (2/3)*u*(1-u):
    # int_0^1 p(u) du = 7/9
    # int_0^1 (u - 0.5)^2 p(u) du = 13/180
    # Var(u) = (13/180) / (7/9) = 13/140 ~= 0.0928571
    analytical_var = 13.0 / 140.0
    uniform_04_06_var = (0.2 ** 2) / 12.0  # ~0.00333

    mean_u = u.mean().item()
    var_u = u.var(unbiased=True).item()

    assert mean_u == pytest.approx(0.50, abs=3e-3)
    assert var_u == pytest.approx(analytical_var, rel=0.02)
    assert var_u > 10.0 * uniform_04_06_var

    # Symmetry around 0.5: fraction below 0.2 equals fraction above 0.8, median ~0.5
    frac_low = (u < 0.2).float().mean().item()
    frac_high = (u > 0.8).float().mean().item()
    assert frac_low == pytest.approx(frac_high, abs=5e-3)
    assert (u < 0.5).float().mean().item() == pytest.approx(0.5, abs=5e-3)


def test_sample_bremsstrahlung_k_distribution_and_stopping_power():
    torch.manual_seed(54321)
    n = 200_000
    k = sample_bremsstrahlung_k(n, device="cpu", dtype=torch.float32, k_min=BREM_K_MIN)

    assert k.shape == (n,)
    assert k.dtype == torch.float32
    assert k.min().item() >= BREM_K_MIN - 1e-7
    assert k.max().item() <= 1.0 - 1e-4 + 1e-7

    # Analytical <k> = int_{k_min}^1 ((4/3)*(1-k) + k^2) dk / BREM_RATE_PER_X0
    k0 = BREM_K_MIN
    numerator = (
        (4.0 / 3.0) * (1.0 - k0)
        - (2.0 / 3.0) * (1.0 - k0 ** 2)
        + (1.0 / 3.0) * (1.0 - k0 ** 3)
    )
    analytical_mean_k = numerator / BREM_RATE_PER_X0

    sample_mean_k = k.mean().item()
    assert sample_mean_k == pytest.approx(analytical_mean_k, rel=0.01)

    # Exact total radiative stopping power: hard + soft == 1.0 per X0
    total_stopping_per_x0 = BREM_RATE_PER_X0 * sample_mean_k + BREM_SOFT_LOSS_PER_X0
    assert total_stopping_per_x0 == pytest.approx(1.0, abs=0.01)


def test_sample_hadronic_interaction_conservation_isospin_multiplicity():
    torch.manual_seed(2026)
    n = 2000

    # Mix protons (pid=3) and charged pions (pid=4), and mix energies (20 GeV & 1000 GeV)
    pid_h = torch.where(
        torch.arange(n) % 2 == 0,
        torch.full((n,), 3.0),
        torch.full((n,), 4.0),
    )
    E_h = torch.where(
        torch.arange(n) < n // 2,
        torch.full((n,), 20.0),
        torch.full((n,), 1000.0),
    )
    x_h = torch.zeros(n)
    y_h = torch.zeros(n)
    z_h = torch.full((n,), 15000.0)
    # Inclined shower at 30 deg
    px_h = torch.full((n,), math.sin(math.radians(30.0)))
    py_h = torch.zeros(n)
    pz_h = torch.full((n,), -math.cos(math.radians(30.0)))
    gen_h = torch.full((n,), 2.0)
    evt_h = torch.arange(n, dtype=torch.float32)

    children_list = sample_hadronic_interaction(
        pid_h, E_h, x_h, y_h, z_h, px_h, py_h, pz_h, gen_h, evt_h
    )
    assert isinstance(children_list, list)
    assert len(children_list) > 1

    # First entry is leading hadron
    lead = children_list[0]
    assert lead.shape == (n, 10)
    # Leading particle keeps original PID (proton->3, pi±->4)
    torch.testing.assert_close(lead[:, 0], pid_h)
    # Leading particle energy fraction (1 - K) in [0.3, 0.7]
    lead_frac = lead[:, 1] / E_h
    assert lead_frac.min().item() >= 0.3 - 1e-6
    assert lead_frac.max().item() <= 0.7 + 1e-6

    all_children = torch.cat(children_list, dim=0)
    assert all_children.shape[1] == 10
    assert torch.all(all_children[:, 8] == 3.0)

    # Exact total energy conservation per parent event: E_lead + sum(E_pions) == E_h
    evt_idx = all_children[:, 9].to(torch.int64)
    total_E_per_parent = torch.zeros(n, dtype=E_h.dtype)
    total_E_per_parent.scatter_add_(0, evt_idx, all_children[:, 1])
    rel_err = torch.abs(total_E_per_parent - E_h) / E_h
    assert rel_err.max().item() < 1e-4

    # All direction vectors are unit length
    dir_norms = torch.sqrt(
        all_children[:, 5] ** 2 + all_children[:, 6] ** 2 + all_children[:, 7] ** 2
    )
    np.testing.assert_allclose(dir_norms.numpy(), np.ones(all_children.shape[0]), atol=1e-5)

    # Secondary pions: isospin fraction pi0 (pid=5) ~= 1/3, pi_charged (pid=4) ~= 2/3
    secondaries = torch.cat(children_list[1:], dim=0)
    sec_pids = secondaries[:, 0]
    assert torch.all((sec_pids == 4.0) | (sec_pids == 5.0))
    pi0_frac = (sec_pids == 5.0).float().mean().item()
    assert abs(pi0_frac - (1.0 / 3.0)) < 0.03

    # Multiplicity grows with energy: 1000 GeV > 20 GeV
    sec_evt_idx = secondaries[:, 9].to(torch.int64)
    counts_per_parent = torch.bincount(sec_evt_idx, minlength=n).float()
    mult_20 = counts_per_parent[: n // 2].mean().item()
    mult_1000 = counts_per_parent[n // 2 :].mean().item()
    assert mult_1000 > mult_20
    assert mult_20 == pytest.approx(8.0)
    assert mult_1000 == pytest.approx(12.0)


def test_sample_pion_decay_to_muon_kinematics():
    torch.manual_seed(777)
    n = 50_000
    E_val = 20.0
    E_pi = torch.full((n,), E_val, dtype=torch.float32)
    x_pi = torch.zeros(n)
    y_pi = torch.zeros(n)
    z_pi = torch.full((n,), 10000.0)
    # 45-deg inclined pion beam
    px_pi = torch.full((n,), math.sin(math.radians(45.0)))
    py_pi = torch.zeros(n)
    pz_pi = torch.full((n,), -math.cos(math.radians(45.0)))
    gen_pi = torch.full((n,), 4.0)
    evt_pi = torch.arange(n, dtype=torch.float32)

    muons = sample_pion_decay_to_muon(
        E_pi, x_pi, y_pi, z_pi, px_pi, py_pi, pz_pi, gen_pi, evt_pi
    )

    assert muons.shape == (n, 10)
    assert muons.dtype == torch.float32
    assert torch.all(muons[:, 0] == 6.0)
    assert torch.all(muons[:, 8] == 5.0)
    torch.testing.assert_close(muons[:, 9], evt_pi)

    r = (M_MUON_GEV / M_PION_CHARGED_GEV) ** 2
    frac_mu = muons[:, 1] / E_pi
    assert frac_mu.min().item() >= 0.573
    assert frac_mu.max().item() <= 1.0
    assert frac_mu.mean().item() == pytest.approx(0.5 * (1.0 + r), abs=2e-3)

    # Direction vectors are unit length
    norms = torch.sqrt(muons[:, 5] ** 2 + muons[:, 6] ** 2 + muons[:, 7] ** 2)
    np.testing.assert_allclose(norms.numpy(), np.ones(n), atol=1e-5)

    # Opening angle small, positive, finite, and scales as ~p_t / E_mu (~m_pi*(1-r)/(2*E_mu))
    dots = torch.clamp(
        muons[:, 5] * px_pi + muons[:, 6] * py_pi + muons[:, 7] * pz_pi, -1.0, 1.0
    )
    theta_mu = torch.acos(dots)
    assert torch.all(torch.isfinite(theta_mu))
    assert theta_mu.mean().item() > 0.0
    # Max rest-frame momentum p* = 0.5 * m_pi * (1 - r) ~= 0.0298 GeV
    # At E_pi = 20 GeV, E_mu >= 11.46 GeV, so theta_mu <= ~0.0026 rad
    max_expected_theta = (0.5 * M_PION_CHARGED_GEV * (1.0 - r)) / (r * E_val)
    assert theta_mu.max().item() <= max_expected_theta * 1.05 + 1e-4
    # Typical opening angle order of magnitude ~ p* / <E_mu>
    mean_expected_order = (0.5 * M_PION_CHARGED_GEV * (1.0 - r)) / (0.5 * (1.0 + r) * E_val)
    assert 0.5 * mean_expected_order < theta_mu.mean().item() < 1.2 * mean_expected_order


@pytest.mark.parametrize("device", _available_devices())
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_device_dtype_preservation_seeding_and_empty_inputs(device, dtype):
    # Reproducibility with torch.manual_seed
    torch.manual_seed(99)
    u1 = sample_pair_production_u(100, device=device, dtype=dtype)
    k1 = sample_bremsstrahlung_k(100, device=device, dtype=dtype)
    torch.manual_seed(99)
    u2 = sample_pair_production_u(100, device=device, dtype=dtype)
    k2 = sample_bremsstrahlung_k(100, device=device, dtype=dtype)

    assert u1.device.type == device.type and u1.dtype == dtype
    assert k1.device.type == device.type and k1.dtype == dtype
    torch.testing.assert_close(u1, u2)
    torch.testing.assert_close(k1, k2)

    # Hadronic & pion decay preserve device & dtype
    n = 16
    pid_h = torch.full((n,), 3, device=device, dtype=torch.int32)
    E_h = torch.full((n,), 100.0, device=device, dtype=dtype)
    zeros = torch.zeros(n, device=device, dtype=dtype)
    z_h = torch.full((n,), 12000.0, device=device, dtype=dtype)
    pz_h = torch.full((n,), -1.0, device=device, dtype=dtype)
    evt_h = torch.arange(n, device=device, dtype=dtype)

    had_out = sample_hadronic_interaction(pid_h, E_h, zeros, zeros, z_h, zeros, zeros, pz_h, zeros, evt_h)
    assert all(t.device.type == device.type and t.dtype == dtype for t in had_out)

    mu_out = sample_pion_decay_to_muon(E_h, zeros, zeros, z_h, zeros, zeros, pz_h, zeros, evt_h)
    assert mu_out.device.type == device.type and mu_out.dtype == dtype

    # Empty inputs
    assert sample_pair_production_u(0, device=device, dtype=dtype).shape == (0,)
    assert sample_bremsstrahlung_k(0, device=device, dtype=dtype).shape == (0,)
    empty = torch.empty(0, device=device, dtype=dtype)
    assert sample_hadronic_interaction(
        empty, empty, empty, empty, empty, empty, empty, empty, empty, empty
    ) == []
    empty_mu = sample_pion_decay_to_muon(
        empty, empty, empty, empty, empty, empty, empty, empty, empty
    )
    assert empty_mu.shape == (0, 10)
    assert empty_mu.device.type == device.type and empty_mu.dtype == dtype

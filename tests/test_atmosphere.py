"""Unit tests for the 5-layer CORSIKA US Standard Atmosphere (sim.atmosphere)."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from sim.atmosphere import (
    ETA_SEA_LEVEL,
    RHO_SEA_LEVEL,
    SITE_ALTITUDES_M,
    SITE_MAGNETIC_FIELDS_T,
    altitude_from_grammage,
    arclength_from_grammage,
    atmospheric_density,
    atmospheric_density_np,
    atmospheric_transmission,
    atmospheric_transmission_np,
    optical_path_delay_m,
    refractive_index_excess,
    refractive_index_excess_np,
    vertical_grammage,
    vertical_grammage_np,
)


# ── Constants & observatory site verification ────────────────────────────────


def test_site_altitudes_and_geomagnetic_fields_z_up():
    assert SITE_ALTITUDES_M == {
        "VERITAS": 1268.0,
        "HESS": 1800.0,
        "MAGIC": 2200.0,
        "CTA_NORTH": 2200.0,
    }

    assert set(SITE_MAGNETIC_FIELDS_T.keys()) == {"VERITAS", "HESS", "MAGIC", "CTA_NORTH"}

    # Northern Hemisphere sites (Z-up: downward vertical component => Bz < 0)
    assert SITE_MAGNETIC_FIELDS_T["VERITAS"][2] < 0.0
    assert SITE_MAGNETIC_FIELDS_T["MAGIC"][2] < 0.0
    assert SITE_MAGNETIC_FIELDS_T["CTA_NORTH"][2] < 0.0

    # Southern Hemisphere site (Z-up: upward vertical component => Bz > 0)
    assert SITE_MAGNETIC_FIELDS_T["HESS"][2] > 0.0

    # Exact values
    assert SITE_MAGNETIC_FIELDS_T["VERITAS"] == pytest.approx((23.9e-6, 4.1e-6, -40.8e-6))
    assert SITE_MAGNETIC_FIELDS_T["MAGIC"] == pytest.approx((30.8e-6, -2.1e-6, -22.9e-6))
    assert SITE_MAGNETIC_FIELDS_T["CTA_NORTH"] == pytest.approx((30.8e-6, -2.1e-6, -22.9e-6))
    assert SITE_MAGNETIC_FIELDS_T["HESS"] == pytest.approx((11.1e-6, -2.8e-6, 25.6e-6))

    assert RHO_SEA_LEVEL == pytest.approx(1.2298058e-3, rel=1e-6)
    assert ETA_SEA_LEVEL == pytest.approx(2.93e-4, rel=1e-12)


# ── 5-layer CORSIKA profile: vertical grammage & density ─────────────────────


def test_vertical_grammage_sea_level_and_layer_continuity():
    t0 = vertical_grammage(torch.tensor(0.0, dtype=torch.float64)).item()
    assert t0 == pytest.approx(1036.1, abs=1e-4)

    eps = 1e-4  # 0.1 mm on either side of each layer boundary
    for z_boundary in (4000.0, 10000.0, 40000.0, 100000.0):
        z_pair = torch.tensor([z_boundary - eps, z_boundary + eps], dtype=torch.float64)
        t_left, t_right = vertical_grammage(z_pair).tolist()
        assert abs(t_left - t_right) < 0.01, (
            f"Discontinuity at z={z_boundary} m: |{t_left} - {t_right}| = {abs(t_left - t_right)}"
        )

    # Top of atmosphere and above
    t_top = vertical_grammage(torch.tensor([112829.2, 120000.0, 200000.0], dtype=torch.float64))
    assert torch.all(t_top == 0.0)


def test_vertical_grammage_strictly_decreasing():
    z_grid = torch.linspace(0.0, 112829.2, 5000, dtype=torch.float64)
    t_grid = vertical_grammage(z_grid)
    diffs = t_grid[1:] - t_grid[:-1]
    assert torch.all(diffs < 0.0)


def test_atmospheric_density_matches_negative_grammage_derivative():
    # Interior points of all 5 layers
    z_mid = torch.tensor([1500.0, 7000.0, 25000.0, 65000.0, 105000.0], dtype=torch.float64)
    dz = 1.0  # 1 meter = 100 cm
    t_minus = vertical_grammage(z_mid - dz)
    t_plus = vertical_grammage(z_mid + dz)
    rho_fd = -(t_plus - t_minus) / (2.0 * dz * 100.0)
    rho_analytic = atmospheric_density(z_mid)
    torch.testing.assert_close(rho_analytic, rho_fd, rtol=1e-5, atol=0.0)

    # Sea level, Layer 5 constant density, and above TOA
    assert atmospheric_density(torch.tensor(0.0, dtype=torch.float64)).item() == pytest.approx(
        RHO_SEA_LEVEL, rel=1e-12
    )
    assert atmospheric_density(torch.tensor(105000.0, dtype=torch.float64)).item() == pytest.approx(
        1.0e-9, rel=1e-12
    )
    assert atmospheric_density(torch.tensor(112829.2, dtype=torch.float64)).item() == 0.0
    assert atmospheric_density(torch.tensor(150000.0, dtype=torch.float64)).item() == 0.0


# ── Exact grammage inversion & slant arclength ───────────────────────────────


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_altitude_from_grammage_round_trip_and_bounds(dtype):
    z_samples = torch.tensor(
        [0.0, 1268.0, 3900.0, 4100.0, 9900.0, 10100.0, 25000.0, 39900.0, 40100.0, 80000.0, 105000.0],
        dtype=dtype,
    )
    t_samples = vertical_grammage(z_samples)
    z_rec = altitude_from_grammage(t_samples)
    assert z_rec.dtype == dtype
    rtol = 1e-4 if dtype == torch.float32 else 1e-10
    atol = 1.0 if dtype == torch.float32 else 1e-6
    torch.testing.assert_close(z_rec, z_samples, rtol=rtol, atol=atol)

    # Non-positive grammage maps cleanly to top of atmosphere
    t_zero_neg = torch.tensor([-10.0, 0.0], dtype=dtype)
    z_top = altitude_from_grammage(t_zero_neg)
    torch.testing.assert_close(z_top, torch.full_like(z_top, 112829.2), rtol=1e-6, atol=1e-2)

    # t > T(0) extrapolates smoothly below 0 m using Layer 1
    z_below = altitude_from_grammage(torch.tensor([1100.0, 1300.0], dtype=dtype))
    assert torch.all(z_below < 0.0)
    assert z_below[1] < z_below[0]

    # Global monotonic non-increasing across dense grammage sweep
    t_sweep = torch.cat([
        torch.linspace(0.0, 0.002, 50, dtype=torch.float64),
        torch.linspace(0.002, 5.0, 100, dtype=torch.float64),
        torch.linspace(5.0, 1200.0, 500, dtype=torch.float64),
    ])
    z_sweep = altitude_from_grammage(t_sweep)
    assert torch.all(z_sweep[1:] <= z_sweep[:-1] + 1e-12)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_arclength_from_grammage_round_trip_across_layer_boundaries(dtype):
    z_starts = [50000.0, 25000.0, 12000.0, 6000.0]
    pzs = [-1.0, -0.866, -0.5]
    dxs = [50.0, 200.0, 500.0]

    z_list, pz_list, dx_list = [], [], []
    for z1 in z_starts:
        for pz in pzs:
            for dx in dxs:
                z_list.append(z1)
                pz_list.append(pz)
                dx_list.append(dx)

    z1_t = torch.tensor(z_list, dtype=dtype)
    pz_t = torch.tensor(pz_list, dtype=dtype)
    dx_t = torch.tensor(dx_list, dtype=dtype)

    dist = arclength_from_grammage(z1_t, pz_t, dx_t)
    assert dist.dtype == dtype
    assert torch.all(torch.isfinite(dist))
    assert torch.all(dist > 0.0)

    z2_t = z1_t + pz_t * dist
    dx_recovered = (vertical_grammage(z2_t) - vertical_grammage(z1_t)) / (-pz_t)
    torch.testing.assert_close(dx_recovered, dx_t, rtol=1e-4, atol=1e-3)


def test_arclength_horizontal_upward_and_edge_cases():
    # Horizontal ray (pz = 0) returns positive finite dx / (rho * 100)
    z_h = torch.tensor([0.0, 4000.0, 10000.0, 30000.0, 80000.0], dtype=torch.float64)
    pz_h = torch.zeros_like(z_h)
    dx_h = torch.full_like(z_h, 36.62)
    dist_h = arclength_from_grammage(z_h, pz_h, dx_h)
    expected_h = dx_h / (atmospheric_density(z_h) * 100.0)
    assert torch.all(torch.isfinite(dist_h))
    assert torch.all(dist_h > 0.0)
    torch.testing.assert_close(dist_h, expected_h, rtol=1e-12, atol=0.0)

    # Upward ray within atmosphere satisfies exact round-trip
    z_up = torch.tensor([5000.0, 15000.0], dtype=torch.float64)
    pz_up = torch.tensor([0.5, 1.0], dtype=torch.float64)
    dx_up = torch.tensor([100.0, 50.0], dtype=torch.float64)
    dist_up = arclength_from_grammage(z_up, pz_up, dx_up)
    z2_up = z_up + pz_up * dist_up
    dx_up_rec = (vertical_grammage(z_up) - vertical_grammage(z2_up)) / pz_up
    torch.testing.assert_close(dx_up_rec, dx_up, rtol=1e-6, atol=0.0)

    # Upward ray escaping top of atmosphere terminates cleanly at 112829.2 m
    dist_esc = arclength_from_grammage(
        torch.tensor(50000.0, dtype=torch.float64),
        torch.tensor(0.5, dtype=torch.float64),
        torch.tensor(1000.0, dtype=torch.float64),
    )
    assert dist_esc.item() == pytest.approx((112829.2 - 50000.0) / 0.5, rel=1e-12)

    # Downward ray stepping past sea level smoothly enters z < 0 and recovers dx
    z_low = torch.tensor(2000.0, dtype=torch.float64)
    pz_low = torch.tensor(-1.0, dtype=torch.float64)
    dx_low = torch.tensor(500.0, dtype=torch.float64)
    dist_low = arclength_from_grammage(z_low, pz_low, dx_low)
    z2_low = z_low + pz_low * dist_low
    assert z2_low.item() < 0.0
    dx_low_rec = (vertical_grammage(z2_low) - vertical_grammage(z_low)) / (-pz_low)
    assert dx_low_rec.item() == pytest.approx(500.0, rel=1e-10)

    # Zero delta_x gives zero distance
    dist_zero = arclength_from_grammage(
        torch.tensor([0.0, 10000.0, 50000.0], dtype=torch.float64),
        torch.tensor([-1.0, -0.5, 0.0], dtype=torch.float64),
        torch.zeros(3, dtype=torch.float64),
    )
    assert torch.all(dist_zero == 0.0)

    # Tiny downward step right at the 10000 m layer boundary stays positive and finite
    dist_seam = arclength_from_grammage(
        torch.tensor(10000.0, dtype=torch.float64),
        torch.tensor(-1.0e-5, dtype=torch.float64),
        torch.tensor(1.0, dtype=torch.float64),
    )
    assert torch.isfinite(dist_seam) and dist_seam.item() > 0.0


# ── Refractive index & integrated optical path delay ─────────────────────────


def test_refractive_index_and_optical_path_delay_quadrature():
    assert refractive_index_excess(torch.tensor(0.0, dtype=torch.float64)).item() == pytest.approx(
        ETA_SEA_LEVEL, rel=1e-12
    )
    assert refractive_index_excess(torch.tensor(120000.0, dtype=torch.float64)).item() == 0.0

    # Total vertical refractive delay from TOA to sea level (~2.4685 m / ~8.23 ns)
    delay_full = optical_path_delay_m(
        torch.tensor(112829.2, dtype=torch.float64),
        torch.tensor(0.0, dtype=torch.float64),
        torch.tensor(1.0, dtype=torch.float64),
    ).item()
    assert delay_full == pytest.approx(2.4685, rel=1e-3)

    # Compare analytic optical_path_delay_m against dense trapezoidal quadrature
    cases = [
        (0.0, 10000.0, 1.0),
        (1268.0, 25000.0, 0.8660254),
        (2200.0, 50000.0, 0.5),
        (3000.0, 110000.0, 0.7071068),
    ]
    for z_obs, z_emit, cos_z in cases:
        z_quad = torch.linspace(z_obs, z_emit, 20001, dtype=torch.float64)
        integrand = refractive_index_excess(z_quad) / cos_z
        quad_delay = torch.trapezoid(integrand, z_quad).item()

        analytic_delay = optical_path_delay_m(
            torch.tensor(z_emit, dtype=torch.float64),
            torch.tensor(z_obs, dtype=torch.float64),
            torch.tensor(cos_z, dtype=torch.float64),
        ).item()
        assert analytic_delay == pytest.approx(quad_delay, rel=5e-3)  # < 0.5%

    # Zero delay when z_emit <= z_obs
    delay_zero = optical_path_delay_m(
        torch.tensor([1000.0, 2200.0], dtype=torch.float64),
        torch.tensor([2200.0, 2200.0], dtype=torch.float64),
        torch.tensor([1.0, 0.5], dtype=torch.float64),
    )
    assert torch.all(delay_zero == 0.0)


# ── Atmospheric transmission ─────────────────────────────────────────────────


def test_atmospheric_transmission_physics_and_monotonicity():
    # Equal or below observer altitude returns exactly 1.0
    t_below = atmospheric_transmission(
        torch.tensor([500.0, 2200.0, 0.0], dtype=torch.float64),
        torch.tensor([2200.0, 2200.0, 1268.0], dtype=torch.float64),
        torch.tensor([1.0, 0.5, 0.8], dtype=torch.float64),
    )
    torch.testing.assert_close(t_below, torch.ones_like(t_below), rtol=0.0, atol=0.0)

    # Strictly in (0, 1) for physical Cherenkov emission above observer
    z_emit = torch.tensor([3000.0, 8000.0, 15000.0, 25000.0, 40000.0], dtype=torch.float64)
    z_obs = torch.full_like(z_emit, SITE_ALTITUDES_M["VERITAS"])
    cos_z = torch.full_like(z_emit, 1.0)
    trans_h = atmospheric_transmission(z_emit, z_obs, cos_z)
    assert torch.all((trans_h > 0.0) & (trans_h < 1.0))

    # Monotonically decreases with emission height
    assert torch.all(trans_h[1:] < trans_h[:-1])

    # Monotonically decreases with zenith angle (decreasing cos_zenith)
    zen_deg = torch.tensor([0.0, 15.0, 30.0, 45.0, 60.0, 75.0], dtype=torch.float64)
    cos_zen = torch.cos(torch.deg2rad(zen_deg))
    trans_z = atmospheric_transmission(
        torch.full_like(cos_zen, 10000.0),
        torch.full_like(cos_zen, SITE_ALTITUDES_M["VERITAS"]),
        cos_zen,
    )
    assert torch.all((trans_z > 0.0) & (trans_z < 1.0))
    assert torch.all(trans_z[1:] < trans_z[:-1])

    # Higher transmission at MAGIC (2200 m) than sea level (0 m) for same emission height
    z_em_cmp = torch.tensor([5000.0, 10000.0, 25000.0], dtype=torch.float64)
    t_magic = atmospheric_transmission(z_em_cmp, SITE_ALTITUDES_M["MAGIC"], torch.ones_like(z_em_cmp))
    t_sea = atmospheric_transmission(z_em_cmp, 0.0, torch.ones_like(z_em_cmp))
    assert torch.all(t_magic > t_sea)


# ── Torch / NumPy parity & device/dtype/shape preservation ───────────────────


def test_torch_matches_numpy_equivalents():
    z_np = np.array([-200.0, 0.0, 1268.0, 4000.0, 7500.0, 10000.0, 22000.0, 40000.0, 75000.0, 105000.0, 120000.0])
    z_t = torch.tensor(z_np, dtype=torch.float64)

    np.testing.assert_allclose(atmospheric_density(z_t).numpy(), atmospheric_density_np(z_np), rtol=1e-12, atol=0.0)
    np.testing.assert_allclose(vertical_grammage(z_t).numpy(), vertical_grammage_np(z_np), rtol=1e-12, atol=0.0)
    np.testing.assert_allclose(
        refractive_index_excess(z_t).numpy(),
        refractive_index_excess_np(z_np),
        rtol=1e-12,
        atol=0.0,
    )

    z_emit_np = np.array([1000.0, 2200.0, 5000.0, 12000.0, 25000.0, 45000.0])
    z_obs_np = np.array([2200.0, 2200.0, 1268.0, 0.0, 1800.0, 2200.0])
    cos_z_np = np.array([1.0, 0.8, 0.95, 0.7, 0.5, 0.3])

    t_torch = atmospheric_transmission(
        torch.tensor(z_emit_np, dtype=torch.float32),
        torch.tensor(z_obs_np, dtype=torch.float32),
        torch.tensor(cos_z_np, dtype=torch.float32),
    ).numpy()
    t_numpy = atmospheric_transmission_np(z_emit_np, z_obs_np, cos_z_np)
    np.testing.assert_allclose(t_torch, t_numpy, rtol=1e-4, atol=0.0)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_nd_empty_and_device_preservation(dtype):
    devices = [torch.device("cpu")]
    if torch.cuda.is_available():
        devices.append(torch.device("cuda"))

    for device in devices:
        # 2D tensor preservation
        z2d = torch.tensor([[0.0, 5000.0], [15000.0, 50000.0]], dtype=dtype, device=device)
        rho = atmospheric_density(z2d)
        gram = vertical_grammage(z2d)
        z_inv = altitude_from_grammage(gram)
        eta = refractive_index_excess(z2d)
        dist = arclength_from_grammage(z2d, torch.full_like(z2d, -0.8), torch.full_like(z2d, 50.0))
        delay = optical_path_delay_m(z2d, 1268.0, torch.full_like(z2d, 0.9))
        trans = atmospheric_transmission(z2d, 1268.0, torch.full_like(z2d, 0.9))

        for out in (rho, gram, z_inv, eta, dist, delay, trans):
            assert out.shape == z2d.shape
            assert out.dtype == dtype
            assert out.device.type == device.type
            assert torch.all(torch.isfinite(out))

        # Empty tensor handling
        empty = torch.empty(0, dtype=dtype, device=device)
        assert atmospheric_density(empty).shape == (0,)
        assert vertical_grammage(empty).shape == (0,)
        assert altitude_from_grammage(empty).shape == (0,)
        assert arclength_from_grammage(empty, empty, empty).shape == (0,)
        assert refractive_index_excess(empty).shape == (0,)
        assert optical_path_delay_m(empty, 1268.0, empty).shape == (0,)
        assert atmospheric_transmission(empty, 1268.0, empty).shape == (0,)

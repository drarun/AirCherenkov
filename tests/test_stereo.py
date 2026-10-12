import numpy as np
import pytest
from sim.camera import Camera
from sim.telescope import TelescopeArray
from recon.stereo import cam_to_sky_vector, tailcut_clean, reconstruct_event_geometric
from recon.ml_stereo import extract_stereo_features, MLStereoReconstructor


class TestCamToSkyVector:
    """Round-trip and analytical tests for camera-to-sky coordinate conversion."""

    def test_optical_axis_zenith_zero(self):
        """Pointing straight up (zenith=0, az=0): camera center maps to (0,0,1)."""
        v = cam_to_sky_vector(0.0, 0.0, 0.0, 0.0)
        assert np.allclose(v, [0.0, 0.0, 1.0], atol=1e-10)

    def test_optical_axis_zenith_30(self):
        """Pointing at 30 deg zenith, 0 az: camera center maps to (sin30, 0, cos30)."""
        v30 = cam_to_sky_vector(0.0, 0.0, np.radians(30.0), 0.0)
        expected = [np.sin(np.radians(30)), 0.0, np.cos(np.radians(30))]
        assert np.allclose(v30, expected, atol=1e-10)

    def test_optical_axis_zenith_30_az_180(self):
        """Pointing at zen=30, az=180: camera center maps to (-sin30, 0, cos30)."""
        v = cam_to_sky_vector(0.0, 0.0, np.radians(30.0), np.radians(180.0))
        expected = [-np.sin(np.radians(30)), 0.0, np.cos(np.radians(30))]
        assert np.allclose(v, expected, atol=1e-10)

    def test_round_trip_with_backend(self):
        """Verify cam_to_sky_vector is the exact inverse of the backend projection."""
        import math

        pointing_zenith = 30.0
        pointing_azimuth = 180.0
        theta = math.radians(pointing_zenith)
        phi = math.radians(pointing_azimuth)

        # Source direction in ground frame
        source = np.array([
            np.sin(np.radians(29.3)) * np.cos(np.radians(5.2)),
            np.sin(np.radians(29.3)) * np.sin(np.radians(5.2)),
            np.cos(np.radians(29.3))
        ])

        # Forward projection (replicate backend.py lines 872-886)
        vx, vy, vz = source
        x_prime = vx * np.cos(phi) + vy * np.sin(phi)
        y_prime = -vx * np.sin(phi) + vy * np.cos(phi)
        z_prime = vz
        x_dp = x_prime * np.cos(theta) - z_prime * np.sin(theta)
        y_dp = y_prime
        z_dp = x_prime * np.sin(theta) + z_prime * np.cos(theta)
        u_deg = np.degrees(np.arctan2(x_dp, z_dp))
        v_deg = np.degrees(np.arctan2(y_dp, z_dp))

        # Inverse (cam_to_sky_vector)
        recovered = cam_to_sky_vector(u_deg, v_deg, theta, phi)
        assert np.allclose(recovered, source / np.linalg.norm(source), atol=1e-10)

    def test_unit_vector(self):
        """Output is always a unit vector."""
        for _ in range(20):
            u_deg = np.random.uniform(-2.0, 2.0)
            v_deg = np.random.uniform(-2.0, 2.0)
            zen = np.random.uniform(0.0, np.radians(60.0))
            azi = np.random.uniform(0.0, 2 * np.pi)
            v = cam_to_sky_vector(u_deg, v_deg, zen, azi)
            assert abs(np.linalg.norm(v) - 1.0) < 1e-12


class TestTailcutClean:

    def test_filters_pedestal(self):
        cam = Camera(n_rings=5)
        charges = np.full(cam.n_pixels, 2.0, dtype=np.float32)  # NSB level
        charges[0] = 17.0  # bright core pixel

        c_clean, mask = tailcut_clean(charges, cam, pic_th=4.0, bnd_th=2.0, ped=2.0)
        assert mask[0] is True or mask[0] == True  # central pixel survives
        assert c_clean[0] == pytest.approx(15.0)
        # Most background pixels should be rejected
        assert np.sum(mask) < cam.n_pixels / 2

    def test_boundary_expansion(self):
        """Pixels below picture threshold but above boundary should survive if neighbor is picture."""
        cam = Camera(n_rings=5)
        charges = np.full(cam.n_pixels, 0.0, dtype=np.float32)
        charges[0] = 20.0  # strong picture pixel (well above pic_th + ped)
        # Find a neighbor of pixel 0
        adj = cam.get_neighbor_matrix()
        neighbors = np.where(adj[:, 0])[0]  # or adj[0, :] depending on convention
        if len(neighbors) == 0:
            neighbors = np.where(adj[0, :])[0]
        assert len(neighbors) > 0, "pixel 0 should have neighbors"
        nb_idx = neighbors[0]
        charges[nb_idx] = 5.0  # above bnd_th + ped but below pic_th + ped

        c_clean, mask = tailcut_clean(charges, cam, pic_th=4.0, bnd_th=2.0, ped=2.0)
        assert mask[0]  # picture pixel
        assert mask[nb_idx]  # boundary pixel promoted by neighbor


class TestMLStereoFeatures:

    def test_feature_vector_length(self):
        """Feature vector should be 4*13 + 9 + 4 = 65."""
        cam = Camera(n_rings=12)
        array = TelescopeArray.veritas_array()
        traces = np.zeros((4, cam.n_pixels, 16), dtype=np.float32)
        feat = extract_stereo_features(traces, array.telescopes, cam)
        assert len(feat) == 65

    def test_zero_traces_yield_zero_features(self):
        """All-zero traces should give untriggered features."""
        cam = Camera(n_rings=12)
        array = TelescopeArray.veritas_array()
        traces = np.zeros((4, cam.n_pixels, 16), dtype=np.float32)
        feat = extract_stereo_features(traces, array.telescopes, cam)
        assert feat[52] == 0.0  # n_trig
        assert np.all(feat[61:65] == 0.0)  # no impact distance without images

    def test_gh_mask_excludes_direction_features(self):
        from recon.ml_stereo import gh_feature_mask
        mask = gh_feature_mask()
        assert mask.shape == (65,)
        for idx in (2, 3, 59, 60, 13 + 12):  # T0 cx, T0 cy, geom zen/azi, T1 alpha
            assert not mask[idx]
        for idx in (1, 5, 6, 10, 57, 58, 61):  # size, length, width, asym, core, impact
            assert mask[idx]

    @pytest.mark.skipif(
        not __import__('os').path.exists('data/stereo_eval_v2/sim_batch_00000.pt'),
        reason="requires stereo_eval_v2 simulation data")
    def test_impact_distance_matches_mc_truth(self):
        """Shower-frame impact distances from reconstruction should track MC truth."""
        import torch
        cam = Camera(n_rings=12)
        tels = TelescopeArray.veritas_array().telescopes
        events = torch.load('data/stereo_eval_v2/sim_batch_00000.pt', weights_only=False)[:60]
        errs = []
        for ev in events:
            feat = extract_stereo_features(ev['fadc_traces'], tels, cam,
                                           ev['zenith_deg'], ev['azimuth_deg'])
            if feat[56] < 1.0:
                continue
            zen = np.radians(ev['true_zenith_deg'])
            azi = np.radians(ev['true_azimuth_deg'] + 180.0)
            axis = np.array([np.sin(zen) * np.cos(azi), np.sin(zen) * np.sin(azi), np.cos(zen)])
            core = np.array([float(ev['impact_x']), float(ev['impact_y']), 1275.0])
            for t, tel in enumerate(tels):
                if feat[t * 13] > 0:
                    r = np.array([tel.x_tel, tel.y_tel, tel.z_tel]) - core
                    true_d = np.linalg.norm(r - np.dot(r, axis) * axis)
                    errs.append(feat[61 + t] - true_d)
        assert len(errs) > 50
        assert abs(np.median(errs)) < 15.0      # no gross systematic offset (m)
        assert np.percentile(np.abs(errs), 68) < 80.0


class TestMLStereoReconstructor:

    def test_fit_predict_shapes(self):
        """Smoke test: fit and predict produce correct shapes."""
        rng = np.random.RandomState(42)
        N = 50
        X = rng.randn(N, 61).astype(np.float32)
        y_imp = rng.randn(N, 2).astype(np.float32)
        y_dir = rng.randn(N, 2).astype(np.float32)

        model = MLStereoReconstructor(n_estimators=5, max_depth=3)
        model.fit(X, y_imp, y_dir)
        p_imp, p_dir = model.predict(X)
        assert p_imp.shape == (N, 2)
        assert p_dir.shape == (N, 2)

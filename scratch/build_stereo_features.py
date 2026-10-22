"""
Extract and cache 65-dimensional stereoscopic feature vectors from the
simulated gamma and proton events (750m throw, 30 deg zenith).
Outputs data/stereo_features_750m.npz.
"""
import glob
import os
import sys
sys.path.insert(0, 'src')

import numpy as np
import torch

from sim.camera import Camera
from sim.telescope import TelescopeArray
from recon.ml_stereo import extract_stereo_features


def compute_angular_separation(zen1_deg, azi1_deg, zen2_deg, azi2_deg):
    """Compute spherical angular separation in degrees."""
    z1, a1 = np.radians(zen1_deg), np.radians(azi1_deg)
    z2, a2 = np.radians(zen2_deg), np.radians(azi2_deg)

    v1 = np.column_stack([np.sin(z1) * np.cos(a1), np.sin(z1) * np.sin(a1), np.cos(z1)])
    v2 = np.column_stack([np.sin(z2) * np.cos(a2), np.sin(z2) * np.sin(a2), np.cos(z2)])

    cos_sep = np.sum(v1 * v2, axis=1)
    cos_sep = np.clip(cos_sep, -1.0, 1.0)
    return np.degrees(np.arccos(cos_sep))


def process_dataset():
    cam = Camera(n_rings=12)
    tels = TelescopeArray.veritas_array().telescopes

    gamma_files = sorted(glob.glob('data/gh_gammas_750m/*.pt'))
    proton_files = sorted(glob.glob('data/gh_protons_750m/*.pt'))

    print(f"Found {len(gamma_files)} gamma chunks and {len(proton_files)} proton chunks.")

    X_list = []
    labels = []
    energies_gev = []
    impact_x = []
    impact_y = []
    true_zenith = []
    true_azimuth = []
    geom_ok = []
    theta_deg = []

    # Nominal wobble target direction for wobble_offset=0.5 deg at zen=30, azi=0
    target_zen = 30.0 + 0.5
    target_azi = 0.0

    total_events = 0

    for label_val, file_list in [(1, gamma_files), (0, proton_files)]:
        class_name = "gammas" if label_val == 1 else "protons"
        print(f"\nProcessing {class_name}...")
        for f_idx, fpath in enumerate(file_list):
            chunk = torch.load(fpath, weights_only=False)
            for ev in chunk:
                p_zen = float(ev.get('zenith_deg', 30.0))
                p_azi = float(ev.get('azimuth_deg', 0.0))

                t_zen = float(ev.get('true_zenith_deg', p_zen))
                t_azi = float(ev.get('true_azimuth_deg', p_azi))
                t_e = float(ev.get('energy', 1000.0))
                t_ix = float(ev.get('impact_x', 0.0))
                t_iy = float(ev.get('impact_y', 0.0))

                traces = ev['fadc_traces']

                feat = extract_stereo_features(
                    traces, tels, cam,
                    pointing_zenith_deg=p_zen,
                    pointing_azimuth_deg=p_azi
                )

                X_list.append(feat)
                labels.append(label_val)
                energies_gev.append(t_e)
                impact_x.append(t_ix)
                impact_y.append(t_iy)
                true_zenith.append(t_zen)
                true_azimuth.append(t_azi)

                g_ok = feat[56] > 0.5  # feature 56: geom_ok
                geom_ok.append(g_ok)

                if g_ok:
                    reco_zen = feat[59]
                    reco_azi = feat[60]
                    # For gammas, theta is distance to true direction (or nominal source)
                    # For protons, theta is distance from reco direction to nominal wobble target
                    ref_zen = t_zen if label_val == 1 else target_zen
                    ref_azi = t_azi if label_val == 1 else target_azi
                    th = compute_angular_separation(
                        np.array([reco_zen]), np.array([reco_azi]),
                        np.array([ref_zen]), np.array([ref_azi])
                    )[0]
                    theta_deg.append(th)
                else:
                    theta_deg.append(np.nan)

                total_events += 1

            print(f"  Processed chunk {f_idx+1}/{len(file_list)} ({len(chunk)} events, total: {total_events})")

    X = np.array(X_list, dtype=np.float32)
    labels = np.array(labels, dtype=np.int32)
    energies_gev = np.array(energies_gev, dtype=np.float32)
    log10_e = np.log10(np.maximum(energies_gev, 1.0))
    impact_x = np.array(impact_x, dtype=np.float32)
    impact_y = np.array(impact_y, dtype=np.float32)
    true_zenith = np.array(true_zenith, dtype=np.float32)
    true_azimuth = np.array(true_azimuth, dtype=np.float32)
    geom_ok = np.array(geom_ok, dtype=bool)
    theta_deg = np.array(theta_deg, dtype=np.float32)

    out_file = 'data/stereo_features_750m.npz'
    np.savez_compressed(
        out_file,
        X=X,
        labels=labels,
        energies_gev=energies_gev,
        log10_e=log10_e,
        impact_x=impact_x,
        impact_y=impact_y,
        true_zenith=true_zenith,
        true_azimuth=true_azimuth,
        geom_ok=geom_ok,
        theta_deg=theta_deg
    )
    print(f"\nSuccessfully extracted and saved {total_events} events (shape: {X.shape}) to {out_file}!")


if __name__ == '__main__':
    process_dataset()

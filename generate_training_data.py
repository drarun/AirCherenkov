import os
import sys
import glob
import time
import torch
import json
import numpy as np
from tqdm import tqdm

sys.path.insert(0, 'src')
from sim.shower import ShowerSimulation
from sim.telescope import TelescopeArray
from sim.trigger import CameraTrigger, evaluate_array_trigger

import argparse
import subprocess
from datetime import datetime, timezone

SIM_SCHEMA_VERSION = "v2.0"


def _get_git_provenance():
    """Return (commit_hash, is_dirty) for the repository."""
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        dirty = bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"],
                stderr=subprocess.DEVNULL,
                text=True,
            ).strip()
        )
        return commit, dirty
    except Exception:
        return "unknown", False

# Enable TF32 for up to 3x faster tensor math on RTX 3000/4000/5000 series
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

CARDINAL_WOBBLE_DIRS = (
    ('N', 0.0),
    ('E', 0.5 * np.pi),
    ('S', np.pi),
    ('W', 1.5 * np.pi),
)


def _sample_power_law(e_min: float, e_max: float, spectral_index: float) -> float:
    u = np.random.rand()
    alpha = float(spectral_index)
    if abs(alpha - 1.0) < 1e-5:
        return float(e_min * ((e_max / e_min) ** u))
    return float((u * e_max ** (1.0 - alpha) + (1.0 - u) * e_min ** (1.0 - alpha)) ** (1.0 / (1.0 - alpha)))


def generate_training_data(
    num_gammas=10000,
    num_hadrons=10000,
    batch_size=500,
    save_every=1000,
    output_dir='data/train_raw',
    zenith_deg=30.0,
    azimuth_deg=0.0,
    diffuse_fov=2.5,
    wobble_offset=0.5,
    wobble_mode='cardinal',
    e_min=100.0,
    e_max=30000.0,
    spectral_index=2.0,
    hadron_e_min=None,
    hadron_e_max=None,
    hadron_spectral_index=None,
    impact_radius=750.0,
    append=False,
    sim_version=None,
    debug=False,
):
    """
    Generates Monte Carlo simulation data using the batched GPU tensor pipeline.
    Saves in chunks so the process can be safely interrupted and resumed.

    Gammas are thrown around the source direction using ``wobble_offset``
    (randomized across N/S/E/W when ``wobble_mode='cardinal'``, or uniformly
    around the ring when ``wobble_mode='continuous'``).
    Hadrons (protons) are thrown isotropically within ``diffuse_fov`` around
    the telescope pointing direction.
    """
    if debug:
        os.environ['AIRCHERENKOV_DEBUG'] = '1'
    else:
        os.environ['AIRCHERENKOV_DEBUG'] = '0'
    os.makedirs(output_dir, exist_ok=True)

    if hadron_e_min is None:
        hadron_e_min = e_min
    if hadron_e_max is None:
        hadron_e_max = e_max
    if hadron_spectral_index is None:
        hadron_spectral_index = spectral_index

    zen_rad = np.radians(zenith_deg)
    azi_rad = np.radians(azimuth_deg)
    sin_zen = max(np.sin(zen_rad), 1e-6)

    print(f"Telescope pointing: zenith={zenith_deg:.1f} deg, azimuth={azimuth_deg:.1f} deg")
    if num_gammas > 0 and wobble_offset > 0:
        print(f"Gamma wobble mode: offset={wobble_offset:.2f} deg ({wobble_mode}: N/S/E/W randomized)")
    if num_hadrons > 0 and diffuse_fov > 0:
        print(f"Hadron diffuse mode: scattering isotropically within {diffuse_fov:.2f} deg FOV cone")

    # Top-of-atmosphere injection (50 km, < 0.8 g/cm^2 vertical overburden)
    z_start = 50000.0

    # 1. Resume Logic
    state_file = os.path.join(output_dir, "sim_state.json")
    existing_files = sorted(glob.glob(os.path.join(output_dir, "sim_batch_*.pt")))
    start_chunk_idx = 0
    gammas_thrown = 0
    hadrons_thrown = 0

    if os.path.exists(state_file):
        with open(state_file, 'r') as f:
            state = json.load(f)
            gammas_thrown = state.get('gammas_thrown', 0)
            hadrons_thrown = state.get('hadrons_thrown', 0)

    gammas_saved = 0
    hadrons_saved = 0
    if existing_files:
        indices = [int(os.path.basename(f).split('_')[2].split('.pt')[0]) for f in existing_files]
        start_chunk_idx = max(indices) + 1
        print(f"Found {len(existing_files)} chunks. Resuming from chunk {start_chunk_idx}.")
        for f in existing_files:
            try:
                chunk = torch.load(f, weights_only=False)
                for ev in chunk:
                    if ev.get('label', 1) == 1:
                        gammas_saved += 1
                    else:
                        hadrons_saved += 1
            except Exception:
                pass
        print(f"Loaded existing events: {gammas_saved} Gammas, {hadrons_saved} Hadrons saved.")
        print(f"State file says we have already thrown {gammas_thrown} Gammas and {hadrons_thrown} Hadrons.")

    target_gammas = num_gammas
    target_hadrons = num_hadrons
    if append:
        gammas_needed = max(0, target_gammas)
        hadrons_needed = max(0, target_hadrons)
        total_target_events = gammas_needed + hadrons_needed
    else:
        gammas_needed = max(0, target_gammas - gammas_saved)
        hadrons_needed = max(0, target_hadrons - hadrons_saved)
    events_remaining = gammas_needed + hadrons_needed

    if events_remaining <= 0:
        print("All requested events have already been generated!")
        return

    # Remove stale cached processed/data.pt so downstream scripts rebuild from raw chunks
    processed_cache = os.path.join(os.path.dirname(os.path.normpath(output_dir)), "processed", "data.pt")
    if os.path.exists(processed_cache):
        os.remove(processed_cache)
        print(f"Removed stale processed cache: {processed_cache}")

    git_commit, git_dirty = _get_git_provenance()
    effective_sim_version = sim_version or SIM_SCHEMA_VERSION
    manifest_path = os.path.join(output_dir, "sim_manifest.json")
    session_start_chunk = start_chunk_idx
    session_gammas_thrown_start = gammas_thrown
    session_hadrons_thrown_start = hadrons_thrown
    session_gammas_saved = 0
    session_hadrons_saved = 0
    session_started_at = datetime.now(timezone.utc).isoformat()

    def _update_manifest(last_chunk_idx, status="in_progress"):
        manifest = {"schema_version": SIM_SCHEMA_VERSION, "sessions": []}
        if os.path.exists(manifest_path):
            try:
                with open(manifest_path, "r") as mf:
                    manifest = json.load(mf)
            except Exception:
                pass
        sessions = manifest.setdefault("sessions", [])
        session_entry = {
            "session_index": len(sessions),
            "sim_version": effective_sim_version,
            "git_commit": git_commit,
            "git_dirty": git_dirty,
            "started_at": session_started_at,
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "status": status,
            "append": bool(append),
            "chunk_start": session_start_chunk,
            "chunk_end": last_chunk_idx,
            "gammas_thrown_session": gammas_thrown - session_gammas_thrown_start,
            "hadrons_thrown_session": hadrons_thrown - session_hadrons_thrown_start,
            "gammas_saved_session": session_gammas_saved,
            "hadrons_saved_session": session_hadrons_saved,
            "params": {
                "e_min": e_min,
                "e_max": e_max,
                "spectral_index": spectral_index,
                "hadron_e_min": hadron_e_min,
                "hadron_e_max": hadron_e_max,
                "hadron_spectral_index": hadron_spectral_index,
                "impact_radius": impact_radius,
                "zenith_deg": zenith_deg,
                "azimuth_deg": azimuth_deg,
                "wobble_offset": wobble_offset,
                "wobble_mode": wobble_mode,
                "diffuse_fov": diffuse_fov,
            },
        }
        if sessions and sessions[-1].get("started_at") == session_started_at:
            session_entry["session_index"] = sessions[-1].get("session_index", len(sessions) - 1)
            sessions[-1] = session_entry
        else:
            sessions.append(session_entry)
        with open(manifest_path, "w") as mf:
            json.dump(manifest, mf, indent=2)

    if append:
        print(f"APPENDING +{gammas_needed} gammas and +{hadrons_needed} hadrons (starting at chunk {start_chunk_idx}) in batches of {batch_size}, saving every {save_every} events to {output_dir}.")
    else:
        print(f"Generating {gammas_needed} gammas and {hadrons_needed} hadrons in batches of {batch_size}, saving every {save_every} events to {output_dir}.")
    if gammas_needed > 0:
        print(f"Gamma energy range:  {e_min:.0f} - {e_max:.0f} GeV, spectral index: E^-{spectral_index:.1f}")
    if hadrons_needed > 0:
        print(f"Hadron energy range: {hadron_e_min:.0f} - {hadron_e_max:.0f} GeV, spectral index: E^-{hadron_spectral_index:.1f}")
    print(f"Impact radius: {impact_radius:.0f} m")

    array = TelescopeArray.veritas_array()
    arr_cx = float(np.mean([t.x_tel for t in array.telescopes]))
    arr_cy = float(np.mean([t.y_tel for t in array.telescopes]))
    cam = array.telescopes[0].camera
    cam_trigger = CameraTrigger(
        cam.pixel_x,
        cam.pixel_y,
        threshold_pe=8.0,
        window_ns=5.0,
        min_pixels=3,
        pixel_size=cam.pixel_size,
    )

    current_chunk = []
    chunk_idx = start_chunk_idx
    passed_trigger_count = 0

    total_target_events = events_remaining
    events_completed_session = 0
    last_pct_reported = 0

    pbar = tqdm(total=events_remaining, desc="MC Generation")

    try:
        while events_remaining > 0:
            current_batch_size = batch_size

            pids = []
            energies = []
            z_starts = []
            px_batch = []
            py_batch = []
            pz_batch = []
            evt_zen_batch = []
            evt_azi_batch = []
            wobble_dirs = []

            # Sample parameters for each primary in the batch
            for _ in range(current_batch_size):
                if gammas_needed > 0 and hadrons_needed > 0:
                    is_gamma = np.random.rand() < (gammas_needed / (gammas_needed + hadrons_needed))
                elif gammas_needed > 0:
                    is_gamma = True
                else:
                    is_gamma = False

                if is_gamma:
                    pids.append('gamma')
                    gammas_thrown += 1
                    E = _sample_power_law(e_min, e_max, spectral_index)
                    if wobble_offset > 0:
                        theta_offset = np.radians(wobble_offset)
                        if wobble_mode == 'cardinal':
                            w_label, phi_offset = CARDINAL_WOBBLE_DIRS[np.random.randint(0, 4)]
                        else:
                            w_label = 'ring'
                            phi_offset = np.random.rand() * 2.0 * np.pi
                        evt_zen = zen_rad + theta_offset * np.cos(phi_offset)
                        evt_azi = azi_rad + theta_offset * np.sin(phi_offset) / sin_zen
                    elif diffuse_fov > 0:
                        w_label = 'diffuse'
                        fov_rad = np.radians(diffuse_fov)
                        cos_theta = 1.0 - np.random.rand() * (1.0 - np.cos(fov_rad))
                        theta_offset = np.arccos(cos_theta)
                        phi_offset = np.random.rand() * 2.0 * np.pi
                        evt_zen = zen_rad + theta_offset * np.cos(phi_offset)
                        evt_azi = azi_rad + theta_offset * np.sin(phi_offset) / sin_zen
                    else:
                        w_label = 'on_axis'
                        evt_zen = zen_rad
                        evt_azi = azi_rad
                else:
                    pids.append('proton')
                    hadrons_thrown += 1
                    E = _sample_power_law(hadron_e_min, hadron_e_max, hadron_spectral_index)
                    if diffuse_fov > 0:
                        w_label = 'diffuse'
                        fov_rad = np.radians(diffuse_fov)
                        cos_theta = 1.0 - np.random.rand() * (1.0 - np.cos(fov_rad))
                        theta_offset = np.arccos(cos_theta)
                        phi_offset = np.random.rand() * 2.0 * np.pi
                        evt_zen = zen_rad + theta_offset * np.cos(phi_offset)
                        evt_azi = azi_rad + theta_offset * np.sin(phi_offset) / sin_zen
                    elif wobble_offset > 0:
                        theta_offset = np.radians(wobble_offset)
                        if wobble_mode == 'cardinal':
                            w_label, phi_offset = CARDINAL_WOBBLE_DIRS[np.random.randint(0, 4)]
                        else:
                            w_label = 'ring'
                            phi_offset = np.random.rand() * 2.0 * np.pi
                        evt_zen = zen_rad + theta_offset * np.cos(phi_offset)
                        evt_azi = azi_rad + theta_offset * np.sin(phi_offset) / sin_zen
                    else:
                        w_label = 'on_axis'
                        evt_zen = zen_rad
                        evt_azi = azi_rad

                energies.append(E)
                z_starts.append(z_start)
                wobble_dirs.append(w_label)
                evt_zen_batch.append(evt_zen)
                evt_azi_batch.append(evt_azi)
                px_batch.append(np.sin(evt_zen) * np.cos(evt_azi))
                py_batch.append(np.sin(evt_zen) * np.sin(evt_azi))
                pz_batch.append(-np.cos(evt_zen))

            # Run fully batched simulation with direction injection aimed at (0, 0, z_obs)
            x_init_batch = []
            y_init_batch = []
            z_obs = 1268.0
            for pz_val, px_val, py_val, z_val in zip(pz_batch, px_batch, py_batch, z_starts):
                x_init_batch.append((z_val - z_obs) * (px_val / pz_val))
                y_init_batch.append((z_val - z_obs) * (py_val / pz_val))

            with torch.inference_mode():
                sim = ShowerSimulation(
                    primary_types=pids, energies=energies, z_starts=z_starts,
                    x_init=x_init_batch, y_init=y_init_batch,
                    px_init=px_batch, py_init=py_batch, pz_init=pz_batch
                )
                sim.run(max_generations=60, verbose=False)

            passed_trigger_count_in_batch = 0
            for i in range(current_batch_size):
                photons = sim.cherenkov_photons_by_event.get(i, {})
                if len(photons.get('x_ground', [])) == 0:
                    continue

                # Randomize core impact position within disk centered on the VERITAS array
                r = np.sqrt(np.random.rand()) * impact_radius
                theta = np.random.rand() * 2.0 * np.pi
                ix = arr_cx + r * np.cos(theta)
                iy = arr_cy + r * np.sin(theta)

                for tel in array.telescopes:
                    tel.x_tel -= ix
                    tel.y_tel -= iy

                with torch.inference_mode():
                    img_outputs = array.ray_trace(
                        photons,
                        pointing_zenith=zenith_deg,
                        pointing_azimuth=azimuth_deg + 180.0,
                        shower_start_altitude=z_starts[i]
                    )

                for tel in array.telescopes:
                    tel.x_tel += ix
                    tel.y_tel += iy

                # Hardware Trigger: At least 2 telescopes with compact 3-pixel adjacent coincidence (> 8 PE within 5 ns)
                passed, trigger_count = evaluate_array_trigger(
                    img_outputs,
                    cam,
                    threshold_pe=8.0,
                    min_pixels=3,
                    min_telescopes=2,
                    bin_width_ns=array.telescopes[0].fadc_config.bin_width_ns,
                    low_gain_factor=array.telescopes[0].fadc_config.low_gain_factor,
                    camera_trigger=cam_trigger,
                )

                if passed:
                    passed_trigger_count_in_batch += 1
                    passed_trigger_count += 1
                    if pids[i] == 'gamma':
                        gammas_needed = max(0, gammas_needed - 1)
                        session_gammas_saved += 1
                    else:
                        hadrons_needed = max(0, hadrons_needed - 1)
                        session_hadrons_saved += 1
                    traces = [trace for trace, gain in img_outputs]
                    gains = [gain for trace, gain in img_outputs]

                    current_chunk.append({
                        'fadc_traces': np.array(traces, dtype=np.float32),
                        'gain_flags': np.array(gains, dtype=np.float32),
                        'energy': energies[i],
                        'label': 1 if pids[i] == 'gamma' else 0,
                        'impact_x': ix,
                        'impact_y': iy,
                        'impact_radius_thrown': float(impact_radius),
                        'zenith_deg': zenith_deg,
                        'azimuth_deg': azimuth_deg,
                        'true_zenith_deg': float(np.degrees(evt_zen_batch[i])),
                        'true_azimuth_deg': float(np.degrees(evt_azi_batch[i])),
                        'wobble_dir': wobble_dirs[i],
                        'sim_version': effective_sim_version,
                        'sim_git_commit': git_commit,
                    })

            events_remaining = gammas_needed + hadrons_needed
            events_completed_session += passed_trigger_count_in_batch
            pbar.update(passed_trigger_count_in_batch)

            current_pct = min(100, int((events_completed_session / total_target_events) * 100))
            if current_pct > last_pct_reported:
                pbar.write(f"[Progress {current_pct}%] {events_completed_session}/{total_target_events} showers triggered")
                last_pct_reported = current_pct

            # Save chunk
            if len(current_chunk) >= save_every or events_remaining == 0:
                out_filename = os.path.join(output_dir, f"sim_batch_{chunk_idx:05d}.pt")
                torch.save(current_chunk, out_filename)

                with open(state_file, 'w') as f:
                    json.dump({'gammas_thrown': gammas_thrown, 'hadrons_thrown': hadrons_thrown}, f)
                _update_manifest(chunk_idx, status="completed" if events_remaining == 0 else "in_progress")

                current_chunk = []
                chunk_idx += 1

            # Graceful pause-file check after each GPU batch
            pause_file = os.path.join(output_dir, "PAUSE")
            if os.path.exists(pause_file):
                pbar.write(f"\nPAUSE file detected ({pause_file}). Flushing progress and pausing cleanly...")
                if len(current_chunk) > 0:
                    out_filename = os.path.join(output_dir, f"sim_batch_{chunk_idx:05d}.pt")
                    torch.save(current_chunk, out_filename)
                    with open(state_file, 'w') as f:
                        json.dump({'gammas_thrown': gammas_thrown, 'hadrons_thrown': hadrons_thrown}, f)
                    _update_manifest(chunk_idx, status="paused")
                    current_chunk = []
                    chunk_idx += 1
                elif chunk_idx > session_start_chunk:
                    with open(state_file, 'w') as f:
                        json.dump({'gammas_thrown': gammas_thrown, 'hadrons_thrown': hadrons_thrown}, f)
                    _update_manifest(chunk_idx - 1, status="paused")
                try:
                    os.remove(pause_file)
                except OSError:
                    pass
                break

    except KeyboardInterrupt:
        print("\nGeneration interrupted by user.")
        if len(current_chunk) > 0:
            out_filename = os.path.join(output_dir, f"sim_batch_{chunk_idx:05d}.pt")
            print(f"Saving remaining {len(current_chunk)} events to {out_filename} before exiting...")
            torch.save(current_chunk, out_filename)
            with open(state_file, 'w') as f:
                json.dump({'gammas_thrown': gammas_thrown, 'hadrons_thrown': hadrons_thrown}, f)
            _update_manifest(chunk_idx, status="interrupted")

    finally:
        pbar.close()
        print(f"\nGeneration stopped. {passed_trigger_count} new events passed the array hardware trigger in this session.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate MC data")
    parser.add_argument('--num_gammas', type=int, default=10000, help='Number of gamma events to generate')
    parser.add_argument('--num_hadrons', type=int, default=10000, help='Number of hadron (proton) events to generate')
    parser.add_argument('--batch_size', type=int, default=100, help='Batch size for GPU pipeline')
    parser.add_argument('--save_every', type=int, default=1000, help='Save chunk every N events')
    parser.add_argument('--output_dir', type=str, default='data/train_raw', help='Output directory')
    parser.add_argument('--zenith_deg', type=float, default=30.0, help='Zenith angle in degrees')
    parser.add_argument('--azimuth_deg', type=float, default=0.0, help='Azimuth angle in degrees (0=N, 90=E, 180=S)')
    parser.add_argument('--diffuse_fov', type=float, default=2.5, help='Isotropic FOV radius (degrees) for diffuse hadrons')
    parser.add_argument('--wobble_offset', type=float, default=0.5, help='Wobble offset in degrees for point-source gammas')
    parser.add_argument('--wobble_mode', type=str, default='cardinal', choices=['cardinal', 'continuous'],
                        help='Wobble sampling mode: cardinal (random N/S/E/W) or continuous ring')
    parser.add_argument('--e_min', type=float, default=100.0, help='Min gamma energy in GeV')
    parser.add_argument('--e_max', type=float, default=30000.0, help='Max gamma energy in GeV')
    parser.add_argument('--spectral_index', type=float, default=2.0, help='Gamma spectral index for E^-alpha sampling')
    parser.add_argument('--hadron_e_min', type=float, default=None, help='Min hadron energy in GeV (defaults to e_min)')
    parser.add_argument('--hadron_e_max', type=float, default=None, help='Max hadron energy in GeV (defaults to e_max)')
    parser.add_argument('--hadron_spectral_index', type=float, default=None, help='Hadron spectral index (defaults to spectral_index)')
    parser.add_argument('--impact_radius', type=float, default=750.0, help='Max impact parameter radius in meters')
    parser.add_argument('--append', action='store_true', help='Append +num_gammas/+num_hadrons to existing dataset instead of resuming to total')
    parser.add_argument('--sim_version', type=str, default=None, help='Override simulation version tag in manifest and event metadata')
    parser.add_argument('--debug', action='store_true', help='Enable verbose DEBUG output')
    args = parser.parse_args()

    generate_training_data(
        num_gammas=args.num_gammas,
        num_hadrons=args.num_hadrons,
        batch_size=args.batch_size,
        save_every=args.save_every,
        output_dir=args.output_dir,
        zenith_deg=args.zenith_deg,
        azimuth_deg=args.azimuth_deg,
        diffuse_fov=args.diffuse_fov,
        wobble_offset=args.wobble_offset,
        wobble_mode=args.wobble_mode,
        e_min=args.e_min,
        e_max=args.e_max,
        spectral_index=args.spectral_index,
        hadron_e_min=args.hadron_e_min,
        hadron_e_max=args.hadron_e_max,
        hadron_spectral_index=args.hadron_spectral_index,
        impact_radius=args.impact_radius,
        append=args.append,
        sim_version=args.sim_version,
        debug=args.debug,
    )

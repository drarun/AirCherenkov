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

import argparse

# Enable TF32 for up to 3x faster tensor math on RTX 3000/4000/5000 series
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

def generate_training_data(num_gammas=10000, num_hadrons=10000, batch_size=500, save_every=1000, 
                           output_dir='data/train_raw', zenith_deg=0.0, azimuth_deg=0.0, diffuse_fov=0.0,
                           wobble_offset=0.0,
                           e_min=100.0, e_max=10000.0, spectral_index=2.0, impact_radius=250.0,
                           debug=False):
    """
    Generates Monte Carlo simulation data using the batched GPU tensor pipeline.
    Saves in chunks so the process can be safely interrupted and resumed.
    """
    if debug:
        os.environ['AIRCHERENKOV_DEBUG'] = '1'
    else:
        os.environ['AIRCHERENKOV_DEBUG'] = '0'
    os.makedirs(output_dir, exist_ok=True)
    
    zen_rad = np.radians(zenith_deg)
    azi_rad = np.radians(azimuth_deg)
    
    print(f"Shower base direction: zenith={zenith_deg:.1f}°, azimuth={azimuth_deg:.1f}°")
    if diffuse_fov > 0:
        print(f"Isotropic Background Mode: Scattering directions within {diffuse_fov:.2f}° FOV")
    
    # Higher z_start for inclined showers (longer slant depth)
    z_start = 20000.0 if zenith_deg < 25.0 else 25000.0
    
    # 1. Resume Logic
    state_file = os.path.join(output_dir, "sim_state.json")
    existing_files = glob.glob(os.path.join(output_dir, "sim_batch_*.pt"))
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
    gammas_needed = max(0, target_gammas - gammas_saved)
    hadrons_needed = max(0, target_hadrons - hadrons_saved)
    events_remaining = gammas_needed + hadrons_needed
    
    if events_remaining <= 0:
        print("All requested events have already been generated!")
        return

    print(f"Generating {gammas_needed} gammas and {hadrons_needed} hadrons in batches of {batch_size}, saving every {save_every} events to {output_dir}.")
    print(f"Energy range: {e_min:.0f} – {e_max:.0f} GeV, spectral index: E^-{spectral_index:.1f}")
    print(f"Impact radius: {impact_radius:.0f} m")
    
    array = TelescopeArray.veritas_array()
    
    current_chunk = []
    chunk_idx = start_chunk_idx
    passed_trigger_count = 0
    
    # Main generation loop
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
            
            # Sample parameters
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
                else:
                    pids.append('proton')
                    hadrons_thrown += 1
                    
                # Generalized power-law sampling: E^(-alpha) from e_min to e_max
                u = np.random.rand()
                alpha = spectral_index
                if abs(alpha - 1.0) < 1e-5:
                    E = e_min * ((e_max / e_min) ** u)
                else:
                    E = (u * e_max**(1-alpha) + (1-u) * e_min**(1-alpha)) ** (1.0/(1-alpha))
                energies.append(E)
                z_starts.append(z_start)
                
                # Calculate specific direction for this shower
                if diffuse_fov > 0:
                    # Randomize direction within FOV cone uniformly in solid angle
                    fov_rad = np.radians(diffuse_fov)
                    # Random angle from the center (cosine distribution for uniform solid angle)
                    cos_theta = 1.0 - np.random.rand() * (1.0 - np.cos(fov_rad))
                    theta_offset = np.arccos(cos_theta)
                    phi_offset = np.random.rand() * 2 * np.pi
                    
                    # Perturb zenith and azimuth (approximate for small FOVs)
                    evt_zen = zen_rad + theta_offset * np.cos(phi_offset)
                    evt_azi = azi_rad + theta_offset * np.sin(phi_offset)
                elif wobble_offset > 0:
                    theta_offset = np.radians(wobble_offset)
                    phi_offset = np.random.rand() * 2 * np.pi
                    evt_zen = zen_rad + theta_offset * np.cos(phi_offset)
                    evt_azi = azi_rad + theta_offset * np.sin(phi_offset)
                else:
                    evt_zen = zen_rad
                    evt_azi = azi_rad
                    
                px_batch.append(np.sin(evt_zen) * np.cos(evt_azi))
                py_batch.append(np.sin(evt_zen) * np.sin(evt_azi))
                pz_batch.append(-np.cos(evt_zen))
                
            # Run fully batched simulation with direction injection
            x_init_batch = []
            y_init_batch = []
            z_obs = 1275.0
            for pz_val, px_val, py_val, z_val in zip(pz_batch, px_batch, py_batch, z_starts):
                x_init_batch.append((z_val - z_obs) * (px_val / pz_val))
                y_init_batch.append((z_val - z_obs) * (py_val / pz_val))
                
            with torch.inference_mode():
                sim = ShowerSimulation(
                    primary_types=pids, energies=energies, z_starts=z_starts,
                    x_init=x_init_batch, y_init=y_init_batch,
                    px_init=px_batch, py_init=py_batch, pz_init=pz_batch
                )
                sim.run(max_generations=30, verbose=False)
            
            passed_trigger_count_in_batch = 0
            for i in range(current_batch_size):
                photons = sim.cherenkov_photons_by_event.get(i, {})
                if len(photons.get('x_ground', [])) == 0:
                    continue
                    
                # Randomize impact parameter within configurable radius
                r = np.sqrt(np.random.rand()) * impact_radius
                theta = np.random.rand() * 2 * np.pi
                ix = r * np.cos(theta)
                iy = r * np.sin(theta)
                
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
                    
                # Hardware Trigger: At least 2 telescopes with >= 3 pixels having > 8 PE total
                trigger_count = 0
                for trace, gain in img_outputs:
                    total_pe_per_pixel = np.sum(trace, axis=1)
                    if np.sum(total_pe_per_pixel > 8.0) >= 3:
                        trigger_count += 1
                        
                if trigger_count >= 2:
                    passed_trigger_count_in_batch += 1
                    passed_trigger_count += 1
                    if pids[i] == 'gamma':
                        gammas_needed = max(0, gammas_needed - 1)
                    else:
                        hadrons_needed = max(0, hadrons_needed - 1)
                    traces = [trace for trace, gain in img_outputs]
                    gains = [gain for trace, gain in img_outputs]
                    
                    current_chunk.append({
                        'fadc_traces': np.array(traces, dtype=np.float32),
                        'gain_flags': np.array(gains, dtype=np.float32),
                        'energy': energies[i],
                        'label': 1 if pids[i] == 'gamma' else 0,
                        'impact_x': ix,
                        'impact_y': iy,
                        'zenith_deg': zenith_deg,
                        'azimuth_deg': azimuth_deg
                    })
            
            # events_remaining is now decremented by triggers!
            events_remaining = gammas_needed + hadrons_needed
            events_completed_session += passed_trigger_count_in_batch
            pbar.update(passed_trigger_count_in_batch)
            
            current_pct = int((events_completed_session / total_target_events) * 100)
            if current_pct > last_pct_reported:
                for p in range(last_pct_reported + 1, current_pct + 1):
                    pbar.write(f"[Progress {p}%] {events_completed_session}/{total_target_events} showers triggered")
                last_pct_reported = current_pct
            
            # Save chunk
            if len(current_chunk) >= save_every or events_remaining == 0:
                out_filename = os.path.join(output_dir, f"sim_batch_{chunk_idx:05d}.pt")
                torch.save(current_chunk, out_filename)
                
                # Save state
                with open(state_file, 'w') as f:
                    json.dump({'gammas_thrown': gammas_thrown, 'hadrons_thrown': hadrons_thrown}, f)
                    
                current_chunk = []
                chunk_idx += 1
                
    except KeyboardInterrupt:
        print("\nGeneration interrupted by user.")
        if len(current_chunk) > 0:
            out_filename = os.path.join(output_dir, f"sim_batch_{chunk_idx:05d}.pt")
            print(f"Saving remaining {len(current_chunk)} events to {out_filename} before exiting...")
            torch.save(current_chunk, out_filename)
            with open(state_file, 'w') as f:
                json.dump({'gammas_thrown': gammas_thrown, 'hadrons_thrown': hadrons_thrown}, f)
            
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
    parser.add_argument('--zenith_deg', type=float, default=0.0, help='Zenith angle in degrees')
    parser.add_argument('--azimuth_deg', type=float, default=0.0, help='Azimuth angle in degrees (0=N, 90=E, 180=S)')
    parser.add_argument('--diffuse_fov', type=float, default=0.0, help='If > 0, scatters directions isotropically within this FOV radius (degrees)')
    parser.add_argument('--wobble_offset', type=float, default=0.0, help='Wobble offset in degrees')
    parser.add_argument('--e_min', type=float, default=80.0, help='Min energy in GeV')
    parser.add_argument('--e_max', type=float, default=30000.0, help='Max energy in GeV')
    parser.add_argument('--spectral_index', type=float, default=2.0, help='Spectral index for E^-alpha sampling')
    parser.add_argument('--impact_radius', type=float, default=350.0, help='Max impact parameter radius in meters')
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
        e_min=args.e_min,
        e_max=args.e_max,
        spectral_index=args.spectral_index,
        impact_radius=args.impact_radius,
        debug=args.debug
    )

import os
import sys
sys.path.insert(0, 'src')
import time
import numpy as np
import torch
import multiprocessing
import glob
from tqdm import tqdm
import multiprocessing
from sim.shower import ShowerSimulation
from sim.telescope import Telescope, TelescopeArray
from sim.backend import compute_cherenkov_pool_gpu

def simulate_shower_cpu(args):
    """
    Run the CPU-intensive cascade generation and return the 
    segments needed for the GPU Cherenkov calculation.
    """
    event_id, e_min, e_max = args
    
    # Sample particle type: 50% gamma, 50% proton
    is_gamma = (np.random.rand() < 0.5)
    pid = 'gamma' if is_gamma else 'proton'
    
    # Sample energy from power law E^-2
    u = np.random.rand()
    e_inv = (1.0/e_max) + u * ((1.0/e_min) - (1.0/e_max))
    energy = 1.0 / e_inv
    
    # Sample impact parameter
    r = np.sqrt(np.random.rand()) * 250.0
    theta = np.random.rand() * 2 * np.pi
    ix = r * np.cos(theta)
    iy = r * np.sin(theta)
    
    # Run simulation (CPU part only)
    sim = ShowerSimulation(pid, energy=energy)
    
    for gen in range(16):
        if not sim.active_particles:
            break
        sim.step()
        
    sim.all_particles = sim.dead_particles + sim.active_particles
    
    # Extract tracks for Cherenkov pool calculation
    track_arrays = []
    energy_counts = []
    for p in sim.all_particles:
        if p.pid not in ['e+', 'e-']:
            continue
        n_seg = len(p.track) - 1
        if n_seg <= 0:
            continue
        arr = np.array(p.track)
        track_arrays.append(arr)
        energy_counts.append((p.energy, n_seg))
        
    if not track_arrays:
        return {
            'empty': True,
            'is_gamma': is_gamma,
            'energy': energy,
            'ix': ix,
            'iy': iy
        }
        
    all_starts = np.concatenate([arr[:-1] for arr in track_arrays])
    all_ends = np.concatenate([arr[1:] for arr in track_arrays])
    seg_energy = np.concatenate([np.full(n, e) for e, n in energy_counts])
    
    seg_x1 = all_starts[:, 0]
    seg_y1 = all_starts[:, 1]
    seg_z1 = all_starts[:, 2]
    seg_x2 = all_ends[:, 0]
    seg_y2 = all_ends[:, 1]
    seg_z2 = all_ends[:, 2]
    
    seg_px = all_starts[:, 3]
    seg_py = all_starts[:, 4]
    seg_pz = all_starts[:, 5]

    return {
        'empty': False,
        'is_gamma': is_gamma,
        'energy': energy,
        'ix': ix,
        'iy': iy,
        'segments': (seg_x1, seg_y1, seg_z1, seg_x2, seg_y2, seg_z2, seg_px, seg_py, seg_pz, seg_energy, sim.photon_yield_factor)
    }

def generate_dataset_batched(num_events, filename_prefix, batch_size=100, e_min=100.0, e_max=10000.0):
    
    # --- Resume Logic ---
    existing_batches = glob.glob(f"{filename_prefix}_batch*.pt")
    start_batch_idx = 0
    if existing_batches:
        # Extract batch indices
        indices = [int(f.split('_batch')[-1].split('.pt')[0]) for f in existing_batches]
        start_batch_idx = max(indices) + 1
        print(f"Found {len(existing_batches)} existing batches. Resuming from batch {start_batch_idx}...")
    else:
        print(f"Generating {num_events} events for {filename_prefix} in batches of {batch_size}...")
        
    array = TelescopeArray.veritas_array()
    
    # Offset the total events and tasks based on where we are starting
    events_to_generate = num_events - (start_batch_idx * batch_size)
    if events_to_generate <= 0:
        print("All requested events have already been generated!")
        return
        
    tasks = [(i, e_min, e_max) for i in range(events_to_generate)]
    
    t0 = time.time()
    
    # Use multiprocessing for CPU tracking, and main thread for GPU pool+raytracing
    num_cores = max(1, multiprocessing.cpu_count() - 1)
    
    total_passing = 0
    
    with multiprocessing.Pool(processes=num_cores) as pool:
        batch_generator = range(0, events_to_generate, batch_size)
        # Add tqdm progress bar over the remaining batches
        for local_batch_idx, i in tqdm(enumerate(batch_generator), total=len(batch_generator), desc="Batches"):
            global_batch_idx = start_batch_idx + local_batch_idx
            batch_tasks = tasks[i:i+batch_size]
            
            # 1. CPU Phase: cascade generation in parallel
            results = pool.map(simulate_shower_cpu, batch_tasks)
            
            # 2. GPU Phase: run Cherenkov & ray trace sequentially on the main thread for the batch
            
            images = []
            energies = []
            labels = []
            impact_x = []
            impact_y = []
            
            for res in results:
                if res['empty']:
                    continue
                
                seg_x1, seg_y1, seg_z1, seg_x2, seg_y2, seg_z2, seg_px, seg_py, seg_pz, seg_energy, pyf = res['segments']
                
                cherenkov_photons = compute_cherenkov_pool_gpu(
                    seg_x1, seg_y1, seg_z1, seg_x2, seg_y2, seg_z2, 
                    seg_px, seg_py, seg_pz, seg_energy, pyf
                )
                
                # Offset the array relative to the shower core
                for tel in array.telescopes:
                    tel.x_tel -= res['ix']
                    tel.y_tel -= res['iy']
                
                imgs = array.ray_trace(cherenkov_photons)
                
                # Restore the array positions
                for tel in array.telescopes:
                    tel.x_tel += res['ix']
                    tel.y_tel += res['iy']
                
                # Stereo trigger condition: require at least 2 telescopes with > 20 PE
                trigger_count = sum(1 for img in imgs if np.sum(img) > 20)
                if trigger_count < 2:
                    continue
                    
                images.append(imgs)  # List of 4 images
                energies.append(res['energy'])
                labels.append(1 if res['is_gamma'] else 0)
                impact_x.append(res['ix'])
                impact_y.append(res['iy'])
            
            if len(images) > 0:
                # Save batch file
                out_filename = f"{filename_prefix}_batch{global_batch_idx}.pt"
                torch.save({
                    'images': torch.tensor(np.array(images), dtype=torch.float32),
                    'energies': torch.tensor(energies, dtype=torch.float32),
                    'labels': torch.tensor(labels, dtype=torch.long),
                    'impact_x': torch.tensor(impact_x, dtype=torch.float32),
                    'impact_y': torch.tensor(impact_y, dtype=torch.float32),
                    'pixel_x': torch.tensor(array.telescopes[0].camera.pixel_x, dtype=torch.float32),
                    'pixel_y': torch.tensor(array.telescopes[0].camera.pixel_y, dtype=torch.float32)
                }, out_filename)
                
                total_passing += len(images)
    
    print(f"Finished generating. {total_passing} total events passed trigger.")

if __name__ == '__main__':
    os.makedirs('data', exist_ok=True)
    generate_dataset_batched(100, 'data/train_events', batch_size=50)

import os
import sys
import glob
import time
import torch
import numpy as np
from tqdm import tqdm

sys.path.insert(0, 'src')
from sim.shower import ShowerSimulation
from sim.telescope import TelescopeArray

def generate_training_data(total_events=1_000_000, batch_size=100, save_every=1000, output_dir='data/raw'):
    """
    Generates Monte Carlo simulation data using the batched GPU tensor pipeline.
    Saves in chunks so the process can be safely interrupted and resumed.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Resume Logic
    existing_files = glob.glob(os.path.join(output_dir, "sim_batch_*.pt"))
    start_chunk_idx = 0
    events_already_generated = 0
    
    if existing_files:
        indices = [int(os.path.basename(f).split('_')[2].split('.pt')[0]) for f in existing_files]
        start_chunk_idx = max(indices) + 1
        
        # Estimate events already generated (assuming all previous chunks are full)
        events_already_generated = start_chunk_idx * save_every
        print(f"Found {len(existing_files)} existing chunks. Resuming from chunk {start_chunk_idx} (~{events_already_generated} events).")
    
    events_remaining = total_events - events_already_generated
    if events_remaining <= 0:
        print("All requested events have already been generated!")
        return

    print(f"Generating {events_remaining} events in batches of {batch_size}, saving every {save_every} events.")
    
    array = TelescopeArray.veritas_array()
    
    current_chunk = []
    chunk_idx = start_chunk_idx
    passed_trigger_count = 0
    
    # Main generation loop
    pbar = tqdm(total=events_remaining, desc="MC Generation")
    
    while events_remaining > 0:
        current_batch_size = min(batch_size, events_remaining)
        
        pids = []
        energies = []
        z_starts = []
        
        # Sample parameters
        for _ in range(current_batch_size):
            is_gamma = np.random.rand() < 0.5
            pids.append('gamma' if is_gamma else 'proton')
            # Power law energy distribution E^-2, roughly 100 to 10000 GeV
            e_min, e_max = 100.0, 10000.0
            u = np.random.rand()
            e_inv = (1.0/e_max) + u * ((1.0/e_min) - (1.0/e_max))
            energies.append(1.0 / e_inv)
            z_starts.append(20000.0)
            
        # Run fully batched simulation
        sim = ShowerSimulation(primary_types=pids, energies=energies, z_starts=z_starts)
        sim.run(max_generations=16, verbose=False)
        
        for i in range(current_batch_size):
            photons = sim.cherenkov_photons_by_event.get(i, {})
            if len(photons.get('x_ground', [])) == 0:
                continue
                
            # Randomize impact parameter within 250m radius
            r = np.sqrt(np.random.rand()) * 250.0
            theta = np.random.rand() * 2 * np.pi
            ix = r * np.cos(theta)
            iy = r * np.sin(theta)
            
            for tel in array.telescopes:
                tel.x_tel -= ix
                tel.y_tel -= iy
                
            imgs = array.ray_trace(photons)
            
            for tel in array.telescopes:
                tel.x_tel += ix
                tel.y_tel += iy
                
            # Hardware Trigger: At least 2 telescopes with > 20 PE
            trigger_count = sum(1 for img in imgs if np.sum(img) > 20)
            if trigger_count >= 2:
                passed_trigger_count += 1
                current_chunk.append({
                    'images': np.array(imgs, dtype=np.float32),
                    'energy': energies[i],
                    'label': 1 if pids[i] == 'gamma' else 0,
                    'impact_x': ix,
                    'impact_y': iy
                })
        
        events_remaining -= current_batch_size
        pbar.update(current_batch_size)
        
        # Save chunk
        if len(current_chunk) >= save_every or events_remaining == 0:
            out_filename = os.path.join(output_dir, f"sim_batch_{chunk_idx:05d}.pt")
            torch.save(current_chunk, out_filename)
            current_chunk = []
            chunk_idx += 1
            
    pbar.close()
    print(f"\nGeneration complete! {passed_trigger_count} new events passed the array hardware trigger.")

if __name__ == '__main__':
    # Start the continuous MC generation
    # Change total_events to whatever size dataset is desired.
    generate_training_data(total_events=1_000_000, batch_size=100, save_every=1000)

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

import argparse

def generate_training_data(num_gammas=10000, num_hadrons=10000, batch_size=100, save_every=1000, output_dir='data/train_raw'):
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
        
        # Assume an equal split for previously generated events
        gammas_generated = events_already_generated // 2
        hadrons_generated = events_already_generated - gammas_generated
        num_gammas -= gammas_generated
        num_hadrons -= hadrons_generated
        
    num_gammas = max(0, num_gammas)
    num_hadrons = max(0, num_hadrons)
    events_remaining = num_gammas + num_hadrons
    
    if events_remaining <= 0:
        print("All requested events have already been generated!")
        return

    print(f"Generating {num_gammas} gammas and {num_hadrons} hadrons in batches of {batch_size}, saving every {save_every} events to {output_dir}.")
    
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
            if num_gammas > 0 and num_hadrons > 0:
                is_gamma = np.random.rand() < (num_gammas / (num_gammas + num_hadrons))
            elif num_gammas > 0:
                is_gamma = True
            else:
                is_gamma = False
                
            if is_gamma:
                pids.append('gamma')
                num_gammas -= 1
            else:
                pids.append('proton')
                num_hadrons -= 1
                
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
                
            img_outputs = array.ray_trace(photons)
            
            for tel in array.telescopes:
                tel.x_tel += ix
                tel.y_tel += iy
                
            # Hardware Trigger: At least 2 telescopes with > 20 PE
            trigger_count = sum(1 for img, timing in img_outputs if np.sum(img) > 20)
            if trigger_count >= 2:
                passed_trigger_count += 1
                images = [img for img, timing in img_outputs]
                timings = [timing for img, timing in img_outputs]
                
                current_chunk.append({
                    'images': np.array(images, dtype=np.float32),
                    'timing': np.array(timings, dtype=np.float32),
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
    parser = argparse.ArgumentParser(description="Generate MC data")
    parser.add_argument('--num_gammas', type=int, default=10000, help='Number of gamma events to generate')
    parser.add_argument('--num_hadrons', type=int, default=10000, help='Number of hadron (proton) events to generate')
    parser.add_argument('--batch_size', type=int, default=100, help='Batch size for GPU pipeline')
    parser.add_argument('--save_every', type=int, default=1000, help='Save chunk every N events')
    parser.add_argument('--output_dir', type=str, default='data/train_raw', help='Output directory')
    args = parser.parse_args()
    
    generate_training_data(
        num_gammas=args.num_gammas,
        num_hadrons=args.num_hadrons,
        batch_size=args.batch_size,
        save_every=args.save_every,
        output_dir=args.output_dir
    )

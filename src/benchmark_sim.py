import os
import sys
import time
import torch
import numpy as np
from sim.shower import ShowerSimulation
from sim.telescope import TelescopeArray

def run_batched_benchmark(num_events=100):
    print(f"Starting BATCHED GPU Benchmark for {num_events} Monte Carlo events...")
    
    array = TelescopeArray.veritas_array()
    
    t0 = time.time()
    
    pids = []
    energies = []
    z_starts = []
    
    for i in range(num_events):
        is_gamma = np.random.rand() < 0.5
        pids.append('gamma' if is_gamma else 'proton')
        energies.append(100.0 + np.random.rand() * 900.0)
        z_starts.append(20000.0)
        
    print(f"Instantiating fully batched state machine for {num_events} particles...")
    sim = ShowerSimulation(primary_types=pids, energies=energies, z_starts=z_starts)
    
    # Run physics completely in one pass
    sim.run(max_generations=16, verbose=True)
    
    total_photons = 0
    passed_trigger = 0
    test_batch = []
    
    for i in range(num_events):
        photons = sim.cherenkov_photons_by_event.get(i, {})
        n_photons = len(photons.get('x_ground', []))
        total_photons += n_photons
        
        if n_photons == 0:
            continue
            
        ix = np.sqrt(np.random.rand()) * 250.0 * np.cos(np.random.rand() * 2 * np.pi)
        iy = np.sqrt(np.random.rand()) * 250.0 * np.sin(np.random.rand() * 2 * np.pi)
        
        for tel in array.telescopes:
            tel.x_tel -= ix
            tel.y_tel -= iy
            
        imgs = array.ray_trace(photons)
        
        for tel in array.telescopes:
            tel.x_tel += ix
            tel.y_tel += iy
            
        trigger_count = sum(1 for img in imgs if np.sum(img) > 20)
        if trigger_count >= 2:
            passed_trigger += 1
            if len(test_batch) < 100:
                test_batch.append({
                    'images': np.array(imgs, dtype=np.float32),
                    'energy': energies[i],
                    'label': 1 if pids[i] == 'gamma' else 0
                })
    
    t1 = time.time()
    elapsed = t1 - t0
    events_per_sec = num_events / elapsed
    
    print("\n" + "="*40)
    print("BATCHED BENCHMARK RESULTS")
    print("="*40)
    print(f"Total Events Simulated: {num_events}")
    print(f"Elapsed Time:           {elapsed:.2f} seconds")
    print(f"Throughput:             {events_per_sec:.2f} events/second")
    print(f"Total Photons Traced:   {total_photons:,}")
    print(f"Events Passed Trigger:  {passed_trigger}")
    
    if len(test_batch) > 0:
        import io
        buffer = io.BytesIO()
        torch.save(test_batch, buffer)
        size_bytes = len(buffer.getvalue())
        avg_bytes_per_event = size_bytes / len(test_batch)
        print(f"Avg Disk Storage/Event: {avg_bytes_per_event / 1024:.2f} KB")
        
        # Extrapolate to 1 Million Events
        million_time_hrs = (1_000_000 / events_per_sec) / 3600
        million_trigger = int(passed_trigger * (1_000_000 / num_events))
        million_storage_gb = (million_trigger * avg_bytes_per_event) / (1024**3)
        
        print("\nEXTRAPOLATION (1,000,000 Events):")
        print(f"Estimated Time:         {million_time_hrs:.2f} hours")
        print(f"Estimated Triggers:     {million_trigger:,}")
        print(f"Est. Storage (Triggers):{million_storage_gb:.2f} GB")

if __name__ == '__main__':
    run_batched_benchmark(100)

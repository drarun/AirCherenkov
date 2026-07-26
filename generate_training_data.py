import os
import sys
sys.path.insert(0, 'src')
import time
import numpy as np
import torch
from sim.shower import ShowerSimulation
from sim.telescope import Telescope

def generate_dataset(num_events, filename, e_min=100.0, e_max=10000.0):
    print(f"Generating {num_events} events for {filename}...")
    
    # We will store the data in PyTorch tensors
    images = []
    energies = []
    labels = []  # 1 for gamma, 0 for proton
    impact_x = []
    impact_y = []
    
    tel = Telescope(x_tel=0.0, y_tel=0.0) # Telescope position can stay at 0, and we randomize impact
    
    t0 = time.time()
    for i in range(num_events):
        if i > 0 and i % 50 == 0:
            print(f"  Generated {i}/{num_events} events ({(time.time()-t0)/60:.1f} min)")
            
        # Sample particle type: 50% gamma, 50% proton
        is_gamma = (np.random.rand() < 0.5)
        pid = 'gamma' if is_gamma else 'proton'
        
        # Sample energy from power law E^-2 (in linear space, sample uniform from E^-1)
        # E = (E_max^-1 + u * (E_min^-1 - E_max^-1))^-1
        u = np.random.rand()
        e_inv = (1.0/e_max) + u * ((1.0/e_min) - (1.0/e_max))
        energy = 1.0 / e_inv
        
        # Sample impact parameter (x, y) uniformly in area up to R=250m
        r = np.sqrt(np.random.rand()) * 250.0
        theta = np.random.rand() * 2 * np.pi
        ix = r * np.cos(theta)
        iy = r * np.sin(theta)
        
        # We simulate the shower at (0,0) and move the telescope to (-ix, -iy)
        tel.x_tel = -ix
        tel.y_tel = -iy
        
        # Run simulation
        sim = ShowerSimulation(pid, energy=energy)
        sim.run(max_generations=16, verbose=False)
        sim.get_cherenkov_dataframe() # trigger cherenkov pool
        
        # Ray trace
        img = tel.ray_trace(sim.cherenkov_photons)
        
        # Skip events that produce zero signal (below threshold)
        if np.sum(img) < 20: # arbitrary minimum PE cut
            continue
            
        images.append(img)
        energies.append(energy)
        labels.append(1 if is_gamma else 0)
        impact_x.append(ix)
        impact_y.append(iy)
        
    print(f"Finished generating {len(images)} events passing trigger.")
    
    # Save dataset
    torch.save({
        'images': torch.tensor(np.array(images), dtype=torch.float32),
        'energies': torch.tensor(energies, dtype=torch.float32),
        'labels': torch.tensor(labels, dtype=torch.long),
        'impact_x': torch.tensor(impact_x, dtype=torch.float32),
        'impact_y': torch.tensor(impact_y, dtype=torch.float32),
        'pixel_x': torch.tensor(tel.camera.pixel_x, dtype=torch.float32),
        'pixel_y': torch.tensor(tel.camera.pixel_y, dtype=torch.float32)
    }, filename)
    print(f"Saved to {filename}")

if __name__ == '__main__':
    os.makedirs('data', exist_ok=True)
    # Generate a small test set for now
    generate_dataset(100, 'data/train_events.pt')

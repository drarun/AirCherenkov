import sys
sys.path.insert(0, 'src')
from sim.shower import ShowerSimulation

print("Starting 30 TeV proton test...")
sim = ShowerSimulation(['proton'], [30000.0], [20000.0])
print("Running shower...")
try:
    sim.run(max_generations=16, verbose=True)
    print("Shower complete. Cherenkov pool photons:", len(sim.cherenkov_photons_by_event[0].get('x_ground', [])))
except Exception as e:
    print(f"Exception caught: {e}")

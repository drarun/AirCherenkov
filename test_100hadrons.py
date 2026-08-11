import sys
sys.path.insert(0, 'src')
from sim.shower import ShowerSimulation

print("Starting 100x 30 TeV proton test...")
sim = ShowerSimulation(['proton']*100, [30000.0]*100, [20000.0]*100)
print("Running shower...")
sim.run(max_generations=16, verbose=True)
print("Shower complete.")

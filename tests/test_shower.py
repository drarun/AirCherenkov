import numpy as np
import torch
from sim.shower import ShowerSimulation

def test_shower_initialization():
    sim = ShowerSimulation(primary_types=['gamma'], energies=[100.0])
    assert sim.batch_size == 1
    assert sim.active.shape == (1, 10)
    assert sim.active[0, 0].item() == sim.PID_MAP['gamma']
    assert sim.active[0, 1].item() == 100.0
    
def test_tensor_step():
    sim = ShowerSimulation(primary_types=['gamma'], energies=[100.0])
    sim.step()
    # After one generation, the gamma (if it survived) may have pair produced.
    # We just want to ensure step runs without error and active particles tensor is updated.
    assert isinstance(sim.active, torch.Tensor)
    assert sim.active.shape[1] == 10

import numpy as np
from sim.shower import ShowerSimulation, Particle

def test_shower_initialization():
    sim = ShowerSimulation('gamma', energy=100.0)
    assert sim.primary_type == 'gamma'
    assert sim.energy == 100.0
    assert len(sim.active_particles) == 1
    assert sim.active_particles[0].pid == 'gamma'

def test_particle_update():
    p = Particle('e+', 10, 0, 0, 100, 0, 0, -1)
    p.update_position(0, 0, -10, 0, 0, -1)
    assert p.z == 90
    assert len(p.track) == 2

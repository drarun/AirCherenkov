"""Small repeatable benchmark for the simulation and detector hot paths."""

from __future__ import annotations

import argparse
import time

import numpy as np
import torch

from sim.backend import device_info
from sim.shower import ShowerSimulation
from sim.telescope import TelescopeArray


def _synchronize(device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


def run_batched_benchmark(
    num_events=100,
    *,
    energy_gev=100.0,
    max_generations=10,
    seed=7,
    device='auto',
):
    """Benchmark cascade, packet construction, and four-telescope tracing."""
    rng = np.random.default_rng(seed)
    primary_types = np.where(
        rng.random(num_events) < 0.5, 'gamma', 'proton'
    ).tolist()
    simulation = ShowerSimulation(
        primary_types,
        energies=energy_gev,
        z_starts=20_000.0,
        seed=seed,
        device=device,
    )
    array = TelescopeArray.veritas_array(
        device=simulation.device,
        nsb_rate=0.0,
        pedestal_std=0.0,
    )

    print(f"Backend: {device_info(simulation.device)}")
    print(
        f"Workload: {num_events} events at {energy_gev:g} GeV, "
        f"{max_generations} generations"
    )

    _synchronize(simulation.device)
    started = time.perf_counter()
    for _ in range(max_generations):
        if simulation.active.shape[0] == 0:
            break
        simulation.step()
    _synchronize(simulation.device)
    cascade_seconds = time.perf_counter() - started

    started = time.perf_counter()
    simulation.calculate_cherenkov_pool()
    _synchronize(simulation.device)
    packet_seconds = time.perf_counter() - started

    packet_count = 0
    represented_photons = 0.0
    packet_bytes = 0
    started = time.perf_counter()
    traced_events = 0
    for event_id in range(num_events):
        packets = simulation.cherenkov_photons_by_event[event_id]
        packet_count += len(packets['x_ground'])
        represented_photons += float(np.sum(packets['weight']))
        packet_bytes += sum(np.asarray(values).nbytes for values in packets.values())
        if len(packets['x_ground']) == 0:
            continue
        outputs = array.ray_trace(packets, device=simulation.device)
        if not all(np.isfinite(trace).all() for trace, _ in outputs):
            raise RuntimeError("Detector produced non-finite traces")
        traced_events += 1
    _synchronize(simulation.device)
    detector_seconds = time.perf_counter() - started

    total_seconds = cascade_seconds + packet_seconds + detector_seconds
    results = {
        'events': num_events,
        'traced_events': traced_events,
        'cascade_seconds': cascade_seconds,
        'packet_seconds': packet_seconds,
        'detector_seconds': detector_seconds,
        'total_seconds': total_seconds,
        'events_per_second': num_events / total_seconds,
        'packets': packet_count,
        'represented_photons': represented_photons,
        'packet_bytes': packet_bytes,
    }

    print(f"Cascade:         {cascade_seconds:9.4f} s")
    print(f"Photon packets:  {packet_seconds:9.4f} s")
    print(f"Detector array:  {detector_seconds:9.4f} s")
    print(f"Total:           {total_seconds:9.4f} s")
    print(f"Throughput:      {results['events_per_second']:9.2f} events/s")
    print(f"Packets:         {packet_count:9,d}")
    print(f"Physical photons:{represented_photons:12,.0f}")
    print(f"Packet storage:  {packet_bytes / 1024**2:9.2f} MiB")
    return results


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--num-events', type=int, default=100)
    parser.add_argument('--energy-gev', type=float, default=100.0)
    parser.add_argument('--max-generations', type=int, default=10)
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--device', choices=('auto', 'cpu', 'cuda'), default='auto')
    args = parser.parse_args(argv)
    return run_batched_benchmark(
        args.num_events,
        energy_gev=args.energy_gev,
        max_generations=args.max_generations,
        seed=args.seed,
        device=args.device,
    )


if __name__ == '__main__':
    main()

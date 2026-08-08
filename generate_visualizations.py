"""Root-level runner for all AirCherenkov visualization products."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import sys
import time
from typing import List, Optional, Union


PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / 'src'
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


def _energy_label(energy_gev: float) -> str:
    if energy_gev >= 1000.0:
        return f"{energy_gev / 1000.0:g} TeV"
    return f"{energy_gev:g} GeV"


def _write_html(figure, output_path: Path, *, plotlyjs: str = 'inline') -> None:
    include_plotlyjs = True if plotlyjs == 'inline' else plotlyjs
    figure.write_html(
        output_path,
        include_plotlyjs=include_plotlyjs,
        config={'responsive': True},
    )
    print(f"      [OK] {output_path}")


def _simulate_shower(
    primary: str,
    energy_gev: float,
    start_altitude_m: float,
    max_generations: int,
    max_tracks: int,
    seed: Optional[int],
    device: str,
):
    from sim.shower import ShowerSimulation

    print(f"Simulating {_energy_label(energy_gev)} {primary} shower...")
    started = time.perf_counter()
    simulation = ShowerSimulation(
        primary_type=primary,
        energy=energy_gev,
        z_start=start_altitude_m,
        record_tracks=True,
        seed=seed,
        device=device,
    )
    simulation.run(max_generations=max_generations)

    tracks = simulation.get_tracks_dataframe(event_idx=0, max_tracks=max_tracks)
    pool = simulation.get_cherenkov_dataframe(event_idx=0)
    photons = simulation.cherenkov_photons_by_event[0]
    packet_count = len(photons['x_ground'])
    represented_photons = sum(photons.get('weight', [1.0] * packet_count))
    print(
        f"      {tracks['particle_id'].nunique():,} plotted segments, "
        f"{packet_count:,} packets representing {represented_photons:,.0f} photons "
        f"({time.perf_counter() - started:.1f}s)"
    )
    return tracks, pool, photons


def _write_camera_analysis(
    photons, energy_label: str, output_path: Path, *, device: str = 'auto'
) -> None:
    import matplotlib
    import numpy as np

    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Ellipse
    from recon.cleaning import double_pass_clean, tail_cut_clean
    from recon.hillas import compute_hillas, print_hillas
    from sim.fadc import restore_low_gain_traces
    from sim.telescope import Telescope

    print("Ray-tracing gamma shower and building camera comparison...")
    telescope = Telescope(x_tel=120.0, y_tel=0.0, device=device)
    fadc_traces, gain_flags = telescope.ray_trace(photons)
    calibrated_traces = restore_low_gain_traces(
        fadc_traces,
        gain_flags,
        low_gain_factor=telescope.fadc_config.low_gain_factor,
    )
    integrated_image = np.sum(calibrated_traces, axis=1)

    tail_mask = tail_cut_clean(telescope.camera, integrated_image)
    double_mask = double_pass_clean(telescope.camera, integrated_image)
    tail_image = np.where(tail_mask, integrated_image, 0.0)
    double_image = np.where(double_mask, integrated_image, 0.0)

    tail_hillas = compute_hillas(telescope.camera, integrated_image, tail_mask)
    double_hillas = compute_hillas(telescope.camera, integrated_image, double_mask)

    print(f"      Tail-cut: {np.sum(tail_mask)} pixels survived")
    print(f"      Double-pass: {np.sum(double_mask)} pixels survived")
    if tail_hillas is not None:
        print("\n-- Standard Tail-Cut --")
        print_hillas(tail_hillas)
    if double_hillas is not None:
        print("\n-- Double-Pass --")
        print_hillas(double_hillas)

    figure, axes = plt.subplots(1, 3, figsize=(22, 7))
    figure.suptitle(
        f"AirCherenkov: Gamma-ray Shower ({energy_label})",
        fontsize=16,
        fontweight='bold',
    )
    telescope.camera.plot_image(
        integrated_image,
        ax=axes[0],
        title="Calibrated Integrated FADC Image",
        cmap="inferno",
    )
    telescope.camera.plot_image(
        tail_image, ax=axes[1], title="Standard Tail-Cut (5/2.5)", cmap="inferno"
    )
    telescope.camera.plot_image(
        double_image,
        ax=axes[2],
        title="Double-Pass (5/2.5 > 2/1.0)",
        cmap="inferno",
    )

    if double_hillas is not None:
        ellipse = Ellipse(
            xy=(double_hillas.centroid_x, double_hillas.centroid_y),
            width=2 * double_hillas.length,
            height=2 * double_hillas.width,
            angle=np.degrees(double_hillas.psi),
            edgecolor='lime',
            facecolor='none',
            linewidth=2,
            linestyle='--',
        )
        axes[2].add_patch(ellipse)
        axes[2].plot(
            double_hillas.centroid_x,
            double_hillas.centroid_y,
            'x',
            color='lime',
            markersize=12,
            markeredgewidth=2,
        )

    figure.tight_layout()
    figure.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(figure)
    print(f"      [OK] {output_path}")


def generate_visualizations(
    *,
    energy_gev: float = 5000.0,
    start_altitude_m: float = 20000.0,
    max_generations: int = 18,
    max_tracks: int = 5000,
    output_dir: Union[Path, str] = PROJECT_ROOT,
    seed: Optional[int] = None,
    device: str = 'auto',
    plotlyjs: str = 'inline',
    include_camera: bool = True,
) -> List[Path]:
    """Simulate gamma/proton showers and write all requested visual products."""
    if not math.isfinite(energy_gev) or energy_gev <= 0:
        raise ValueError("energy_gev must be finite and greater than zero")
    if not math.isfinite(start_altitude_m) or start_altitude_m <= 0:
        raise ValueError("start_altitude_m must be finite and greater than zero")
    if max_generations < 0:
        raise ValueError("max_generations must be non-negative")
    if max_tracks <= 0:
        raise ValueError("max_tracks must be greater than zero")
    if device not in {'auto', 'cpu', 'cuda'}:
        raise ValueError("device must be one of: auto, cpu, cuda")
    if plotlyjs not in {'inline', 'directory', 'cdn'}:
        raise ValueError("plotlyjs must be one of: inline, directory, cdn")

    from sim.backend import device_info
    from sim.visualize import plot_cherenkov_pool, plot_shower

    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if seed is not None:
        import numpy as np
        import torch

        np.random.seed(seed)
        torch.manual_seed(seed)

    print("=" * 60)
    print("AirCherenkov -- Visualization Runner")
    print("=" * 60)
    print(f"Compute backend: {device_info(device)}")
    print(f"Output directory: {output_dir}")

    gamma_tracks, gamma_pool, gamma_photons = _simulate_shower(
        'gamma', energy_gev, start_altitude_m, max_generations, max_tracks,
        seed, device,
    )
    proton_tracks, proton_pool, _proton_photons = _simulate_shower(
        'proton', energy_gev, start_altitude_m, max_generations, max_tracks,
        None if seed is None else seed + 1, device,
    )
    if gamma_tracks.empty or proton_tracks.empty:
        raise RuntimeError("Track recording produced no segments; cannot generate 3D plots")

    label = _energy_label(energy_gev)
    artifacts = []

    gamma_shower_path = output_dir / 'gamma_shower.html'
    _write_html(
        plot_shower(
            gamma_tracks,
            f"Gamma-ray Shower - {label} (Electromagnetic Cascade)",
        ),
        gamma_shower_path,
        plotlyjs=plotlyjs,
    )
    artifacts.append(gamma_shower_path)

    proton_shower_path = output_dir / 'proton_shower.html'
    _write_html(
        plot_shower(
            proton_tracks,
            f"Proton Shower - {label} (Hadronic Cascade)",
        ),
        proton_shower_path,
        plotlyjs=plotlyjs,
    )
    artifacts.append(proton_shower_path)

    for pool, filename, title in (
        (gamma_pool, 'gamma_cherenkov_pool.html', f"Gamma-ray Cherenkov Pool - {label}"),
        (proton_pool, 'proton_cherenkov_pool.html', f"Proton Cherenkov Pool - {label}"),
    ):
        if pool.empty:
            print(f"      [SKIP] {filename}: no Cherenkov photons reached the ground")
            continue
        output_path = output_dir / filename
        _write_html(
            plot_cherenkov_pool(pool, title), output_path, plotlyjs=plotlyjs
        )
        artifacts.append(output_path)

    if include_camera:
        camera_path = output_dir / 'camera_cleaning_comparison.png'
        _write_camera_analysis(gamma_photons, label, camera_path, device=device)
        artifacts.append(camera_path)

    print("=" * 60)
    print(f"Generated {len(artifacts)} visualization artifact(s).")
    print("=" * 60)
    return artifacts


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate interactive AirCherenkov shower visualizations."
    )
    parser.add_argument(
        '--energy-gev', '--energy', dest='energy_gev',
        type=float, default=5000.0, metavar='GEV',
        help='Primary energy in GeV (default: 5000).',
    )
    parser.add_argument(
        '--start-altitude-m', '--start-altitude', dest='start_altitude_m',
        type=float, default=20000.0, metavar='METERS',
        help='Primary starting altitude in meters (default: 20000).',
    )
    parser.add_argument(
        '--max-generations', type=int, default=18,
        help='Maximum cascade generations (default: 18).',
    )
    parser.add_argument(
        '--max-tracks', type=int, default=5000,
        help='Maximum plotted segments per shower (default: 5000).',
    )
    parser.add_argument(
        '--output-dir', type=Path, default=PROJECT_ROOT,
        help='Artifact output directory (default: repository root).',
    )
    parser.add_argument('--seed', type=int, help='Optional NumPy/PyTorch random seed.')
    parser.add_argument(
        '--device', choices=('auto', 'cpu', 'cuda'), default='auto',
        help='Compute device (default: auto).',
    )
    parser.add_argument(
        '--plotlyjs', choices=('inline', 'directory', 'cdn'), default='inline',
        help=(
            'Plotly JavaScript mode: inline is self-contained, directory shares '
            'one local bundle, and cdn creates the smallest online-only files.'
        ),
    )
    parser.add_argument(
        '--skip-camera', action='store_true',
        help='Generate only the four interactive HTML files.',
    )
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    return generate_visualizations(
        energy_gev=args.energy_gev,
        start_altitude_m=args.start_altitude_m,
        max_generations=args.max_generations,
        max_tracks=args.max_tracks,
        output_dir=args.output_dir,
        seed=args.seed,
        device=args.device,
        plotlyjs=args.plotlyjs,
        include_camera=not args.skip_camera,
    )


if __name__ == '__main__':
    main()

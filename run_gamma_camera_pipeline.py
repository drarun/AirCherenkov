"""Run one gamma shower through telescope imaging and Hillas analysis."""

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / 'src'
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import numpy as np
from sim.shower import ShowerSimulation
from sim.telescope import Telescope
from sim.backend import device_info
from sim.fadc import restore_low_gain_traces
from recon.cleaning import tail_cut_clean, double_pass_clean
from recon.hillas import compute_hillas, print_hillas


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--energy-gev', type=float, default=5000.0)
    parser.add_argument('--start-altitude-m', type=float, default=25000.0)
    parser.add_argument('--max-generations', type=int, default=18)
    parser.add_argument('--telescope-x-m', type=float, default=0.0)
    parser.add_argument('--nsb-rate', type=float, default=2.0)
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--device', choices=('auto', 'cpu', 'cuda'), default='auto')
    parser.add_argument(
        '--output', type=Path, default=Path('camera_cleaning_comparison.png')
    )
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        import matplotlib

        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError(
            'Camera plotting requires the visualization dependencies; '
            'install aircherenkov[viz]'
        ) from exc
    print("=" * 60)
    print("AirCherenkov -- Gamma Camera Analysis Pipeline")
    print("=" * 60)
    print(f"Compute backend: {device_info(args.device)}")

    # -- Shower Simulation -------------------------------------------
    print(f"\n[1/4] Simulating a {args.energy_gev:g} GeV gamma-ray shower...")
    sim = ShowerSimulation(
        primary_type='gamma',
        energy=args.energy_gev,
        z_start=args.start_altitude_m,
        seed=args.seed,
        device=args.device,
    )
    sim.run(max_generations=args.max_generations)

    photons = sim.cherenkov_photons_by_event[0]
    n_packets = len(photons['x_ground'])
    represented_photons = np.sum(photons.get('weight', np.ones(n_packets)))
    print(
        f"      Generated {n_packets:,} weighted packets representing "
        f"{represented_photons:,.0f} Cherenkov photons on ground."
    )

    # -- Ray-tracing -------------------------------------------------
    print("\n[2/4] Ray-tracing through telescope optics...")
    tel = Telescope(
        x_tel=args.telescope_x_m,
        y_tel=0.0,
        mirror_radius=6.0,
        focal_length=15.0,
        shower_start_altitude=args.start_altitude_m,
        device=args.device,
    )
    fadc_traces, gain_flags = tel.ray_trace(
        photons, nsb_rate=args.nsb_rate,
        generator=sim.generator,
    )
    calibrated_traces = restore_low_gain_traces(
        fadc_traces,
        gain_flags,
        low_gain_factor=tel.fadc_config.low_gain_factor,
    )
    integrated_image = np.sum(calibrated_traces, axis=1)

    integrated_charge = np.sum(integrated_image)
    print(
        f"      Camera: {tel.camera.n_pixels} pixels, "
        f"~{integrated_charge:.0f} calibrated integrated counts"
    )

    # -- Image Cleaning ----------------------------------------------
    print("\n[3/4] Cleaning images...")

    mask_std = tail_cut_clean(
        tel.camera, integrated_image, picture_thresh=5.0, boundary_thresh=2.5
    )
    img_std = np.where(mask_std, integrated_image, 0.0)

    mask_dp = double_pass_clean(
        tel.camera,
        integrated_image,
        pic1=5.0,
        bnd1=2.5,
        pic2=2.0,
        bnd2=1.0,
        dist_tolerance=0.2,
    )
    img_dp = np.where(mask_dp, integrated_image, 0.0)

    print(f"      Tail-cut: {np.sum(mask_std)} pixels survived")
    print(f"      Double-pass: {np.sum(mask_dp)} pixels survived")

    # -- Hillas Parameterization -------------------------------------
    print("\n[4/4] Computing Hillas parameters...")

    hillas_std = compute_hillas(tel.camera, integrated_image, mask_std)
    hillas_dp = compute_hillas(tel.camera, integrated_image, mask_dp)

    print("\n-- Standard Tail-Cut --")
    print_hillas(hillas_std)

    print("\n-- Double-Pass --")
    print_hillas(hillas_dp)

    # -- Plotting ----------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(22, 7))

    tel.camera.plot_image(
        integrated_image,
        ax=axes[0],
        title="Calibrated Integrated FADC Image",
        cmap="inferno",
    )
    tel.camera.plot_image(
        img_std, ax=axes[1], title="Standard Tail-Cut (5/2.5)", cmap="inferno"
    )
    tel.camera.plot_image(
        img_dp,
        ax=axes[2],
        title="Double-Pass (5/2.5 > 2/1.0)",
        cmap="inferno",
    )

    # Draw the Hillas ellipse on the Double-Pass plot if available
    if hillas_dp is not None:
        ax = axes[2]
        from matplotlib.patches import Ellipse
        ellipse = Ellipse(
            xy=(hillas_dp.centroid_x, hillas_dp.centroid_y),
            width=2 * hillas_dp.length,
            height=2 * hillas_dp.width,
            angle=np.degrees(hillas_dp.psi),
            edgecolor='lime',
            facecolor='none',
            linewidth=2,
            linestyle='--',
        )
        ax.add_patch(ellipse)
        ax.plot(
            hillas_dp.centroid_x,
            hillas_dp.centroid_y,
            'x',
            color='lime',
            markersize=12,
            markeredgewidth=2,
        )

    plt.suptitle(
        f"AirCherenkov: Gamma-ray Shower ({args.energy_gev:g} GeV)",
        fontsize=14,
        fontweight='bold',
    )
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output, bbox_inches='tight', dpi=150)
    plt.close(fig)
    print(f"\nSaved: {output}")
    return output

if __name__ == "__main__":
    main()

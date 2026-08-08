# AirCherenkov

AirCherenkov is a project for simulating, visualizing, and reconstructing atmospheric Cherenkov radiation from gamma-ray and proton showers.

## Installation

AirCherenkov supports Python 3.10+ on Windows, macOS, and Linux. From a cloned
repository, create an environment and install the project itself:

```bash
python -m venv .venv
python -m pip install -e ".[viz,dev]"
```

Optional stacks are installed explicitly:

```bash
python -m pip install -e ".[ml]"       # PyTorch Geometric and scikit-learn
python -m pip install -e ".[sbi]"      # simulation-based inference example
python -m pip install -e ".[io]"       # HDF5 reader
```

`requirements.txt` remains a convenience environment for core simulation,
visualization, and tests. `pyproject.toml` is the canonical package definition.

> **Hardware:** Select `--device auto`, `cpu`, or `cuda`. CUDA is recommended for multi-TeV production, but CPU and CUDA now execute the same packet, camera, timing, noise, and gain model.

> **Scientific scope:** This is a visualization and reconstruction research prototype. Its shower and detector models are simplified and are not a replacement for a validated package such as CORSIKA/sim_telarray.

## Physics Overview
AirCherenkov simulates the complex physics of extensive air showers and the resulting Cherenkov radiation:
- **Particle Cascades**: Simulates electromagnetic pair production/bremsstrahlung and a simplified proton/pion branching model.
- **Atmospheric Optics**: Uses beta- and altitude-dependent Cherenkov thresholds, angles, and weighted Frank-Tamm photon yield.
- **Multiple Coulomb Scattering**: Models Highland scattering using vectorized tensor cross-products to capture the lateral spread of the shower.
- **Ray-Tracing**: Batches compatible telescope arrays and accumulates weighted photon packets into a shared FADC detector model.

Cherenkov output rows are weighted transport packets. The `weight` field is the
number of physical photons represented by a packet. Density calculations and
custom consumers must sum weights rather than count rows. This keeps total light
normalization when a packet budget is applied and greatly reduces memory use.

## Pipeline Overview

The project provides simulation and classical image reconstruction, plus experimental GNN and SBI components.

### 1. Physics Simulation
You can run the core physics simulations and generate event outputs using:
- `generate_visualizations.py`: Root-level runner that simulates gamma and proton events and generates all five visualization products.
- `run_gamma_camera_pipeline.py`: Runs one gamma event through the shower, telescope, cleaning, and Hillas-analysis stages.
- `regenerate_all.py` and `run_full_sim.py`: Compatibility aliases for the older command names.

An editable installation also provides `aircherenkov-visualize`,
`aircherenkov-generate`, and `aircherenkov-camera`; these work from any current
directory without setting `PYTHONPATH`.

`src/sim/visualize.py` is the reusable plotting library; it does not run simulations or write files itself.

Particle tracks are retained on CPU only when `ShowerSimulation(..., record_tracks=True)` is requested. Production/training simulations leave this disabled to avoid unnecessary host transfers and memory use.

### 2. Training Data Generation
Generate reproducible, trigger-selected raw events with:

```bash
aircherenkov-generate --num-gammas 1000 --num-hadrons 1000 --seed 7 --device auto
```

The default output is `data/train/raw`, matching `CherenkovDataset(root="data/train")`.
Each production has an atomic `.aircherenkov/manifest.json`, pending-event
checkpoints, and per-batch throw ledgers. The ledgers retain rejected trials,
sampling probabilities, trigger decisions, seeds, and configuration hashes so
trigger efficiency and sampling weights remain auditable.

### 3. Graph Neural Network (GNN) Training
After installing the `ml` extra, train and evaluate the independent energy and
classification networks with:

```bash
aircherenkov-train --data-root data/train --seed 7
aircherenkov-evaluate --dataset-root data/test
aircherenkov-event-display --dataset-root data/test --event-index 0
```

Processed graphs preserve event/telescope IDs, array positions, impact metadata,
trigger provenance, and sampling weights. Train/validation splitting is grouped
by shower event, so telescope views from the same shower cannot leak across the
split. Low-gain attenuation is restored before charge and feature extraction.

### 4. Simulation-Based Inference (SBI)
`src/recon/sbi_fit.py` is currently a standalone SBI power-law example. Connecting SBI to shower and camera observations remains future work.

## Visual Analysis Products

The simulation and analysis tools generate several rich visual products for inspection and analysis:
- **Interactive 3D Plotly HTML files**: Explore 3D representations of shower development and the Cherenkov photon pool on the ground (e.g., `gamma_shower.html`, `proton_shower.html`, `gamma_cherenkov_pool.html`, `proton_cherenkov_pool.html`).
- **LDF Density Plots**: Lateral Distribution Function visualizations showing photon density versus distance from the shower core (e.g., `ldf_comparison.png`).
- **Hexagonal Camera Images**: 2D visualizations of the simulated camera focal plane, depicting individual pixels, signal intensity, and the effects of camera image cleaning algorithms (e.g., `camera_image_gamma.png`, `camera_cleaning_comparison.png`).

Run the installed visualization entry point from any directory:

```bash
aircherenkov-visualize --seed 7 --device auto
```

For a quicker lower-energy run without the camera PNG:

```bash
aircherenkov-visualize --energy-gev 500 --max-generations 10 --skip-camera
```

The default `--plotlyjs inline` produces fully self-contained HTML. Use
`--plotlyjs directory` to share one offline Plotly bundle among all four files,
or `--plotlyjs cdn` for the smallest files when internet access is acceptable.

The runner creates:

- `gamma_shower.html` and `proton_shower.html` — responsive, self-contained interactive 3D particle tracks.
- `gamma_cherenkov_pool.html` and `proton_cherenkov_pool.html` — interactive ground footprints.
- `camera_cleaning_comparison.png` — raw/cleaned camera images and Hillas ellipse.

## Benchmarking

Run the phase-separated benchmark from the repository checkout with:

```bash
PYTHONPATH=src python src/benchmark_sim.py --num-events 50 --energy-gev 100 --device cpu
```

It reports cascade time, weighted-packet construction, four-telescope detector
time, represented physical photons, packet count, and packet storage. Use fixed
seeds and identical arguments when comparing changes.

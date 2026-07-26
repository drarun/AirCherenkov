# AirCherenkov

AirCherenkov is a project for simulating, visualizing, and reconstructing atmospheric Cherenkov radiation from gamma-ray and proton showers.

## Installation

The project uses Python and can be run on Windows, Mac, or Linux. 

1. Ensure you have Python 3.8+ installed.
2. Clone this repository and navigate to the project directory.
3. Install the required dependencies using `pip`:
```bash
pip install -r requirements.txt
```

> **Note on Hardware:** The AirCherenkov simulation engine (`src/sim/shower.py` and `src/sim/backend.py`) has been completely refactored into a **fully tensorized PyTorch state machine**. It assumes a CUDA-enabled GPU to run efficiently, abandoning CPU-fallback in favor of massively parallel batch processing of physics interactions in VRAM.

## Physics Overview
AirCherenkov simulates the complex physics of extensive air showers and the resulting Cherenkov radiation:
- **Particle Cascades**: Simulates electromagnetic (pair production, Bremsstrahlung) and hadronic (pion decay) interactions as primary cosmic rays strike the upper atmosphere.
- **Atmospheric Optics**: Tracks the density-dependent Cherenkov emission threshold and the photon yield factor based on the Frank-Tamm formula.
- **Multiple Coulomb Scattering**: Models Highland scattering using vectorized tensor cross-products to capture the lateral spread of the shower.
- **Ray-Tracing**: Implements highly parallelized projections to trace emitted Cherenkov photons from the 3D shower track down to the focal plane of ground-based telescope arrays (like VERITAS or CTA).

## Pipeline Overview

The project provides an end-to-end pipeline from simulation to neural network-based and SBI-based event reconstruction.

### 1. Physics Simulation
You can run the core physics simulations and generate event outputs using:
- `run_full_sim.py`: Runs a full simulation for a specific event configuration.
- `regenerate_all.py`: Orchestrates and runs a larger batch of simulations to regenerate full event libraries and visual plots.

### 2. Training Data Generation
Generate PyTorch Geometric graphs and formatted data for neural network training using:
- `generate_training_data.py`: Processes simulation outputs (like hexagonal camera images and shower parameters) into a format suitable for the GNN.

### 3. Graph Neural Network (GNN) Training
Train the Graph Neural Network for energy estimation and primary particle classification:
- `train_gnn.py`: Trains the GNN on the generated training data.

### 4. Simulation-Based Inference (SBI)
Perform simulation-based inference to fit physical parameters of the events:
- `src/recon/sbi_fit.py`: Runs the SBI fitting procedure to estimate shower parameters (like energy and direction) based on simulated observations and camera images.

## Visual Analysis Products

The simulation and analysis tools generate several rich visual products for inspection and analysis:
- **Interactive 3D Plotly HTML files**: Explore 3D representations of shower development and the Cherenkov photon pool on the ground (e.g., `gamma_shower.html`, `proton_shower.html`, `gamma_cherenkov_pool.html`, `proton_cherenkov_pool.html`).
- **LDF Density Plots**: Lateral Distribution Function visualizations showing photon density versus distance from the shower core (e.g., `ldf_comparison.png`).
- **Hexagonal Camera Images**: 2D visualizations of the simulated camera focal plane, depicting individual pixels, signal intensity, and the effects of camera image cleaning algorithms (e.g., `camera_image_gamma.png`, `camera_cleaning_comparison.png`).

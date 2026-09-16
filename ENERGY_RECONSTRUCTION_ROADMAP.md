# Energy Reconstruction Roadmap & Root-Cause Analysis (SpatiotemporalGNN v5)

## 1. Executive Summary & Benchmark Status

Following 25 full training epochs on `data/train_large` (311,692 events), the `SpatiotemporalGNN v5` model successfully resolved the catastrophic zero-variance mode collapse that plagued earlier iterations. However, when benchmarked against official **VERITAS** performance standards, the current energy reconstruction is **suboptimal**:

| Metric | VERITAS Benchmark | SpatiotemporalGNN v5 | Status / Assessment |
| :--- | :---: | :---: | :--- |
| **Core Energy Resolution (68%)** | **`15% – 25%`** | **`45% – 65%`** | **~2–3× worse** than standard lookup tables |
| **Calibrated Safe Energy Window** | $|\mathrm{Bias}| \le 10\%$ across **0.2 – 10+ TeV** | $|\mathrm{Bias}| \le 10\%$ across **0.18 – 0.45 TeV** | Bounded in narrow sliver; collapses $>1$ TeV |
| **High-Energy Bias (>3 TeV)** | $< 10\%$ | **$-60\%$ to $-89\%$** | Severe compression / underprediction |
| **Linear Correlation ($r$)** | $> 0.95$ | **`0.750`** | Good baseline, but broad dispersion |
| **Classification AUC** | $\sim 0.95$ | **`0.950`** | State-of-the-art $\gamma/h$ separation |

---

## 2. In-Depth Root-Cause Analysis

### Root Cause 1: Steep Power-Law Training Distribution ($E^{-2.5}$)
- **The Issue**: Showers in `train_large` follow a steep astrophysical power law:
  - 98 GeV: 53,510 events
  - 162 GeV: 42,135 events
  - 1.2 TeV: 6,983 events
  - 5.4 TeV: 1,629 events
  - 24.4 TeV: 337 events
- **The Consequence**: Over 90% of training events are $<500$ GeV. Any standard regression loss (MSE or Huber) minimizes global error by predicting conservatively toward the dataset median (~250 GeV). In a 64-event mini-batch, events $>3$ TeV appear rarely (or not at all), so the optimizer rarely receives gradients pushing predictions into the multi-TeV regime.
- **IACT Standard Practice**: Observatories (VERITAS, CTA, MAGIC, H.E.S.S.) **never train energy estimators on a power-law spectrum**. Training is always performed on a **flat logarithmic spectrum** ($\mathrm{d}N/\mathrm{d}\log E = \text{const}$, or $E^{-1}$) so each decade of energy contributes equally to gradient updates.

### Root Cause 2: Residual `GraphNorm` in Layers 1, 2, and 3
- **The Issue**: While `GraphNorm` was bypassed at Layer 4, Layers 1, 2, and 3 continue to apply `GraphNorm(x, batch)`.
- **The Consequence**: `GraphNorm` zero-centers the node feature mean graph-by-graph. By the time features reach Layer 4, the GNN's deep spatial representation has had its graph-wide light yield subtracted three times consecutively. The energy head is forced to rely almost entirely on the raw skip connections rather than learned graph representations.

### Root Cause 3: Opposing Gradient Objectives in Shared Backbone
- **Classification**: Demands **scale-invariance** (a 1 TeV proton and a 100 GeV gamma must be separated purely by transverse shower morphology, compactness, and timing spread, independent of total brightness).
- **Energy Estimation**: Demands **scale-dependence** (total Cherenkov light yield is the fundamental observable for primary particle energy).
- **The Consequence**: Forcing both tasks through the identical 4-layer GAT backbone creates gradient interference, where classification pulls the filters toward shape invariance while regression fights for scale preservation.

---

## 3. Concrete Action Plan for Next Session

### Priority 1: Log-Uniform Resampling / Dataset Re-weighting
- **Option A (Zero Simulation Cost)**: Implement a `WeightedRandomSampler` in the PyTorch `DataLoader` that inversely samples events based on global (dataset-wide) $\log_{10}(E)$ density. Every mini-batch will contain an equal distribution of showers from 100 GeV to 30 TeV.
- **Option B**: Run `generate_training_data.py` with an explicit flat spectral index ($\gamma = 1.0$) specifically for energy regression training.

### Priority 2: Decoupled Backbone or Scale-Preserving Normalization
- **Option A (Channel-Wise LayerNorm)**: Replace `GraphNorm` with `nn.LayerNorm(hidden_channels * heads)` across all layers. `LayerNorm` normalizes across feature channels for each node independently without subtracting the graph-wide light intensity.
- **Option B (Dual-Stream / Dedicated Architecture)**:
  - Branch into two dedicated networks or split after the 1D temporal convolution:
    1. `MorphologyGNN`: GAT + GraphNorm $\to$ Classification Head ($\gamma/h$).
    2. `CalorimetryGNN`: GCN/GAT + LayerNorm + Multi-scale Light Aggregation $\to$ Energy Head.

### Priority 3: Explicit Array Geometry Features
- Feed telescope-level integrated observables directly into the energy readout head:
  - Total array light yield: $\log_{10}\left(\sum_{\mathrm{tels}} \mathrm{Size}\right)$
  - Number of triggered telescopes ($N_\mathrm{tels}$)
  - Max single-telescope Size and distance from reconstructed core position.

### Priority 4: Post-Hoc Empirical Energy Correction
- Standard practice in observatory analysis: fit a 2D spline / polynomial $E_\mathrm{calibrated} = f(E_\mathrm{pred}, \text{Impact})$ to systematically flatten any residual threshold and high-energy roll-off bias.

---

## 4. Key Artifacts & Diagnostic Files
- **Final Evaluation PDF**: [`C:\Users\aruns\OneDrive\Desktop\VERITAS_GNN_Energy_Performance.pdf`](file:///C:/Users/aruns/OneDrive/Desktop/VERITAS_GNN_Energy_Performance.pdf)
- **High-Res Diagnostic Plot**: [`data/energy_performance.png`](file:///C:/Users/aruns/Projects/AirCherenkov/data/energy_performance.png)
- **Standalone Resolution Plot**: [`data/energy_resolution_vs_veritas.png`](file:///C:/Users/aruns/Projects/AirCherenkov/data/energy_resolution_vs_veritas.png)
- **Standalone Bias Plot**: [`data/energy_bias_vs_veritas.png`](file:///C:/Users/aruns/Projects/AirCherenkov/data/energy_bias_vs_veritas.png)
- **Saved Best Model Weights**: [`data/spatiotemporal_gnn_v5.pt`](file:///C:/Users/aruns/Projects/AirCherenkov/data/spatiotemporal_gnn_v5.pt) (Epoch 25, `val_loss = 0.4690`)

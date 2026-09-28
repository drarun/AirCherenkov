# Energy Reconstruction Improvement Roadmap (Revised)

After a thorough code audit by Claude Opus 4.6 (Thinking), this plan has been revised. The core finding is that our previous binned metric was conflating bias with scatter, and the optimizer was actively ignoring high-energy events due to a combination of power-law spectrum imbalance and an overly aggressive HuberLoss delta.

## Prioritized Action Plan

| Priority | Action | Description | Status |
| :--- | :--- | :--- | :--- |
| **P0** | Replace `HuberLoss(delta=0.3)` with `MSELoss` | `HuberLoss` in log-space clamped the gradient for any event with >2x error. This actively suppressed learning for the worst-predicted high-energy events. Switched to `MSELoss`. | ✅ Implemented |
| **P1** | Log-uniform `WeightedRandomSampler` | The $E^{-2.0}$ spectrum meant >90% of events were low energy. We replaced the unstable per-batch inverse-frequency weights with a proper PyTorch `WeightedRandomSampler` to ensure every mini-batch sees a flat distribution across the decades. | ✅ Implemented |
| **P2** | Fix Train/Val split & metric tracking | Fixed RNG non-determinism that could leak data across resumes. Fixed validation loop to use the unweighted `.mean()` loss matching the training loop. | ✅ Implemented |
| **P3** | Replace `GraphNorm` with `LayerNorm` | `GraphNorm` strips absolute scale information per-graph by subtracting the mean. Replaced all 4 normalization layers on the GNN backbone with `LayerNorm` to preserve telescope-level amplitude while stabilizing training. | ✅ Implemented |
| **P4** | Fix Resolution Metric | The 68th percentile of absolute fractional error overestimates resolution when bias is large. Switched to standard IACT metric: $\frac{1}{2}(Q_{84} - Q_{16})$. | ✅ Implemented |

## Deferred (Future Enhancements)

- **Dual-stream architecture**: Previously considered, but multi-task learning with a shared backbone should be fine given the independent energy pooling head and skip connections. Wait to see if P0-P3 fix the issue.
- **Explicit array-level geometry**: `impact_x`/`impact_y` and `N_tels` are still missing from the node features. This is a very strong proxy for energy and should be added to `generate_training_data.py` and `dataset.py` if performance is still lacking.
- **Post-hoc calibration spline**: Only necessary if uncorrectable systematic biases remain.

## Next Steps

1. Launch overnight training run with the new code.
2. Monitor training/val loss curves.
3. Re-run `evaluate_gnns.py` to check the unbiased resolution metric.

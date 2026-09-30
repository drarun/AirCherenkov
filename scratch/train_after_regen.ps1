$ErrorActionPreference = "Stop"

Write-Host "Waiting for data regeneration to complete..."
Wait-Process -Name "python" -ErrorAction SilentlyContinue | Where-Object { $_.CommandLine -like "*generate_training_data*" } | Wait-Process

Write-Host "Starting v8 training on clean data..."
# Run the training
python train_gnn.py --model_path data/spatiotemporal_gnn_v8_clean.pt --finetune_from data/spatiotemporal_gnn_v7.pt --epochs 15 --extra_root data/train_large_highE

Write-Host "Evaluating..."
python src/analysis/evaluate_gnns.py --model data/spatiotemporal_gnn_v8_clean.pt
python scratch/plot_veritas_comparison.py

Write-Host "All done!"

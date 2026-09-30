$ErrorActionPreference = "Continue"
$env:PYTHONPATH = "src"

Write-Host "Starting high-E generation..."
python generate_training_data.py --output_dir data/train_large_highE/raw --num_gammas 100000 --num_hadrons 0 --batch_size 500 --spectral_index 1.0 --e_min 1000.0 --e_max 30000.0

Write-Host "Compiling graphs for baseline..."
python src/analysis/dataset.py --root data/train_large

Write-Host "Compiling graphs for high-E..."
python src/analysis/dataset.py --root data/train_large_highE

Write-Host "Starting v8 training..."
python train_gnn.py --model_path data/spatiotemporal_gnn_v8.pt --finetune_from data/spatiotemporal_gnn_v7.pt --epochs 15 --extra_root data/train_large_highE

Write-Host "Evaluating..."
python src/analysis/evaluate_gnns.py --model data/spatiotemporal_gnn_v8.pt
python scratch/plot_veritas_comparison.py

Write-Host "All done!"

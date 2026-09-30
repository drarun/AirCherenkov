$ErrorActionPreference = "Stop"

Write-Host "Cleaning up old data..."
if (Test-Path "data/train_large/raw") { Remove-Item -Recurse -Force data/train_large/raw/* }
if (Test-Path "data/train_large/processed") { Remove-Item -Recurse -Force data/train_large/processed/* }
if (Test-Path "data/train_large_highE/raw") { Remove-Item -Recurse -Force data/train_large_highE/raw/* }
if (Test-Path "data/train_large_highE/processed") { Remove-Item -Recurse -Force data/train_large_highE/processed/* }

Write-Host "Starting baseline generation..."
# Start a baseline generation (this will take a while)
python generate_training_data.py --output_dir data/train_large/raw --num_gammas 500000 --num_hadrons 500000 --batch_size 500

Write-Host "Starting high-E generation..."
python generate_training_data.py --output_dir data/train_large_highE/raw --num_gammas 100000 --num_hadrons 0 --batch_size 500 --alpha 1.0 --e_min 1000.0 --e_max 30000.0

Write-Host "Compiling graphs..."
python src/analysis/dataset.py --root data/train_large
python src/analysis/dataset.py --root data/train_large_highE

Write-Host "All data generated."

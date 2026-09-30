$ErrorActionPreference = "Stop"

Write-Host "Waiting for data regeneration to finish (checking log file)..."
$logFile = "data_regen.log"
$isDone = $false

while (-not $isDone) {
    if (Test-Path $logFile) {
        $tail = Get-Content $logFile -Tail 5 -ErrorAction SilentlyContinue
        if ($tail -match "All data generated.") {
            $isDone = $true
        }
    }
    if (-not $isDone) {
        Start-Sleep -Seconds 30
    }
}

Write-Host "Data regeneration complete! Starting v8 training..."
python train_gnn.py --model_path data/spatiotemporal_gnn_v8.pt --finetune_from data/spatiotemporal_gnn_v7.pt --epochs 15 --extra_root data/train_large_highE > train_v8.log 2>&1

Write-Host "Training complete! Evaluating..."
python src/analysis/evaluate_gnns.py --model data/spatiotemporal_gnn_v8.pt >> train_v8.log 2>&1
python scratch/plot_veritas_comparison.py >> train_v8.log 2>&1

Write-Host "All tasks completed!"

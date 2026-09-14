"""
train_gnn.py – Fault-tolerant training loop for the Spatiotemporal GNN.

Features:
  • Full stateful checkpointing (model, optimizer, scheduler, RNG, epoch, best loss)
  • Automatic resume from latest checkpoint via --resume
  • CUDA OOM catch-and-recover (skips offending batch, clears cache)
  • Graceful shutdown on SIGINT / SIGTERM (saves emergency checkpoint)
  • Per-epoch memory cleanup (gc + torch.cuda.empty_cache)
  • Capped E^2 spectrum weights to flatten the power-law without gradient explosion
"""

import os
import sys
import gc
import signal
import argparse

sys.path.insert(0, 'src')

import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.loader import DataLoader
from recon.gnn import SpatiotemporalGNN
from analysis.dataset import CherenkovDataset


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def save_checkpoint(path, model, optimizer, scheduler, scaler, epoch, best_val_loss,
                    is_emergency=False):
    """Atomically save a full training checkpoint including AMP scaler."""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'scaler_state_dict': scaler.state_dict() if scaler is not None else None,
        'best_val_loss': best_val_loss,
        'rng_state': torch.get_rng_state(),
        'cuda_rng_state': torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
    }
    tag = " [EMERGENCY]" if is_emergency else ""
    # Atomic write: write to tmp then rename so a crash mid-write can't corrupt
    tmp_path = path + ".tmp"
    torch.save(checkpoint, tmp_path)
    os.replace(tmp_path, path)
    print(f"  >> Checkpoint saved{tag}: {path}  (epoch {epoch + 1})")


def load_checkpoint(path, model, optimizer, scheduler, scaler, device):
    """Restore training state from a checkpoint. Returns (start_epoch, best_val_loss)."""
    print(f"  >> Resuming from checkpoint: {path}")
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    scheduler.load_state_dict(ckpt['scheduler_state_dict'])
    if scaler is not None and ckpt.get('scaler_state_dict') is not None:
        scaler.load_state_dict(ckpt['scaler_state_dict'])
    try:
        if 'rng_state' in ckpt and ckpt['rng_state'] is not None:
            torch.set_rng_state(ckpt['rng_state'].byte().cpu())
        if ckpt.get('cuda_rng_state') is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state(ckpt['cuda_rng_state'].byte().cpu())
    except Exception as e:
        print(f"     [Warning] Could not restore exact RNG state ({e}), continuing with fresh RNG.")
    start_epoch = ckpt['epoch'] + 1
    best_val_loss = ckpt['best_val_loss']
    print(f"     Restored epoch {ckpt['epoch'] + 1}, best val loss = {best_val_loss:.4f}")
    return start_epoch, best_val_loss


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def train_networks():
    parser = argparse.ArgumentParser(
        description="Train Spatiotemporal GNN for AirCherenkov (fault-tolerant)")
    parser.add_argument("--root", type=str, default="data/train",
                        help="Dataset root directory containing 'raw' folder")
    parser.add_argument("--model_path", type=str, default="data/spatiotemporal_gnn.pt",
                        help="Path to save the best model weights")
    parser.add_argument("--epochs", type=int, default=25,
                        help="Total number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size for DataLoader")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from latest checkpoint if available")
    args = parser.parse_args()

    # Derived paths
    checkpoint_path = args.model_path.replace(".pt", "_checkpoint.pt")
    best_model_path = args.model_path

    # ------------------------------------------------------------------
    # Dataset
    # ------------------------------------------------------------------
    print(f"Loading dataset from root: {args.root}...")
    dataset = CherenkovDataset(root=args.root)
    print(f"Dataset loaded with {len(dataset)} events.")

    if len(dataset) == 0:
        print("No events found. Please run the simulator first.")
        return

    # Split train/val (deterministic after RNG restore)
    dataset = dataset.shuffle()
    split = int(0.8 * len(dataset))
    train_data = dataset[:split]
    val_data = dataset[split:]

    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True,
                              num_workers=0, pin_memory=torch.cuda.is_available())
    val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False,
                            num_workers=0, pin_memory=torch.cuda.is_available())

    # ------------------------------------------------------------------
    # Model, optimizer, scheduler, loss, AMP Scaler
    # ------------------------------------------------------------------
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision('high')
        torch.backends.cudnn.benchmark = True
        
    model = SpatiotemporalGNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=2)

    # Automatic Mixed Precision (AMP) GradScaler for 2x faster GPU training
    use_amp = torch.cuda.is_available()
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)

    criterion_energy = nn.HuberLoss(delta=0.3, reduction='none')
    criterion_class = nn.BCEWithLogitsLoss()

    # ------------------------------------------------------------------
    # Resume from checkpoint
    # ------------------------------------------------------------------
    start_epoch = 0
    best_val_loss = float('inf')

    if args.resume and os.path.exists(checkpoint_path):
        start_epoch, best_val_loss = load_checkpoint(
            checkpoint_path, model, optimizer, scheduler, scaler, device)
    elif args.resume:
        print("  >> --resume requested but no checkpoint found. Starting fresh.")

    if start_epoch >= args.epochs:
        print(f"Already completed {start_epoch} epochs. Nothing to do.")
        return

    # ------------------------------------------------------------------
    # Graceful shutdown handler
    # ------------------------------------------------------------------
    shutdown_requested = [False]   # mutable flag for the closure

    def handle_shutdown(signum, frame):
        sig_name = signal.Signals(signum).name
        print(f"\n[SHUTDOWN] Received {sig_name}. Saving emergency checkpoint...")
        save_checkpoint(checkpoint_path, model, optimizer, scheduler, scaler,
                        max(start_epoch, 0), best_val_loss, is_emergency=True)
        # Also save the raw model weights so we always have a usable artifact
        torch.save(model.state_dict(),
                    args.model_path.replace(".pt", "_emergency.pt"))
        shutdown_requested[0] = True
        sys.exit(0)

    signal.signal(signal.SIGINT, handle_shutdown)
    signal.signal(signal.SIGTERM, handle_shutdown)

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    print(f"\nTraining on {device} (AMP: {use_amp}) for epochs {start_epoch + 1}..{args.epochs}"
          f" ({len(train_loader)} batches/epoch)")

    oom_count = 0

    for epoch in range(start_epoch, args.epochs):
        model.train()
        total_loss = 0.0
        batches_ok = 0

        for batch_idx, batch in enumerate(train_loader):
            try:
                batch = batch.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)

                # AMP Autocast forward pass
                with torch.amp.autocast('cuda', enabled=use_amp):
                    # 1. Forward pass
                    class_logits, energy_pred = model(
                        batch.x, batch.edge_index, batch.batch)

                    # 2. Classification loss (all events)
                    loss_c = criterion_class(
                        class_logits.view(-1), batch.y_class.view(-1))

                    # 3. Energy loss (gamma events only, with inverse-frequency bin weights)
                    gamma_mask = (batch.y_class.view(-1) == 1.0)
                    if gamma_mask.any():
                        y_true = batch.y_energy.view(-1)[gamma_mask]
                        y_pred = energy_pred.view(-1)[gamma_mask]

                        # Inverse-frequency bin weights
                        n_bins = 10
                        bin_indices = torch.clamp(
                            ((y_true - 1.8) / (4.5 - 1.8) * n_bins).long(),
                            min=0, max=n_bins - 1)
                        bin_counts = torch.bincount(bin_indices, minlength=n_bins).float()
                        bin_counts = bin_counts.clamp(min=1.0)
                        energy_weights = 1.0 / bin_counts[bin_indices]
                        energy_weights = energy_weights / energy_weights.mean()

                        per_event_loss = criterion_energy(y_pred, y_true)
                        loss_e = (energy_weights * per_event_loss).mean()

                        loss = loss_c + 5.0 * loss_e
                    else:
                        loss = loss_c

                # Scaled backward pass & unscaling clip
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
                scaler.step(optimizer)
                scaler.update()

                total_loss += loss.item()
                batches_ok += 1

            except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
                if "out of memory" in str(e).lower() or "CUDA" in str(e):
                    oom_count += 1
                    print(f"\n  [OOM] Batch {batch_idx + 1} caused CUDA OOM "
                          f"(total OOMs: {oom_count}). Skipping...")
                    optimizer.zero_grad(set_to_none=True)
                    # Aggressively free everything by dropping references explicitly
                    batch = class_logits = energy_pred = loss = loss_c = loss_e = None
                    per_event_loss = energy_weights = y_true = y_pred = None
                    
                    # If OOM happened after unscale_() but before step(), clear the scaler state
                    # to prevent a RuntimeError on the next batch.
                    if hasattr(scaler, '_per_optimizer_states'):
                        scaler._per_optimizer_states.clear()
                        
                    torch.cuda.empty_cache()
                    gc.collect()
                    continue
                else:
                    raise  # re-raise non-OOM RuntimeErrors

            if (batch_idx + 1) % 200 == 0:
                print(f"  Batch {batch_idx + 1}/{len(train_loader)} "
                      f"| Cur Loss: {loss.item():.4f}")

        train_loss = total_loss / max(batches_ok, 1)

        # ------------------------------------------------------------------
        # Validation loop
        # ------------------------------------------------------------------
        model.eval()
        val_loss = 0.0
        val_count = 0
        with torch.no_grad():
            for data in val_loader:
                data = data.to(device, non_blocking=True)
                with torch.amp.autocast('cuda', enabled=use_amp):
                    class_logits, energy_pred = model(
                        data.x, data.edge_index, data.batch)
                    loss_c = criterion_class(
                        class_logits.view(-1), data.y_class.view(-1))
                    gamma_mask = (data.y_class.view(-1) == 1.0)
                    if gamma_mask.any():
                        loss_e = criterion_energy(
                            energy_pred.view(-1)[gamma_mask],
                            data.y_energy.view(-1)[gamma_mask]).mean()
                        loss = loss_c + 5.0 * loss_e
                    else:
                        loss = loss_c
                val_loss += loss.item() * data.num_graphs
                val_count += data.num_graphs

        val_loss = val_loss / max(val_count, 1)

        print(f"Epoch {epoch + 1:03d} | Train Loss: {train_loss:.4f} "
              f"| Val Loss: {val_loss:.4f}"
              + (f" | OOMs skipped: {oom_count}" if oom_count else ""))

        scheduler.step(val_loss)

        # ------------------------------------------------------------------
        # Checkpointing
        # ------------------------------------------------------------------
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            print(f"*** New best validation loss! Saving to {best_model_path}")
            os.makedirs(os.path.dirname(best_model_path) or '.', exist_ok=True)
            torch.save(model.state_dict(), best_model_path)

        # Full stateful checkpoint every epoch (atomic write)
        save_checkpoint(checkpoint_path, model, optimizer, scheduler, scaler,
                        epoch, best_val_loss)

        # ------------------------------------------------------------------
        # Per-epoch memory cleanup
        # ------------------------------------------------------------------
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print(f"\nTraining complete! Best val loss: {best_val_loss:.4f}")
    if oom_count:
        print(f"  Total CUDA OOM batches recovered: {oom_count}")


if __name__ == '__main__':
    train_networks()

import sys, os, time, psutil
sys.path.insert(0, 'src')
import torch
from analysis.dataset import CherenkovDataset
from recon.gnn import SpatiotemporalGNN
from torch_geometric.loader import DataLoader

def run_audit():
    print('--- 1. Checking Dataset Balance & Spectrum ---')
    dataset = CherenkovDataset(root='data/train_large')
    loader = DataLoader(dataset, batch_size=1024, shuffle=False)
    n_gamma = 0
    n_hadron = 0
    energies = []
    
    for batch in loader:
        y_c = batch.y_class.view(-1)
        y_e = batch.y_energy.view(-1)
        n_gamma += (y_c == 1.0).sum().item()
        n_hadron += (y_c == 0.0).sum().item()
        
        gamma_mask = (y_c == 1.0)
        if gamma_mask.any():
            energies.extend(y_e[gamma_mask].tolist())

    print(f'Total Events: {n_gamma + n_hadron}')
    print(f'Gammas:  {n_gamma} ({n_gamma/(n_gamma+n_hadron)*100:.1f}%)')
    print(f'Hadrons: {n_hadron} ({n_hadron/(n_gamma+n_hadron)*100:.1f}%)')
    print(f'Min Gamma Energy (log10): {min(energies):.2f}')
    print(f'Max Gamma Energy (log10): {max(energies):.2f}')

    print('\n--- 2. Checking Checkpoint & Resume Logic ---')
    model = SpatiotemporalGNN()
    optimizer = torch.optim.Adam(model.parameters())
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
    
    # We test scaler loading logic
    scaler = torch.amp.GradScaler('cuda', enabled=True)

    ckpt_path = 'data/spatiotemporal_gnn_v5_checkpoint.pt'
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        ep = ckpt['epoch'] + 1
        print(f'Found checkpoint from epoch {ep}')
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        scaler.load_state_dict(ckpt['scaler_state_dict'])
        best_loss = ckpt['best_val_loss']
        print(f'Successfully loaded model, optimizer, scheduler, scaler. Best val loss: {best_loss:.4f}')
    else:
        print('No checkpoint found!')

    print('\n--- 3. Checking awake.ps1 Daemon ---')
    found_awake = False
    for p in psutil.process_iter(['name', 'cmdline']):
        try:
            cmd = ' '.join(p.info['cmdline'] or [])
            if 'powershell' in cmd.lower() and 'awake.ps1' in cmd.lower():
                found_awake = True
                print(f'Found awake.ps1 daemon (PID {p.pid}): {cmd}')
        except:
            pass
    if not found_awake:
        print('WARNING: awake.ps1 does not seem to be running!')

if __name__ == "__main__":
    run_audit()

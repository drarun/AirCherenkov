import os
import sys
sys.path.insert(0, 'src')
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import DataLoader
from recon.gnn import EnergyGNN, ClassGNN
from analysis.dataset import CherenkovDataset

def train_networks():
    print("Loading datasets...")
    
    # Generate the edge index for the hexagonal camera grid
    from sim.camera import Camera
    cam = Camera(n_rings=12)
    pixel_x, pixel_y = cam.pixel_x, cam.pixel_y
    pos = torch.stack([torch.tensor(pixel_x, dtype=torch.float32), 
                       torch.tensor(pixel_y, dtype=torch.float32)], dim=1)
    
    dist = torch.cdist(pos, pos)
    adj_matrix = (dist > 0.01) & (dist < 0.105)
    edge_index = adj_matrix.nonzero(as_tuple=False).t().contiguous()
    
    # Pre-transform to attach edge_index to every Data object
    import torch_geometric.transforms as T
    
    class AddEdgeIndex(object):
        def __init__(self, edge_idx):
            self.edge_idx = edge_idx
        def __call__(self, data):
            data.edge_index = self.edge_idx
            return data
            
    dataset = CherenkovDataset(root='data/train', pre_transform=AddEdgeIndex(edge_index))
    print(f"Dataset loaded with {len(dataset)} events.")
    
    if len(dataset) == 0:
        print("No events found. Please run the simulator first.")
        return
    
    # Split train/val
    dataset = dataset.shuffle()
    split = int(0.8 * len(dataset))
    train_data = dataset[:split]
    val_data = dataset[split:]
    
    train_loader = DataLoader(train_data, batch_size=128, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=128, shuffle=False)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize the two independent networks
    energy_model = EnergyGNN().to(device)
    class_model = ClassGNN().to(device)
    
    opt_energy = optim.Adam(energy_model.parameters(), lr=0.001)
    opt_class = optim.Adam(class_model.parameters(), lr=0.001)
    
    criterion_energy = nn.MSELoss()
    criterion_class = nn.BCEWithLogitsLoss()
    
    epochs = 25
    print(f"\nTraining on {device} for {epochs} epochs...")
    
    for epoch in range(epochs):
        energy_model.train()
        class_model.train()
        
        total_loss_e, total_loss_c = 0, 0
        
        for batch in train_loader:
            batch = batch.to(device)
            
            # 1. Train Classification Model (on all events)
            opt_class.zero_grad()
            class_logits = class_model(batch.x, batch.edge_index, batch.batch)
            loss_c = criterion_class(class_logits.view(-1), batch.y_class.view(-1))
            loss_c.backward()
            torch.nn.utils.clip_grad_norm_(class_model.parameters(), max_norm=2.0)
            opt_class.step()
            total_loss_c += loss_c.item()
            
            # 2. Train Energy Model (ONLY on Gamma events: y_class == 1)
            gamma_mask = (batch.y_class.view(-1) == 1.0)
            
            if gamma_mask.any():
                opt_energy.zero_grad()
                # Re-run forward pass just to filter out protons from the loss
                energy_pred = energy_model(batch.x, batch.edge_index, batch.batch)
                
                loss_e = criterion_energy(energy_pred.view(-1)[gamma_mask], batch.y_energy.view(-1)[gamma_mask])
                loss_e.backward()
                torch.nn.utils.clip_grad_norm_(energy_model.parameters(), max_norm=2.0)
                opt_energy.step()
                total_loss_e += loss_e.item()
            
        print(f"Epoch {epoch+1}/{epochs} | Class Loss: {total_loss_c/len(train_loader):.4f} | Energy Loss: {total_loss_e/len(train_loader):.4f}")
        
    print("\nTraining complete! Saving models...")
    torch.save(energy_model.state_dict(), 'data/energy_gnn.pt')
    torch.save(class_model.state_dict(), 'data/class_gnn.pt')

if __name__ == '__main__':
    train_networks()

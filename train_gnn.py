import os
import sys
sys.path.insert(0, 'src')
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.loader import DataLoader
from recon.gnn import SpatiotemporalGNN
from analysis.dataset import CherenkovDataset

def train_networks():
    print("Loading datasets...")
    
    # Generate the edge index for the hexagonal camera grid
    from sim.camera import Camera
    cam = Camera(n_rings=12)
    edge_index = cam.edge_index
    
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
    
    # Initialize the unified spatiotemporal network
    model = SpatiotemporalGNN().to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    criterion_energy = nn.MSELoss()
    criterion_class = nn.BCEWithLogitsLoss()
    
    epochs = 25
    print(f"\nTraining on {device} for {epochs} epochs...")
    
    for epoch in range(epochs):
        model.train()
        
        total_loss_e, total_loss_c = 0, 0
        
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            
            # 1. Single Forward Pass
            class_logits, energy_pred = model(batch.x, batch.edge_index, batch.batch)
            
            # 2. Classification Loss (all events)
            loss_c = criterion_class(class_logits.view(-1), batch.y_class.view(-1))
            total_loss_c += loss_c.item()
            
            # 3. Energy Loss (only gamma events)
            gamma_mask = (batch.y_class.view(-1) == 1.0)
            if gamma_mask.any():
                loss_e = criterion_energy(energy_pred.view(-1)[gamma_mask], batch.y_energy.view(-1)[gamma_mask])
                total_loss_e += loss_e.item()
                loss = loss_c + loss_e
            else:
                loss = loss_c
                
            # 4. Single Backward Pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()
            
        print(f"Epoch {epoch+1}/{epochs} | Class Loss: {total_loss_c/len(train_loader):.4f} | Energy Loss: {total_loss_e/len(train_loader):.4f}")
        
    print("\nTraining complete! Saving models...")
    torch.save(model.state_dict(), 'data/spatiotemporal_gnn.pt')

if __name__ == '__main__':
    train_networks()

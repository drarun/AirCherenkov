import os
import sys
sys.path.insert(0, 'src')
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import radius_graph
from recon.gnn import HexCameraGNN
from analysis.dataset import CherenkovDataset
import numpy as np

def train_gnn():
    print("Loading dataset from data/...")
    
    # Generate the edge index for the hexagonal camera grid
    from sim.camera import Camera
    cam = Camera()
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
            
    dataset = CherenkovDataset(root='data', pre_transform=AddEdgeIndex(edge_index))
    
    print(f"Dataset loaded with {len(dataset)} events.")
    
    if len(dataset) == 0:
        print("No events found in 'data/raw/'. Please run the simulator first.")
        return
    
    # Split train/val
    dataset = dataset.shuffle()
    split = int(0.8 * len(dataset))
    train_data = dataset[:split]
    val_data = dataset[split:]
    
    train_loader = DataLoader(train_data, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=16, shuffle=False)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = HexCameraGNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Loss functions
    criterion_energy = nn.MSELoss()
    criterion_class = nn.BCEWithLogitsLoss()
    
    epochs = 10
    print(f"Training on {device} for {epochs} epochs...")
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            
            energy_pred, class_logits = model(batch.x, batch.edge_index, batch.batch)
            
            loss_e = criterion_energy(energy_pred.view(-1), batch.y_energy.view(-1))
            loss_c = criterion_class(class_logits.view(-1), batch.y_class.view(-1))
            
            loss = loss_e + loss_c
            loss.backward()
            
            # Gradient clipping to prevent NaNs
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            
            optimizer.step()
            total_loss += loss.item()
            
        print(f"Epoch {epoch+1}/{epochs} | Loss: {total_loss/len(train_loader):.4f}")
        
    print("Training complete! Model saved to data/gnn_model.pt")
    torch.save(model.state_dict(), 'data/gnn_model.pt')

if __name__ == '__main__':
    train_gnn()

import os
import sys
sys.path.insert(0, 'src')
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import radius_graph
from recon.gnn import HexCameraGNN
import numpy as np

def create_edge_index(pixel_x, pixel_y, radius=0.105):
    """Create edge_index for the hexagonal camera grid without relying on pyg-lib."""
    pos = torch.stack([pixel_x, pixel_y], dim=1)
    # Compute pairwise distances
    dist = torch.cdist(pos, pos)
    # Connect nodes within `radius` (excluding self loops if dist > 0)
    # Due to floating point errors, dist to self is 0
    adj_matrix = (dist > 0.01) & (dist < radius)
    edge_index = adj_matrix.nonzero(as_tuple=False).t().contiguous()
    return edge_index

def prepare_dataset(data_path):
    dataset_raw = torch.load(data_path, weights_only=False)
    images = dataset_raw['images'] # [N, num_pixels]
    energies = dataset_raw['energies'] # [N]
    labels = dataset_raw['labels'] # [N]
    
    pixel_x = dataset_raw['pixel_x']
    pixel_y = dataset_raw['pixel_y']
    
    edge_index = create_edge_index(pixel_x, pixel_y)
    
    data_list = []
    for i in range(len(images)):
        # Node features: just the pixel amplitude, shape [num_pixels, 1]
        # We should normalize the image? Clamp to 0 to remove negative electronic noise, then log10
        img_clamped = torch.clamp(images[i], min=0.0)
        x = torch.log10(img_clamped.unsqueeze(1) + 1.0)
        
        y_energy = torch.log10(energies[i]).unsqueeze(0)
        y_class = labels[i].float().unsqueeze(0)
        
        data = Data(x=x, edge_index=edge_index, y_energy=y_energy, y_class=y_class)
        data_list.append(data)
        
    return data_list

def train_gnn():
    print("Loading dataset...")
    data_list = prepare_dataset('data/train_events.pt')
    
    # Split train/val
    np.random.shuffle(data_list)
    split = int(0.8 * len(data_list))
    train_data = data_list[:split]
    val_data = data_list[split:]
    
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

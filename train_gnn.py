import os
import sys
sys.path.insert(0, 'src')
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.loader import DataLoader
from recon.gnn import SpatiotemporalGNN
from analysis.dataset import CherenkovDataset
import argparse

def train_networks():
    parser = argparse.ArgumentParser(description="Train Spatiotemporal GNN for AirCherenkov")
    parser.add_argument("--root", type=str, default="data/train", help="Dataset root directory containing 'raw' folder")
    parser.add_argument("--model_path", type=str, default="data/spatiotemporal_gnn.pt", help="Path to save the trained model")
    parser.add_argument("--epochs", type=int, default=25, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
    args = parser.parse_args()

    print(f"Loading dataset from root: {args.root}...")
    
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
            
    dataset = CherenkovDataset(root=args.root, pre_transform=AddEdgeIndex(edge_index))
    print(f"Dataset loaded with {len(dataset)} events.")
    
    if len(dataset) == 0:
        print("No events found. Please run the simulator first.")
        return
    
    # Split train/val
    dataset = dataset.shuffle()
    split = int(0.8 * len(dataset))
    train_data = dataset[:split]
    val_data = dataset[split:]
    
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False)
import os
import sys
sys.path.insert(0, 'src')
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.loader import DataLoader
from recon.gnn import SpatiotemporalGNN
from analysis.dataset import CherenkovDataset
import argparse

def train_networks():
    parser = argparse.ArgumentParser(description="Train Spatiotemporal GNN for AirCherenkov")
    parser.add_argument("--root", type=str, default="data/train", help="Dataset root directory containing 'raw' folder")
    parser.add_argument("--model_path", type=str, default="data/spatiotemporal_gnn.pt", help="Path to save the trained model")
    parser.add_argument("--epochs", type=int, default=25, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
    args = parser.parse_args()

    print(f"Loading dataset from root: {args.root}...")
    
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
            
    dataset = CherenkovDataset(root=args.root, pre_transform=AddEdgeIndex(edge_index))
    print(f"Dataset loaded with {len(dataset)} events.")
    
    if len(dataset) == 0:
        print("No events found. Please run the simulator first.")
        return
    
    # Split train/val
    dataset = dataset.shuffle()
    split = int(0.8 * len(dataset))
    train_data = dataset[:split]
    val_data = dataset[split:]
    
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize the unified spatiotemporal network
    model = SpatiotemporalGNN().to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=True)
    
    criterion_energy = nn.MSELoss()
    criterion_class = nn.BCEWithLogitsLoss()
    
    epochs = args.epochs
    print(f"\nTraining on {device} for {epochs} epochs...")
    
    for epoch in range(epochs):
        model.train()
        
        total_loss = 0
        
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            
            # 1. Single Forward Pass
            class_logits, energy_pred = model(batch.x, batch.edge_index, batch.batch)
            
            # 2. Classification Loss (all events)
            loss_c = criterion_class(class_logits.view(-1), batch.y_class.view(-1))
            
            # 3. Energy Loss (only gamma events)
            gamma_mask = (batch.y_class.view(-1) == 1.0)
            if gamma_mask.any():
                loss_e = criterion_energy(energy_pred.view(-1)[gamma_mask], batch.y_energy.view(-1)[gamma_mask])
                # Loss weighting: give more weight to energy
                loss = loss_c + 5.0 * loss_e
            else:
                loss = loss_c
                
            # 4. Single Backward Pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()
            total_loss += loss.item()
            
        train_loss = total_loss / len(train_loader)
        
        # Validation Loop
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for data in val_loader:
                data = data.to(device)
                class_logits, energy_pred = model(data.x, data.edge_index, data.batch)
                loss_c = criterion_class(class_logits.view(-1), data.y_class.view(-1))
                gamma_mask = (data.y_class.view(-1) == 1.0)
                if gamma_mask.any():
                    loss_e = criterion_energy(energy_pred.view(-1)[gamma_mask], data.y_energy.view(-1)[gamma_mask])
                    loss = loss_c + 5.0 * loss_e
                else:
                    loss = loss_c
                val_loss += loss.item() * data.num_graphs
        
        val_loss = val_loss / len(val_loader.dataset)
        
        print(f"Epoch {epoch+1:03d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        
        # Step scheduler
        scheduler.step(val_loss)
        
    print(f"\nTraining complete! Saving model to {args.model_path}...")
    os.makedirs(os.path.dirname(args.model_path), exist_ok=True)
    torch.save(model.state_dict(), args.model_path)

if __name__ == '__main__':
    train_networks()

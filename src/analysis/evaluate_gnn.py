import os
import sys
sys.path.insert(0, 'src')
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from torch_geometric.data import DataLoader
from recon.gnn import HexCameraGNN
from analysis.dataset import CherenkovDataset
from sim.camera import Camera

def evaluate_gnn():
    print("Setting up camera and edge index...")
    cam = Camera()
    pixel_x, pixel_y = cam.pixel_x, cam.pixel_y
    pos = torch.stack([torch.tensor(pixel_x, dtype=torch.float32), 
                       torch.tensor(pixel_y, dtype=torch.float32)], dim=1)
    
    dist = torch.cdist(pos, pos)
    adj_matrix = (dist > 0.01) & (dist < 0.105)
    edge_index = adj_matrix.nonzero(as_tuple=False).t().contiguous()
    
    class AddEdgeIndex(object):
        def __init__(self, edge_idx):
            self.edge_idx = edge_idx
        def __call__(self, data):
            data.edge_index = self.edge_idx
            return data
            
    print("Loading dataset...")
    dataset = CherenkovDataset(root='data', pre_transform=AddEdgeIndex(edge_index))
    
    print(f"Dataset loaded with {len(dataset)} events.")
    
    if len(dataset) == 0:
        print("No events found in dataset.")
        return

    loader = DataLoader(dataset, batch_size=32, shuffle=False)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = HexCameraGNN().to(device)
    
    print("Loading model weights...")
    model.load_state_dict(torch.load('data/gnn_model.pt', map_location=device, weights_only=True))
    model.eval()
    
    true_energy = []
    pred_energy = []
    true_class = []
    pred_class_prob = []
    
    print("Evaluating...")
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            energy_pred, class_logits = model(batch.x, batch.edge_index, batch.batch)
            
            # Extract true and predicted energies (in original scale)
            # Ensure proper reshaping when dealing with multi-dimensional tensors
            true_e = (10 ** batch.y_energy).cpu().numpy().flatten()
            pred_e = (10 ** energy_pred).cpu().numpy().flatten()
            
            t_class = batch.y_class.cpu().numpy().flatten()
            # probabilities via sigmoid
            p_class = torch.sigmoid(class_logits).cpu().numpy().flatten()
            
            true_energy.extend(true_e)
            pred_energy.extend(pred_e)
            true_class.extend(t_class)
            pred_class_prob.extend(p_class)
            
    true_energy = np.array(true_energy)
    pred_energy = np.array(pred_energy)
    true_class = np.array(true_class)
    pred_class_prob = np.array(pred_class_prob)
    
    # --- ROC AUC ---
    fpr, tpr, thresholds = roc_curve(true_class, pred_class_prob)
    roc_auc = auc(fpr, tpr)
    print(f"ROC AUC: {roc_auc:.4f}")
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (Gamma/Hadron Separation)')
    plt.legend(loc="lower right")
    plt.savefig('data/roc_curve.png')
    plt.close()
    print("Saved ROC curve to data/roc_curve.png")
    
    # --- Energy Resolution ---
    resolution = (pred_energy - true_energy) / true_energy
    
    plt.figure(figsize=(8, 6))
    plt.hist(resolution, bins=50, range=(-1, 1), alpha=0.75, color='blue', edgecolor='black')
    plt.xlabel('(E_pred - E_true) / E_true')
    plt.ylabel('Counts')
    plt.title('Energy Resolution')
    plt.grid(True)
    plt.savefig('data/energy_resolution.png')
    plt.close()
    print("Saved energy resolution to data/energy_resolution.png")
    
if __name__ == '__main__':
    evaluate_gnn()

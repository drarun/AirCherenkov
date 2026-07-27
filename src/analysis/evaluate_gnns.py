import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve

from recon.gnn import EnergyGNN, ClassGNN
from analysis.dataset import CherenkovDataset
from torch_geometric.data import DataLoader

def evaluate():
    # Pre-transform
    from sim.camera import Camera
    cam = Camera(n_rings=12)
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

    dataset = CherenkovDataset(root='data/test', pre_transform=AddEdgeIndex(edge_index))
    loader = DataLoader(dataset, batch_size=128, shuffle=False)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    energy_model = EnergyGNN().to(device)
    energy_model.load_state_dict(torch.load('data/energy_gnn.pt', map_location=device, weights_only=True))
    energy_model.eval()
    
    class_model = ClassGNN().to(device)
    class_model.load_state_dict(torch.load('data/class_gnn.pt', map_location=device, weights_only=True))
    class_model.eval()
    
    true_e, pred_e = [], []
    true_c, pred_c = [], []
    
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            e_out = energy_model(batch.x, batch.edge_index, batch.batch)
            c_out = class_model(batch.x, batch.edge_index, batch.batch)
            
            true_e.extend(batch.y_energy.cpu().numpy())
            pred_e.extend(e_out.view(-1).cpu().numpy())
            
            true_c.extend(batch.y_class.cpu().numpy())
            pred_c.extend(torch.sigmoid(c_out).view(-1).cpu().numpy())
            
    true_e = np.array(true_e)
    pred_e = np.array(pred_e)
    true_c = np.array(true_c)
    pred_c = np.array(pred_c)
    
    # 1. Energy Resolution (only on gammas)
    gamma_mask = (true_c == 1.0)
    te = true_e[gamma_mask]
    pe = pred_e[gamma_mask]
    
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.hist2d(te, pe, bins=50, cmap='viridis')
    plt.plot([2, 4], [2, 4], 'r--', label='Ideal')
    plt.xlabel('True Log10(Energy) [GeV]')
    plt.ylabel('Predicted Log10(Energy) [GeV]')
    plt.title('Energy Reconstruction (Gamma only)')
    plt.legend()
    plt.colorbar()
    
    # 2. ROC Curve
    plt.subplot(1, 2, 2)
    try:
        fpr, tpr, _ = roc_curve(true_c, pred_c)
        auc = roc_auc_score(true_c, pred_c)
        plt.plot(fpr, tpr, label=f'AUC = {auc:.3f}')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Gamma/Hadron Separation')
        plt.legend()
    except Exception as e:
        plt.title(f'ROC Failed: {str(e)}')
        
    plt.tight_layout()
    plt.savefig('data/evaluation.png', dpi=300)
    
    print(f"Evaluation complete. Energy RMSE: {np.sqrt(np.mean((te - pe)**2)):.3f}")

if __name__ == '__main__':
    evaluate()

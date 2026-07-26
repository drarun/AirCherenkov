import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt

# Add project root and src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from analysis.dataset import CherenkovDataset
from recon.gnn import HexCameraGNN
from sim.camera import Camera

def main():
    print("Loading dataset...")
    # Add edge_index transform as in train_gnn.py
    cam = Camera(n_rings=12)
    pos = torch.stack([torch.tensor(cam.pixel_x, dtype=torch.float32), 
                       torch.tensor(cam.pixel_y, dtype=torch.float32)], dim=1)
    dist = torch.cdist(pos, pos)
    adj_matrix = (dist > 0.01) & (dist < 0.105)
    edge_index = adj_matrix.nonzero(as_tuple=False).t().contiguous()
    
    class AddEdgeIndex(object):
        def __init__(self, edge_idx):
            self.edge_idx = edge_idx
        def __call__(self, data):
            data.edge_index = self.edge_idx
            return data
            
    dataset = CherenkovDataset(root='data', pre_transform=AddEdgeIndex(edge_index))
    if len(dataset) == 0:
        print("No events in dataset.")
        return

    # Take the first event
    data = dataset[0]

    print("Loading model...")
    device = torch.device('cpu')
    model = HexCameraGNN().to(device)
    model.load_state_dict(torch.load('data/gnn_model.pt', map_location=device))
    model.eval()

    print("Running inference...")
    with torch.no_grad():
        # Add batch dimension for global pooling
        batch = torch.zeros(data.x.shape[0], dtype=torch.long)
        
        energy_pred, class_logits = model(data.x, data.edge_index, batch)
        
        # Probabilities
        prob = torch.sigmoid(class_logits).item()
        
        # Energies are in log10 scale
        pred_energy = 10 ** energy_pred.item()
        true_energy = 10 ** data.y_energy.item()
        
        true_class = data.y_class.item()

    print("Plotting...")
    # The image features are in data.x
    # In dataset.py, it's x = torch.log10(img_clamped.unsqueeze(1) + 1.0)
    image_amplitudes = data.x[:, 0].numpy()
    
    # We can invert log10 to show real PE, or just plot log10(PE).
    # Camera.plot_image typically plots PE
    pe_amplitudes = (10 ** image_amplitudes) - 1.0

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111)
    
    class_name = "Gamma" if true_class == 1.0 else "Proton"
    title = (
        f"Simulated Event Display\n"
        f"True Class: {class_name} | Pred Gamma Prob: {prob:.1%}\n"
        f"True Energy: {true_energy:.2f} TeV | Pred Energy: {pred_energy:.2f} TeV"
    )
    cam.plot_image(pe_amplitudes, ax=ax, title=title)
    
    plt.tight_layout()
    output_path = 'data/event_display.png'
    plt.savefig(output_path, dpi=150)
    print(f"Plot saved to {output_path}")

if __name__ == '__main__':
    main()

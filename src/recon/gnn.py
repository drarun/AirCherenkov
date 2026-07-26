import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool

class HexCameraGNN(torch.nn.Module):
    def __init__(self, num_node_features=1, hidden_channels=32):
        super(HexCameraGNN, self).__init__()
        # Graph Attention Layers
        self.conv1 = GATConv(num_node_features, hidden_channels)
        self.conv2 = GATConv(hidden_channels, hidden_channels)
        self.conv3 = GATConv(hidden_channels, hidden_channels)
        
        # Multi-task heads
        # 1. Energy Regression (Log Energy)
        self.energy_head = nn.Sequential(
            nn.Linear(hidden_channels, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )
        
        # 2. Gamma/Hadron Classification
        self.class_head = nn.Sequential(
            nn.Linear(hidden_channels, 16),
            nn.ReLU(),
            nn.Linear(16, 1) # Logits for Binary Cross Entropy
        )

    def forward(self, x, edge_index, batch):
        # x shape: [num_nodes_total, num_features]
        # x is the pixel amplitude
        
        # Node embeddings
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.conv3(x, edge_index)
        x = F.relu(x)
        
        # Global pooling (aggregate all pixel embeddings into a single graph embedding)
        # We use global mean pooling, but max pooling is also an option
        x = global_mean_pool(x, batch)
        
        # Multi-task outputs
        energy_pred = self.energy_head(x)
        class_logits = self.class_head(x)
        
        return energy_pred, class_logits

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_max_pool

class EnergyGNN(torch.nn.Module):
    def __init__(self, num_node_features=2, hidden_channels=32):
        super(EnergyGNN, self).__init__()
        # Graph Attention Layers
        self.conv1 = GATConv(num_node_features, hidden_channels)
        self.conv2 = GATConv(hidden_channels, hidden_channels)
        self.conv3 = GATConv(hidden_channels, hidden_channels)
        
        # Energy Regression (Log Energy)
        self.energy_head = nn.Sequential(
            nn.Linear(hidden_channels, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )

    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.conv3(x, edge_index)
        x = F.relu(x)
        
        x = global_max_pool(x, batch)
        
        return self.energy_head(x)

class ClassGNN(torch.nn.Module):
    def __init__(self, num_node_features=2, hidden_channels=32):
        super(ClassGNN, self).__init__()
        # Graph Attention Layers
        self.conv1 = GATConv(num_node_features, hidden_channels)
        self.conv2 = GATConv(hidden_channels, hidden_channels)
        self.conv3 = GATConv(hidden_channels, hidden_channels)
        
        # Gamma/Hadron Classification
        self.class_head = nn.Sequential(
            nn.Linear(hidden_channels, 16),
            nn.ReLU(),
            nn.Linear(16, 1) # Logits for Binary Cross Entropy
        )

    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.conv3(x, edge_index)
        x = F.relu(x)
        
        x = global_max_pool(x, batch)
        
        return self.class_head(x)

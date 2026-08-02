import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_max_pool, global_mean_pool

class EnergyGNN(torch.nn.Module):
    def __init__(self, num_node_features=22, hidden_channels=64):
        super(EnergyGNN, self).__init__()
        
        # Phase 1: 1D-CNN Temporal Extractor
        self.conv1d_1 = nn.Conv1d(in_channels=1, out_channels=8, kernel_size=3, padding=1)
        self.conv1d_2 = nn.Conv1d(in_channels=8, out_channels=16, kernel_size=3, padding=1)
        self.pool1d = nn.MaxPool1d(kernel_size=2)
        
        # 16 time bins -> pool by 2 = 8 bins. 8 bins * 16 channels = 128 temporal features.
        # Plus 6 static features (gain + 5 spatial) = 134 graph features.
        gat_in_channels = 128 + 6
        
        # Phase 3: Spatial Graph Attention
        self.conv1 = GATConv(gat_in_channels, hidden_channels)
        self.conv2 = GATConv(hidden_channels, hidden_channels)
        self.conv3 = GATConv(hidden_channels, hidden_channels)
        self.conv4 = GATConv(hidden_channels, hidden_channels)
        
        # Phase 4: Regression
        self.energy_head = nn.Sequential(
            nn.Linear(hidden_channels, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )

    def forward(self, x, edge_index, batch):
        trace = x[:, :16].unsqueeze(1) # [N, 1 (channel), 16 (bins)]
        spatial = x[:, 16:] # [N, 6]
        
        trace = F.relu(self.conv1d_1(trace))
        trace = F.relu(self.conv1d_2(trace))
        trace = self.pool1d(trace) # [N, 16, 8]
        trace_embed = trace.view(trace.size(0), -1) # [N, 128]
        
        # Re-fuse Time and Space
        x_fused = torch.cat([trace_embed, spatial], dim=1)
        
        x = F.relu(self.conv1(x_fused, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = F.relu(self.conv3(x, edge_index))
        x = F.relu(self.conv4(x, edge_index))
        
        x = global_max_pool(x, batch)
        
        return self.energy_head(x)

class ClassGNN(torch.nn.Module):
    def __init__(self, num_node_features=22, hidden_channels=64):
        super(ClassGNN, self).__init__()
        
        # Phase 1: 1D-CNN Temporal Extractor
        self.conv1d_1 = nn.Conv1d(in_channels=1, out_channels=8, kernel_size=3, padding=1)
        self.conv1d_2 = nn.Conv1d(in_channels=8, out_channels=16, kernel_size=3, padding=1)
        self.pool1d = nn.MaxPool1d(kernel_size=2)
        
        gat_in_channels = 128 + 6
        
        # Phase 3: Spatial Graph Attention
        self.conv1 = GATConv(gat_in_channels, hidden_channels)
        self.conv2 = GATConv(hidden_channels, hidden_channels)
        self.conv3 = GATConv(hidden_channels, hidden_channels)
        self.conv4 = GATConv(hidden_channels, hidden_channels)
        
        # Phase 4: Classification
        self.class_head = nn.Sequential(
            nn.Linear(hidden_channels * 2, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x, edge_index, batch):
        trace = x[:, :16].unsqueeze(1)
        spatial = x[:, 16:]
        
        trace = F.relu(self.conv1d_1(trace))
        trace = F.relu(self.conv1d_2(trace))
        trace = self.pool1d(trace)
        trace_embed = trace.view(trace.size(0), -1)
        
        # Re-fuse Time and Space
        x_fused = torch.cat([trace_embed, spatial], dim=1)
        
        x = F.relu(self.conv1(x_fused, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = F.relu(self.conv3(x, edge_index))
        x = F.relu(self.conv4(x, edge_index))
        
        x_mean = global_mean_pool(x, batch)
        x_max = global_max_pool(x, batch)
        x = torch.cat([x_mean, x_max], dim=1)
        
        return self.class_head(x)

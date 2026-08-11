import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GraphNorm, global_max_pool, global_mean_pool, global_add_pool

class SpatiotemporalGNN(torch.nn.Module):
    def __init__(self, num_node_features=23, hidden_channels=64, heads=4, dropout=0.1):
        super(SpatiotemporalGNN, self).__init__()
        self.dropout = dropout
        
        # Phase 1: 1D-CNN Temporal Extractor
        self.conv1d_1 = nn.Conv1d(in_channels=1, out_channels=8, kernel_size=3, padding=1)
        self.conv1d_2 = nn.Conv1d(in_channels=8, out_channels=16, kernel_size=3, padding=1)
        self.pool1d = nn.MaxPool1d(kernel_size=2)
        
        # 16 time bins -> pool by 2 = 8 bins. 8 bins * 16 channels = 128 temporal features.
        # Plus remaining features (gain, spatial, timing)
        gat_in_channels = 128 + (num_node_features - 16)
        
        # Phase 3: Spatial Graph Attention (Multi-head)
        self.conv1 = GATConv(gat_in_channels, hidden_channels, heads=heads)
        self.norm1 = GraphNorm(hidden_channels * heads)
        
        self.conv2 = GATConv(hidden_channels * heads, hidden_channels, heads=heads)
        self.norm2 = GraphNorm(hidden_channels * heads)
        
        self.conv3 = GATConv(hidden_channels * heads, hidden_channels, heads=heads)
        self.norm3 = GraphNorm(hidden_channels * heads)
        
        # Output layer of GAT consolidates back to hidden_channels
        self.conv4 = GATConv(hidden_channels * heads, hidden_channels, heads=1)
        self.norm4 = GraphNorm(hidden_channels)
        
        # Phase 4: Shared Heads
        self.energy_head = nn.Sequential(
            nn.Linear(hidden_channels, 16),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(16, 1)
        )
        
        self.class_head = nn.Sequential(
            nn.Linear(hidden_channels * 2, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1)
        )

    def forward(self, x, edge_index, batch):
        x = x.float()
        trace = x[:, :16].unsqueeze(1) # [N, 1 (channel), 16 (bins)]
        spatial = x[:, 16:] # [N, 6]
        
        trace = F.relu(self.conv1d_1(trace))
        trace = F.relu(self.conv1d_2(trace))
        trace = self.pool1d(trace) # [N, 16, 8]
        trace_embed = trace.view(trace.size(0), -1) # [N, 128]
        
        # Re-fuse Time and Space
        x_fused = torch.cat([trace_embed, spatial], dim=1)
        
        # GAT Blocks
        x = self.conv1(x_fused, edge_index)
        x = self.norm1(x, batch)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        x = self.conv2(x, edge_index)
        x = self.norm2(x, batch)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        x = self.conv3(x, edge_index)
        x = self.norm3(x, batch)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        x = self.conv4(x, edge_index)
        x = self.norm4(x, batch)
        x = F.relu(x)
        
        # Energy output uses sum pooling (light is additive)
        x_add = global_add_pool(x, batch)
        energy_out = self.energy_head(x_add)
        
        # Class output uses mean and max pooling
        x_mean = global_mean_pool(x, batch)
        x_concat = torch.cat([x_mean, x_max], dim=1)
        class_out = self.class_head(x_concat)
        
        return class_out, energy_out

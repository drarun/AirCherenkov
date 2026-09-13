import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GraphNorm, global_max_pool, global_mean_pool, global_add_pool

class SpatiotemporalGNN(torch.nn.Module):
    def __init__(self, num_node_features=25, hidden_channels=64, heads=4, dropout=0.1):
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
        # Energy head receives:
        # 1. Unnormalized GNN pooling: sum (hidden_channels) + mean (hidden_channels) + max (hidden_channels) = hidden_channels * 3
        # 2. Raw charge/size skip-connections: global sum of log1p(charge) [1], max log1p(charge) [1], sum log_size [1], max log_size [1] = 4
        energy_in_dim = hidden_channels * 3 + 4
        self.energy_head = nn.Sequential(
            nn.Linear(energy_in_dim, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.GELU(),
            nn.Linear(32, 1)
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
        spatial = x[:, 16:] # [N, remaining features]
        
        trace = F.relu(self.conv1d_1(trace))
        trace = F.relu(self.conv1d_2(trace))
        trace = self.pool1d(trace) # [N, 16, 8]
        trace_embed = trace.view(trace.size(0), -1) # [N, 128]
        
        # Re-fuse Time and Space
        x_fused = torch.cat([trace_embed, spatial], dim=1)
        
        # GAT Blocks
        x_conv = self.conv1(x_fused, edge_index)
        x_conv = self.norm1(x_conv, batch)
        x_conv = F.relu(x_conv)
        x_conv = F.dropout(x_conv, p=self.dropout, training=self.training)
        
        x_conv = self.conv2(x_conv, edge_index)
        x_conv = self.norm2(x_conv, batch)
        x_conv = F.relu(x_conv)
        x_conv = F.dropout(x_conv, p=self.dropout, training=self.training)
        
        x_conv = self.conv3(x_conv, edge_index)
        x_conv = self.norm3(x_conv, batch)
        x_conv = F.relu(x_conv)
        x_conv = F.dropout(x_conv, p=self.dropout, training=self.training)
        
        # Layer 4: Extract unnormalized representation BEFORE norm4 for energy estimation
        x_unnorm = self.conv4(x_conv, edge_index)
        
        # Classification path uses normalized representation
        x_norm = self.norm4(x_unnorm, batch)
        x_norm = F.relu(x_norm)
        
        # Energy Pooling: multi-scale pooling on unnormalized GAT features (preserves light scale)
        e_sum = global_add_pool(x_unnorm, batch)
        e_mean = global_mean_pool(x_unnorm, batch)
        e_max = global_max_pool(x_unnorm, batch)
        
        # Raw charge skip-connection directly from input node features:
        # pixel_charge is at index 23, log_size is at index 24
        raw_charge = x[:, 23:24]
        raw_size = x[:, 24:25]
        charge_sum = global_add_pool(raw_charge, batch)
        charge_max = global_max_pool(raw_charge, batch)
        size_sum = global_add_pool(raw_size, batch)
        size_max = global_max_pool(raw_size, batch)
        
        energy_features = torch.cat([e_sum, e_mean, e_max, charge_sum, charge_max, size_sum, size_max], dim=1)
        energy_out = self.energy_head(energy_features)
        
        # Class output uses normalized mean and max pooling
        x_mean = global_mean_pool(x_norm, batch)
        x_max = global_max_pool(x_norm, batch)
        x_concat = torch.cat([x_mean, x_max], dim=1)
        class_out = self.class_head(x_concat)
        
        return class_out, energy_out

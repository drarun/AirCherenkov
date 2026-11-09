import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GraphNorm, global_max_pool, global_mean_pool, global_add_pool


class SpatiotemporalGNN(torch.nn.Module):
    """Physics-informed Spatiotemporal Graph Attention Network for IACT event reconstruction.

    Node feature contract (up to 37 features; inputs with <37 features are zero-padded):
      0..15 : 16-bin FADC waveform log1p(trace)
      16    : low-gain channel flag
      17    : DoublePass relative pixel arrival time (T0 - T_centroid) / 10 ns
      18..22: CoG-centered pixel coordinates (px, py, px^2, py^2, px*py)
      23    : log10(1 + DoublePass pixel charge)
      24    : log10(DoublePass cleaned telescope size)
      25..30: Direction-agnostic DoublePass morphology & timing
              (dp_mask, time_gradient/10, time_rms/5, hillas_length, hillas_width, wl_ratio)
      31..36: Stereoscopic & containment energy calibration features
              (rt_tel/200, loss1, loss2, log_slope_parallax, depth_corr_log_size, mean_r/200)
    """

    TOTAL_NODE_FEATURES = 37
    GAT_SPATIAL_FEATURES = 21  # indices 16..36 (all direction-agnostic spatial, timing & stereo features)
    STEREO_AUX_FEATURES = 13   # indices 24..36 (log_size + DoublePass + stereo energy/morphology features)
    CALO_SKIP_FEATURES = 14    # explicit O(1) physical calorimetric & lateral-halo skip features

    def __init__(self, num_node_features=37, hidden_channels=64, heads=4, dropout=0.1):
        super(SpatiotemporalGNN, self).__init__()
        self.num_node_features = num_node_features
        self.dropout = dropout

        # Phase 1: 1D-CNN Temporal Extractor on 16 FADC time bins
        self.conv1d_1 = nn.Conv1d(in_channels=1, out_channels=8, kernel_size=3, padding=1)
        self.conv1d_2 = nn.Conv1d(in_channels=8, out_channels=16, kernel_size=3, padding=1)
        self.pool1d = nn.MaxPool1d(kernel_size=2)

        # 16 time bins -> pool by 2 = 8 bins * 16 channels = 128 temporal features
        # Plus 21 direction-agnostic spatial/timing/DoublePass/stereo features (indices 16..36)
        gat_in_channels = 128 + self.GAT_SPATIAL_FEATURES

        # Phase 2: Spatial Graph Attention (Multi-head)
        self.conv1 = GATConv(gat_in_channels, hidden_channels, heads=heads)
        self.norm1 = GraphNorm(hidden_channels * heads)

        self.conv2 = GATConv(hidden_channels * heads, hidden_channels, heads=heads)
        self.norm2 = GraphNorm(hidden_channels * heads)

        self.conv3 = GATConv(hidden_channels * heads, hidden_channels, heads=heads)
        self.norm3 = GraphNorm(hidden_channels * heads)

        # Output layer of GAT consolidates back to hidden_channels
        self.conv4 = GATConv(hidden_channels * heads, hidden_channels, heads=1)
        self.norm4 = GraphNorm(hidden_channels)

        # Phase 3: Scale-Normalized Physics-Informed Energy Head (outputs [pred_log10_E, log_var_E])
        # Normalizes the 192-d GAT pooled embedding so it cannot drown out O(1) core-distance features
        self.energy_pool_norm = nn.LayerNorm(hidden_channels * 3)
        calo_aux_dim = self.CALO_SKIP_FEATURES + (self.STEREO_AUX_FEATURES * 2)
        energy_in_dim = hidden_channels * 3 + calo_aux_dim
        self.energy_head = nn.Sequential(
            nn.Linear(energy_in_dim, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.GELU(),
            nn.Linear(32, 2),
        )
        # Direct linear calorimetric residual skip connection (dropout-free physical prior)
        self.energy_skip = nn.Linear(calo_aux_dim, 1)

        # Phase 4: Direction-Agnostic Classification Head
        # Receives normalized GAT mean & max pooling + full stereo/DoublePass summary (indices 24..36)
        class_in_dim = hidden_channels * 2 + (self.STEREO_AUX_FEATURES * 2)
        self.class_norm = nn.LayerNorm(class_in_dim)
        self.class_head = nn.Sequential(
            nn.Linear(class_in_dim, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def forward(self, x, edge_index, batch, return_uncertainty: bool = False):
        x = x.float()
        if x.size(1) < self.TOTAL_NODE_FEATURES:
            x = F.pad(x, (0, self.TOTAL_NODE_FEATURES - x.size(1)))

        trace = x[:, :16].unsqueeze(1)          # [N, 1, 16]
        spatial_agnostic = x[:, 16:37]          # [N, 21] all direction-agnostic node & stereo features
        stereo_aux = x[:, 24:37]                # [N, 13] log_size + DoublePass + stereo features

        trace = F.relu(self.conv1d_1(trace))
        trace = F.relu(self.conv1d_2(trace))
        trace = self.pool1d(trace)              # [N, 16, 8]
        trace_embed = trace.view(trace.size(0), -1)  # [N, 128]

        # Fuse temporal and all 21 direction-agnostic spatial/stereoscopic features
        x_fused = torch.cat([trace_embed, spatial_agnostic], dim=1)

        # GAT Blocks (message passing conditioned on impact distance R_t, size, and leakage)
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

        # Energy Pooling: scale pixel sums by 1/469.0 (telescope-scale O(1) instead of pixel-scale O(5000))
        e_sum = global_add_pool(x_unnorm, batch) / 469.0
        e_mean = global_mean_pool(x_unnorm, batch)
        e_max = global_max_pool(x_unnorm, batch)
        e_pool_norm = self.energy_pool_norm(torch.cat([e_sum, e_mean, e_max], dim=1))
        e_sum_lognorm = torch.log1p(torch.norm(e_sum, dim=1, keepdim=True))

        # Telescope-normalized charge, size, and multiplicity features
        raw_charge = x[:, 23:24]
        raw_size = x[:, 24:25]
        charge_sum = global_add_pool(raw_charge, batch) / 469.0
        charge_max = global_max_pool(raw_charge, batch)
        size_sum = global_add_pool(raw_size, batch) / 469.0
        size_max = global_max_pool(raw_size, batch)
        n_tel_norm = global_add_pool(x.new_ones((x.size(0), 1)), batch) / (469.0 * 4.0)

        # Stereoscopic & DoublePass auxiliary pooling
        aux_mean = global_mean_pool(stereo_aux, batch)
        aux_max = global_max_pool(stereo_aux, batch)

        # Explicit physical leakage compensation & lateral-halo attenuation (R_t > 120 m)
        dp_mask = x[:, 25:26].clamp(min=0.0, max=1.0)
        rt_norm = x[:, 31:32].clamp(min=0.0, max=6.0)   # R_t / 200 m (0.60 == 120 m Cherenkov shoulder)
        loss1 = x[:, 32:33].clamp(min=0.0, max=1.0)
        loss2 = x[:, 33:34].clamp(min=0.0, max=1.0)

        keep_frac = (1.0 - 1.15 * loss1 - 0.35 * loss2).clamp(min=0.20, max=1.0)
        leak_corr_size = raw_size - torch.log10(keep_frac)
        halo_excess = F.relu(rt_norm - 0.60)            # > 0 outside 120 m Cherenkov light pool
        halo_sq = halo_excess ** 2
        calo_proxy = leak_corr_size + 0.85 * halo_excess

        # Containment-weighted pooling across cleaned pixels (prioritizes contained telescopes)
        w_contain = (dp_mask + 1e-3) * ((1.0 - 0.85 * loss1).clamp(min=0.1) ** 2) * torch.exp(-1.2 * halo_excess)
        w_sum = global_add_pool(w_contain, batch).clamp(min=1e-6)
        w_log_size = global_add_pool(w_contain * raw_size, batch) / w_sum
        w_leak_corr_size = global_add_pool(w_contain * leak_corr_size, batch) / w_sum
        w_calo_proxy = global_add_pool(w_contain * calo_proxy, batch) / w_sum
        w_rt = global_add_pool(w_contain * rt_norm, batch) / w_sum
        w_halo_excess = global_add_pool(w_contain * halo_excess, batch) / w_sum
        w_halo_sq = global_add_pool(w_contain * halo_sq, batch) / w_sum
        rt_min = -global_max_pool(-rt_norm, batch)
        w_tgrad = global_add_pool(w_contain * torch.abs(x[:, 26:27]), batch) / w_sum

        calo_skip_vec = torch.cat([
            charge_sum, charge_max, size_sum, size_max, n_tel_norm, e_sum_lognorm,
            w_log_size, w_leak_corr_size, w_calo_proxy,
            w_rt, w_halo_excess, w_halo_sq, rt_min, w_tgrad,
            aux_mean, aux_max,
        ], dim=1)

        energy_features = torch.cat([e_pool_norm, calo_skip_vec], dim=1)
        energy_raw = self.energy_head(energy_features)
        energy_out = energy_raw[:, :1] + self.energy_skip(calo_skip_vec)
        energy_logvar = torch.clamp(energy_raw[:, 1:2], min=-5.0, max=1.0)

        # Classification output uses normalized GAT pooling + full stereo/DoublePass summary (cols 24..36)
        x_mean = global_mean_pool(x_norm, batch)
        x_max = global_max_pool(x_norm, batch)
        x_concat = self.class_norm(torch.cat([x_mean, x_max, aux_mean, aux_max], dim=1))
        class_out = self.class_head(x_concat)

        if return_uncertainty:
            return class_out, energy_out, energy_logvar
        return class_out, energy_out

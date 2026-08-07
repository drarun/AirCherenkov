import torch
from typing import Dict, Tuple

try:
    from torch_geometric.data import Data
    HAS_PYG = True
except ImportError:
    HAS_PYG = False
    # Fallback placeholder if PyG is not installed yet
    class Data:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)

class GraphBuilder:
    """
    Transforms processed pixel charge/timing data (from simulation or BaseDataReader)
    into PyTorch Geometric Data graphs for the downstream GNN.
    """
    def __init__(self, pixel_positions: torch.Tensor, edge_index: torch.Tensor):
        """
        Initializes with a specific camera geometry.
        
        Parameters:
        - pixel_positions: (num_pixels, 2) Tensor of physical (x, y) coordinates
        - edge_index: (2, num_edges) Tensor of adjacency indices
        """
        self.pos = pixel_positions
        self.edge_index = edge_index
        
    def build_graph(self, fadc_traces: torch.Tensor, gain_flags: torch.Tensor = None) -> Data:
        """
        Constructs the PyG Data object with 22-dimensional spatiotemporal features.
        
        Parameters:
        - fadc_traces: (num_pixels, 16) Tensor
        - gain_flags: (num_pixels,) Tensor, optional. Will default to zeros if not provided.
        """
        if gain_flags is None:
            gain_flags = torch.zeros(fadc_traces.size(0), dtype=torch.float32, device=fadc_traces.device)
            
        # Calculate Center of Gravity for translation invariance
        img = fadc_traces.sum(dim=1)
        total_charge = img.sum()
        
        if total_charge > 0:
            cog_x = (img * self.pos[:, 0].to(img.device)).sum() / total_charge
            cog_y = (img * self.pos[:, 1].to(img.device)).sum() / total_charge
        else:
            cog_x, cog_y = 0.0, 0.0
            
        px_shifted = (self.pos[:, 0].to(fadc_traces.device) - cog_x).unsqueeze(1)
        py_shifted = (self.pos[:, 1].to(fadc_traces.device) - cog_y).unsqueeze(1)
        
        px_squared = px_shifted ** 2
        py_squared = py_shifted ** 2
        pxy = px_shifted * py_shifted
        
        gain_feat = gain_flags.unsqueeze(1)
        
        x = torch.cat([
            fadc_traces.float(), 
            gain_feat.float(), 
            px_shifted.float(), 
            py_shifted.float(), 
            px_squared.float(), 
            py_squared.float(), 
            pxy.float()
        ], dim=1)
        
        return Data(x=x, edge_index=self.edge_index, pos=self.pos)

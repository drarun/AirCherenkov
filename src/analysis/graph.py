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
        
    def build_graph(self, charge: torch.Tensor, timing: torch.Tensor) -> Data:
        """
        Constructs the PyG Data object.
        
        Parameters:
        - charge: (num_pixels,) Tensor
        - timing: (num_pixels,) Tensor
        """
        # Node features: [charge, timing]
        x = torch.stack([charge, timing], dim=-1)
        
        return Data(x=x, edge_index=self.edge_index, pos=self.pos)

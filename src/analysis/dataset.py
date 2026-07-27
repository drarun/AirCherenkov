import os
import torch
import numpy as np
from typing import List, Callable, Union

try:
    from torch_geometric.data import InMemoryDataset, Data
except ImportError:
    class InMemoryDataset:
        def __init__(self, *args, **kwargs): pass
    class Data:
        pass

from sim.backend import get_device
from analysis.io import BaseDataReader, TraceProcessor
from analysis.graph import GraphBuilder

class CherenkovDataset(InMemoryDataset):
    """
    Unified PyTorch Geometric Dataset for Cherenkov events.
    Can ingest real data files (via VBFReader/HDF5Reader) or 
    pre-processed simulation graphs (.pt files).
    """
    def __init__(self, root: str, 
                 reader_class: type = None, 
                 graph_builder: GraphBuilder = None,
                 transform: Callable = None, 
                 pre_transform: Callable = None):
        """
        Parameters:
        - root: Directory containing data files.
        - reader_class: Class of the reader (e.g., VBFReader) to parse raw files.
                        If None, assumes .pt simulation files are in the raw dir.
        - graph_builder: GraphBuilder instance with the correct camera geometry.
        """
        self.reader_class = reader_class
        self.graph_builder = graph_builder
        self.processor = TraceProcessor()
        self.device = get_device() or torch.device('cpu')
        
        # PyG automatically calls process() if the processed files don't exist
        super().__init__(root, transform, pre_transform)
        
        # Load the processed data
        self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)

    @property
    def raw_file_names(self) -> List[str]:
        # List all files in the raw directory
        raw_dir = os.path.join(self.root, 'raw')
        if not os.path.exists(raw_dir):
            return []
        return [f for f in os.listdir(raw_dir) if os.path.isfile(os.path.join(raw_dir, f))]

    @property
    def processed_file_names(self) -> str:
        return 'data.pt'

    def process(self):
        data_list = []
        
        for raw_file in self.raw_paths:
            print(f"Processing {raw_file}...")
            
            if self.reader_class is not None:
                # Process real telescope raw data (VBF/HDF5)
                reader = self.reader_class(raw_file, device=self.device)
                for event in reader.read_event():
                    if "fadc_traces" not in event:
                        continue
                    processed = self.processor.process(event["fadc_traces"])
                    
                    if self.graph_builder:
                        data = self.graph_builder.build_graph(processed["charge"], processed["timing"])
                    else:
                        data = Data(x=torch.stack([processed["charge"], processed["timing"]], dim=-1))
                    
                    # Dummy labels since it's real unlabeled data
                    data.y_energy = torch.tensor([0.0], dtype=torch.float32)
                    data.y_class = torch.tensor([0.0], dtype=torch.float32)
                    data_list.append(data)
            else:
                # Process simulation `.pt` files
                # Assumes files were saved by `benchmark_sim.py` or `generate_training_data.py`
                # which save list of dicts: [{'images': ..., 'energy': ..., 'label': ...}]
                sim_data = torch.load(raw_file, weights_only=False)
                
                # Support both old dictionary format and new list format
                if isinstance(sim_data, dict) and 'images' in sim_data:
                    # Old batched format from generate_training_data.py
                    images = sim_data['images']
                    energies = sim_data['energies']
                    labels = sim_data['labels']
                    pixel_x = sim_data['pixel_x']
                    pixel_y = sim_data['pixel_y']
                    
                    for i in range(len(images)):
                        # Just take the first telescope image for simplistic training
                        img = torch.tensor(images[i][0] if len(images[i].shape) > 1 else images[i], dtype=torch.float32)
                        img_clamped = torch.clamp(img, min=0.0)
                        x = torch.log10(img_clamped.unsqueeze(1) + 1.0)
                        
                        y_e = torch.log10(torch.tensor([energies[i]], dtype=torch.float32))
                        y_c = torch.tensor([labels[i]], dtype=torch.float32)
                        
                        # Fallback edge index generation if graph_builder not provided
                        if self.graph_builder:
                            data = self.graph_builder.build_graph(img_clamped, torch.zeros_like(img_clamped))
                        else:
                            data = Data(x=x, y_energy=y_e, y_class=y_c)
                        data_list.append(data)
                        
                elif isinstance(sim_data, list):
                    # New format from benchmark_sim.py
                    for evt in sim_data:
                        # evt['images'] shape is (4 telescopes, 469 pixels). 
                        # We must NOT sum them, as that destroys stereoscopic spatial correlations.
                        # Instead, we treat each telescope as a separate graph for the single-camera GNN.
                        for tel_idx in range(len(evt['images'])):
                            img = torch.tensor(evt['images'][tel_idx], dtype=torch.float32)
                            
                            # Only train on telescopes that actually triggered / saw the shower
                            if torch.sum(img) > 20.0:
                                img_clamped = torch.clamp(img, min=0.0)
                                charge_feat = torch.log10(img_clamped + 1.0).unsqueeze(1)
                                
                                timing_arr = torch.tensor(evt['timing'][tel_idx], dtype=torch.float32)
                                valid_timing = timing_arr > 0
                                if valid_timing.any():
                                    t_min = timing_arr[valid_timing].min()
                                    timing_arr[valid_timing] = timing_arr[valid_timing] - t_min + 1.0
                                
                                timing_feat = timing_arr.unsqueeze(1)
                                
                                x = torch.cat([charge_feat, timing_feat], dim=1)
                        
                                y_e = torch.log10(torch.tensor([evt['energy']], dtype=torch.float32))
                                y_c = torch.tensor([evt['label']], dtype=torch.float32)
                                
                                data = Data(x=x, y_energy=y_e, y_class=y_c)
                                data_list.append(data)
        
        if self.pre_transform is not None:
            data_list = [self.pre_transform(d) for d in data_list]
            
        print(f"Processed {len(data_list)} events. Saving to {self.processed_paths[0]}...")
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

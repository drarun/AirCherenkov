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
        
        from sim.camera import Camera
        from sim.trigger import CameraTrigger
        cam = Camera(n_rings=12)
        pixel_x, pixel_y = cam.pixel_x, cam.pixel_y
        
        px_feat = torch.tensor(pixel_x, dtype=torch.float32).unsqueeze(1)
        py_feat = torch.tensor(pixel_y, dtype=torch.float32).unsqueeze(1)
        trigger = CameraTrigger(pixel_x, pixel_y)
        
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
                        data = self.graph_builder.build_graph(event["fadc_traces"])
                    else:
                        data = Data(x=event["fadc_traces"])
                    
                    # Dummy labels since it's real unlabeled data
                    data.y_energy = torch.tensor([0.0], dtype=torch.float32)
                    data.y_class = torch.tensor([0.0], dtype=torch.float32)
                    data_list.append(data)
            else:
                # Process simulation `.pt` files (saved by `generate_training_data.py`)
                # which save list of dicts: [{'fadc_traces': ..., 'energy': ..., 'label': ...}]
                sim_data = torch.load(raw_file, weights_only=False)
                
                if isinstance(sim_data, list):
                    for evt in sim_data:
                        event_x_list = []
                        for tel_idx in range(len(evt['fadc_traces'])):
                            trace_np = evt['fadc_traces'][tel_idx]
                            gain_np = evt['gain_flags'][tel_idx]
                            
                            img_np = np.sum(trace_np, axis=1)
                            timing_np = np.argmax(trace_np, axis=1) * 2.0
                            
                            is_triggered, t0 = trigger.evaluate(img_np, timing_np)
                            if is_triggered:
                                # Apply log1p normalization (Critical 1)
                                trace_feat = torch.log1p(torch.tensor(trace_np, dtype=torch.float32))
                                gain_feat = torch.tensor(gain_np, dtype=torch.float32).unsqueeze(1)
                                timing_feat = torch.tensor(timing_np, dtype=torch.float32).unsqueeze(1) # Advisory 9
                                
                                # CoG with mask (Advisory 8)
                                mask = img_np > 5.0
                                total_charge = np.sum(img_np[mask])
                                if total_charge > 0:
                                    cog_x = np.sum(img_np[mask] * pixel_x[mask]) / total_charge
                                    cog_y = np.sum(img_np[mask] * pixel_y[mask]) / total_charge
                                else:
                                    cog_x, cog_y = 0.0, 0.0
                                    
                                px_shifted = torch.tensor(pixel_x - cog_x, dtype=torch.float32).unsqueeze(1)
                                py_shifted = torch.tensor(pixel_y - cog_y, dtype=torch.float32).unsqueeze(1)
                                
                                px_squared = px_shifted ** 2
                                py_squared = py_shifted ** 2
                                pxy = px_shifted * py_shifted
                                
                                x = torch.cat([
                                    trace_feat, 
                                    gain_feat, 
                                    timing_feat,
                                    px_shifted, 
                                    py_shifted, 
                                    px_squared, 
                                    py_squared, 
                                    pxy
                                ], dim=1)
                                event_x_list.append(x)
                                
                        if len(event_x_list) > 0:
                            # Stereoscopic (Important 4)
                            combined_x = torch.cat(event_x_list, dim=0).half()  # Halve memory usage
                            
                            num_nodes = len(pixel_x)
                            edge_index_list = []
                            for i in range(len(event_x_list)):
                                offset_edge = cam.edge_index + (i * num_nodes)
                                edge_index_list.append(offset_edge)
                            combined_edge_index = torch.cat(edge_index_list, dim=1)
                            
                            y_e = torch.log10(torch.tensor([evt['energy']], dtype=torch.float32))
                            y_c = torch.tensor([evt['label']], dtype=torch.float32)
                            
                            data = Data(x=combined_x, edge_index=combined_edge_index, y_energy=y_e, y_class=y_c)
                            data_list.append(data)
        
        # We removed self.pre_transform logic here since edge_index is built-in
            
        print(f"Processed {len(data_list)} events. Saving to {self.processed_paths[0]}...")
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

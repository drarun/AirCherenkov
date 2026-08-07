from abc import ABC, abstractmethod
from typing import Dict, Any, Generator
import torch

import numpy as np
class BaseDataReader(ABC):
    """
    Unified abstract interface for reading IACT raw data formats 
    (VBF, HESS, CTA HDF5) and yielding standardized tensors.
    """
    def __init__(self, file_path: str, device: torch.device = None, **kwargs):
        self.file_path = file_path
        self.config = kwargs
        self.device = device if device else torch.device('cpu')
    
    @abstractmethod
    def read_event(self) -> Generator[Dict[str, Any], None, None]:
        """
        Yields a dictionary containing event data.
        Expected keys:
        - 'fadc_traces': Tensor of shape (num_pixels, num_samples)
        - 'pixel_ids': Tensor of shape (num_pixels,)
        - 'event_metadata': Dict containing timestamps, telescope ID, etc.
        """
        pass

class TraceProcessor:
    """
    Applies preprocessing to raw FADC traces. Designed to run on GPU
    for maximum throughput using batched PyTorch operations.
    """
    def __init__(self, calibration_data: Dict[str, torch.Tensor] = None):
        self.calibration_data = calibration_data or {}
        
    def process(self, fadc_traces: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Applies pedestal subtraction and integrates the trace to extract
        charge and timing.
        
        Parameters:
        - fadc_traces: (num_pixels, num_samples) Tensor
        
        Returns:
        Dict with 'charge' and 'timing' Tensors of shape (num_pixels,)
        """
        # Simplistic placeholder logic for integration window
        # In a real implementation, this would be a learnable or dynamic sliding window
        
        # Pedestal subtraction: Average the LAST 3 bins to avoid eating the main Cherenkov pulse 
        # (which typically arrives around bin 1 or 2 since the window starts just 2ns before the first photon)
        baseline = fadc_traces[:, -3:].mean(dim=1, keepdim=True)
        cleaned_traces = fadc_traces - baseline
        
        # Zero out negative fluctuations
        cleaned_traces = torch.clamp(cleaned_traces, min=0.0)
        
        # Extract integrated charge
        charge = cleaned_traces.sum(dim=1)
        
        # Extract time-of-peak (argmax)
        timing = cleaned_traces.argmax(dim=1).float()
        
        return {"charge": charge, "timing": timing}

class VBFReader(BaseDataReader):
    """
    Reader for proprietary VERITAS Bank Format (.vbf) files.
    This relies on the pyvbf library (C++ bindings).
    """
    def __init__(self, file_path: str, device: torch.device = None, **kwargs):
        super().__init__(file_path, device, **kwargs)
        try:
            import pyvbf
            self.vbf_file = pyvbf.VBFArchive(self.file_path)
        except ImportError:
            self.vbf_file = None
            print("Warning: pyvbf not installed. VBFReader will yield empty events.")
        
    def read_event(self) -> Generator[Dict[str, Any], None, None]:
        if self.vbf_file is None:
            yield {}
            return
            
        for event in self.vbf_file.events():
            # Get raw traces: shape (num_pixels, num_samples)
            traces = np.array(event.get_fadc_traces(), dtype=np.float32)
            fadc = torch.tensor(traces, device=self.device)
            
            pixel_ids = torch.tensor(event.get_pixel_ids(), device=self.device)
            
            metadata = {
                "event_id": event.get_event_number(),
                "timestamp": event.get_timestamp(),
                "telescope_id": event.get_telescope_id()
            }
            yield {"fadc_traces": fadc, "pixel_ids": pixel_ids, "event_metadata": metadata}

class HDF5Reader(BaseDataReader):
    """
    Reader for standardized HDF5 files (e.g., CTA DL0/DL1 data).
    Utilizes h5py for chunked, fast reading.
    """
    def __init__(self, file_path: str, device: torch.device = None, **kwargs):
        super().__init__(file_path, device, **kwargs)
        try:
            import h5py
            self.h5_file = h5py.File(self.file_path, 'r')
        except ImportError:
            self.h5_file = None
            print("Warning: h5py not installed. HDF5Reader will yield empty events.")
        
    def read_event(self) -> Generator[Dict[str, Any], None, None]:
        if self.h5_file is None:
            yield {}
            return
            
        # Assuming typical CTA DL0 structure: /dl0/event/telescope/waveform/tel_001
        events = self.h5_file.get('dl0/event/telescope/waveform/tel_001', None)
        if events is None:
            yield {}
            return
            
        for i in range(events.shape[0]):
            waveform = events[i]['waveform'] # (num_pixels, num_samples)
            fadc = torch.tensor(waveform, dtype=torch.float32, device=self.device)
            
            metadata = {
                "event_id": events[i]['event_id'],
                "obs_id": events[i]['obs_id']
            }
            yield {"fadc_traces": fadc, "event_metadata": metadata}

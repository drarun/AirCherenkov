from abc import ABC, abstractmethod
from typing import Dict, Any, Generator
import torch

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
        
        # Pedestal subtraction (assuming a basic fixed baseline for now)
        baseline = fadc_traces[:, :10].mean(dim=1, keepdim=True)
        cleaned_traces = fadc_traces - baseline
        
        # Zero out negative fluctuations
        cleaned_traces = torch.clamp(cleaned_traces, min=0.0)
        
        # Extract integrated charge
        charge = cleaned_traces.sum(dim=1)
        
        # Extract time-of-peak (argmax)
        timing = cleaned_traces.argmax(dim=1).float()
        
        return {"charge": charge, "timing": timing}

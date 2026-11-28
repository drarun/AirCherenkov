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


class MAGICRootReader(BaseDataReader):
    """
    Reader for MAGIC MARS / VERITAS VEGAS & EventDisplay ROOT files via ``uproot``.

    Supports calibrated FADC waveform branches or integrated pixel-charge branches
    and yields standardized event dictionaries compatible with AirCherenkov.
    """

    def __init__(self, file_path: str, device: torch.device = None, **kwargs):
        super().__init__(file_path, device, **kwargs)
        self.uproot_file = None
        try:
            import uproot
            self.uproot_file = uproot.open(self.file_path)
        except ImportError:
            print("Warning: uproot not installed. MAGICRootReader cannot read .root files.")
        except Exception as exc:
            print(f"Warning: Could not open ROOT file {self.file_path}: {exc}")

    def close(self):
        if self.uproot_file is not None:
            try:
                self.uproot_file.close()
            except Exception:
                pass
            self.uproot_file = None

    def read_event(self) -> Generator[Dict[str, Any], None, None]:
        if self.uproot_file is None:
            return
        tree = None
        for candidate in ("Events", "T", "EventTree", "DL1", "ShowerEvents"):
            if candidate in self.uproot_file:
                tree = self.uproot_file[candidate]
                break
        if tree is None:
            for key in self.uproot_file.keys():
                obj = self.uproot_file[key]
                if hasattr(obj, "arrays") and hasattr(obj, "keys"):
                    tree = obj
                    break
        if tree is None:
            return

        branch_names = list(tree.keys())
        waveform_key = next(
            (k for k in branch_names if any(tok in k.lower() for tok in ("fadc", "waveform", "trace", "samples"))),
            None,
        )
        charge_key = next(
            (k for k in branch_names if any(tok in k.lower() for tok in ("charge", "signal", "phe", "image"))),
            None,
        )
        target_key = waveform_key or charge_key
        if target_key is None:
            return

        arrays = tree.arrays([target_key], library="np")
        data = arrays[target_key]
        n_events = len(data)
        for idx in range(n_events):
            arr = np.asarray(data[idx], dtype=np.float32)
            if arr.ndim == 1:
                arr = arr[None, :, None]
            elif arr.ndim == 2:
                arr = arr[None, :, :]
            yield {
                "fadc_traces": torch.tensor(arr, dtype=torch.float32, device=self.device),
                "gain_flags": torch.zeros(arr.shape[:2], dtype=torch.float32, device=self.device),
                "event_metadata": {"event_id": int(idx), "source": "MAGIC_ROOT"},
            }


class HESSDataReader(BaseDataReader):
    """
    Reader for H.E.S.S. / CTA DL0 and DL1 HDF5 files using ``h5py``.

    Supports multi-telescope groups under:
      - ``/dl0/event/telescope/waveform/tel_XXX``
      - ``/dl1/event/telescope/image/tel_XXX``
      - Top-level ``fadc_traces`` / ``waveforms`` arrays
    """

    def __init__(self, file_path: str, device: torch.device = None, **kwargs):
        super().__init__(file_path, device, **kwargs)
        self.h5_file = None
        try:
            import h5py
            self.h5_file = h5py.File(self.file_path, "r")
        except ImportError:
            print("Warning: h5py not installed. HESSDataReader cannot read HDF5 files.")
        except Exception as exc:
            print(f"Warning: Could not open HDF5 file {self.file_path}: {exc}")

    def close(self):
        if self.h5_file is not None:
            try:
                self.h5_file.close()
            except Exception:
                pass
            self.h5_file = None

    def read_event(self) -> Generator[Dict[str, Any], None, None]:
        if self.h5_file is None:
            return

        # 1. Check direct AirCherenkov-compatible HDF5 arrays
        if "fadc_traces" in self.h5_file:
            traces_ds = self.h5_file["fadc_traces"]
            gains_ds = self.h5_file.get("gain_flags", None)
            energy_ds = self.h5_file.get("energy", None)
            label_ds = self.h5_file.get("label", None)
            n_events = traces_ds.shape[0]
            for i in range(n_events):
                arr = np.asarray(traces_ds[i], dtype=np.float32)
                if arr.ndim == 2:
                    arr = arr[None, :, :]
                gains = (
                    np.asarray(gains_ds[i], dtype=np.float32)
                    if gains_ds is not None
                    else np.zeros(arr.shape[:2], dtype=np.float32)
                )
                yield {
                    "fadc_traces": torch.tensor(arr, dtype=torch.float32, device=self.device),
                    "gain_flags": torch.tensor(gains, dtype=torch.float32, device=self.device),
                    "event_metadata": {
                        "event_id": i,
                        "energy": float(energy_ds[i]) if energy_ds is not None else 0.0,
                        "label": int(label_ds[i]) if label_ds is not None else -1,
                        "source": "HESS_HDF5",
                    },
                }
            return

        # 2. Check CTA / H.E.S.S. DL0 waveform groups
        dl0_group = self.h5_file.get("dl0/event/telescope/waveform", None)
        if dl0_group is not None and len(dl0_group.keys()) > 0:
            tel_keys = sorted(dl0_group.keys())
            tel_datasets = [dl0_group[k] for k in tel_keys]
            n_events = min(ds.shape[0] for ds in tel_datasets)
            for i in range(n_events):
                tel_waveforms = []
                ev_id = i
                obs_id = 0
                for ds in tel_datasets:
                    row = ds[i]
                    wf = row["waveform"] if getattr(row, "dtype", None) is not None and row.dtype.names and "waveform" in row.dtype.names else row
                    arr = np.asarray(wf, dtype=np.float32)
                    if arr.ndim == 3:
                        arr = arr[0]
                    tel_waveforms.append(arr)
                    if getattr(row, "dtype", None) is not None and row.dtype.names:
                        if "event_id" in row.dtype.names:
                            ev_id = int(row["event_id"])
                        if "obs_id" in row.dtype.names:
                            obs_id = int(row["obs_id"])
                stacked = np.stack(tel_waveforms, axis=0)
                yield {
                    "fadc_traces": torch.tensor(stacked, dtype=torch.float32, device=self.device),
                    "gain_flags": torch.zeros(stacked.shape[:2], dtype=torch.float32, device=self.device),
                    "event_metadata": {"event_id": ev_id, "obs_id": obs_id, "source": "HESS_DL0"},
                }
            return

        # 3. Check CTA / H.E.S.S. DL1 calibrated image groups
        dl1_group = self.h5_file.get("dl1/event/telescope/image", None)
        if dl1_group is not None and len(dl1_group.keys()) > 0:
            tel_keys = sorted(dl1_group.keys())
            tel_datasets = [dl1_group[k] for k in tel_keys]
            n_events = min(ds.shape[0] for ds in tel_datasets)
            for i in range(n_events):
                tel_imgs = []
                for ds in tel_datasets:
                    row = ds[i]
                    img = row["image"] if getattr(row, "dtype", None) is not None and row.dtype.names and "image" in row.dtype.names else row
                    arr = np.asarray(img, dtype=np.float32).reshape(-1, 1)
                    # Distribute integrated DL1 charge into an 16-bin nominal pulse shape
                    wf = np.zeros((arr.shape[0], 16), dtype=np.float32)
                    wf[:, 4] = arr[:, 0]
                    tel_imgs.append(wf)
                stacked = np.stack(tel_imgs, axis=0)
                yield {
                    "fadc_traces": torch.tensor(stacked, dtype=torch.float32, device=self.device),
                    "gain_flags": torch.zeros(stacked.shape[:2], dtype=torch.float32, device=self.device),
                    "event_metadata": {"event_id": i, "source": "HESS_DL1"},
                }


def _normalize_event_dict(raw_ev: Dict[str, Any], default_zenith: float = 30.0, source_format: str = "aircherenkov_pt") -> Dict[str, Any]:
    traces = raw_ev.get("fadc_traces")
    if isinstance(traces, torch.Tensor):
        traces_np = traces.detach().cpu().numpy().astype(np.float32)
    else:
        traces_np = np.asarray(traces, dtype=np.float32)
    if traces_np.ndim == 2:
        traces_np = traces_np[None, :, :]

    gains = raw_ev.get("gain_flags")
    if gains is None:
        gains_np = np.zeros(traces_np.shape[:2], dtype=np.float32)
    elif isinstance(gains, torch.Tensor):
        gains_np = gains.detach().cpu().numpy().astype(np.float32)
    else:
        gains_np = np.asarray(gains, dtype=np.float32)
    if gains_np.ndim == 1:
        gains_np = gains_np[None, :]

    from sim.provenance import clean_float, clean_str

    meta = raw_ev.get("event_metadata", {}) or {}
    zen = clean_float(raw_ev.get("zenith_deg", meta.get("zenith_deg", default_zenith)))
    zen = zen if zen is not None else float(default_zenith)
    azi = clean_float(raw_ev.get("azimuth_deg", meta.get("azimuth_deg", 0.0)))
    azi = azi if azi is not None else 0.0
    t_zen = clean_float(raw_ev.get("true_zenith_deg", meta.get("true_zenith_deg", zen)))
    t_azi = clean_float(raw_ev.get("true_azimuth_deg", meta.get("true_azimuth_deg", azi)))
    e_val = clean_float(raw_ev.get("energy", meta.get("energy", 0.0)))
    ix_val = clean_float(raw_ev.get("impact_x", meta.get("impact_x", 0.0)))
    iy_val = clean_float(raw_ev.get("impact_y", meta.get("impact_y", 0.0)))
    return {
        "fadc_traces": traces_np,
        "gain_flags": gains_np,
        "energy": e_val if e_val is not None else 0.0,
        "label": int(raw_ev.get("label", meta.get("label", -1))),
        "impact_x": ix_val if ix_val is not None else 0.0,
        "impact_y": iy_val if iy_val is not None else 0.0,
        "impact_radius_thrown": clean_float(raw_ev.get("impact_radius_thrown", meta.get("impact_radius_thrown"))),
        "zenith_deg": zen,
        "azimuth_deg": azi,
        "true_zenith_deg": t_zen if t_zen is not None else zen,
        "true_azimuth_deg": t_azi if t_azi is not None else azi,
        "wobble_dir": clean_str(raw_ev.get("wobble_dir", meta.get("wobble_dir"))) or "Not recorded",
        "source_format": clean_str(meta.get("source", source_format)) or source_format,
        "sim_version": clean_str(raw_ev.get("sim_version", meta.get("sim_version"))),
        "sim_git_commit": clean_str(raw_ev.get("sim_git_commit", meta.get("sim_git_commit"))),
        "sim_created_utc": clean_str(raw_ev.get("sim_created_utc", meta.get("sim_created_utc"))),
        "array_preset": clean_str(raw_ev.get("array_preset", meta.get("array_preset"))),
    }


def load_events_universal(
    file_or_dir_path: str,
    array_preset: str = "VERITAS",
    max_events: int | None = None,
) -> list[Dict[str, Any]]:
    """Load events from a directory of ``.pt`` batches or an experimental data file.

    Supported formats:
      - Directory containing ``sim_batch_*.pt``, ``*.pt``, or ``raw/*.pt``
      - Single ``.pt`` batch file
      - HDF5 (``.h5``, ``.hdf5``) via :class:`HESSDataReader`
      - ROOT (``.root``) via :class:`MAGICRootReader`
      - VBF (``.vbf``) via :class:`VBFReader`
      - NumPy archive (``.npz``)
    """
    import glob
    import os

    if not file_or_dir_path or not os.path.exists(file_or_dir_path):
        raise FileNotFoundError(f"Path not found: {file_or_dir_path}")

    events: list[Dict[str, Any]] = []
    target = os.path.abspath(file_or_dir_path)

    if os.path.isdir(target):
        pt_files = sorted(glob.glob(os.path.join(target, "sim_batch_*.pt")))
        if not pt_files:
            pt_files = sorted(glob.glob(os.path.join(target, "*.pt")))
        if not pt_files and os.path.isdir(os.path.join(target, "raw")):
            pt_files = sorted(glob.glob(os.path.join(target, "raw", "sim_batch_*.pt")))
            if not pt_files:
                pt_files = sorted(glob.glob(os.path.join(target, "raw", "*.pt")))
        prov = None
        if pt_files:
            from sim.provenance import ensure_dataset_tagged

            prov = ensure_dataset_tagged(os.path.dirname(pt_files[0]))
        retro_ver = (prov or {}).get("latest_version")
        retro_sha = (prov or {}).get("latest_commit")
        for fpath in pt_files:
            chunk = torch.load(fpath, weights_only=False)
            if isinstance(chunk, list):
                for ev in chunk:
                    if isinstance(ev, dict) and "fadc_traces" in ev:
                        norm_ev = _normalize_event_dict(ev, source_format="aircherenkov_pt")
                        if norm_ev.get("sim_version") is None and retro_ver is not None:
                            norm_ev["sim_version"] = retro_ver
                            if norm_ev.get("sim_git_commit") is None and retro_sha is not None:
                                norm_ev["sim_git_commit"] = retro_sha
                        events.append(norm_ev)
                        if max_events is not None and len(events) >= max_events:
                            return events
        return events

    lower = target.lower()
    if lower.endswith(".pt"):
        from sim.provenance import ensure_dataset_tagged

        prov = ensure_dataset_tagged(os.path.dirname(target))
        retro_ver = prov.get("latest_version")
        retro_sha = prov.get("latest_commit")
        chunk = torch.load(target, weights_only=False)
        if isinstance(chunk, dict) and "fadc_traces" in chunk:
            norm_ev = _normalize_event_dict(chunk, source_format="aircherenkov_pt")
            if norm_ev.get("sim_version") is None and retro_ver is not None:
                norm_ev["sim_version"] = retro_ver
                if norm_ev.get("sim_git_commit") is None and retro_sha is not None:
                    norm_ev["sim_git_commit"] = retro_sha
            events.append(norm_ev)
        elif isinstance(chunk, list):
            for ev in chunk:
                if isinstance(ev, dict) and "fadc_traces" in ev:
                    norm_ev = _normalize_event_dict(ev, source_format="aircherenkov_pt")
                    if norm_ev.get("sim_version") is None and retro_ver is not None:
                        norm_ev["sim_version"] = retro_ver
                        if norm_ev.get("sim_git_commit") is None and retro_sha is not None:
                            norm_ev["sim_git_commit"] = retro_sha
                    events.append(norm_ev)
                    if max_events is not None and len(events) >= max_events:
                        break
        return events

    if lower.endswith((".h5", ".hdf5")):
        reader = HESSDataReader(target)
        try:
            for ev in reader.read_event():
                if ev and "fadc_traces" in ev:
                    events.append(_normalize_event_dict(ev, source_format="hess_hdf5"))
                    if max_events is not None and len(events) >= max_events:
                        break
        finally:
            reader.close()
        return events

    if lower.endswith(".root"):
        reader = MAGICRootReader(target)
        try:
            for ev in reader.read_event():
                if ev and "fadc_traces" in ev:
                    events.append(_normalize_event_dict(ev, source_format="magic_root"))
                    if max_events is not None and len(events) >= max_events:
                        break
        finally:
            reader.close()
        return events

    if lower.endswith(".vbf"):
        reader = VBFReader(target)
        for ev in reader.read_event():
            if ev and "fadc_traces" in ev:
                events.append(_normalize_event_dict(ev, source_format="veritas_vbf"))
                if max_events is not None and len(events) >= max_events:
                    break
        return events

    if lower.endswith(".npz"):
        data = np.load(target, allow_pickle=False)
        traces = data["fadc_traces"]
        gains = data["gain_flags"] if "gain_flags" in data else None
        energies = data["energy"] if "energy" in data else None
        labels = data["label"] if "label" in data else None
        if traces.ndim == 3:
            traces = traces[None, ...]
        for idx in range(traces.shape[0]):
            ev_dict = {
                "fadc_traces": traces[idx],
                "gain_flags": gains[idx] if gains is not None else np.zeros(traces[idx].shape[:2], dtype=np.float32),
                "energy": float(energies[idx]) if energies is not None else 0.0,
                "label": int(labels[idx]) if labels is not None else -1,
            }
            events.append(_normalize_event_dict(ev_dict, source_format=f"{array_preset.lower()}_npz"))
            if max_events is not None and len(events) >= max_events:
                break
        return events

    raise ValueError(f"Unsupported event file format: {target}")

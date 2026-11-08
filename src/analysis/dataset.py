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
from sim.fadc import restore_low_gain_traces
from analysis.io import BaseDataReader, TraceProcessor
from analysis.graph import GraphBuilder


class CherenkovDataset(InMemoryDataset):
    """
    Unified PyTorch Geometric Dataset for Cherenkov events.
    Can ingest real data files (via VBFReader/HDF5Reader) or 
    pre-processed simulation graphs (.pt files).
    """
    # VERITAS array centroid (m)
    _ARR_CX = 19.46
    _ARR_CY = 48.32

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

        from sim.camera import Camera
        cam = Camera(n_rings=12, pixel_size=0.15)
        self._n_pix = cam.n_pixels
        self._cam_edges_by_ntel = {
            k: torch.cat([cam.edge_index + (i * self._n_pix) for i in range(k)], dim=1)
            for k in range(1, 5)
        }
        
        # PyG automatically calls process() if the processed files don't exist
        super().__init__(root, transform, pre_transform)
        
        # Load the processed data
        self.data, self.slices = torch.load(
            self.processed_paths[0], weights_only=False
        )

    def get(self, idx: int) -> Data:
        data = super().get(idx)
        if getattr(data, 'edge_index', None) is None and getattr(data, 'x', None) is not None:
            n_nodes = data.x.shape[0]
            if n_nodes % self._n_pix == 0:
                n_tel = n_nodes // self._n_pix
                if n_tel in self._cam_edges_by_ntel:
                    data.edge_index = self._cam_edges_by_ntel[n_tel]
        return data

    @property
    def raw_file_names(self) -> List[str]:
        raw_dir = os.path.join(self.root, 'raw')
        if not os.path.exists(raw_dir):
            return []
        return sorted([f for f in os.listdir(raw_dir) if os.path.isfile(os.path.join(raw_dir, f)) and f.endswith('.pt')])

    @property
    def processed_file_names(self) -> str:
        return 'data.pt'

    def process(self):
        data_list = []
        
        from sim.camera import Camera
        from sim.telescope import TelescopeArray
        from sim.trigger import CameraTrigger
        from recon.cleaning import double_pass_fadc_clean
        from recon.hillas import compute_hillas
        from recon.ml_stereo import extract_stereo_features, StereoEnergyRegressor

        cam = Camera(n_rings=12, pixel_size=0.15)
        telescopes = TelescopeArray.veritas_array().telescopes
        pixel_x, pixel_y = cam.pixel_x, cam.pixel_y
        n_pix = len(pixel_x)
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
                    
                    data.y_energy = torch.tensor([0.0], dtype=torch.float32)
                    data.y_class = torch.tensor([0.0], dtype=torch.float32)
                    data_list.append(data)
            else:
                # Process simulation `.pt` files (saved by `generate_training_data.py`)
                sim_data = torch.load(raw_file, weights_only=False)
                
                if isinstance(sim_data, list):
                    for evt in sim_data:
                        traces_all = np.asarray(evt['fadc_traces'], dtype=np.float32)
                        gains_all = np.asarray(evt['gain_flags'], dtype=np.float32)
                        zen_deg = float(evt.get('zenith_deg', 30.0))
                        azi_deg = float(evt.get('azimuth_deg', 0.0))

                        # Extract stereoscopic & DoublePass array features once per event
                        stereo_vec = extract_stereo_features(
                            traces_all,
                            telescopes,
                            cam,
                            pointing_zenith_deg=zen_deg,
                            pointing_azimuth_deg=azi_deg,
                            gain_flags=gains_all,
                            include_extended=True,
                        )
                        # Require valid geometric stereoscopic reconstruction (geom_avail == 1)
                        if stereo_vec[56] < 0.5:
                            continue

                        aug_vec = StereoEnergyRegressor._augment_features(stereo_vec[None, :])[0]
                        log_slope = float(aug_vec[87])
                        depth_corr_log_sz = float(aug_vec[89])
                        reco_r_arr = float(
                            np.hypot(float(stereo_vec[57]) - self._ARR_CX, float(stereo_vec[58]) - self._ARR_CY)
                        )

                        event_x_list = []
                        for tel_idx in range(len(traces_all)):
                            gain_np = gains_all[tel_idx]
                            trace_np = restore_low_gain_traces(traces_all[tel_idx], gain_np)
                            dp = double_pass_fadc_clean(trace_np, cam)

                            n_clean_pix = int(np.count_nonzero(dp.mask))
                            clean_charge = float(np.sum(dp.charges[dp.mask])) if n_clean_pix > 0 else 0.0
                            if n_clean_pix < 3 or clean_charge < 35.0:
                                continue

                            img_sum = np.sum(trace_np, axis=1)
                            is_trig_hw, _ = trigger.evaluate(img_sum, dp.arrival_times_ns)
                            if not is_trig_hw and clean_charge < 60.0:
                                continue

                            h = compute_hillas(cam, dp.charges, dp.mask)
                            if h is None or h.size < 35.0:
                                continue

                            trace_feat = torch.log1p(
                                torch.clamp(torch.tensor(trace_np, dtype=torch.float32), min=0.0)
                            )
                            gain_feat = torch.tensor(gain_np, dtype=torch.float32).unsqueeze(1)

                            t_ref = float(dp.time_intercept)
                            timing_feat = torch.tensor(
                                (dp.arrival_times_ns - t_ref) / 10.0, dtype=torch.float32
                            ).unsqueeze(1)

                            cog_x, cog_y = float(h.centroid_x), float(h.centroid_y)
                            h_len = float(h.length)
                            h_wid = float(h.width)
                            h_wl = float(h_wid / max(h_len, 1e-4))
                            total_charge = float(h.size)

                            px_shifted = torch.tensor(pixel_x - cog_x, dtype=torch.float32).unsqueeze(1)
                            py_shifted = torch.tensor(pixel_y - cog_y, dtype=torch.float32).unsqueeze(1)
                            px_squared = px_shifted ** 2
                            py_squared = py_shifted ** 2
                            pxy = px_shifted * py_shifted

                            pixel_charge = np.maximum(dp.charges, 0.0)
                            log_pixel_charge = torch.tensor(
                                np.log10(pixel_charge + 1.0), dtype=torch.float32
                            ).unsqueeze(1)
                            log_size = np.log10(max(total_charge, 1.0))
                            log_size_feat = torch.full((n_pix, 1), log_size, dtype=torch.float32)

                            # 6 direction-agnostic DoublePass morphology & timing features (25..30)
                            dp_mask_feat = torch.tensor(dp.mask.astype(np.float32)).unsqueeze(1)
                            tgrad_feat = torch.full((n_pix, 1), float(dp.time_gradient) / 10.0, dtype=torch.float32)
                            trms_feat = torch.full((n_pix, 1), float(dp.time_rms) / 5.0, dtype=torch.float32)
                            len_feat = torch.full((n_pix, 1), h_len, dtype=torch.float32)
                            wid_feat = torch.full((n_pix, 1), h_wid, dtype=torch.float32)
                            wl_feat = torch.full((n_pix, 1), h_wl, dtype=torch.float32)

                            # 6 stereoscopic & containment energy calibration features (31..36)
                            rt_tel = float(stereo_vec[61 + tel_idx]) if tel_idx < 4 else reco_r_arr
                            rt_feat = torch.full((n_pix, 1), rt_tel / 200.0, dtype=torch.float32)
                            loss1_feat = torch.full((n_pix, 1), float(dp.loss1), dtype=torch.float32)
                            loss2_feat = torch.full((n_pix, 1), float(dp.loss2), dtype=torch.float32)
                            slope_feat = torch.full((n_pix, 1), log_slope, dtype=torch.float32)
                            dcorr_sz_feat = torch.full((n_pix, 1), depth_corr_log_sz, dtype=torch.float32)
                            reco_r_feat = torch.full((n_pix, 1), reco_r_arr / 200.0, dtype=torch.float32)

                            x = torch.cat([
                                trace_feat,        # 0..15
                                gain_feat,         # 16
                                timing_feat,       # 17
                                px_shifted,        # 18
                                py_shifted,        # 19
                                px_squared,        # 20
                                py_squared,        # 21
                                pxy,               # 22
                                log_pixel_charge,  # 23
                                log_size_feat,     # 24
                                dp_mask_feat,      # 25
                                tgrad_feat,        # 26
                                trms_feat,         # 27
                                len_feat,          # 28
                                wid_feat,          # 29
                                wl_feat,           # 30
                                rt_feat,           # 31
                                loss1_feat,        # 32
                                loss2_feat,        # 33
                                slope_feat,        # 34
                                dcorr_sz_feat,     # 35
                                reco_r_feat,       # 36
                            ], dim=1)
                            event_x_list.append(x)

                        # Enforce stereoscopic multiplicity (>= 2 triggered telescopes)
                        if len(event_x_list) >= 2:
                            combined_x = torch.cat(event_x_list, dim=0).half()
                            y_e = torch.log10(torch.tensor([evt['energy']], dtype=torch.float32))
                            y_c = torch.tensor([evt['label']], dtype=torch.float32)

                            # Omit edge_index on disk; attached dynamically in get() to save ~4 GB RAM/disk
                            data = Data(x=combined_x, y_energy=y_e, y_class=y_c)
                            data_list.append(data)

        print(f"Processed {len(data_list)} events. Saving to {self.processed_paths[0]}...")
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

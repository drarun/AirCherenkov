import concurrent.futures
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

_WORKER_CACHE = {}
_CHUNK_CACHE_SCHEMA = 37


def _get_worker_geometry():
    if "geom" not in _WORKER_CACHE:
        from sim.camera import Camera
        from sim.telescope import TelescopeArray
        from sim.trigger import CameraTrigger

        cam = Camera(n_rings=12, pixel_size=0.15)
        telescopes = TelescopeArray.veritas_array().telescopes
        pixel_x, pixel_y = cam.pixel_x, cam.pixel_y
        n_pix = len(pixel_x)
        trigger = CameraTrigger(pixel_x, pixel_y)
        _WORKER_CACHE["geom"] = (cam, telescopes, pixel_x, pixel_y, n_pix, trigger)
    return _WORKER_CACHE["geom"]


def _process_sim_chunk_file(raw_file: str, cache_dir: str, arr_cx: float, arr_cy: float) -> str:
    """Process one simulation .pt chunk into PyG Data objects with mtime/size caching."""
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, f"{os.path.basename(raw_file)}.cache.pt")
    st = os.stat(raw_file)

    if os.path.isfile(cache_path):
        try:
            cached = torch.load(cache_path, map_location="cpu", weights_only=False)
            if (
                isinstance(cached, dict)
                and cached.get("schema") == _CHUNK_CACHE_SCHEMA
                and cached.get("mtime_ns") == int(st.st_mtime_ns)
                and cached.get("size") == int(st.st_size)
            ):
                return cache_path
        except Exception:
            pass

    from recon.ml_stereo import StereoEnergyRegressor, extract_stereo_features

    cam, telescopes, pixel_x, pixel_y, n_pix, trigger = _get_worker_geometry()
    sim_data = torch.load(raw_file, map_location="cpu", weights_only=False)
    chunk_data = []

    if isinstance(sim_data, list):
        for evt in sim_data:
            traces_all = np.asarray(evt["fadc_traces"], dtype=np.float32)
            gains_all = np.asarray(evt["gain_flags"], dtype=np.float32)
            zen_deg = float(evt.get("zenith_deg", 30.0))
            azi_deg = float(evt.get("azimuth_deg", 0.0))

            stereo_vec, restored_traces, dp_results, hillas_results = extract_stereo_features(
                traces_all,
                telescopes,
                cam,
                pointing_zenith_deg=zen_deg,
                pointing_azimuth_deg=azi_deg,
                gain_flags=gains_all,
                include_extended=True,
                return_intermediates=True,
            )
            if stereo_vec[56] < 0.5:
                continue

            aug_vec = StereoEnergyRegressor._augment_features(stereo_vec[None, :])[0]
            log_slope = float(aug_vec[87])
            depth_corr_log_sz = float(aug_vec[89])
            reco_r_arr = float(np.hypot(float(stereo_vec[57]) - arr_cx, float(stereo_vec[58]) - arr_cy))

            event_x_list = []
            for tel_idx in range(len(restored_traces)):
                gain_np = gains_all[tel_idx]
                trace_np = restored_traces[tel_idx]
                dp = dp_results[tel_idx]
                h = hillas_results[tel_idx]

                n_clean_pix = int(np.count_nonzero(dp.mask))
                clean_charge = float(np.sum(dp.charges[dp.mask])) if n_clean_pix > 0 else 0.0
                if n_clean_pix < 3 or clean_charge < 35.0:
                    continue

                img_sum = np.sum(trace_np, axis=1)
                is_trig_hw, _ = trigger.evaluate(img_sum, dp.arrival_times_ns)
                if not is_trig_hw and clean_charge < 60.0:
                    continue

                if h is None or h.size < 35.0:
                    continue

                trace_feat = torch.log1p(torch.clamp(torch.tensor(trace_np, dtype=torch.float32), min=0.0))
                gain_feat = torch.tensor(gain_np, dtype=torch.float32).unsqueeze(1)
                t_ref = float(dp.time_intercept)
                timing_feat = torch.tensor((dp.arrival_times_ns - t_ref) / 10.0, dtype=torch.float32).unsqueeze(1)

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
                log_pixel_charge = torch.tensor(np.log10(pixel_charge + 1.0), dtype=torch.float32).unsqueeze(1)
                log_size = np.log10(max(total_charge, 1.0))
                log_size_feat = torch.full((n_pix, 1), log_size, dtype=torch.float32)

                dp_mask_feat = torch.tensor(dp.mask.astype(np.float32)).unsqueeze(1)
                tgrad_feat = torch.full((n_pix, 1), float(dp.time_gradient) / 10.0, dtype=torch.float32)
                trms_feat = torch.full((n_pix, 1), float(dp.time_rms) / 5.0, dtype=torch.float32)
                len_feat = torch.full((n_pix, 1), h_len, dtype=torch.float32)
                wid_feat = torch.full((n_pix, 1), h_wid, dtype=torch.float32)
                wl_feat = torch.full((n_pix, 1), h_wl, dtype=torch.float32)

                rt_tel = float(stereo_vec[61 + tel_idx]) if tel_idx < 4 else reco_r_arr
                rt_feat = torch.full((n_pix, 1), rt_tel / 200.0, dtype=torch.float32)
                loss1_feat = torch.full((n_pix, 1), float(dp.loss1), dtype=torch.float32)
                loss2_feat = torch.full((n_pix, 1), float(dp.loss2), dtype=torch.float32)
                slope_feat = torch.full((n_pix, 1), log_slope, dtype=torch.float32)
                dcorr_sz_feat = torch.full((n_pix, 1), depth_corr_log_sz, dtype=torch.float32)
                reco_r_feat = torch.full((n_pix, 1), reco_r_arr / 200.0, dtype=torch.float32)

                x = torch.cat([
                    trace_feat, gain_feat, timing_feat,
                    px_shifted, py_shifted, px_squared, py_squared, pxy,
                    log_pixel_charge, log_size_feat,
                    dp_mask_feat, tgrad_feat, trms_feat, len_feat, wid_feat, wl_feat,
                    rt_feat, loss1_feat, loss2_feat, slope_feat, dcorr_sz_feat, reco_r_feat,
                ], dim=1)
                event_x_list.append(x)

            if len(event_x_list) >= 2:
                combined_x = torch.cat(event_x_list, dim=0).half()
                y_e = torch.log10(torch.tensor([evt["energy"]], dtype=torch.float32))
                y_c = torch.tensor([evt["label"]], dtype=torch.float32)
                chunk_data.append(Data(x=combined_x, y_energy=y_e, y_class=y_c))

    torch.save(
        {
            "schema": _CHUNK_CACHE_SCHEMA,
            "mtime_ns": int(st.st_mtime_ns),
            "size": int(st.st_size),
            "data_list": chunk_data,
        },
        cache_path,
    )
    return cache_path


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

        if self.reader_class is not None:
            for raw_file in self.raw_paths:
                print(f"Processing {raw_file}...")
                reader = self.reader_class(raw_file, device=self.device)
                for event in reader.read_event():
                    if "fadc_traces" not in event:
                        continue
                    _ = self.processor.process(event["fadc_traces"])

                    if self.graph_builder:
                        data = self.graph_builder.build_graph(event["fadc_traces"])
                    else:
                        data = Data(x=event["fadc_traces"])

                    data.y_energy = torch.tensor([0.0], dtype=torch.float32)
                    data.y_class = torch.tensor([0.0], dtype=torch.float32)
                    data_list.append(data)
        elif self.raw_paths:
            cache_dir = os.path.join(self.processed_dir, "chunks")
            n_files = len(self.raw_paths)
            print(f"Processing {n_files} simulation chunk(s) (per-chunk cache + parallel workers)...")
            if n_files >= 2:
                max_workers = min(max(1, (os.cpu_count() or 2) - 1), 8, n_files)
                try:
                    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as pool:
                        cache_paths = list(
                            pool.map(
                                _process_sim_chunk_file,
                                self.raw_paths,
                                [cache_dir] * n_files,
                                [self._ARR_CX] * n_files,
                                [self._ARR_CY] * n_files,
                            )
                        )
                except Exception:
                    cache_paths = [
                        _process_sim_chunk_file(rf, cache_dir, self._ARR_CX, self._ARR_CY)
                        for rf in self.raw_paths
                    ]
            else:
                cache_paths = [
                    _process_sim_chunk_file(self.raw_paths[0], cache_dir, self._ARR_CX, self._ARR_CY)
                ]

            for cp in cache_paths:
                cached = torch.load(cp, map_location="cpu", weights_only=False)
                data_list.extend(cached.get("data_list", []))

        print(f"Processed {len(data_list)} events. Saving to {self.processed_paths[0]}...")
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

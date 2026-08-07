"""Dataset adapters for real and simulated Cherenkov-camera events."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Callable, List

import numpy as np
import torch

try:
    from torch_geometric.data import Data, InMemoryDataset
except ImportError:
    # Keep the pure preprocessing helpers importable in CPU-only/core installs.
    # Constructing CherenkovDataset itself still requires torch-geometric.
    class InMemoryDataset:
        def __init__(self, *args, **kwargs):
            raise ImportError(
                "CherenkovDataset requires torch-geometric; install aircherenkov[ml]"
            )

    class Data:
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)


from analysis.graph import GraphBuilder
from analysis.io import TraceProcessor
from sim.backend import get_device
from sim.fadc import restore_low_gain_traces as _restore_low_gain_traces


MODEL_TRACE_BINS = 16
DEFAULT_LOW_GAIN_FACTOR = 10.0
DEFAULT_BIN_WIDTH_NS = 2.0
_VERITAS_POSITIONS_M = np.asarray(
    ((0.0, 0.0, 0.0), (100.0, 0.0, 0.0),
     (0.0, 100.0, 0.0), (100.0, 100.0, 0.0)),
    dtype=np.float32,
)


def _as_numpy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def restore_low_gain_traces(traces, gain_flags, low_gain_factor=DEFAULT_LOW_GAIN_FACTOR):
    """Return waveforms in physical photoelectron-equivalent units.

    The detector stores a saturated pixel through its attenuated low-gain
    channel and marks that pixel with ``gain_flags == 1``.  That attenuation
    must be undone before trigger, charge, timing, or ML feature calculations.
    """
    return _restore_low_gain_traces(
        _as_numpy(traces),
        _as_numpy(gain_flags),
        low_gain_factor=low_gain_factor,
    )


def _nested_value(mapping, paths, default=None):
    for path in paths:
        current = mapping
        for key in path:
            if not isinstance(current, dict) or key not in current:
                break
            current = current[key]
        else:
            if current is not None:
                return current
    return default


def _readout_value(event, generation_config, name, default):
    paths = (
        (name,),
        ("fadc", name),
        ("fadc_config", name),
        ("readout", name),
        ("config", "camera_trigger", name),
    )
    value = _nested_value(event, paths)
    if value is None:
        value = _nested_value(generation_config or {}, (("camera_trigger", name),))
    return default if value is None else float(value)


def _event_impact_xy(event):
    impact_x = _nested_value(
        event,
        (("impact_x_m",), ("impact_x",), ("thrown", "impact_x_m"),
         ("thrown", "impact_x")),
        np.nan,
    )
    impact_y = _nested_value(
        event,
        (("impact_y_m",), ("impact_y",), ("thrown", "impact_y_m"),
         ("thrown", "impact_y")),
        np.nan,
    )
    return float(impact_x), float(impact_y)


def _telescope_position(event, telescope_index, telescope_count):
    positions = _nested_value(
        event,
        (("telescope_positions_m",), ("telescope_positions",),
         ("array", "telescope_positions_m")),
    )
    if positions is not None:
        positions = _as_numpy(positions)
        if positions.ndim == 1:
            positions = positions[None, :]
        if positions.ndim == 2 and telescope_index < len(positions):
            position = np.asarray(positions[telescope_index], dtype=np.float32).reshape(-1)
            if position.size == 2:
                position = np.append(position, np.float32(0.0))
            if position.size == 3:
                return position

    # Historical training files did not save geometry, but were generated with
    # the fixed four-telescope VERITAS layout.
    if telescope_count == len(_VERITAS_POSITIONS_M):
        return _VERITAS_POSITIONS_M[telescope_index].copy()
    return np.full(3, np.nan, dtype=np.float32)


def extract_simulation_metadata(event, telescope_index, telescope_count, event_id):
    """Build graph-level provenance tensors for one telescope view."""
    telescope_ids = event.get("telescope_ids")
    if telescope_ids is None:
        telescope_id = telescope_index
    else:
        telescope_ids = _as_numpy(telescope_ids).reshape(-1)
        telescope_id = telescope_ids[telescope_index]

    position = _telescope_position(event, telescope_index, telescope_count)
    impact_x, impact_y = _event_impact_xy(event)
    if np.isfinite(position[:2]).all() and np.isfinite((impact_x, impact_y)).all():
        impact_distance = float(
            np.hypot(position[0] - impact_x, position[1] - impact_y)
        )
    else:
        impact_distance = np.nan

    trigger = event.get("trigger") if isinstance(event.get("trigger"), dict) else {}
    multiplicity = event.get(
        "telescope_multiplicity",
        trigger.get("triggered_telescopes", telescope_count),
    )
    sampling_weight = event.get(
        "sampling_weight",
        _nested_value(event, (("weights", "inverse_sampling_density"),), 1.0),
    )

    camera_triggered = trigger.get("camera_triggered")
    if camera_triggered is not None and telescope_index < len(camera_triggered):
        view_triggered = bool(camera_triggered[telescope_index])
    else:
        view_triggered = True
    camera_times = trigger.get("camera_trigger_times_ns")
    if camera_times is not None and telescope_index < len(camera_times):
        view_time = camera_times[telescope_index]
        view_time = np.nan if view_time is None else float(view_time)
    else:
        view_time = np.nan

    return {
        "event_id": torch.tensor([int(event_id)], dtype=torch.long),
        "telescope_id": torch.tensor([int(telescope_id)], dtype=torch.long),
        # Leading graph dimension makes PyG batches shape these as (B, 3)/(B, 2).
        "telescope_position_m": torch.tensor(position, dtype=torch.float32).reshape(1, 3),
        "telescope_multiplicity": torch.tensor([int(multiplicity)], dtype=torch.long),
        "impact_xy_m": torch.tensor([[impact_x, impact_y]], dtype=torch.float32),
        "impact_x_m": torch.tensor([impact_x], dtype=torch.float32),
        "impact_y_m": torch.tensor([impact_y], dtype=torch.float32),
        "impact_distance_m": torch.tensor([impact_distance], dtype=torch.float32),
        "sampling_weight": torch.tensor([float(sampling_weight)], dtype=torch.float64),
        "camera_triggered": torch.tensor([view_triggered], dtype=torch.bool),
        "camera_trigger_time_ns": torch.tensor([view_time], dtype=torch.float32),
    }


def _integrated_image_trace(image):
    """Adapt legacy integrated images to the existing 16-bin model contract."""
    image = _as_numpy(image).astype(np.float32, copy=False).reshape(-1)
    trace = np.zeros((len(image), MODEL_TRACE_BINS), dtype=np.float32)
    trace[:, 0] = np.clip(image, 0.0, None)
    return trace


def _legacy_event_value(payload, key, event_index, event_count):
    """Select event-scoped metadata from a historical batched dictionary."""
    if key not in payload:
        return None
    value = payload[key]
    array = _as_numpy(value)
    if array.ndim == 0:
        return array.item()

    shared_array_fields = {"telescope_positions_m", "telescope_ids"}
    if key in shared_array_fields:
        # Positions/IDs were commonly saved once for the whole array. A leading
        # event dimension unambiguously marks the newer per-event variant.
        minimum_event_rank = 3 if key == "telescope_positions_m" else 2
        if array.ndim >= minimum_event_rank and len(array) == event_count:
            return value[event_index]
        return value

    if len(array) == event_count:
        return value[event_index]
    return value


def _trace_features(traces, gain_flags, pixel_x, pixel_y, low_gain_factor):
    corrected = restore_low_gain_traces(traces, gain_flags, low_gain_factor)
    if corrected.shape[1] != MODEL_TRACE_BINS:
        raise ValueError(
            f"the current GNN expects {MODEL_TRACE_BINS} trace bins, "
            f"but the event contains {corrected.shape[1]}"
        )

    charge = corrected.sum(axis=1)
    timing = corrected.argmax(axis=1).astype(np.float32)
    pixel_x = _as_numpy(pixel_x).astype(np.float32, copy=False).reshape(-1)
    pixel_y = _as_numpy(pixel_y).astype(np.float32, copy=False).reshape(-1)
    if charge.shape != pixel_x.shape or charge.shape != pixel_y.shape:
        raise ValueError("camera coordinates and waveform pixel counts do not match")

    # Negative pedestal fluctuations are not physical image weights.
    cog_charge = np.clip(charge, 0.0, None)
    total_charge = float(cog_charge.sum())
    if total_charge > 0:
        cog_x = float(np.sum(cog_charge * pixel_x) / total_charge)
        cog_y = float(np.sum(cog_charge * pixel_y) / total_charge)
    else:
        cog_x = cog_y = 0.0

    px_shifted = torch.from_numpy(pixel_x - cog_x).unsqueeze(1)
    py_shifted = torch.from_numpy(pixel_y - cog_y).unsqueeze(1)
    trace_feature = torch.from_numpy(corrected)
    gain_feature = torch.from_numpy(
        _as_numpy(gain_flags).astype(np.float32, copy=False).reshape(-1)
    ).unsqueeze(1)
    x = torch.cat(
        (
            trace_feature,
            gain_feature,
            px_shifted,
            py_shifted,
            px_shifted.square(),
            py_shifted.square(),
            px_shifted * py_shifted,
        ),
        dim=1,
    )
    return x, charge, timing


def _generation_config(raw_dir):
    manifest_path = Path(raw_dir) / ".aircherenkov" / "manifest.json"
    if not manifest_path.is_file():
        return {}
    try:
        with manifest_path.open("r", encoding="utf-8") as handle:
            manifest = json.load(handle)
    except (OSError, ValueError):
        return {}
    config = manifest.get("config", {})
    return config if isinstance(config, dict) else {}


class CherenkovDataset(InMemoryDataset):
    """PyG dataset for real camera records or saved simulation events."""

    def __init__(
        self,
        root: str,
        reader_class: type = None,
        graph_builder: GraphBuilder = None,
        transform: Callable = None,
        pre_transform: Callable = None,
    ):
        self.reader_class = reader_class
        self.graph_builder = graph_builder
        self.processor = TraceProcessor()
        self.device = get_device() or torch.device("cpu")
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(
            self.processed_paths[0], weights_only=False
        )

    @property
    def raw_file_names(self) -> List[str]:
        raw_dir = os.path.join(self.root, "raw")
        if not os.path.exists(raw_dir):
            return []
        # Stable order is required for deterministic synthetic IDs in old files.
        return sorted(
            name
            for name in os.listdir(raw_dir)
            if os.path.isfile(os.path.join(raw_dir, name))
        )

    @property
    def processed_file_names(self) -> str:
        # Do not silently reuse processed graphs that predate provenance and
        # low-gain restoration.
        return "data_v2.pt"

    def _attach_graph_geometry(self, data):
        if self.graph_builder is not None:
            data.edge_index = self.graph_builder.edge_index
            data.pos = self.graph_builder.pos

    def _simulation_graph(
        self,
        event,
        telescope_index,
        telescope_count,
        event_id,
        trace,
        gain_flags,
        pixel_x,
        pixel_y,
        low_gain_factor,
    ):
        x, charge, timing_bins = _trace_features(
            trace, gain_flags, pixel_x, pixel_y, low_gain_factor
        )
        data = Data(
            x=x,
            y_energy=torch.log10(
                torch.tensor([float(event["energy"])], dtype=torch.float32)
            ),
            y_class=torch.tensor([float(event["label"])], dtype=torch.float32),
        )
        self._attach_graph_geometry(data)
        for name, value in extract_simulation_metadata(
            event, telescope_index, telescope_count, event_id
        ).items():
            setattr(data, name, value)
        return data, charge, timing_bins

    def process(self):
        data_list = []

        from sim.camera import Camera
        from sim.trigger import CameraTrigger

        camera = Camera(n_rings=12)
        default_pixel_x, default_pixel_y = camera.pixel_x, camera.pixel_y
        generation_config = _generation_config(self.raw_dir)
        trigger_config = generation_config.get("camera_trigger", {})
        trigger = CameraTrigger(
            default_pixel_x,
            default_pixel_y,
            threshold_pe=float(trigger_config.get("threshold_pe", 5.0)),
            window_ns=float(trigger_config.get("window_ns", 5.0)),
            min_pixels=int(trigger_config.get("min_pixels", 3)),
            pixel_size=camera.pixel_size,
        )

        used_event_ids = set()
        next_synthetic_id = 0

        def event_id_for(value=None):
            nonlocal next_synthetic_id
            if value is not None:
                event_id = int(value)
                used_event_ids.add(event_id)
                return event_id
            while next_synthetic_id in used_event_ids:
                next_synthetic_id += 1
            event_id = next_synthetic_id
            used_event_ids.add(event_id)
            next_synthetic_id += 1
            return event_id

        for raw_file in self.raw_paths:
            print(f"Processing {raw_file}...")

            if self.reader_class is not None:
                reader = self.reader_class(raw_file, device=self.device)
                for event in reader.read_event():
                    if "fadc_traces" not in event:
                        continue
                    processed = self.processor.process(event["fadc_traces"])
                    if self.graph_builder:
                        data = self.graph_builder.build_graph(
                            processed["charge"], processed["timing"]
                        )
                    else:
                        data = Data(
                            x=torch.stack(
                                [processed["charge"], processed["timing"]], dim=-1
                            )
                        )
                    data.y_energy = torch.tensor([0.0], dtype=torch.float32)
                    data.y_class = torch.tensor([0.0], dtype=torch.float32)

                    metadata = event.get("event_metadata", {})
                    raw_event_id = metadata.get("event_id")
                    real_event_id = event_id_for(raw_event_id)
                    synthetic_event = {
                        "telescope_ids": [metadata.get("telescope_id", 0)],
                        "telescope_positions_m": [
                            metadata.get("telescope_position_m", (np.nan,) * 3)
                        ],
                        "impact_x_m": metadata.get("impact_x_m", np.nan),
                        "impact_y_m": metadata.get("impact_y_m", np.nan),
                        "telescope_multiplicity": metadata.get(
                            "telescope_multiplicity", 1
                        ),
                    }
                    for name, value in extract_simulation_metadata(
                        synthetic_event, 0, 1, real_event_id
                    ).items():
                        setattr(data, name, value)
                    data_list.append(data)
                continue

            sim_data = torch.load(raw_file, weights_only=False)

            if isinstance(sim_data, dict) and "images" in sim_data:
                # Historical batched dictionary. Integrated images are adapted
                # to 16-bin traces so they still satisfy the current model API.
                images = sim_data["images"]
                energies = sim_data["energies"]
                labels = sim_data["labels"]
                pixel_x = sim_data.get("pixel_x", default_pixel_x)
                pixel_y = sim_data.get("pixel_y", default_pixel_y)
                event_ids = sim_data.get("event_ids", sim_data.get("event_id"))

                for index in range(len(images)):
                    if event_ids is None:
                        raw_event_id = None
                    elif _as_numpy(event_ids).ndim == 0:
                        raw_event_id = event_ids
                    else:
                        raw_event_id = event_ids[index]
                    event_id = event_id_for(raw_event_id)
                    event_images = _as_numpy(images[index])
                    if event_images.ndim == 1:
                        event_images = event_images[None, :]
                    if event_images.ndim != 2:
                        raise ValueError("legacy images must have shape (events, telescopes, pixels)")
                    event = {
                        "energy": energies[index],
                        "label": labels[index],
                    }
                    for key in (
                        "telescope_positions_m", "telescope_ids",
                        "telescope_multiplicity", "impact_x", "impact_y",
                        "impact_x_m", "impact_y_m", "sampling_weight",
                    ):
                        value = _legacy_event_value(
                            sim_data, key, index, len(images)
                        )
                        if value is not None:
                            event[key] = value

                    telescope_count = len(event_images)
                    for telescope_index, image in enumerate(event_images):
                        trace = _integrated_image_trace(image)
                        gain_flags = np.zeros(trace.shape[0], dtype=np.float32)
                        data, _, _ = self._simulation_graph(
                            event, telescope_index, telescope_count, event_id,
                            trace, gain_flags, pixel_x, pixel_y, 1.0,
                        )
                        data_list.append(data)
                continue

            if not isinstance(sim_data, list):
                raise ValueError(
                    f"unsupported simulation payload in {raw_file}: "
                    f"expected an event list or legacy image dictionary"
                )

            for event in sim_data:
                if not isinstance(event, dict):
                    raise ValueError("simulation event lists must contain dictionaries")
                event_id = event_id_for(event.get("event_id"))
                low_gain_factor = _readout_value(
                    event,
                    generation_config,
                    "low_gain_factor",
                    DEFAULT_LOW_GAIN_FACTOR,
                )
                bin_width_ns = _readout_value(
                    event,
                    generation_config,
                    "trace_bin_width_ns",
                    DEFAULT_BIN_WIDTH_NS,
                )

                if "fadc_traces" in event:
                    traces = _as_numpy(event["fadc_traces"])
                    if traces.ndim == 2:
                        traces = traces[None, :, :]
                    if traces.ndim != 3:
                        raise ValueError(
                            "fadc_traces must have shape (telescopes, pixels, bins)"
                        )
                    if "gain_flags" in event:
                        gains = _as_numpy(event["gain_flags"])
                        if gains.ndim == 1:
                            gains = gains[None, :]
                    else:
                        gains = np.zeros(traces.shape[:2], dtype=np.float32)
                    if gains.shape != traces.shape[:2]:
                        raise ValueError("gain_flags and fadc_traces shapes do not match")
                    integrated_legacy = False
                elif "images" in event:
                    images = _as_numpy(event["images"])
                    if images.ndim == 1:
                        images = images[None, :]
                    traces = np.stack([_integrated_image_trace(image) for image in images])
                    gains = np.zeros(traces.shape[:2], dtype=np.float32)
                    integrated_legacy = True
                else:
                    continue

                telescope_count = len(traces)
                pixel_x = event.get("pixel_x", default_pixel_x)
                pixel_y = event.get("pixel_y", default_pixel_y)
                recorded_trigger = event.get("trigger", {})
                recorded_flags = (
                    recorded_trigger.get("camera_triggered")
                    if isinstance(recorded_trigger, dict) else None
                )

                for telescope_index, (trace, gain_flags) in enumerate(zip(traces, gains)):
                    data, charge, timing_bins = self._simulation_graph(
                        event, telescope_index, telescope_count, event_id,
                        trace, gain_flags, pixel_x, pixel_y, low_gain_factor,
                    )

                    if integrated_legacy:
                        is_triggered = True
                    elif recorded_flags is not None and telescope_index < len(recorded_flags):
                        is_triggered = bool(recorded_flags[telescope_index])
                    else:
                        is_triggered, _ = trigger.evaluate(
                            charge, timing_bins * bin_width_ns
                        )
                    if is_triggered:
                        data_list.append(data)

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        print(
            f"Processed {len(data_list)} telescope graphs. "
            f"Saving to {self.processed_paths[0]}..."
        )
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

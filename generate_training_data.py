"""Generate reproducible, trigger-selected AirCherenkov training events."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import sys
import tempfile

import numpy as np
import torch
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from sim.shower import ShowerSimulation
from sim.telescope import TelescopeArray
from sim.trigger import CameraTrigger
from sim.backend import get_device


MANIFEST_SCHEMA = "aircherenkov.training-manifest.v1"
EVENT_SCHEMA = "aircherenkov.training-event.v2"
THROW_LEDGER_SCHEMA = "aircherenkov.throw-ledger.v1"
METADATA_DIR_NAME = ".aircherenkov"
MANIFEST_FILE_NAME = "manifest.json"


def _derive_seed(root_seed, *components):
    """Derive a stable independent 63-bit seed for one event/stage."""
    material = ":".join([str(int(root_seed)), *(str(value) for value in components)])
    digest = hashlib.blake2b(material.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "little") & ((1 << 63) - 1)


def _seed_global_rngs(seed):
    """Seed APIs whose current simulation interfaces do not accept generators."""
    np.random.seed(int(seed) % (1 << 32))
    torch.manual_seed(int(seed))


def _atomic_json_dump(payload, destination):
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, destination)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def _atomic_torch_save(payload, destination):
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    os.close(descriptor)
    try:
        torch.save(payload, temporary_name)
        os.replace(temporary_name, destination)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def _configuration_hash(config):
    encoded = json.dumps(config, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _build_config(
    num_gammas,
    num_hadrons,
    batch_size,
    save_every,
    max_generations,
    impact_radius_m,
    camera_threshold_pe,
    camera_window_ns,
    trigger_min_pixels,
    array_min_telescopes,
    array_coincidence_window_ns,
    low_gain_factor,
    nsb_rate,
    requested_device,
    resolved_device,
    instrument=None,
):
    return {
        "targets": {"gamma": int(num_gammas), "proton": int(num_hadrons)},
        "batch_size": int(batch_size),
        "save_every_accepted_events": int(save_every),
        "primary_altitude_m": 20_000.0,
        "max_generations": int(max_generations),
        "energy_throw": {
            "distribution": "power_law",
            "spectral_index": -2.0,
            "minimum_gev": 100.0,
            "maximum_gev": 10_000.0,
        },
        "impact_throw": {
            "distribution": "uniform_area_disk",
            "radius_m": float(impact_radius_m),
        },
        "camera_trigger": {
            "threshold_pe": float(camera_threshold_pe),
            "window_ns": float(camera_window_ns),
            "min_pixels": int(trigger_min_pixels),
            "trace_bin_width_ns": 2.0,
            "low_gain_factor": float(low_gain_factor),
        },
        "array_trigger": {
            "type": "telescope_time_coincidence",
            "min_telescopes": int(array_min_telescopes),
            "coincidence_window_ns": float(array_coincidence_window_ns),
        },
        "ray_trace": {
            "n_time_bins": 16,
            "bin_width_ns": 2.0,
            "nsb_rate_per_pixel_window": float(nsb_rate),
            "pedestal_rms_per_integrated_window": 0.5,
            "saturation_limit_per_bin": 250.0,
            "low_gain_factor": float(low_gain_factor),
        },
        "device": {"requested": requested_device, "resolved": resolved_device},
        "instrument": instrument,
        "software": {
            "aircherenkov": "0.2.0",
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": torch.__version__,
        },
    }


def _instrument_metadata(array):
    telescopes = []
    for index, telescope in enumerate(array.telescopes):
        telescopes.append({
            "id": int(index),
            "position_m": [
                float(telescope.x_tel),
                float(telescope.y_tel),
                float(telescope.z_tel),
            ],
            "mirror_radius_m": float(telescope.mirror_radius),
            "mirror_reflectivity": float(telescope.mirror_reflectivity),
            "quantum_efficiency": float(telescope.quantum_efficiency),
            "camera": {
                "n_pixels": int(telescope.camera.n_pixels),
                "n_rings": int(telescope.camera.n_rings),
                "pixel_spacing_deg": float(telescope.camera.pixel_size),
            },
        })
    payload = {"name": "veritas-like", "telescopes": telescopes}
    payload["geometry_hash"] = _configuration_hash(payload)
    return payload


def _new_manifest(seed, config):
    return {
        "schema": MANIFEST_SCHEMA,
        "event_schema": EVENT_SCHEMA,
        "seed": int(seed),
        "config": config,
        "config_hash": _configuration_hash(config),
        "counts": {
            "attempted": {"gamma": 0, "proton": 0, "total": 0},
            "accepted": {"gamma": 0, "proton": 0, "total": 0},
            "rejected": {"gamma": 0, "proton": 0, "total": 0},
        },
        "next_event_id": 0,
        "next_chunk_id": 0,
        "pending_checkpoint": None,
        "throw_ledger_files": [],
        "complete": False,
    }


def _load_or_create_state(output_dir, seed, config):
    metadata_dir = output_dir / METADATA_DIR_NAME
    manifest_path = metadata_dir / MANIFEST_FILE_NAME

    if manifest_path.exists():
        with manifest_path.open("r", encoding="utf-8") as handle:
            manifest = json.load(handle)
        if manifest.get("schema") != MANIFEST_SCHEMA:
            raise RuntimeError(
                f"Unsupported training manifest schema: {manifest.get('schema')!r}"
            )
        if manifest.get("seed") != int(seed):
            raise ValueError(
                f"Cannot resume {output_dir}: seed {seed} does not match manifest "
                f"seed {manifest.get('seed')}"
            )
        if manifest.get("config_hash") != _configuration_hash(config):
            raise ValueError(
                f"Cannot resume {output_dir}: generation configuration differs "
                "from the existing manifest"
            )

        for chunk_id in range(int(manifest["next_chunk_id"])):
            chunk_path = output_dir / f"sim_batch_{chunk_id:05d}.pt"
            if not chunk_path.is_file():
                raise RuntimeError(f"Committed training chunk is missing: {chunk_path}")

        for ledger_name in manifest.get("throw_ledger_files", []):
            ledger_path = metadata_dir / ledger_name
            if not ledger_path.is_file():
                raise RuntimeError(f"Committed throw ledger is missing: {ledger_path}")

        pending = []
        pending_name = manifest.get("pending_checkpoint")
        if pending_name:
            pending_path = metadata_dir / pending_name
            if not pending_path.is_file():
                raise RuntimeError(f"Pending training checkpoint is missing: {pending_path}")
            pending = torch.load(pending_path, weights_only=False)
            if not isinstance(pending, list):
                raise RuntimeError(f"Invalid pending training checkpoint: {pending_path}")
        return manifest, pending, metadata_dir, manifest_path

    legacy_chunks = sorted(output_dir.glob("sim_batch_*.pt"))
    if legacy_chunks:
        raise RuntimeError(
            f"Found {len(legacy_chunks)} legacy chunk(s) in {output_dir} but no manifest. "
            "Their attempted-event and RNG state cannot be reconstructed safely; "
            "resume in a new output directory. The existing .pt chunks remain readable."
        )

    metadata_dir.mkdir(parents=True, exist_ok=True)
    manifest = _new_manifest(seed, config)
    _atomic_json_dump(manifest, manifest_path)
    return manifest, [], metadata_dir, manifest_path


def _checkpoint_state(manifest, pending_events, metadata_dir, manifest_path):
    previous_name = manifest.get("pending_checkpoint")
    if pending_events:
        pending_name = (
            f"pending_event_{int(manifest['next_event_id']):012d}_"
            f"chunk_{int(manifest['next_chunk_id']):06d}.pt"
        )
        _atomic_torch_save(pending_events, metadata_dir / pending_name)
    else:
        pending_name = None

    manifest["pending_checkpoint"] = pending_name
    _atomic_json_dump(manifest, manifest_path)

    if previous_name and previous_name != pending_name:
        previous_path = metadata_dir / previous_name
        try:
            previous_path.unlink()
        except FileNotFoundError:
            pass


def _flush_chunks(events, manifest, output_dir, save_every, flush_partial=False):
    while len(events) >= save_every or (flush_partial and events):
        count = min(save_every, len(events))
        chunk = events[:count]
        del events[:count]
        chunk_id = int(manifest["next_chunk_id"])
        destination = output_dir / f"sim_batch_{chunk_id:05d}.pt"
        _atomic_torch_save(chunk, destination)
        manifest["next_chunk_id"] = chunk_id + 1


def _commit_throw_ledger(records, batch_start, batch_end, manifest, metadata_dir):
    """Atomically retain every thrown trial, including trigger rejections."""
    ledger_name = f"throws_{batch_start:012d}_{batch_end - 1:012d}.json"
    _atomic_json_dump(
        {
            "schema": THROW_LEDGER_SCHEMA,
            "event_start": int(batch_start),
            "event_end_exclusive": int(batch_end),
            "events": records,
        },
        metadata_dir / ledger_name,
    )
    ledgers = manifest.setdefault("throw_ledger_files", [])
    if ledger_name not in ledgers:
        ledgers.append(ledger_name)


def _primary_schedule(num_gammas, num_hadrons, seed):
    schedule = np.concatenate(
        [
            np.ones(int(num_gammas), dtype=np.int8),
            np.zeros(int(num_hadrons), dtype=np.int8),
        ]
    )
    rng = np.random.default_rng(_derive_seed(seed, "primary-schedule"))
    rng.shuffle(schedule)
    return schedule


def _sample_throw(seed, event_id, primary_type, config):
    throw_seed = _derive_seed(seed, "throw", event_id)
    rng = np.random.default_rng(throw_seed)

    energy_config = config["energy_throw"]
    energy_min = energy_config["minimum_gev"]
    energy_max = energy_config["maximum_gev"]
    inverse_energy = (1.0 / energy_max) + rng.random() * (
        (1.0 / energy_min) - (1.0 / energy_max)
    )
    energy_gev = 1.0 / inverse_energy

    impact_radius = config["impact_throw"]["radius_m"]
    radius = math.sqrt(rng.random()) * impact_radius
    azimuth = rng.random() * 2.0 * math.pi
    impact_x_m = radius * math.cos(azimuth)
    impact_y_m = radius * math.sin(azimuth)

    normalization = 1.0 / ((1.0 / energy_min) - (1.0 / energy_max))
    energy_pdf = normalization / (energy_gev * energy_gev)
    impact_pdf = 1.0 / (math.pi * impact_radius * impact_radius)
    total_target = config["targets"]["gamma"] + config["targets"]["proton"]
    class_probability = config["targets"][primary_type] / total_target
    sampling_density = class_probability * energy_pdf * impact_pdf

    return {
        "event_id": int(event_id),
        "primary_type": primary_type,
        "energy_gev": float(energy_gev),
        "impact_x_m": float(impact_x_m),
        "impact_y_m": float(impact_y_m),
        "impact_radius_m": float(radius),
        "impact_azimuth_rad": float(azimuth),
        "throw_seed": int(throw_seed),
        "weights": {
            "class_sampling_probability": float(class_probability),
            "energy_pdf_per_gev": float(energy_pdf),
            "impact_pdf_per_m2": float(impact_pdf),
            "joint_sampling_density": float(sampling_density),
            "inverse_sampling_density": float(1.0 / sampling_density),
        },
    }


def _translate_photons(cherenkov_photons, impact_x_m, impact_y_m):
    """Translate the shower footprint without mutating telescope geometry."""
    translated = dict(cherenkov_photons)
    for key in ("x_emit", "x_ground"):
        translated[key] = np.asarray(cherenkov_photons[key]) + impact_x_m
    for key in ("y_emit", "y_ground"):
        translated[key] = np.asarray(cherenkov_photons[key]) + impact_y_m
    return translated


def _calibrated_trace(trace, gain_flags, low_gain_factor):
    trace = np.asarray(trace, dtype=np.float32)
    gain_flags = np.asarray(gain_flags)
    if trace.ndim != 2 or gain_flags.shape != (trace.shape[0],):
        raise ValueError("ray tracing returned incompatible trace and gain arrays")

    # The backend stores saturated pixels in the low-gain channel after dividing
    # them by this factor. Restore physical charge before applying PE thresholds.
    gain_scale = np.where(gain_flags > 0.5, low_gain_factor, 1.0).astype(np.float32)
    return trace * gain_scale[:, None]


def _camera_trigger_inputs(trace, gain_flags, bin_width_ns, low_gain_factor):
    corrected_trace = _calibrated_trace(trace, gain_flags, low_gain_factor)
    integrated_charge = np.sum(corrected_trace, axis=1)
    peak_time_ns = np.argmax(corrected_trace, axis=1).astype(np.float32) * bin_width_ns
    return integrated_charge, peak_time_ns


def _evaluate_array_coincidence(
    camera_triggered, camera_trigger_times_ns, min_telescopes, window_ns
):
    """Find the first camera-trigger group contained in the array time window."""
    trigger_times = sorted(
        float(trigger_time)
        for triggered, trigger_time in zip(camera_triggered, camera_trigger_times_ns)
        if triggered and trigger_time is not None and np.isfinite(trigger_time)
    )
    if len(trigger_times) < min_telescopes:
        return False, None, 0

    largest_group = 0
    right = 0
    for left, start_time in enumerate(trigger_times):
        right = max(right, left)
        while right + 1 < len(trigger_times) and trigger_times[right + 1] - start_time <= window_ns:
            right += 1
        group_size = right - left + 1
        largest_group = max(largest_group, group_size)
        if group_size >= min_telescopes:
            # The array trigger fires when the last telescope in the first valid
            # coincidence crosses its camera trigger.
            return True, float(trigger_times[left + min_telescopes - 1]), group_size
    return False, None, largest_group


def _resolve_device(requested_device):
    if requested_device not in {"auto", "cpu", "cuda"}:
        raise ValueError("device must be one of: auto, cpu, cuda")
    selected = get_device(requested_device)
    if selected is None:
        raise RuntimeError("PyTorch is required for shower generation")
    return selected.type


def _validate_parameters(
    num_gammas,
    num_hadrons,
    batch_size,
    save_every,
    seed,
    max_generations,
    impact_radius_m,
    camera_threshold_pe,
    camera_window_ns,
    trigger_min_pixels,
    array_min_telescopes,
    array_coincidence_window_ns,
    low_gain_factor,
    nsb_rate,
):
    for name, value in (("num_gammas", num_gammas), ("num_hadrons", num_hadrons)):
        if not isinstance(value, (int, np.integer)) or isinstance(value, bool) or value < 0:
            raise ValueError(f"{name} must be a non-negative integer")
    for name, value in (("batch_size", batch_size), ("save_every", save_every)):
        if not isinstance(value, (int, np.integer)) or isinstance(value, bool) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if num_gammas + num_hadrons <= 0:
        raise ValueError("at least one gamma or proton event must be requested")
    if not isinstance(seed, (int, np.integer)) or isinstance(seed, bool) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    if (
        not isinstance(max_generations, (int, np.integer))
        or isinstance(max_generations, bool)
        or max_generations < 0
    ):
        raise ValueError("max_generations must be a non-negative integer")
    if not np.isfinite(impact_radius_m) or impact_radius_m <= 0:
        raise ValueError("impact_radius_m must be finite and greater than zero")
    if not np.isfinite(camera_threshold_pe) or camera_threshold_pe < 0:
        raise ValueError("camera_threshold_pe must be finite and non-negative")
    if not np.isfinite(camera_window_ns) or camera_window_ns < 0:
        raise ValueError("camera_window_ns must be finite and non-negative")
    if (
        not isinstance(trigger_min_pixels, (int, np.integer))
        or isinstance(trigger_min_pixels, bool)
        or trigger_min_pixels != 3
    ):
        raise ValueError("trigger_min_pixels must be 3; other cluster sizes are unsupported")
    if (
        not isinstance(array_min_telescopes, (int, np.integer))
        or isinstance(array_min_telescopes, bool)
        or not 1 <= array_min_telescopes <= 4
    ):
        raise ValueError("array_min_telescopes must be between 1 and 4")
    if not np.isfinite(array_coincidence_window_ns) or array_coincidence_window_ns < 0:
        raise ValueError("array_coincidence_window_ns must be finite and non-negative")
    if not np.isfinite(low_gain_factor) or low_gain_factor <= 0:
        raise ValueError("low_gain_factor must be finite and greater than zero")
    if not np.isfinite(nsb_rate) or nsb_rate < 0:
        raise ValueError("nsb_rate must be finite and non-negative")


def generate_training_data(
    num_gammas=10_000,
    num_hadrons=10_000,
    batch_size=100,
    save_every=1_000,
    output_dir="data/train/raw",
    seed=0,
    max_generations=16,
    impact_radius_m=250.0,
    camera_threshold_pe=5.0,
    camera_window_ns=5.0,
    trigger_min_pixels=3,
    array_min_telescopes=2,
    array_coincidence_window_ns=50.0,
    low_gain_factor=10.0,
    nsb_rate=2.0,
    device="auto",
):
    """Generate trigger-selected events with manifest-backed resumability."""
    _validate_parameters(
        num_gammas,
        num_hadrons,
        batch_size,
        save_every,
        seed,
        max_generations,
        impact_radius_m,
        camera_threshold_pe,
        camera_window_ns,
        trigger_min_pixels,
        array_min_telescopes,
        array_coincidence_window_ns,
        low_gain_factor,
        nsb_rate,
    )

    resolved_device = _resolve_device(device)

    array = TelescopeArray.veritas_array(
        device=resolved_device,
        nsb_rate=nsb_rate,
        low_gain_factor=low_gain_factor,
        shower_start_altitude=20_000.0,
    )

    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    config = _build_config(
        num_gammas,
        num_hadrons,
        batch_size,
        save_every,
        max_generations,
        impact_radius_m,
        camera_threshold_pe,
        camera_window_ns,
        trigger_min_pixels,
        array_min_telescopes,
        array_coincidence_window_ns,
        low_gain_factor,
        nsb_rate,
        device,
        resolved_device,
        _instrument_metadata(array),
    )
    manifest, current_chunk, metadata_dir, manifest_path = _load_or_create_state(
        output_dir, seed, config
    )

    schedule = _primary_schedule(num_gammas, num_hadrons, seed)
    next_event_id = int(manifest["next_event_id"])
    if next_event_id > len(schedule):
        raise RuntimeError("manifest next_event_id exceeds the configured event target")
    expected_gamma_attempts = int(np.sum(schedule[:next_event_id]))
    expected_proton_attempts = next_event_id - expected_gamma_attempts
    attempted = manifest["counts"]["attempted"]
    if (
        attempted["gamma"] != expected_gamma_attempts
        or attempted["proton"] != expected_proton_attempts
        or attempted["total"] != next_event_id
    ):
        raise RuntimeError("manifest attempted-event counters do not match its event position")

    events_remaining = len(schedule) - next_event_id
    if events_remaining == 0:
        manifest["complete"] = True
        _flush_chunks(current_chunk, manifest, output_dir, save_every, flush_partial=True)
        _checkpoint_state(manifest, current_chunk, metadata_dir, manifest_path)
        print("All requested events have already been generated!")
        return

    if array_min_telescopes > len(array.telescopes):
        raise ValueError(
            f"array_min_telescopes={array_min_telescopes} exceeds the "
            f"{len(array.telescopes)}-telescope array"
        )
    camera_triggers = [
        CameraTrigger(
            telescope.camera.pixel_x,
            telescope.camera.pixel_y,
            threshold_pe=camera_threshold_pe,
            window_ns=camera_window_ns,
            min_pixels=trigger_min_pixels,
            pixel_size=telescope.camera.pixel_size,
        )
        for telescope in array.telescopes
    ]

    remaining_gamma = num_gammas - attempted["gamma"]
    remaining_proton = num_hadrons - attempted["proton"]
    print(
        f"Generating {remaining_gamma} gammas and {remaining_proton} protons in "
        f"batches of {batch_size}; accepted chunks contain up to {save_every} events."
    )
    print(f"Output: {output_dir}")
    print(f"Seed: {seed}; resume event: {next_event_id}")

    new_accepted = 0
    pbar = tqdm(total=events_remaining, desc="MC Generation")
    try:
        while next_event_id < len(schedule):
            batch_start = next_event_id
            batch_end = min(batch_start + batch_size, len(schedule))
            batch_throws = []
            for event_id in range(batch_start, batch_end):
                primary_type = "gamma" if schedule[event_id] == 1 else "proton"
                batch_throws.append(_sample_throw(seed, event_id, primary_type, config))
            batch_ledger = []

            primary_types = [throw["primary_type"] for throw in batch_throws]
            energies = [throw["energy_gev"] for throw in batch_throws]
            shower_seed = _derive_seed(seed, "shower-batch", batch_start, batch_end)
            _seed_global_rngs(shower_seed)
            simulation = ShowerSimulation(
                primary_types=primary_types,
                energies=energies,
                z_starts=[config["primary_altitude_m"]] * len(batch_throws),
                device=resolved_device,
                seed=shower_seed,
            )
            simulation.run(max_generations=max_generations, verbose=False)

            for local_index, thrown in enumerate(batch_throws):
                primary_type = thrown["primary_type"]
                event_id = thrown["event_id"]
                manifest["counts"]["attempted"][primary_type] += 1
                manifest["counts"]["attempted"]["total"] += 1
                manifest["next_event_id"] = event_id + 1

                photons = simulation.cherenkov_photons_by_event[local_index]
                ledger_record = {
                    "event_id": int(event_id),
                    "primary_type": primary_type,
                    "energy_gev": float(thrown["energy_gev"]),
                    "impact_x_m": float(thrown["impact_x_m"]),
                    "impact_y_m": float(thrown["impact_y_m"]),
                    "throw_seed": int(thrown["throw_seed"]),
                    "weights": thrown["weights"],
                }
                if len(photons.get("x_ground", [])) == 0:
                    manifest["counts"]["rejected"][primary_type] += 1
                    manifest["counts"]["rejected"]["total"] += 1
                    ledger_record.update({
                        "accepted": False,
                        "rejection_reason": "no_ground_photon_packets",
                        "trigger": None,
                    })
                    batch_ledger.append(ledger_record)
                    continue

                translated_photons = _translate_photons(
                    photons, thrown["impact_x_m"], thrown["impact_y_m"]
                )
                traces = []
                gains = []
                camera_triggered = []
                camera_trigger_times = []
                camera_total_charge = []

                trigger_config = config["camera_trigger"]
                optics_seed = _derive_seed(seed, "optics-array", event_id)
                optics_generator = torch.Generator(device=torch.device(resolved_device))
                optics_generator.manual_seed(optics_seed)
                ray_trace_outputs = array.ray_trace(
                    translated_photons,
                    nsb_rate=nsb_rate,
                    device=resolved_device,
                    generator=optics_generator,
                    shower_start_altitude=config["primary_altitude_m"],
                )
                for (trace, gain), camera_trigger in zip(
                    ray_trace_outputs, camera_triggers
                ):
                    charge, timing = _camera_trigger_inputs(
                        trace,
                        gain,
                        trigger_config["trace_bin_width_ns"],
                        trigger_config["low_gain_factor"],
                    )
                    calibrated_trace = _calibrated_trace(
                        trace, gain, trigger_config["low_gain_factor"]
                    )
                    triggered, trigger_time = camera_trigger.evaluate_traces(
                        calibrated_trace,
                        trigger_config["trace_bin_width_ns"],
                    )
                    traces.append(np.asarray(trace, dtype=np.float32))
                    gains.append(np.asarray(gain, dtype=np.float32))
                    camera_triggered.append(bool(triggered))
                    camera_trigger_times.append(
                        None if trigger_time is None else float(trigger_time)
                    )
                    camera_total_charge.append(float(np.sum(charge)))

                array_triggered, array_trigger_time, coincident_telescopes = (
                    _evaluate_array_coincidence(
                        camera_triggered,
                        camera_trigger_times,
                        array_min_telescopes,
                        array_coincidence_window_ns,
                    )
                )
                trigger_metadata = {
                    "array_triggered": bool(array_triggered),
                    "array_min_telescopes": int(array_min_telescopes),
                    "array_coincidence_window_ns": float(array_coincidence_window_ns),
                    "array_trigger_time_ns": array_trigger_time,
                    "triggered_telescopes": int(sum(camera_triggered)),
                    "coincident_telescopes": int(coincident_telescopes),
                    "camera_triggered": camera_triggered,
                    "camera_trigger_times_ns": camera_trigger_times,
                    "camera_total_charge_pe": camera_total_charge,
                }
                if not array_triggered:
                    manifest["counts"]["rejected"][primary_type] += 1
                    manifest["counts"]["rejected"]["total"] += 1
                    ledger_record.update({
                        "accepted": False,
                        "rejection_reason": "array_trigger",
                        "trigger": trigger_metadata,
                    })
                    batch_ledger.append(ledger_record)
                    continue

                label = 1 if primary_type == "gamma" else 0
                event = {
                    # Existing keys are retained for the legacy CherenkovDataset reader.
                    "fadc_traces": np.asarray(traces, dtype=np.float32),
                    "gain_flags": np.asarray(gains, dtype=np.float32),
                    "energy": thrown["energy_gev"],
                    "label": label,
                    "impact_x": thrown["impact_x_m"],
                    "impact_y": thrown["impact_y_m"],
                    "telescope_positions_m": np.asarray(
                        [
                            [telescope.x_tel, telescope.y_tel, telescope.z_tel]
                            for telescope in array.telescopes
                        ],
                        dtype=np.float32,
                    ),
                    "telescope_multiplicity": len(array.telescopes),
                    # Versioned metadata makes the accepted sample auditable.
                    "schema": EVENT_SCHEMA,
                    "event_id": event_id,
                    "primary_type": primary_type,
                    "thrown": {
                        key: value
                        for key, value in thrown.items()
                        if key not in {"event_id", "primary_type", "weights"}
                    },
                    "weights": thrown["weights"],
                    "sampling_weight": thrown["weights"]["inverse_sampling_density"],
                    "trigger": trigger_metadata,
                    "rng": {
                        "root_seed": int(seed),
                        "throw_seed": thrown["throw_seed"],
                        "shower_batch_seed": int(shower_seed),
                        "optics_array_seed": int(optics_seed),
                    },
                    "config": {
                        "manifest_schema": MANIFEST_SCHEMA,
                        "config_hash": manifest["config_hash"],
                    },
                }
                current_chunk.append(event)
                manifest["counts"]["accepted"][primary_type] += 1
                manifest["counts"]["accepted"]["total"] += 1
                new_accepted += 1
                ledger_record.update({
                    "accepted": True,
                    "rejection_reason": None,
                    "trigger": trigger_metadata,
                })
                batch_ledger.append(ledger_record)

            next_event_id = batch_end
            _commit_throw_ledger(
                batch_ledger, batch_start, batch_end, manifest, metadata_dir
            )
            _flush_chunks(current_chunk, manifest, output_dir, save_every)
            _checkpoint_state(manifest, current_chunk, metadata_dir, manifest_path)
            pbar.update(batch_end - batch_start)
    finally:
        pbar.close()

    _flush_chunks(current_chunk, manifest, output_dir, save_every, flush_partial=True)
    manifest["complete"] = True
    _checkpoint_state(manifest, current_chunk, metadata_dir, manifest_path)
    accepted_total = manifest["counts"]["accepted"]["total"]
    attempted_total = manifest["counts"]["attempted"]["total"]
    print(
        f"Generation complete: {new_accepted} new events accepted; "
        f"{accepted_total}/{attempted_total} accepted overall."
    )


def build_parser():
    parser = argparse.ArgumentParser(description="Generate reproducible Cherenkov MC data")
    parser.add_argument(
        "--num-gammas", "--num_gammas", dest="num_gammas", type=int, default=10_000,
        help="Total number of thrown gamma events (default: 10000).",
    )
    parser.add_argument(
        "--num-hadrons", "--num_hadrons", dest="num_hadrons", type=int, default=10_000,
        help="Total number of thrown proton events (default: 10000).",
    )
    parser.add_argument(
        "--batch-size", "--batch_size", dest="batch_size", type=int, default=100,
        help="Shower simulation batch size (default: 100).",
    )
    parser.add_argument(
        "--save-every", "--save_every", dest="save_every", type=int, default=1_000,
        help="Maximum accepted events per .pt chunk (default: 1000).",
    )
    parser.add_argument(
        "--output-dir", "--output_dir", dest="output_dir", type=Path,
        default=Path("data/train/raw"),
        help="Raw event output directory (default: data/train/raw).",
    )
    parser.add_argument("--seed", type=int, default=0, help="Root RNG seed (default: 0).")
    parser.add_argument("--max-generations", type=int, default=16)
    parser.add_argument("--impact-radius-m", type=float, default=250.0)
    parser.add_argument("--camera-threshold-pe", type=float, default=5.0)
    parser.add_argument("--camera-window-ns", type=float, default=5.0)
    parser.add_argument("--trigger-min-pixels", type=int, default=3)
    parser.add_argument("--array-min-telescopes", type=int, default=2)
    parser.add_argument("--array-coincidence-window-ns", type=float, default=50.0)
    parser.add_argument("--low-gain-factor", type=float, default=10.0)
    parser.add_argument("--nsb-rate", type=float, default=2.0)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    return generate_training_data(
        num_gammas=args.num_gammas,
        num_hadrons=args.num_hadrons,
        batch_size=args.batch_size,
        save_every=args.save_every,
        output_dir=args.output_dir,
        seed=args.seed,
        max_generations=args.max_generations,
        impact_radius_m=args.impact_radius_m,
        camera_threshold_pe=args.camera_threshold_pe,
        camera_window_ns=args.camera_window_ns,
        trigger_min_pixels=args.trigger_min_pixels,
        array_min_telescopes=args.array_min_telescopes,
        array_coincidence_window_ns=args.array_coincidence_window_ns,
        low_gain_factor=args.low_gain_factor,
        nsb_rate=args.nsb_rate,
        device=args.device,
    )


if __name__ == "__main__":
    main()

"""Live simulation progress monitor (pure-Python scanner + PySide6 dashboard widget)."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import glob
import json
import os
import time
from typing import Any, Dict, List, Optional, Tuple

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import numpy as np
import torch

try:
    from PySide6.QtCore import Qt, QThread, QTimer, Signal, Slot
    from PySide6.QtWidgets import (
        QCheckBox,
        QFileDialog,
        QGroupBox,
        QHBoxLayout,
        QLabel,
        QLineEdit,
        QProgressBar,
        QPushButton,
        QSpinBox,
        QVBoxLayout,
        QWidget,
    )
    HAS_PYSIDE6 = True
except ImportError:  # pragma: no cover
    HAS_PYSIDE6 = False


@dataclass
class TimeseriesPoint:
    """Single point on the cumulative-events vs. active-wall-clock timeseries."""

    active_elapsed_s: float
    total_events: int
    gammas: int
    hadrons: int
    chunk_idx: int
    session_index: int


@dataclass
class SessionProgress:
    """Progress summary for a single simulation session."""

    session_index: int
    sim_version: str
    status: str
    chunk_start: int
    chunk_end: int
    gammas_saved: int
    hadrons_saved: int
    target_gammas: Optional[int]
    target_hadrons: Optional[int]
    gammas_thrown: int
    hadrons_thrown: int
    active_elapsed_s: float
    started_at: Optional[float]
    updated_at: Optional[float]
    git_commit: Optional[str] = None


@dataclass
class SimulationProgressSnapshot:
    """Complete snapshot of current run, active cohort, and timeseries."""

    directory: str
    status: str
    current_session: Optional[SessionProgress]
    cohort_sessions: List[SessionProgress] = field(default_factory=list)
    cohort_label: str = ""
    cohort_gammas_saved: int = 0
    cohort_hadrons_saved: int = 0
    cohort_target_gammas: Optional[int] = None
    cohort_target_hadrons: Optional[int] = None
    total_dataset_events: int = 0
    total_chunks: int = 0
    active_elapsed_s: float = 0.0
    paused_time_excluded_s: float = 0.0
    rate_events_per_hr: float = 0.0
    eta_seconds: Optional[float] = None
    timeseries: List[TimeseriesPoint] = field(default_factory=list)
    pause_markers_s: List[float] = field(default_factory=list)
    sample_energies: Optional[np.ndarray] = None
    sample_labels: Optional[np.ndarray] = None
    sample_impact_x: Optional[np.ndarray] = None
    sample_impact_y: Optional[np.ndarray] = None


def resolve_sim_raw_dir(path: str) -> str:
    """Resolve either a raw/ simulation directory or its parent dataset directory."""
    clean = os.path.normpath(str(path or "").strip())
    if not clean:
        return ""
    if os.path.isdir(clean):
        has_here = (
            os.path.isfile(os.path.join(clean, "sim_manifest.json"))
            or os.path.isfile(os.path.join(clean, "sim_state.json"))
            or bool(glob.glob(os.path.join(clean, "sim_batch_*.pt")))
        )
        if has_here:
            return clean
        raw_sub = os.path.join(clean, "raw")
        if os.path.isdir(raw_sub):
            has_raw = (
                os.path.isfile(os.path.join(raw_sub, "sim_manifest.json"))
                or os.path.isfile(os.path.join(raw_sub, "sim_state.json"))
                or bool(glob.glob(os.path.join(raw_sub, "sim_batch_*.pt")))
            )
            if has_raw:
                return raw_sub
    return clean


def _parse_iso_timestamp(ts_str: Optional[str]) -> Optional[float]:
    if not ts_str or not isinstance(ts_str, str):
        return None
    try:
        s = ts_str.strip()
        if s.endswith("Z"):
            s = s[:-1] + "+00:00"
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return float(dt.timestamp())
    except Exception:
        return None


def _chunk_index_from_path(path: str) -> Optional[int]:
    base = os.path.basename(path)
    if not (base.startswith("sim_batch_") and base.endswith(".pt")):
        return None
    try:
        return int(base[len("sim_batch_") : -len(".pt")])
    except ValueError:
        return None


def _extract_chunk_summary(
    path: str, chunk_cache: Optional[Dict[Tuple[str, int, int], Tuple[Any, ...]]] = None
) -> Tuple[int, int, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    st = os.stat(path)
    key = (os.path.abspath(path), int(st.st_mtime_ns), int(st.st_size))
    if chunk_cache is not None and key in chunk_cache:
        return chunk_cache[key]
    chunk = torch.load(path, map_location="cpu", weights_only=False)
    gammas = 0
    hadrons = 0
    energies: List[float] = []
    labels: List[int] = []
    ixs: List[float] = []
    iys: List[float] = []
    for ev in chunk:
        if not isinstance(ev, dict):
            continue
        lbl = 1 if int(ev.get("label", 1)) == 1 else 0
        if lbl == 1:
            gammas += 1
        else:
            hadrons += 1
        energies.append(float(ev.get("energy", 0.0)))
        labels.append(lbl)
        ixs.append(float(ev.get("impact_x", 0.0)))
        iys.append(float(ev.get("impact_y", 0.0)))
    result = (
        gammas,
        hadrons,
        np.asarray(energies, dtype=np.float32),
        np.asarray(labels, dtype=np.int64),
        np.asarray(ixs, dtype=np.float32),
        np.asarray(iys, dtype=np.float32),
    )
    if chunk_cache is not None:
        chunk_cache[key] = result
    return result


def _count_chunk_events(
    path: str, chunk_cache: Optional[Dict[Tuple[str, int, int], Tuple[Any, ...]]] = None
) -> Tuple[int, int]:
    g, h, _, _, _, _ = _extract_chunk_summary(path, chunk_cache)
    return g, h


def _params_cohort_key(session: Dict[str, Any]) -> Tuple[Any, ...]:
    sim_ver = str(session.get("sim_version") or "")
    params = session.get("params") or {}
    if not isinstance(params, dict) or not params:
        return (sim_ver, None)
    keys = (
        "e_min",
        "e_max",
        "spectral_index",
        "hadron_e_min",
        "hadron_e_max",
        "hadron_spectral_index",
        "impact_radius",
        "zenith_deg",
        "azimuth_deg",
    )
    vals = tuple(round(float(params[k]), 6) if params.get(k) is not None else None for k in keys)
    return (sim_ver, vals)


def _format_duration(seconds: Optional[float]) -> str:
    if seconds is None or not np.isfinite(seconds) or seconds < 0:
        return "--"
    total = int(round(seconds))
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    if h > 0:
        return f"{h}h {m:02d}m {s:02d}s"
    if m > 0:
        return f"{m}m {s:02d}s"
    return f"{s}s"


def scan_simulation_progress(
    directory: str,
    *,
    chunk_cache: Optional[Dict[Tuple[str, int, int], Tuple[int, int]]] = None,
    target_gammas_override: Optional[int] = None,
    target_hadrons_override: Optional[int] = None,
    cohort_target_gammas_override: Optional[int] = None,
    cohort_target_hadrons_override: Optional[int] = None,
    now_ts: Optional[float] = None,
    auto_tag: bool = True,
) -> SimulationProgressSnapshot:
    """Scan a simulation raw directory for session, cohort, and timeseries progress."""
    raw_dir = resolve_sim_raw_dir(directory)
    now = float(time.time() if now_ts is None else now_ts)
    if not raw_dir or not os.path.isdir(raw_dir):
        return SimulationProgressSnapshot(directory=raw_dir or str(directory), status="IDLE")

    chunk_files: Dict[int, str] = {}
    for p in sorted(glob.glob(os.path.join(raw_dir, "sim_batch_*.pt"))):
        idx = _chunk_index_from_path(p)
        if idx is not None:
            chunk_files[idx] = p

    manifest_path = os.path.join(raw_dir, "sim_manifest.json")
    if auto_tag and chunk_files and not os.path.isfile(manifest_path):
        try:
            from sim.provenance import ensure_dataset_tagged

            ensure_dataset_tagged(raw_dir)
        except Exception:
            pass

    manifest: Optional[Dict[str, Any]] = None
    if os.path.isfile(manifest_path):
        try:
            with open(manifest_path, "r", encoding="utf-8") as mf:
                manifest = json.load(mf)
        except Exception:
            manifest = None

    sessions_raw = (manifest or {}).get("sessions") or []
    if not sessions_raw:
        # Manifest-less fallback: treat all chunks as one synthetic session
        if not chunk_files:
            return SimulationProgressSnapshot(directory=raw_dir, status="IDLE")
        sorted_idxs = sorted(chunk_files.keys())
        mtimes = [os.path.getmtime(chunk_files[i]) for i in sorted_idxs]
        t0 = min(mtimes)
        cum_g = 0
        cum_h = 0
        ts_points: List[TimeseriesPoint] = [
            TimeseriesPoint(0.0, 0, 0, 0, sorted_idxs[0], 0)
        ]
        sample_e_chunks: List[np.ndarray] = []
        sample_lbl_chunks: List[np.ndarray] = []
        sample_ix_chunks: List[np.ndarray] = []
        sample_iy_chunks: List[np.ndarray] = []
        for idx, mt in zip(sorted_idxs, mtimes):
            g, h, e_arr, lbl_arr, ix_arr, iy_arr = _extract_chunk_summary(chunk_files[idx], chunk_cache)
            cum_g += g
            cum_h += h
            sample_e_chunks.append(e_arr)
            sample_lbl_chunks.append(lbl_arr)
            sample_ix_chunks.append(ix_arr)
            sample_iy_chunks.append(iy_arr)
            ts_points.append(
                TimeseriesPoint(max(0.0, mt - t0), cum_g + cum_h, cum_g, cum_h, idx, 0)
            )
        last_mt = max(mtimes)
        active_s = max(0.0, (now if (now - last_mt) < 1200.0 else last_mt) - t0)
        status = "RUNNING" if (now - last_mt) < 1200.0 else "IDLE"
        rate_hr = ((cum_g + cum_h) / active_s * 3600.0) if active_s > 1.0 else 0.0
        tgt_g = target_gammas_override if (target_gammas_override and target_gammas_override > 0) else None
        tgt_h = target_hadrons_override if (target_hadrons_override and target_hadrons_override > 0) else None
        sess = SessionProgress(
            session_index=0,
            sim_version="Unversioned",
            status=status.lower(),
            chunk_start=sorted_idxs[0],
            chunk_end=sorted_idxs[-1],
            gammas_saved=cum_g,
            hadrons_saved=cum_h,
            target_gammas=tgt_g,
            target_hadrons=tgt_h,
            gammas_thrown=0,
            hadrons_thrown=0,
            active_elapsed_s=active_s,
            started_at=t0,
            updated_at=last_mt,
        )
        eta_s = None
        if tgt_g is not None or tgt_h is not None:
            tot_tgt = (tgt_g or 0) + (tgt_h or 0)
            rem = max(0, tot_tgt - (cum_g + cum_h))
            if rate_hr > 0 and rem > 0:
                eta_s = rem / rate_hr * 3600.0
        return SimulationProgressSnapshot(
            directory=raw_dir,
            status=status,
            current_session=sess,
            cohort_sessions=[sess],
            cohort_label="Detected Chunks",
            cohort_gammas_saved=cum_g,
            cohort_hadrons_saved=cum_h,
            cohort_target_gammas=tgt_g,
            cohort_target_hadrons=tgt_h,
            total_dataset_events=cum_g + cum_h,
            total_chunks=len(chunk_files),
            active_elapsed_s=active_s,
            paused_time_excluded_s=0.0,
            rate_events_per_hr=rate_hr,
            eta_seconds=eta_s,
            timeseries=ts_points,
            pause_markers_s=[],
            sample_energies=np.concatenate(sample_e_chunks) if sample_e_chunks else None,
            sample_labels=np.concatenate(sample_lbl_chunks) if sample_lbl_chunks else None,
            sample_impact_x=np.concatenate(sample_ix_chunks) if sample_ix_chunks else None,
            sample_impact_y=np.concatenate(sample_iy_chunks) if sample_iy_chunks else None,
        )

    # Group sessions into cohorts matching the latest session's version & physics params
    last_raw = sessions_raw[-1]
    target_key = _params_cohort_key(last_raw)
    cohort_raw = [s for s in sessions_raw if _params_cohort_key(s) == target_key]
    non_cohort_events = 0
    for s in sessions_raw:
        if _params_cohort_key(s) != target_key:
            non_cohort_events += int(s.get("gammas_saved_session", 0)) + int(
                s.get("hadrons_saved_session", 0)
            )

    prior_active_s = 0.0
    cum_g = 0
    cum_h = 0
    ts_points = []
    pause_markers: List[float] = []
    cohort_sessions: List[SessionProgress] = []
    first_t0: Optional[float] = None
    last_wall_ts: Optional[float] = None
    cohort_chunk_idxs: List[int] = []

    for i, s in enumerate(cohort_raw):
        s_idx = int(s.get("session_index", i))
        c_start = int(s.get("chunk_start", 0))
        c_end = int(s.get("chunk_end", c_start - 1))
        is_latest = (i == len(cohort_raw) - 1)

        if is_latest and chunk_files:
            max_disk_idx = max(chunk_files.keys())
            if max_disk_idx > c_end:
                c_end = max_disk_idx

        s_chunk_idxs = [
            idx for idx in sorted(chunk_files.keys()) if c_start <= idx <= max(c_end, c_start - 1)
        ]
        cohort_chunk_idxs.extend(s_chunk_idxs)

        t0_s = _parse_iso_timestamp(s.get("started_at"))
        t_upd = _parse_iso_timestamp(s.get("updated_at"))
        history = s.get("chunk_history")

        if t0_s is None:
            if s_chunk_idxs:
                t0_s = os.path.getmtime(chunk_files[s_chunk_idxs[0]])
            elif t_upd is not None:
                t0_s = t_upd
            else:
                t0_s = now

        if first_t0 is None:
            first_t0 = t0_s

        if i > 0:
            pause_markers.append(prior_active_s)
        ts_points.append(
            TimeseriesPoint(
                active_elapsed_s=prior_active_s,
                total_events=cum_g + cum_h,
                gammas=cum_g,
                hadrons=cum_h,
                chunk_idx=c_start,
                session_index=s_idx,
            )
        )

        sess_g = 0
        sess_h = 0
        last_chunk_ts = t0_s

        if isinstance(history, list) and history:
            for item in history:
                c_i = int(item.get("chunk_idx", c_start))
                g_i = int(item.get("gammas_saved", 0))
                h_i = int(item.get("hadrons_saved", 0))
                ts_i = _parse_iso_timestamp(item.get("timestamp_utc"))
                if ts_i is None and c_i in chunk_files:
                    ts_i = os.path.getmtime(chunk_files[c_i])
                if ts_i is None:
                    ts_i = last_chunk_ts
                last_chunk_ts = max(last_chunk_ts, ts_i)
                sess_g += g_i
                sess_h += h_i
                cum_g += g_i
                cum_h += h_i
                ts_points.append(
                    TimeseriesPoint(
                        active_elapsed_s=prior_active_s + max(0.0, ts_i - t0_s),
                        total_events=cum_g + cum_h,
                        gammas=cum_g,
                        hadrons=cum_h,
                        chunk_idx=c_i,
                        session_index=s_idx,
                    )
                )
        elif s_chunk_idxs:
            for c_i in s_chunk_idxs:
                p = chunk_files[c_i]
                ts_i = os.path.getmtime(p)
                g_i, h_i = _count_chunk_events(p, chunk_cache)
                last_chunk_ts = max(last_chunk_ts, ts_i)
                sess_g += g_i
                sess_h += h_i
                cum_g += g_i
                cum_h += h_i
                ts_points.append(
                    TimeseriesPoint(
                        active_elapsed_s=prior_active_s + max(0.0, ts_i - t0_s),
                        total_events=cum_g + cum_h,
                        gammas=cum_g,
                        hadrons=cum_h,
                        chunk_idx=c_i,
                        session_index=s_idx,
                    )
                )
        else:
            sess_g = int(s.get("gammas_saved_session", 0))
            sess_h = int(s.get("hadrons_saved_session", 0))
            cum_g += sess_g
            cum_h += sess_h
            end_ts = t_upd if t_upd is not None else t0_s
            last_chunk_ts = max(last_chunk_ts, end_ts)
            ts_points.append(
                TimeseriesPoint(
                    active_elapsed_s=prior_active_s + max(0.0, end_ts - t0_s),
                    total_events=cum_g + cum_h,
                    gammas=cum_g,
                    hadrons=cum_h,
                    chunk_idx=c_end,
                    session_index=s_idx,
                )
            )

        s_status = str(s.get("status", "unknown")).lower()
        t_end_s = max(t0_s, last_chunk_ts, t_upd or t0_s)
        if is_latest and s_status == "in_progress" and (now - t_end_s) < 1800.0:
            t_end_s = max(t_end_s, now)
        sess_active_s = max(0.0, t_end_s - t0_s)
        prior_active_s += sess_active_s
        last_wall_ts = t_end_s

        tgt_g_s = s.get("target_gammas_session")
        tgt_h_s = s.get("target_hadrons_session")
        if is_latest:
            if target_gammas_override and target_gammas_override > 0:
                tgt_g_s = int(target_gammas_override)
            if target_hadrons_override and target_hadrons_override > 0:
                tgt_h_s = int(target_hadrons_override)

        from sim.provenance import clean_str

        cohort_sessions.append(
            SessionProgress(
                session_index=s_idx,
                sim_version=clean_str(s.get("sim_version")) or "Unversioned",
                status=s_status,
                chunk_start=c_start,
                chunk_end=c_end,
                gammas_saved=sess_g,
                hadrons_saved=sess_h,
                target_gammas=int(tgt_g_s) if tgt_g_s is not None else None,
                target_hadrons=int(tgt_h_s) if tgt_h_s is not None else None,
                gammas_thrown=int(s.get("gammas_thrown_session", 0)),
                hadrons_thrown=int(s.get("hadrons_thrown_session", 0)),
                active_elapsed_s=sess_active_s,
                started_at=t0_s,
                updated_at=t_end_s,
                git_commit=clean_str(s.get("git_commit")),
            )
        )

    current_sess = cohort_sessions[-1] if cohort_sessions else None
    total_wall_span_s = max(0.0, (last_wall_ts - first_t0) if (first_t0 and last_wall_ts) else 0.0)
    paused_excluded_s = max(0.0, total_wall_span_s - prior_active_s)

    if current_sess is None:
        overall_status = "IDLE"
    elif current_sess.status == "in_progress":
        overall_status = (
            "RUNNING"
            if (current_sess.updated_at is not None and (now - current_sess.updated_at) < 1800.0)
            else "PAUSED"
        )
    elif current_sess.status in ("paused", "interrupted"):
        overall_status = "PAUSED"
    elif current_sess.status == "completed":
        overall_status = "COMPLETED"
    else:
        overall_status = current_sess.status.upper()

    rate_hr = 0.0
    if current_sess and current_sess.active_elapsed_s > 1.0 and (current_sess.gammas_saved + current_sess.hadrons_saved) > 0:
        rate_hr = (current_sess.gammas_saved + current_sess.hadrons_saved) / current_sess.active_elapsed_s * 3600.0
    elif prior_active_s > 1.0:
        rate_hr = (cum_g + cum_h) / prior_active_s * 3600.0

    # Cohort targets: explicit override, or sum of session targets if all sessions specify them
    if cohort_target_gammas_override and cohort_target_gammas_override > 0:
        coh_tgt_g: Optional[int] = int(cohort_target_gammas_override)
    elif cohort_sessions and all(cs.target_gammas is not None for cs in cohort_sessions):
        coh_tgt_g = sum(cs.target_gammas or 0 for cs in cohort_sessions)
    else:
        coh_tgt_g = None

    if cohort_target_hadrons_override and cohort_target_hadrons_override > 0:
        coh_tgt_h: Optional[int] = int(cohort_target_hadrons_override)
    elif cohort_sessions and all(cs.target_hadrons is not None for cs in cohort_sessions):
        coh_tgt_h = sum(cs.target_hadrons or 0 for cs in cohort_sessions)
    else:
        coh_tgt_h = None

    eta_s = None
    if current_sess and (current_sess.target_gammas is not None or current_sess.target_hadrons is not None):
        sess_tot_tgt = (current_sess.target_gammas or 0) + (current_sess.target_hadrons or 0)
        sess_done = current_sess.gammas_saved + current_sess.hadrons_saved
        if rate_hr > 0 and sess_tot_tgt > sess_done:
            eta_s = (sess_tot_tgt - sess_done) / rate_hr * 3600.0
    elif (coh_tgt_g is not None or coh_tgt_h is not None) and rate_hr > 0:
        coh_tot_tgt = (coh_tgt_g or 0) + (coh_tgt_h or 0)
        if coh_tot_tgt > (cum_g + cum_h):
            eta_s = (coh_tot_tgt - (cum_g + cum_h)) / rate_hr * 3600.0

    from sim.provenance import clean_float, clean_str

    params = last_raw.get("params") or {}
    e_lo = clean_float(params.get("e_min"))
    e_hi = clean_float(params.get("e_max"))
    zen = clean_float(params.get("zenith_deg"))
    ver = clean_str(last_raw.get("sim_version")) or "Unversioned"
    parts = [ver]
    if e_lo is not None and e_hi is not None:
        parts.append(f"γ {e_lo:g}–{e_hi:g} GeV")
    if zen is not None:
        parts.append(f"zen {zen:g}°")
    cohort_label = " | ".join(parts)

    # Gather lightweight energy/core arrays from cohort chunks (cached per chunk, capped at 15k events)
    s_e_list: List[np.ndarray] = []
    s_l_list: List[np.ndarray] = []
    s_ix_list: List[np.ndarray] = []
    s_iy_list: List[np.ndarray] = []
    if cohort_chunk_idxs:
        # Take up to 35 most recent chunks to keep initial UI plot snappy
        for c_i in cohort_chunk_idxs[-35:]:
            if c_i in chunk_files:
                _, _, e_arr, l_arr, ix_arr, iy_arr = _extract_chunk_summary(chunk_files[c_i], chunk_cache)
                if len(e_arr) > 0:
                    s_e_list.append(e_arr)
                    s_l_list.append(l_arr)
                    s_ix_list.append(ix_arr)
                    s_iy_list.append(iy_arr)
    sample_e = np.concatenate(s_e_list) if s_e_list else None
    sample_l = np.concatenate(s_l_list) if s_l_list else None
    sample_ix = np.concatenate(s_ix_list) if s_ix_list else None
    sample_iy = np.concatenate(s_iy_list) if s_iy_list else None
    if sample_e is not None and len(sample_e) > 15000:
        stride = max(1, len(sample_e) // 15000)
        sample_e = sample_e[::stride]
        sample_l = sample_l[::stride]
        sample_ix = sample_ix[::stride]
        sample_iy = sample_iy[::stride]

    return SimulationProgressSnapshot(
        directory=raw_dir,
        status=overall_status,
        current_session=current_sess,
        cohort_sessions=cohort_sessions,
        cohort_label=cohort_label,
        cohort_gammas_saved=cum_g,
        cohort_hadrons_saved=cum_h,
        cohort_target_gammas=coh_tgt_g,
        cohort_target_hadrons=coh_tgt_h,
        total_dataset_events=non_cohort_events + cum_g + cum_h,
        total_chunks=len(chunk_files),
        active_elapsed_s=prior_active_s,
        paused_time_excluded_s=paused_excluded_s,
        rate_events_per_hr=rate_hr,
        eta_seconds=eta_s,
        timeseries=ts_points,
        pause_markers_s=pause_markers,
        sample_energies=sample_e,
        sample_labels=sample_l,
        sample_impact_x=sample_ix,
        sample_impact_y=sample_iy,
    )


if HAS_PYSIDE6:

    class _MonitorScanWorker(QThread):
        """Background thread that scans simulation manifests/chunks without blocking UI."""

        snapshot_ready = Signal(object)

        def __init__(
            self,
            directory: str,
            chunk_cache: Dict[Tuple[str, int, int], Tuple[int, int]],
            sess_tgt_g: Optional[int],
            sess_tgt_h: Optional[int],
            coh_tgt_g: Optional[int],
            coh_tgt_h: Optional[int],
            parent: Optional[QWidget] = None,
        ):
            super().__init__(parent)
            self.directory = directory
            self.chunk_cache = chunk_cache
            self.sess_tgt_g = sess_tgt_g
            self.sess_tgt_h = sess_tgt_h
            self.coh_tgt_g = coh_tgt_g
            self.coh_tgt_h = coh_tgt_h

        def run(self) -> None:
            snap = scan_simulation_progress(
                self.directory,
                chunk_cache=self.chunk_cache,
                target_gammas_override=self.sess_tgt_g,
                target_hadrons_override=self.sess_tgt_h,
                cohort_target_gammas_override=self.coh_tgt_g,
                cohort_target_hadrons_override=self.coh_tgt_h,
            )
            self.snapshot_ready.emit(snap)


    def _styled_bar(accent: str, height: int = 16) -> QProgressBar:
        bar = QProgressBar()
        bar.setRange(0, 100)
        bar.setValue(0)
        bar.setTextVisible(True)
        bar.setFixedHeight(height)
        bar.setStyleSheet(
            f"""
            QProgressBar {{
                border: 1px solid #334155;
                border-radius: 4px;
                background-color: #0f172a;
                color: #f1f5f9;
                font-size: 11px;
                font-weight: 600;
                text-align: center;
            }}
            QProgressBar::chunk {{
                background-color: {accent};
                border-radius: 3px;
            }}
            """
        )
        return bar


    class SimulationMonitorWidget(QWidget):
        """Real-time progress bars and active-time timeseries for running or paused simulations."""

        cohort_data_ready = Signal(dict)

        def __init__(self, initial_dir: str = "", parent: Optional[QWidget] = None):
            super().__init__(parent)
            self._chunk_cache: Dict[Tuple[str, int, int], Tuple[int, int]] = {}
            self._scan_worker: Optional[_MonitorScanWorker] = None
            self._last_snapshot: Optional[SimulationProgressSnapshot] = None

            self._build_ui(initial_dir)

            self._timer = QTimer(self)
            self._timer.setInterval(5000)
            self._timer.timeout.connect(self.refresh_now)
            self._timer.start()

            if initial_dir:
                self.refresh_now()

        def _build_ui(self, initial_dir: str) -> None:
            root = QVBoxLayout(self)
            root.setContentsMargins(8, 8, 8, 8)
            root.setSpacing(6)

            # Top control bar
            ctrl = QHBoxLayout()
            ctrl.setSpacing(6)
            ctrl.addWidget(QLabel("Monitored Dir:"))
            self.edit_dir = QLineEdit(initial_dir)
            self.edit_dir.editingFinished.connect(self.refresh_now)
            ctrl.addWidget(self.edit_dir, stretch=1)

            self.btn_browse = QPushButton("Browse...")
            self.btn_browse.clicked.connect(self._browse_dir)
            ctrl.addWidget(self.btn_browse)

            self.lbl_status_badge = QLabel("IDLE")
            self.lbl_status_badge.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.lbl_status_badge.setMinimumWidth(86)
            self._set_status_badge("IDLE")
            ctrl.addWidget(self.lbl_status_badge)

            self.chk_auto = QCheckBox("Auto-Refresh (5s)")
            self.chk_auto.setChecked(True)
            self.chk_auto.toggled.connect(self._on_auto_toggled)
            ctrl.addWidget(self.chk_auto)

            self.btn_refresh = QPushButton("Refresh")
            self.btn_refresh.clicked.connect(self.refresh_now)
            ctrl.addWidget(self.btn_refresh)

            self.btn_pause_ext = QPushButton("Pause External (PAUSE)")
            self.btn_pause_ext.setToolTip(
                "Create a PAUSE sentinel file in the monitored directory to gracefully checkpoint and stop external runs."
            )
            self.btn_pause_ext.clicked.connect(self._write_pause_file)
            ctrl.addWidget(self.btn_pause_ext)
            root.addLayout(ctrl)

            # Optional target overrides (0 = auto-detect from sim_manifest.json)
            tgt_row = QHBoxLayout()
            tgt_row.setSpacing(6)
            tgt_row.addWidget(QLabel("Targets (0=auto)  Run γ:"))
            self.spin_sess_g = QSpinBox(); self.spin_sess_g.setRange(0, 5_000_000); self.spin_sess_g.setValue(0)
            self.spin_sess_g.valueChanged.connect(self.refresh_now)
            tgt_row.addWidget(self.spin_sess_g)

            tgt_row.addWidget(QLabel("Run h:"))
            self.spin_sess_h = QSpinBox(); self.spin_sess_h.setRange(0, 5_000_000); self.spin_sess_h.setValue(0)
            self.spin_sess_h.valueChanged.connect(self.refresh_now)
            tgt_row.addWidget(self.spin_sess_h)

            tgt_row.addWidget(QLabel("Cohort γ:"))
            self.spin_coh_g = QSpinBox(); self.spin_coh_g.setRange(0, 5_000_000); self.spin_coh_g.setValue(0)
            self.spin_coh_g.valueChanged.connect(self.refresh_now)
            tgt_row.addWidget(self.spin_coh_g)

            tgt_row.addWidget(QLabel("Cohort h:"))
            self.spin_coh_h = QSpinBox(); self.spin_coh_h.setRange(0, 5_000_000); self.spin_coh_h.setValue(0)
            self.spin_coh_h.valueChanged.connect(self.refresh_now)
            tgt_row.addWidget(self.spin_coh_h)
            tgt_row.addStretch(1)
            root.addLayout(tgt_row)

            # Progress Bar 1: Current Run / Session
            grp_sess = QGroupBox("Current Run / Session Progress")
            vl_s = QVBoxLayout(grp_sess)
            vl_s.setSpacing(4)
            self.lbl_sess_info = QLabel("No active session detected.")
            self.lbl_sess_info.setStyleSheet("color: #cbd5e1; font-size: 11px;")
            self.bar_sess_total = _styled_bar("#22c55e", height=18)
            sub_s = QHBoxLayout()
            self.bar_sess_g = _styled_bar("#38bdf8", height=14)
            self.bar_sess_h = _styled_bar("#f43f5e", height=14)
            sub_s.addWidget(self.bar_sess_g)
            sub_s.addWidget(self.bar_sess_h)
            vl_s.addWidget(self.lbl_sess_info)
            vl_s.addWidget(self.bar_sess_total)
            vl_s.addLayout(sub_s)
            root.addWidget(grp_sess)

            # Progress Bar 2: Cumulative Active Cohort
            grp_coh = QGroupBox("Cumulative Active Cohort Progress (Across Resumed Runs)")
            vl_c = QVBoxLayout(grp_coh)
            vl_c.setSpacing(4)
            self.lbl_coh_info = QLabel("No cohort data detected.")
            self.lbl_coh_info.setStyleSheet("color: #cbd5e1; font-size: 11px;")
            self.bar_coh_total = _styled_bar("#a855f7", height=18)
            sub_c = QHBoxLayout()
            self.bar_coh_g = _styled_bar("#38bdf8", height=14)
            self.bar_coh_h = _styled_bar("#f43f5e", height=14)
            sub_c.addWidget(self.bar_coh_g)
            sub_c.addWidget(self.bar_coh_h)
            vl_c.addWidget(self.lbl_coh_info)
            vl_c.addWidget(self.bar_coh_total)
            vl_c.addLayout(sub_c)
            root.addWidget(grp_coh)

            # Timeseries plot
            self.fig = Figure(figsize=(6.4, 4.2), facecolor="#0b1120")
            self.canvas = FigureCanvas(self.fig)
            root.addWidget(self.canvas, stretch=1)
            self._draw_empty_plot()

        def _set_status_badge(self, status: str) -> None:
            colors = {
                "RUNNING": ("#16a34a", "#ffffff"),
                "PAUSED": ("#d97706", "#ffffff"),
                "COMPLETED": ("#0284c7", "#ffffff"),
                "IDLE": ("#334155", "#cbd5e1"),
            }
            bg, fg = colors.get(status.upper(), ("#334155", "#cbd5e1"))
            self.lbl_status_badge.setText(status.upper())
            self.lbl_status_badge.setStyleSheet(
                f"background-color: {bg}; color: {fg}; font-weight: bold; "
                "border-radius: 4px; padding: 3px 8px;"
            )

        def _draw_empty_plot(self) -> None:
            self.fig.clear()
            ax = self.fig.add_subplot(111)
            ax.set_facecolor("#0f172a")
            ax.set_title("Cumulative Events vs. Active Wall-Clock Time", color="#38bdf8", fontsize=11, fontweight="bold")
            ax.set_xlabel("Active Elapsed Time (hours)", color="#cbd5e1", fontsize=9.5)
            ax.set_ylabel("Cumulative Saved Events", color="#cbd5e1", fontsize=9.5)
            ax.tick_params(colors="#94a3b8", labelsize=8.5)
            ax.grid(True, alpha=0.25, color="#334155")
            for sp in ax.spines.values():
                sp.set_color("#334155")
            self.fig.tight_layout()
            self.canvas.draw_idle()

            if snap.sample_energies is not None and len(snap.sample_energies) > 0:
                self.cohort_data_ready.emit(
                    {
                        "energies": snap.sample_energies,
                        "labels": snap.sample_labels,
                        "impact_x": snap.sample_impact_x,
                        "impact_y": snap.sample_impact_y,
                    }
                )

        @Slot(bool)
        def _on_auto_toggled(self, checked: bool) -> None:
            if checked:
                self._timer.start()
            else:
                self._timer.stop()

        @Slot()
        def _browse_dir(self) -> None:
            d = QFileDialog.getExistingDirectory(self, "Select Simulation Directory", "data")
            if d:
                self.set_directory(d)

        @Slot()
        def _write_pause_file(self) -> None:
            raw_dir = resolve_sim_raw_dir(self.edit_dir.text().strip())
            if raw_dir and os.path.isdir(raw_dir):
                pause_path = os.path.join(raw_dir, "PAUSE")
                with open(pause_path, "w", encoding="utf-8") as pf:
                    pf.write("pause requested from AirCherenkov Studio\n")
                self.lbl_sess_info.setText(f"PAUSE file written to {pause_path} — waiting for active batch to flush...")

        def set_directory(self, path: str) -> None:
            clean = str(path or "").strip()
            if not clean:
                return
            resolved = resolve_sim_raw_dir(clean)
            if self.edit_dir.text().strip() != resolved:
                self.edit_dir.setText(resolved)
            self.refresh_now()

        @Slot()
        def refresh_now(self) -> None:
            raw_dir = self.edit_dir.text().strip()
            if not raw_dir:
                return
            if self._scan_worker is not None and self._scan_worker.isRunning():
                return
            self._scan_worker = _MonitorScanWorker(
                raw_dir,
                self._chunk_cache,
                self.spin_sess_g.value() or None,
                self.spin_sess_h.value() or None,
                self.spin_coh_g.value() or None,
                self.spin_coh_h.value() or None,
                parent=self,
            )
            self._scan_worker.snapshot_ready.connect(self._apply_snapshot)
            self._scan_worker.start()

        @staticmethod
        def _update_bar(bar: QProgressBar, saved: int, target: Optional[int], prefix: str) -> None:
            if target is not None and target > 0:
                pct = min(100, int(round(100.0 * saved / target)))
                bar.setRange(0, 100)
                bar.setValue(pct)
                bar.setFormat(f"{prefix}: {saved:,} / {target:,} ({100.0 * saved / target:.1f}%)")
            else:
                bar.setRange(0, 100)
                bar.setValue(100 if saved > 0 else 0)
                bar.setFormat(f"{prefix}: {saved:,} saved (target unset)")

        @Slot(object)
        def _apply_snapshot(self, snap: SimulationProgressSnapshot) -> None:
            self._last_snapshot = snap
            self._set_status_badge(snap.status)

            # Bar 1: Current Session
            cs = snap.current_session
            if cs is not None:
                sess_tot_saved = cs.gammas_saved + cs.hadrons_saved
                sess_tot_tgt = (
                    (cs.target_gammas or 0) + (cs.target_hadrons or 0)
                    if (cs.target_gammas is not None or cs.target_hadrons is not None)
                    else None
                )
                eta_txt = _format_duration(snap.eta_seconds)
                rate_txt = f"{snap.rate_events_per_hr:,.0f} ev/hr" if snap.rate_events_per_hr > 0 else "--"
                ver_badge = f"{cs.sim_version} @ {cs.git_commit}" if cs.git_commit else cs.sim_version
                self.lbl_sess_info.setText(
                    f"Session {cs.session_index} ({ver_badge}) | Chunks {cs.chunk_start}..{cs.chunk_end} | "
                    f"Active: {_format_duration(cs.active_elapsed_s)} | Rate: {rate_txt} | ETA: {eta_txt}"
                )
                self._update_bar(self.bar_sess_total, sess_tot_saved, sess_tot_tgt, "Current Run Total")
                self._update_bar(self.bar_sess_g, cs.gammas_saved, cs.target_gammas, "Gammas")
                self._update_bar(self.bar_sess_h, cs.hadrons_saved, cs.target_hadrons, "Hadrons")
            else:
                self.lbl_sess_info.setText(f"No simulation chunks found in {snap.directory}")
                self._update_bar(self.bar_sess_total, 0, None, "Current Run Total")
                self._update_bar(self.bar_sess_g, 0, None, "Gammas")
                self._update_bar(self.bar_sess_h, 0, None, "Hadrons")

            # Bar 2: Cumulative Active Cohort
            coh_tot_saved = snap.cohort_gammas_saved + snap.cohort_hadrons_saved
            coh_tot_tgt = (
                (snap.cohort_target_gammas or 0) + (snap.cohort_target_hadrons or 0)
                if (snap.cohort_target_gammas is not None or snap.cohort_target_hadrons is not None)
                else None
            )
            self.lbl_coh_info.setText(
                f"Cohort [{snap.cohort_label}] ({len(snap.cohort_sessions)} session(s)) | "
                f"Cumulative Active: {_format_duration(snap.active_elapsed_s)} "
                f"(Paused Excluded: {_format_duration(snap.paused_time_excluded_s)}) | "
                f"All Dataset Events: {snap.total_dataset_events:,}"
            )
            self._update_bar(self.bar_coh_total, coh_tot_saved, coh_tot_tgt, "Cohort Total")
            self._update_bar(self.bar_coh_g, snap.cohort_gammas_saved, snap.cohort_target_gammas, "Gammas")
            self._update_bar(self.bar_coh_h, snap.cohort_hadrons_saved, snap.cohort_target_hadrons, "Hadrons")

            # Timeseries plot
            self.fig.clear()
            ax = self.fig.add_subplot(111)
            ax.set_facecolor("#0f172a")
            ax.set_title(
                f"Cumulative Generated Events vs. Active Wall-Clock Time ({snap.cohort_label or 'Dataset'})",
                color="#38bdf8",
                fontsize=10.5,
                fontweight="bold",
            )
            if snap.timeseries:
                max_s = max(p.active_elapsed_s for p in snap.timeseries)
                use_hours = max_s >= 5400.0
                scale = 3600.0 if use_hours else 60.0
                unit_lbl = "hours" if use_hours else "minutes"
                t_vals = np.array([p.active_elapsed_s / scale for p in snap.timeseries], dtype=np.float64)
                tot_vals = np.array([p.total_events for p in snap.timeseries], dtype=np.float64)
                g_vals = np.array([p.gammas for p in snap.timeseries], dtype=np.float64)
                h_vals = np.array([p.hadrons for p in snap.timeseries], dtype=np.float64)

                ax.plot(t_vals, tot_vals, color="#a855f7", lw=2.2, label="Total Saved")
                ax.plot(t_vals, g_vals, color="#38bdf8", lw=1.8, label="Gammas")
                ax.plot(t_vals, h_vals, color="#f43f5e", lw=1.8, label="Hadrons")
                for idx_m, pm in enumerate(snap.pause_markers_s):
                    ax.axvline(
                        pm / scale,
                        color="#fbbf24",
                        linestyle="--",
                        alpha=0.65,
                        lw=1.1,
                        label="Resume boundary (pause excluded)" if idx_m == 0 else None,
                    )
                ax.set_xlabel(f"Cumulative Active Wall-Clock Time ({unit_lbl})", color="#cbd5e1", fontsize=9.5)
                ax.legend(fontsize=8.5, facecolor="#0f172a", edgecolor="#334155", labelcolor="#e2e8f0")
            else:
                ax.set_xlabel("Cumulative Active Wall-Clock Time (minutes)", color="#cbd5e1", fontsize=9.5)

            ax.set_ylabel("Cumulative Saved Events", color="#cbd5e1", fontsize=9.5)
            ax.tick_params(colors="#94a3b8", labelsize=8.5)
            ax.grid(True, alpha=0.25, color="#334155")
            for sp in ax.spines.values():
                sp.set_color("#334155")
            self.fig.tight_layout()
            self.canvas.draw_idle()

            if snap.sample_energies is not None and len(snap.sample_energies) > 0:
                self.cohort_data_ready.emit(
                    {
                        "energies": snap.sample_energies,
                        "labels": snap.sample_labels,
                        "impact_x": snap.sample_impact_x,
                        "impact_y": snap.sample_impact_y,
                    }
                )

        def shutdown(self) -> None:
            self._timer.stop()
            if self._scan_worker is not None and self._scan_worker.isRunning():
                self._scan_worker.wait(2500)

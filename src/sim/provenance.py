"""Single source of truth for simulation versioning, manifest I/O, and safe metadata formatting."""

from __future__ import annotations

from datetime import datetime, timezone
import glob
import json
import math
import os
import subprocess
from typing import Any, Dict, List, Optional, Tuple

SIM_SCHEMA_VERSION = "v2.0"

_SLOP_STRINGS = frozenset({"", "unknown", "none", "null", "nan", "n/a", "na", "undefined"})


def clean_str(val: Any) -> Optional[str]:
    """Return a trimmed non-placeholder string, or None for missing/slop values."""
    if val is None:
        return None
    if isinstance(val, float) and not math.isfinite(val):
        return None
    s = str(val).strip()
    if s.lower() in _SLOP_STRINGS:
        return None
    return s


def clean_float(val: Any) -> Optional[float]:
    """Return a finite float, or None for missing/non-numeric/NaN/Inf values."""
    if val is None or isinstance(val, bool):
        return None
    try:
        f = float(val)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def get_git_provenance(cwd: Optional[str] = None) -> Tuple[Optional[str], bool]:
    """Return (short_commit_sha, is_dirty) for the repository, or (None, False)."""
    try:
        commit_raw = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=cwd,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        commit = clean_str(commit_raw)
        dirty = bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"],
                cwd=cwd,
                stderr=subprocess.DEVNULL,
                text=True,
            ).strip()
        )
        return commit, dirty
    except Exception:
        return None, False


def resolve_manifest_dir(path: str) -> str:
    """Resolve either a raw/ simulation directory or its parent dataset directory."""
    clean = os.path.normpath(str(path or "").strip())
    if not clean or clean == ".":
        return ""
    if os.path.isdir(clean):
        if (
            os.path.isfile(os.path.join(clean, "sim_manifest.json"))
            or os.path.isfile(os.path.join(clean, "sim_state.json"))
            or bool(glob.glob(os.path.join(clean, "sim_batch_*.pt")))
        ):
            return clean
        raw_sub = os.path.join(clean, "raw")
        if os.path.isdir(raw_sub) and (
            os.path.isfile(os.path.join(raw_sub, "sim_manifest.json"))
            or os.path.isfile(os.path.join(raw_sub, "sim_state.json"))
            or bool(glob.glob(os.path.join(raw_sub, "sim_batch_*.pt")))
        ):
            return raw_sub
    return clean


def update_sim_manifest(
    output_dir: str,
    session_entry: Dict[str, Any],
    schema_version: str = SIM_SCHEMA_VERSION,
) -> str:
    """Safely create or update ``sim_manifest.json`` in ``output_dir``."""
    os.makedirs(output_dir, exist_ok=True)
    manifest_path = os.path.join(output_dir, "sim_manifest.json")
    manifest: Dict[str, Any] = {"schema_version": schema_version, "sessions": []}
    if os.path.isfile(manifest_path):
        try:
            with open(manifest_path, "r", encoding="utf-8") as mf:
                loaded = json.load(mf)
            if isinstance(loaded, dict):
                manifest = loaded
        except Exception:
            pass

    sessions = manifest.setdefault("sessions", [])
    if not isinstance(sessions, list):
        sessions = []
        manifest["sessions"] = sessions

    entry = dict(session_entry)
    started_at = entry.get("started_at")
    if sessions and started_at and sessions[-1].get("started_at") == started_at:
        entry["session_index"] = sessions[-1].get("session_index", len(sessions) - 1)
        sessions[-1] = entry
    else:
        entry["session_index"] = len(sessions)
        sessions.append(entry)

    tmp_path = f"{manifest_path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as mf:
        json.dump(manifest, mf, indent=2)
    os.replace(tmp_path, manifest_path)
    return manifest_path


def read_dataset_provenance(data_path: str) -> Dict[str, Any]:
    """Safely summarize simulation version provenance for a dataset directory.

    Never raises on missing/corrupt files and never emits ``NaN``, ``None``,
    or ``unknown`` placeholder strings.
    """
    empty = {
        "has_manifest": False,
        "has_legacy_chunks": False,
        "versions": [],
        "latest_version": None,
        "latest_commit": None,
        "is_mixed_versions": False,
        "summary_tag": "",
    }
    raw_dir = resolve_manifest_dir(data_path)
    if not raw_dir or not os.path.isdir(raw_dir):
        return empty

    manifest_path = os.path.join(raw_dir, "sim_manifest.json")
    has_pt = bool(glob.glob(os.path.join(raw_dir, "sim_batch_*.pt")))

    sessions: List[Dict[str, Any]] = []
    has_manifest = False
    if os.path.isfile(manifest_path):
        try:
            with open(manifest_path, "r", encoding="utf-8") as mf:
                data = json.load(mf)
            if isinstance(data, dict) and isinstance(data.get("sessions"), list):
                sessions = [s for s in data["sessions"] if isinstance(s, dict)]
                has_manifest = bool(sessions)
        except Exception:
            sessions = []
            has_manifest = False

    if not has_manifest:
        if has_pt:
            return {
                **empty,
                "has_legacy_chunks": True,
                "summary_tag": "Legacy (unversioned)",
            }
        return empty

    versions: List[str] = []
    latest_version: Optional[str] = None
    latest_commit: Optional[str] = None
    has_unversioned_session = False

    for sess in sessions:
        ver = clean_str(sess.get("sim_version"))
        commit = clean_str(sess.get("git_commit"))
        if ver is not None:
            if ver not in versions:
                versions.append(ver)
            latest_version = ver
            latest_commit = commit
        else:
            has_unversioned_session = True

    is_mixed = len(versions) > 1 or (len(versions) == 1 and has_unversioned_session)

    if not versions:
        summary_tag = "Legacy (unversioned)" if has_pt or has_unversioned_session else ""
    elif is_mixed:
        ver_label = " + ".join(versions)
        if has_unversioned_session:
            ver_label = f"unversioned + {ver_label}"
        summary_tag = f"Mixed ({ver_label}" + (f" @ {latest_commit})" if latest_commit else ")")
    else:
        summary_tag = f"{latest_version} @ {latest_commit}" if latest_commit else str(latest_version)

    return {
        "has_manifest": True,
        "has_legacy_chunks": has_unversioned_session,
        "versions": versions,
        "latest_version": latest_version,
        "latest_commit": latest_commit,
        "is_mixed_versions": is_mixed,
        "summary_tag": summary_tag,
    }


def format_event_provenance(ev: Optional[Dict[str, Any]]) -> str:
    """Format per-event simulation provenance safely without NaN/None/unknown."""
    if not isinstance(ev, dict):
        return "Not recorded (legacy)"

    ver = clean_str(ev.get("sim_version"))
    commit = clean_str(ev.get("sim_git_commit"))
    r_thrown = clean_float(ev.get("impact_radius_thrown"))
    preset = clean_str(ev.get("array_preset"))

    parts: List[str] = []
    if ver is not None:
        parts.append(f"{ver} @ {commit}" if commit else ver)
    elif commit is not None:
        parts.append(f"commit {commit}")

    if preset is not None:
        parts.append(preset)
    if r_thrown is not None and r_thrown > 0:
        parts.append(f"R_thrown={r_thrown:.0f} m")

    return " | ".join(parts) if parts else "Not recorded (legacy)"


def utc_now_iso() -> str:
    """Return current UTC timestamp in ISO-8601 format."""
    return datetime.now(timezone.utc).isoformat()


_TAGGED_DIR_CACHE: set[str] = set()


def ensure_dataset_tagged(
    data_path: str,
    *,
    allow_llm: bool = True,
    force: bool = False,
) -> Dict[str, Any]:
    """Check a dataset directory once per process and retroactively tag if untagged."""
    raw_dir = resolve_manifest_dir(data_path)
    if not raw_dir or not os.path.isdir(raw_dir):
        return read_dataset_provenance(data_path)
    if not force and raw_dir in _TAGGED_DIR_CACHE:
        return read_dataset_provenance(raw_dir)

    from sim.llm_tagger import tag_untagged_dataset

    prov = tag_untagged_dataset(raw_dir, allow_llm=allow_llm)
    _TAGGED_DIR_CACHE.add(raw_dir)
    return prov

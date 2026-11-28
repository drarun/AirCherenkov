"""LLM-assisted retroactive version tagging for untagged AirCherenkov simulation datasets."""

from __future__ import annotations

from datetime import datetime, timezone
import glob
import json
import os
import re
import subprocess
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from sim.provenance import (
    SIM_SCHEMA_VERSION,
    clean_float,
    clean_str,
    read_dataset_provenance,
    resolve_manifest_dir,
    update_sim_manifest,
    utc_now_iso,
)

_TAG_SAFE_RE = re.compile(r"[^A-Za-z0-9._-]+")


def sanitize_version_tag(raw_tag: Any) -> Optional[str]:
    """Sanitize a proposed version tag into a clean ASCII slug or return None."""
    cleaned = clean_str(raw_tag)
    if cleaned is None:
        return None
    slug = _TAG_SAFE_RE.sub("-", cleaned).strip("-._")
    if not slug or len(slug) > 48:
        slug = slug[:48].strip("-._")
    return clean_str(slug)


def _chunk_index(path: str) -> Optional[int]:
    base = os.path.basename(path)
    if not (base.startswith("sim_batch_") and base.endswith(".pt")):
        return None
    try:
        return int(base[len("sim_batch_") : -len(".pt")])
    except ValueError:
        return None


def summarize_untagged_dataset(raw_dir: str) -> Dict[str, Any]:
    """Inspect metadata & first/last chunk of an untagged directory without modifying .pt files."""
    pt_files = sorted(glob.glob(os.path.join(raw_dir, "sim_batch_*.pt")))
    if not pt_files:
        pt_files = sorted(glob.glob(os.path.join(raw_dir, "*.pt")))

    chunk_indices = [idx for idx in (_chunk_index(p) for p in pt_files) if idx is not None]
    mtimes = [os.path.getmtime(p) for p in pt_files] if pt_files else []
    earliest_ts = min(mtimes) if mtimes else None
    earliest_date = (
        datetime.fromtimestamp(earliest_ts, tz=timezone.utc).strftime("%Y-%m-%d")
        if earliest_ts is not None
        else None
    )

    sample_files = [pt_files[0]] + ([pt_files[-1]] if len(pt_files) > 1 else [])
    g_energies: List[float] = []
    h_energies: List[float] = []
    zeniths: List[float] = []
    impact_rs: List[float] = []
    sample_keys: List[str] = []
    n_telescopes: Optional[int] = None
    n_pixels: Optional[int] = None
    n_time_bins: Optional[int] = None

    for sf in sample_files:
        try:
            loaded = torch.load(sf, map_location="cpu", weights_only=False)
        except Exception:
            continue
        if not isinstance(loaded, list):
            continue
        for ev in loaded:
            if not isinstance(ev, dict):
                continue
            if not sample_keys:
                sample_keys = sorted(str(k) for k in ev.keys())
            if n_telescopes is None and "fadc_traces" in ev:
                arr = np.asarray(ev["fadc_traces"])
                if arr.ndim == 3:
                    n_telescopes, n_pixels, n_time_bins = map(int, arr.shape)
            e_val = clean_float(ev.get("energy"))
            lbl = int(ev.get("label", -1))
            if e_val is not None and e_val > 0:
                if lbl == 1:
                    g_energies.append(e_val)
                elif lbl == 0:
                    h_energies.append(e_val)
            z_val = clean_float(ev.get("zenith_deg"))
            if z_val is not None:
                zeniths.append(z_val)
            ix = clean_float(ev.get("impact_x"))
            iy = clean_float(ev.get("impact_y"))
            if ix is not None and iy is not None:
                impact_rs.append(float(np.hypot(ix, iy)))

    gammas_thrown = 0
    hadrons_thrown = 0
    state_file = os.path.join(raw_dir, "sim_state.json")
    if os.path.isfile(state_file):
        try:
            with open(state_file, "r", encoding="utf-8") as sfh:
                st = json.load(sfh)
            gammas_thrown = int(st.get("gammas_thrown", 0))
            hadrons_thrown = int(st.get("hadrons_thrown", 0))
        except Exception:
            pass

    recent_commits: List[str] = []
    if earliest_ts is not None:
        try:
            iso_before = datetime.fromtimestamp(earliest_ts, tz=timezone.utc).isoformat()
            out = subprocess.check_output(
                ["git", "log", f"--before={iso_before}", "-n", "5", "--oneline"],
                stderr=subprocess.DEVNULL,
                text=True,
            ).strip()
            if out:
                recent_commits = [line.strip() for line in out.splitlines() if line.strip()]
        except Exception:
            pass

    parent_name = os.path.basename(os.path.dirname(raw_dir))
    dir_label = parent_name if os.path.basename(raw_dir) == "raw" and parent_name else os.path.basename(raw_dir)

    return {
        "dir_name": dir_label or "sim_dataset",
        "n_chunks": len(pt_files),
        "chunk_start": min(chunk_indices) if chunk_indices else 0,
        "chunk_end": max(chunk_indices) if chunk_indices else max(0, len(pt_files) - 1),
        "earliest_date": earliest_date,
        "gamma_e_min": round(min(g_energies), 2) if g_energies else None,
        "gamma_e_max": round(max(g_energies), 2) if g_energies else None,
        "hadron_e_min": round(min(h_energies), 2) if h_energies else None,
        "hadron_e_max": round(max(h_energies), 2) if h_energies else None,
        "zenith_deg": round(float(np.median(zeniths)), 1) if zeniths else None,
        "max_impact_r_m": round(max(impact_rs), 1) if impact_rs else None,
        "n_telescopes": n_telescopes,
        "n_pixels": n_pixels,
        "n_time_bins": n_time_bins,
        "sample_keys": sample_keys,
        "gammas_thrown": gammas_thrown,
        "hadrons_thrown": hadrons_thrown,
        "recent_commits": recent_commits,
    }


def heuristic_version_tag(summary: Dict[str, Any]) -> Dict[str, str]:
    """Deterministic fallback version tag derived from dataset summary."""
    has_v2_keys = "impact_radius_thrown" in (summary.get("sample_keys") or [])
    prefix = "v2.0" if has_v2_keys else "v1.0-retro"
    parts = [prefix]
    if summary.get("zenith_deg") is not None:
        parts.append(f"zen{int(round(float(summary['zenith_deg'])))}")
    if summary.get("earliest_date"):
        parts.append(str(summary["earliest_date"]))
    tag = sanitize_version_tag("-".join(parts)) or "v1.0-retro"
    notes = (
        f"Retroactively tagged via heuristic fallback for {summary.get('dir_name', 'dataset')} "
        f"({summary.get('n_chunks', 0)} chunk(s))."
    )
    return {"sim_version": tag, "notes": notes, "tag_source": "heuristic-fallback"}


def _extract_json_payload(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped)
        stripped = re.sub(r"\s*```$", "", stripped)
    try:
        obj = json.loads(stripped)
        return obj if isinstance(obj, dict) else None
    except Exception:
        m = re.search(r"\{.*\}", stripped, flags=re.DOTALL)
        if m:
            try:
                obj = json.loads(m.group(0))
                return obj if isinstance(obj, dict) else None
            except Exception:
                return None
        return None


def _build_llm_prompt(summary: Dict[str, Any]) -> str:
    return (
        "Assign a concise, informative simulation version tag to an untagged IACT Cherenkov MC dataset.\n"
        "Respond strictly with JSON: {\"sim_version\": \"<tag>\", \"notes\": \"<1-sentence rationale>\"}.\n"
        "Rules for sim_version: ASCII [A-Za-z0-9._-] only, <= 40 chars, e.g. 'v1.0-pre-audit-2026-11-01' "
        "or 'v1.0-crab-zen30-2026-10-15'. Never output 'unknown', 'none', or 'nan'.\n"
        f"Dataset summary:\n{json.dumps(summary, indent=2)}"
    )


def infer_tag_with_llm(
    summary: Dict[str, Any],
    *,
    allow_llm: bool = True,
    timeout_s: float = 8.0,
) -> Dict[str, str]:
    """Auto-detect an LLM provider (Gemini -> OpenAI -> Anthropic -> Ollama) with deterministic fallback."""
    if not allow_llm:
        return heuristic_version_tag(summary)

    try:
        import requests
    except ImportError:
        return heuristic_version_tag(summary)

    prompt = _build_llm_prompt(summary)
    providers = []

    gemini_key = clean_str(os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"))
    if gemini_key:
        providers.append(("llm-gemini", gemini_key))

    openai_key = clean_str(os.environ.get("OPENAI_API_KEY"))
    if openai_key:
        providers.append(("llm-openai", openai_key))

    anthropic_key = clean_str(os.environ.get("ANTHROPIC_API_KEY"))
    if anthropic_key:
        providers.append(("llm-anthropic", anthropic_key))

    ollama_host = (clean_str(os.environ.get("OLLAMA_HOST")) or "http://localhost:11434").rstrip("/")
    providers.append(("llm-ollama", ollama_host))

    for source, cred in providers:
        raw_text = ""
        try:
            if source == "llm-gemini":
                url = (
                    "https://generativelanguage.googleapis.com/v1beta/models/"
                    f"gemini-2.5-flash:generateContent?key={cred}"
                )
                resp = requests.post(
                    url,
                    json={"contents": [{"parts": [{"text": prompt}]}]},
                    timeout=timeout_s,
                )
                resp.raise_for_status()
                data = resp.json()
                raw_text = (
                    data.get("candidates", [{}])[0]
                    .get("content", {})
                    .get("parts", [{}])[0]
                    .get("text", "")
                )
            elif source == "llm-openai":
                resp = requests.post(
                    "https://api.openai.com/v1/chat/completions",
                    headers={"Authorization": f"Bearer {cred}"},
                    json={
                        "model": "gpt-4o-mini",
                        "messages": [{"role": "user", "content": prompt}],
                        "temperature": 0.1,
                    },
                    timeout=timeout_s,
                )
                resp.raise_for_status()
                raw_text = resp.json()["choices"][0]["message"]["content"]
            elif source == "llm-anthropic":
                resp = requests.post(
                    "https://api.anthropic.com/v1/messages",
                    headers={
                        "x-api-key": cred,
                        "anthropic-version": "2023-06-01",
                    },
                    json={
                        "model": "claude-3-5-haiku-latest",
                        "max_tokens": 200,
                        "messages": [{"role": "user", "content": prompt}],
                    },
                    timeout=timeout_s,
                )
                resp.raise_for_status()
                raw_text = resp.json()["content"][0]["text"]
            elif source == "llm-ollama":
                probe = requests.get(f"{cred}/api/tags", timeout=min(1.5, timeout_s))
                probe.raise_for_status()
                models = probe.json().get("models") or []
                if not models:
                    continue
                model_name = str(models[0].get("name") or "llama3.2")
                resp = requests.post(
                    f"{cred}/api/generate",
                    json={"model": model_name, "prompt": prompt, "stream": False},
                    timeout=timeout_s,
                )
                resp.raise_for_status()
                raw_text = resp.json().get("response", "")
        except Exception:
            continue

        parsed = _extract_json_payload(raw_text)
        if parsed:
            tag = sanitize_version_tag(parsed.get("sim_version"))
            if tag is not None:
                notes = clean_str(parsed.get("notes")) or f"Retroactively tagged by {source}."
                return {"sim_version": tag, "notes": notes, "tag_source": source}

    return heuristic_version_tag(summary)


def tag_untagged_dataset(
    data_path: str,
    *,
    allow_llm: bool = True,
    timeout_s: float = 8.0,
) -> Dict[str, Any]:
    """Ensure untagged simulation directories receive a retroactive sim_manifest.json sidecar."""
    raw_dir = resolve_manifest_dir(data_path)
    prov = read_dataset_provenance(raw_dir)
    if not raw_dir or not os.path.isdir(raw_dir):
        return prov
    if prov["has_manifest"] and prov["latest_version"] is not None:
        return prov
    if not prov["has_legacy_chunks"] and not glob.glob(os.path.join(raw_dir, "sim_batch_*.pt")):
        return prov

    summary = summarize_untagged_dataset(raw_dir)
    inferred = infer_tag_with_llm(summary, allow_llm=allow_llm, timeout_s=timeout_s)

    session_entry: Dict[str, Any] = {
        "sim_version": inferred["sim_version"],
        "status": "completed",
        "append": False,
        "retroactive": True,
        "tag_source": inferred["tag_source"],
        "retroactively_tagged_utc": utc_now_iso(),
        "chunk_start": summary["chunk_start"],
        "chunk_end": summary["chunk_end"],
        "gammas_thrown_session": summary["gammas_thrown"],
        "hadrons_thrown_session": summary["hadrons_thrown"],
        "notes": inferred["notes"],
        "params": {
            "e_min": summary["gamma_e_min"],
            "e_max": summary["gamma_e_max"],
            "hadron_e_min": summary["hadron_e_min"],
            "hadron_e_max": summary["hadron_e_max"],
            "zenith_deg": summary["zenith_deg"],
        },
    }
    update_sim_manifest(raw_dir, session_entry, schema_version=SIM_SCHEMA_VERSION)
    return read_dataset_provenance(raw_dir)

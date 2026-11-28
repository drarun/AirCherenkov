"""Tests for LLM-assisted retroactive tagging of untagged simulation datasets."""

import json
import os
from types import SimpleNamespace

import numpy as np
import requests
import torch

from analysis.io import load_events_universal
from sim.llm_tagger import infer_tag_with_llm, sanitize_version_tag, tag_untagged_dataset


def _make_untagged_chunk(path, energy: float = 500.0) -> None:
    torch.save(
        [
            {
                "fadc_traces": np.zeros((4, 469, 16), dtype=np.float32),
                "gain_flags": np.zeros((4, 469), dtype=np.float32),
                "energy": energy,
                "label": 1,
                "impact_x": 45.0,
                "impact_y": -30.0,
                "zenith_deg": 30.0,
                "azimuth_deg": 0.0,
            }
        ],
        path,
    )


def _clear_llm_env(monkeypatch) -> None:
    for var in ("GEMINI_API_KEY", "GOOGLE_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY", "OLLAMA_HOST"):
        monkeypatch.delenv(var, raising=False)


def test_provider_priority_gemini_openai_anthropic_ollama(monkeypatch):
    summary = {"dir_name": "crab_old", "n_chunks": 2, "zenith_deg": 30.0, "earliest_date": "2026-10-01"}

    # 1. Gemini takes priority when GEMINI_API_KEY is set
    _clear_llm_env(monkeypatch)
    monkeypatch.setenv("GEMINI_API_KEY", "gem-key")
    monkeypatch.setenv("OPENAI_API_KEY", "oai-key")

    def fake_post_gemini(url, **kwargs):
        assert "generativelanguage.googleapis.com" in url
        return SimpleNamespace(
            raise_for_status=lambda: None,
            json=lambda: {
                "candidates": [
                    {"content": {"parts": [{"text": '{"sim_version": "v1.0-crab-gem", "notes": "Gemini tag"}'}]}}
                ]
            },
        )

    monkeypatch.setattr(requests, "post", fake_post_gemini)
    res_g = infer_tag_with_llm(summary)
    assert res_g["sim_version"] == "v1.0-crab-gem"
    assert res_g["tag_source"] == "llm-gemini"

    # 2. OpenAI used when Gemini key is absent
    _clear_llm_env(monkeypatch)
    monkeypatch.setenv("OPENAI_API_KEY", "oai-key")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "ant-key")

    def fake_post_openai(url, **kwargs):
        assert "api.openai.com" in url
        return SimpleNamespace(
            raise_for_status=lambda: None,
            json=lambda: {
                "choices": [{"message": {"content": '{"sim_version": "v1.0-crab-oai", "notes": "OpenAI tag"}'}}]
            },
        )

    monkeypatch.setattr(requests, "post", fake_post_openai)
    res_o = infer_tag_with_llm(summary)
    assert res_o["sim_version"] == "v1.0-crab-oai"
    assert res_o["tag_source"] == "llm-openai"

    # 3. Anthropic used when only ANTHROPIC_API_KEY is set
    _clear_llm_env(monkeypatch)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "ant-key")

    def fake_post_anthropic(url, **kwargs):
        assert "api.anthropic.com" in url
        return SimpleNamespace(
            raise_for_status=lambda: None,
            json=lambda: {
                "content": [{"text": '{"sim_version": "v1.0-crab-ant", "notes": "Anthropic tag"}'}]
            },
        )

    monkeypatch.setattr(requests, "post", fake_post_anthropic)
    res_a = infer_tag_with_llm(summary)
    assert res_a["sim_version"] == "v1.0-crab-ant"
    assert res_a["tag_source"] == "llm-anthropic"

    # 4. Ollama used when no cloud keys are set and local Ollama responds
    _clear_llm_env(monkeypatch)
    monkeypatch.setattr(
        requests,
        "get",
        lambda url, **kwargs: SimpleNamespace(
            raise_for_status=lambda: None, json=lambda: {"models": [{"name": "llama3.2"}]}
        ),
    )
    monkeypatch.setattr(
        requests,
        "post",
        lambda url, **kwargs: SimpleNamespace(
            raise_for_status=lambda: None,
            json=lambda: {"response": '{"sim_version": "v1.0-crab-ollama", "notes": "Local Ollama"}'},
        ),
    )
    res_l = infer_tag_with_llm(summary)
    assert res_l["sim_version"] == "v1.0-crab-ollama"
    assert res_l["tag_source"] == "llm-ollama"


def test_slop_and_corrupt_llm_response_falls_back_cleanly(monkeypatch):
    _clear_llm_env(monkeypatch)
    monkeypatch.setenv("GEMINI_API_KEY", "gem-key")
    monkeypatch.setattr(
        requests,
        "post",
        lambda url, **kwargs: SimpleNamespace(
            raise_for_status=lambda: None,
            json=lambda: {
                "candidates": [{"content": {"parts": [{"text": '{"sim_version": "unknown", "notes": "NaN"}'}]}}]
            },
        ),
    )
    monkeypatch.setattr(requests, "get", lambda *a, **kw: (_ for _ in ()).throw(ConnectionError("no ollama")))

    summary = {"dir_name": "old_crab", "n_chunks": 3, "zenith_deg": 30.0, "earliest_date": "2026-10-12"}
    res = infer_tag_with_llm(summary)
    assert res["tag_source"] == "heuristic-fallback"
    assert res["sim_version"] == "v1.0-retro-zen30-2026-10-12"
    assert sanitize_version_tag("unknown") is None
    assert sanitize_version_tag("NaN") is None


def test_retroactive_sidecar_tagging_on_load_events_universal(tmp_path, monkeypatch):
    _clear_llm_env(monkeypatch)
    monkeypatch.setenv("OPENAI_API_KEY", "oai-key")

    call_count = {"n": 0}

    def fake_post(url, **kwargs):
        call_count["n"] += 1
        return SimpleNamespace(
            raise_for_status=lambda: None,
            json=lambda: {
                "choices": [
                    {
                        "message": {
                            "content": '{"sim_version": "v1.0-retro-crab", "notes": "Tagged on read"}'
                        }
                    }
                ]
            },
        )

    monkeypatch.setattr(requests, "post", fake_post)

    raw = tmp_path / "raw"
    raw.mkdir()
    pt_path = raw / "sim_batch_00000.pt"
    _make_untagged_chunk(pt_path, energy=350.0)
    orig_mtime_ns = os.stat(pt_path).st_mtime_ns

    events1 = load_events_universal(str(tmp_path))
    assert len(events1) == 1
    assert events1[0]["sim_version"] == "v1.0-retro-crab"
    assert call_count["n"] == 1
    # Original .pt file must not be modified
    assert os.stat(pt_path).st_mtime_ns == orig_mtime_ns

    manifest_path = raw / "sim_manifest.json"
    assert manifest_path.is_file()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["sessions"][0]["retroactive"] is True
    assert manifest["sessions"][0]["sim_version"] == "v1.0-retro-crab"
    assert manifest["sessions"][0]["tag_source"] == "llm-openai"

    # Subsequent reads / tag calls must NOT re-call the LLM
    prov2 = tag_untagged_dataset(str(tmp_path))
    assert prov2["latest_version"] == "v1.0-retro-crab"
    assert call_count["n"] == 1

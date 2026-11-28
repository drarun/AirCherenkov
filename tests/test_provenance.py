"""Tests for simulation version provenance and slop-free missing-metadata handling."""

import json
import math

import numpy as np
import pytest
import torch

from analysis.io import _normalize_event_dict
from sim.provenance import (
    clean_float,
    clean_str,
    format_event_provenance,
    read_dataset_provenance,
    update_sim_manifest,
)


def _assert_no_slop(text: str) -> None:
    low = text.lower()
    for bad in ("nan", "none", "unknown", "null"):
        assert bad not in low, f"Found slop token {bad!r} in {text!r}"


def test_clean_helpers():
    assert clean_str(None) is None
    assert clean_str("") is None
    assert clean_str("unknown") is None
    assert clean_str("NaN") is None
    assert clean_str("None") is None
    assert clean_str("v2.0") == "v2.0"

    assert clean_float(None) is None
    assert clean_float(float("nan")) is None
    assert clean_float(float("inf")) is None
    assert clean_float(True) is None
    assert clean_float("300.0") == 300.0


def test_read_dataset_provenance_missing_corrupt_and_poisoned(tmp_path):
    # Non-existent path
    p_missing = read_dataset_provenance(str(tmp_path / "does_not_exist"))
    assert p_missing["summary_tag"] == ""
    assert not p_missing["has_manifest"]

    # Corrupt JSON with legacy .pt chunk
    raw = tmp_path / "raw"
    raw.mkdir()
    torch.save([{"label": 1, "energy": 100.0}], raw / "sim_batch_00000.pt")
    (raw / "sim_manifest.json").write_text("{corrupt json", encoding="utf-8")
    p_corrupt = read_dataset_provenance(str(tmp_path))
    assert p_corrupt["has_legacy_chunks"] is True
    assert p_corrupt["summary_tag"] == "Legacy (unversioned)"
    _assert_no_slop(p_corrupt["summary_tag"])

    # Poisoned manifest with slop values
    (raw / "sim_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "v2.0",
                "sessions": [
                    {"sim_version": "unknown", "git_commit": "NaN"},
                    {"sim_version": None, "git_commit": "none"},
                ],
            }
        ),
        encoding="utf-8",
    )
    p_poison = read_dataset_provenance(str(tmp_path))
    _assert_no_slop(p_poison["summary_tag"])
    assert p_poison["summary_tag"] == "Legacy (unversioned)"


def test_read_dataset_provenance_single_and_mixed(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    update_sim_manifest(
        str(raw),
        {"sim_version": "v1.0", "git_commit": "aaa1111", "started_at": "t0"},
    )
    p1 = read_dataset_provenance(str(tmp_path))
    assert p1["summary_tag"] == "v1.0 @ aaa1111"
    assert not p1["is_mixed_versions"]

    update_sim_manifest(
        str(raw),
        {"sim_version": "v2.0", "git_commit": "bbb2222", "started_at": "t1"},
    )
    p2 = read_dataset_provenance(str(raw))
    assert p2["is_mixed_versions"] is True
    assert p2["summary_tag"] == "Mixed (v1.0 + v2.0 @ bbb2222)"
    _assert_no_slop(p2["summary_tag"])


def test_format_event_provenance_and_normalize_event_dict():
    assert format_event_provenance({}) == "Not recorded (legacy)"
    assert (
        format_event_provenance(
            {
                "sim_version": "unknown",
                "sim_git_commit": "NaN",
                "impact_radius_thrown": float("nan"),
                "array_preset": "None",
            }
        )
        == "Not recorded (legacy)"
    )

    tagged = format_event_provenance(
        {
            "sim_version": "v2.0",
            "sim_git_commit": "0bded55",
            "impact_radius_thrown": 300.0,
            "array_preset": "VERITAS",
        }
    )
    assert tagged == "v2.0 @ 0bded55 | VERITAS | R_thrown=300 m"
    _assert_no_slop(tagged)

    # _normalize_event_dict handles missing/NaN metadata cleanly and preserves valid fields
    legacy_norm = _normalize_event_dict(
        {
            "fadc_traces": np.zeros((4, 469, 16), dtype=np.float32),
            "energy": float("nan"),
            "wobble_dir": "unknown",
        }
    )
    assert math.isfinite(legacy_norm["energy"]) and legacy_norm["energy"] == 0.0
    assert legacy_norm["sim_version"] is None
    assert legacy_norm["impact_radius_thrown"] is None
    assert legacy_norm["wobble_dir"] == "Not recorded"

    v2_norm = _normalize_event_dict(
        {
            "fadc_traces": np.zeros((4, 469, 16), dtype=np.float32),
            "energy": 120.0,
            "impact_radius_thrown": 300.0,
            "sim_version": "v2.0",
            "sim_git_commit": "0bded55",
            "array_preset": "VERITAS",
        }
    )
    assert v2_norm["sim_version"] == "v2.0"
    assert v2_norm["sim_git_commit"] == "0bded55"
    assert v2_norm["impact_radius_thrown"] == 300.0
    assert v2_norm["array_preset"] == "VERITAS"


def test_describe_model_and_dataset_provenance(tmp_path):
    pytest.importorskip("PySide6")
    from gui.workers import describe_model_and_dataset

    raw = tmp_path / "raw"
    raw.mkdir()
    update_sim_manifest(
        str(raw),
        {"sim_version": "v2.0", "git_commit": "0bded55", "started_at": "t0"},
    )
    prov = describe_model_and_dataset(data_path=str(tmp_path), array_preset="VERITAS")
    assert prov["sim_provenance"] == "v2.0 @ 0bded55"
    assert "Sim: v2.0 @ 0bded55" in prov["provenance_banner"]
    _assert_no_slop(prov["provenance_banner"])

    prov_empty = describe_model_and_dataset(data_path=str(tmp_path / "missing"), array_preset="VERITAS")
    assert prov_empty["sim_provenance"] == ""
    _assert_no_slop(prov_empty["provenance_banner"])

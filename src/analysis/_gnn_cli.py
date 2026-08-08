"""Lightweight helpers shared by the optional GNN command-line tools.

This module deliberately avoids importing PyTorch, PyTorch Geometric, sklearn,
or matplotlib at import time.  That keeps ``--help`` available in a core-only
AirCherenkov installation and lets the commands report actionable dependency
errors when inference is actually requested.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


EXPECTED_NODE_FEATURES = 22
DEFAULT_CAMERA_RINGS = 12


class OptionalDependencyError(RuntimeError):
    """Raised when a command needs an optional dependency group."""


@dataclass(frozen=True)
class GNNRuntime:
    """Modules loaded lazily for GNN inference and plotting."""

    torch: Any
    data_loader: Any
    pyplot: Any
    numpy: Any
    roc_auc_score: Any | None = None
    roc_curve: Any | None = None


class AddEdgeIndex:
    """Attach one immutable camera graph to each retrieved event."""

    def __init__(self, edge_index):
        self.edge_index = edge_index

    def __call__(self, data):
        data.edge_index = self.edge_index
        return data


def load_runtime(*, metrics: bool) -> GNNRuntime:
    """Import optional inference dependencies only when a command runs."""
    try:
        import numpy as np
        import torch
    except ModuleNotFoundError as exc:  # pragma: no cover - core dependency guard
        raise OptionalDependencyError(
            "GNN inference requires the core dependencies; install the project "
            "with `python -m pip install -e .`."
        ) from exc

    try:
        from torch_geometric.loader import DataLoader
    except ModuleNotFoundError as exc:
        raise OptionalDependencyError(
            "GNN inference requires PyTorch Geometric; install the ML extras "
            "and plotting support with `python -m pip install -e \".[ml,viz]\"`."
        ) from exc

    try:
        import matplotlib

        # These tools only write image files; a GUI backend is unnecessary and
        # makes remote/headless execution less portable.
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        raise OptionalDependencyError(
            "GNN plots require matplotlib; install the visualization extras "
            "with `python -m pip install -e \".[ml,viz]\"`."
        ) from exc

    roc_auc_score = None
    roc_curve = None
    if metrics:
        try:
            from sklearn.metrics import roc_auc_score, roc_curve
        except ModuleNotFoundError as exc:
            raise OptionalDependencyError(
                "GNN evaluation requires scikit-learn; install the ML extras "
                "with `python -m pip install -e \".[ml,viz]\"`."
            ) from exc

    return GNNRuntime(
        torch=torch,
        data_loader=DataLoader,
        pyplot=plt,
        numpy=np,
        roc_auc_score=roc_auc_score,
        roc_curve=roc_curve,
    )


def resolve_device(torch, requested: str):
    """Resolve an explicit portable device selection."""
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise ValueError(
            "CUDA was requested, but torch.cuda.is_available() is false; "
            "use `--device cpu` or install a CUDA-enabled PyTorch build."
        )
    if requested not in {"cpu", "cuda"}:
        raise ValueError(f"Unsupported device {requested!r}; choose auto, cpu, or cuda")
    return torch.device(requested)


def camera_edge_index(torch, camera):
    """Return directed nearest-neighbour edges for a Camera instance."""
    adjacency = torch.as_tensor(camera.get_neighbor_matrix(), dtype=torch.bool)
    return adjacency.nonzero(as_tuple=False).t().contiguous()


def validate_dataset_root(root: Path) -> Path:
    """Reject misspelled/empty roots before PyG creates cache directories."""
    root = Path(root).expanduser()
    processed = root / "processed" / "data.pt"
    raw = root / "raw"
    has_raw_files = raw.is_dir() and any(path.is_file() for path in raw.iterdir())
    if not processed.is_file() and not has_raw_files:
        raise FileNotFoundError(
            f"No processed dataset or raw event files found under {root}. "
            f"Expected {processed} or files in {raw}."
        )
    return root


def validate_graph(data, *, expected_pixels: int) -> None:
    """Verify the current 16-bin trace + gain + spatial feature contract."""
    features = getattr(data, "x", None)
    if features is None or features.ndim != 2:
        raise ValueError("Dataset events must contain a two-dimensional `x` tensor")
    if features.shape[1] != EXPECTED_NODE_FEATURES:
        raise ValueError(
            "The current EnergyGNN/ClassGNN checkpoints require 22 node features "
            "(16 trace bins, one gain flag, and five spatial features), but the "
            f"dataset provides {features.shape[1]}. Regenerate/reprocess the dataset "
            "with the current pipeline."
        )
    if features.shape[0] != expected_pixels:
        raise ValueError(
            f"The selected camera has {expected_pixels} pixels, but the event has "
            f"{features.shape[0]} graph nodes. Select matching camera geometry."
        )
    for label in ("y_energy", "y_class"):
        if getattr(data, label, None) is None:
            raise ValueError(f"Dataset events must contain a `{label}` target tensor")


def load_checkpoint(torch, model, checkpoint: Path, *, model_name: str) -> None:
    """Load the plain state dictionaries written by ``train_gnn.py``."""
    checkpoint = Path(checkpoint).expanduser()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"{model_name} checkpoint not found: {checkpoint}")
    state_dict = torch.load(checkpoint, map_location="cpu", weights_only=True)
    try:
        model.load_state_dict(state_dict)
    except (TypeError, RuntimeError) as exc:
        raise RuntimeError(
            f"{checkpoint} is not a compatible {model_name} state dictionary. "
            "Use the separate energy_gnn.pt and class_gnn.pt files written by "
            "the current train_gnn.py."
        ) from exc

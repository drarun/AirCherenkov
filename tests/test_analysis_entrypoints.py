import importlib.util

import pytest

from analysis import evaluate_gnn, evaluate_gnns, event_display
from analysis._gnn_cli import OptionalDependencyError, load_runtime


@pytest.mark.parametrize(
    "entrypoint",
    [evaluate_gnn.main, evaluate_gnns.main, event_display.main],
)
def test_analysis_help_does_not_load_optional_runtime(entrypoint, monkeypatch):
    def fail_if_called(*args, **kwargs):
        raise AssertionError("optional runtime was loaded while rendering --help")

    monkeypatch.setattr(evaluate_gnns, "load_runtime", fail_if_called)
    monkeypatch.setattr(event_display, "load_runtime", fail_if_called)
    with pytest.raises(SystemExit) as exit_info:
        entrypoint(["--help"])
    assert exit_info.value.code == 0


def test_singular_evaluator_is_a_compatibility_wrapper(monkeypatch):
    sentinel = {"num_events": 3}
    monkeypatch.setattr(evaluate_gnn, "evaluate", lambda **kwargs: sentinel)
    assert evaluate_gnn.evaluate_gnn(dataset_root="example") is sentinel


def test_missing_torch_geometric_error_is_actionable():
    if importlib.util.find_spec("torch_geometric") is not None:
        pytest.skip("PyTorch Geometric is installed")
    with pytest.raises(OptionalDependencyError, match=r"pip install.*\[ml,viz\]"):
        load_runtime(metrics=False)


def test_current_gnn_output_contracts():
    pytest.importorskip("torch_geometric")
    import torch

    from recon.gnn import ClassGNN, EnergyGNN

    features = torch.zeros((3, 22), dtype=torch.float32)
    edge_index = torch.tensor(
        [[0, 1, 1, 2], [1, 0, 2, 1]],
        dtype=torch.long,
    )
    batch = torch.zeros(3, dtype=torch.long)

    assert EnergyGNN()(features, edge_index, batch).shape == (1, 1)
    assert ClassGNN()(features, edge_index, batch).shape == (1, 1)

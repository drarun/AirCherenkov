"""Compatibility wrapper for the plural two-checkpoint GNN evaluator.

The retired evaluator expected a combined multitask model. Current training
writes independent ``EnergyGNN`` and ``ClassGNN`` state dictionaries, so all
evaluation is implemented by :mod:`analysis.evaluate_gnns`.
"""

from __future__ import annotations

from analysis.evaluate_gnns import evaluate, main


def evaluate_gnn(*args, **kwargs):
    """Backward-compatible function name for :func:`evaluate_gnns.evaluate`."""
    return evaluate(*args, **kwargs)


__all__ = ["evaluate", "evaluate_gnn", "main"]


if __name__ == "__main__":
    raise SystemExit(main())

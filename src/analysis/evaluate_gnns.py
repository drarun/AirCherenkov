"""Evaluate the current independent EnergyGNN and ClassGNN checkpoints."""

from __future__ import annotations

import argparse
from pathlib import Path

from analysis._gnn_cli import (
    AddEdgeIndex,
    DEFAULT_CAMERA_RINGS,
    OptionalDependencyError,
    camera_edge_index,
    load_checkpoint,
    load_runtime,
    resolve_device,
    validate_dataset_root,
    validate_graph,
)


def evaluate(
    *,
    dataset_root: Path | str = Path("data/test"),
    energy_checkpoint: Path | str = Path("data/energy_gnn.pt"),
    class_checkpoint: Path | str = Path("data/class_gnn.pt"),
    output: Path | str = Path("data/evaluation.png"),
    batch_size: int = 128,
    device: str = "auto",
    camera_rings: int = DEFAULT_CAMERA_RINGS,
) -> dict[str, float | int | None | str]:
    """Evaluate separate regression and classification checkpoints.

    ``EnergyGNN`` predicts ``log10(energy / GeV)`` and is scored on gamma
    events, matching its training loss. ``ClassGNN`` returns a gamma logit and
    is scored on all events.
    """
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    if camera_rings < 0:
        raise ValueError("camera_rings must be non-negative")

    runtime = load_runtime(metrics=True)
    torch = runtime.torch
    np = runtime.numpy
    plt = runtime.pyplot

    # Imported after the optional dependency check so importing this module and
    # requesting --help remain available in core-only installations.
    from analysis.dataset import CherenkovDataset
    from recon.gnn import ClassGNN, EnergyGNN
    from sim.camera import Camera

    dataset_root = validate_dataset_root(Path(dataset_root))
    camera = Camera(n_rings=camera_rings)
    edge_index = camera_edge_index(torch, camera)
    dataset = CherenkovDataset(
        root=str(dataset_root),
        transform=AddEdgeIndex(edge_index),
    )
    if len(dataset) == 0:
        raise ValueError(f"Dataset contains no triggered camera graphs: {dataset_root}")
    validate_graph(dataset[0], expected_pixels=camera.n_pixels)

    selected_device = resolve_device(torch, device)
    energy_model = EnergyGNN()
    class_model = ClassGNN()
    load_checkpoint(
        torch,
        energy_model,
        Path(energy_checkpoint),
        model_name="EnergyGNN",
    )
    load_checkpoint(
        torch,
        class_model,
        Path(class_checkpoint),
        model_name="ClassGNN",
    )
    energy_model = energy_model.to(selected_device).eval()
    class_model = class_model.to(selected_device).eval()

    loader = runtime.data_loader(dataset, batch_size=batch_size, shuffle=False)
    true_energy_parts = []
    predicted_energy_parts = []
    true_class_parts = []
    predicted_class_parts = []

    with torch.inference_mode():
        for batch in loader:
            validate_graph(batch, expected_pixels=batch.x.shape[0])
            batch = batch.to(selected_device)
            energy_output = energy_model(batch.x, batch.edge_index, batch.batch)
            class_output = class_model(batch.x, batch.edge_index, batch.batch)

            true_energy_parts.append(batch.y_energy.detach().view(-1).cpu().numpy())
            predicted_energy_parts.append(
                energy_output.detach().view(-1).cpu().numpy()
            )
            true_class_parts.append(batch.y_class.detach().view(-1).cpu().numpy())
            predicted_class_parts.append(
                torch.sigmoid(class_output).detach().view(-1).cpu().numpy()
            )

    true_energy = np.concatenate(true_energy_parts)
    predicted_energy = np.concatenate(predicted_energy_parts)
    true_class = np.concatenate(true_class_parts)
    predicted_class = np.concatenate(predicted_class_parts)
    if not np.all(np.isin(true_class, [0.0, 1.0])):
        raise ValueError("ClassGNN evaluation requires binary y_class labels 0 or 1")

    gamma_mask = true_class == 1.0
    gamma_true_energy = true_energy[gamma_mask]
    gamma_predicted_energy = predicted_energy[gamma_mask]
    if gamma_true_energy.size:
        energy_rmse = float(
            np.sqrt(np.mean((gamma_true_energy - gamma_predicted_energy) ** 2))
        )
    else:
        energy_rmse = None

    class_values = np.unique(true_class)
    if class_values.size == 2:
        false_positive_rate, true_positive_rate, _ = runtime.roc_curve(
            true_class, predicted_class
        )
        roc_auc = float(runtime.roc_auc_score(true_class, predicted_class))
    else:
        false_positive_rate = true_positive_rate = None
        roc_auc = None

    figure, axes = plt.subplots(1, 2, figsize=(12, 5))
    if gamma_true_energy.size:
        histogram = axes[0].hist2d(
            gamma_true_energy,
            gamma_predicted_energy,
            bins=50,
            cmap="viridis",
        )
        figure.colorbar(histogram[3], ax=axes[0], label="Graphs")
        bounds = np.concatenate([gamma_true_energy, gamma_predicted_energy])
        lower = float(np.min(bounds))
        upper = float(np.max(bounds))
        if lower == upper:
            lower -= 0.05
            upper += 0.05
        axes[0].plot([lower, upper], [lower, upper], "r--", label="Ideal")
        axes[0].legend()
    else:
        axes[0].text(
            0.5,
            0.5,
            "No gamma events; energy metric unavailable",
            ha="center",
            va="center",
            transform=axes[0].transAxes,
        )
    axes[0].set_xlabel("True log10(Energy / GeV)")
    axes[0].set_ylabel("Predicted log10(Energy / GeV)")
    axes[0].set_title("Energy reconstruction (gamma only)")

    if roc_auc is not None:
        axes[1].plot(
            false_positive_rate,
            true_positive_rate,
            label=f"AUC = {roc_auc:.3f}",
        )
        axes[1].plot([0, 1], [0, 1], "k--")
        axes[1].legend()
    else:
        axes[1].text(
            0.5,
            0.5,
            "Both classes are required for ROC AUC",
            ha="center",
            va="center",
            transform=axes[1].transAxes,
        )
    axes[1].set_xlabel("False positive rate")
    axes[1].set_ylabel("True positive rate")
    axes[1].set_title("Gamma/hadron separation")

    output = Path(output).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(output, dpi=300)
    plt.close(figure)

    return {
        "num_events": int(true_class.size),
        "num_gamma_events": int(gamma_mask.sum()),
        "energy_rmse_log10_gev": energy_rmse,
        "roc_auc": roc_auc,
        "device": str(selected_device),
        "output": str(output),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="aircherenkov-evaluate",
        description=(
            "Evaluate an EnergyGNN regression checkpoint and a ClassGNN gamma/"
            "hadron checkpoint on a CherenkovDataset."
        ),
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("data/test"),
        help="PyG dataset root containing raw/ or processed/data_v2.pt (default: %(default)s)",
    )
    parser.add_argument(
        "--energy-checkpoint",
        type=Path,
        default=Path("data/energy_gnn.pt"),
        help="EnergyGNN state dictionary written by train_gnn.py (default: %(default)s)",
    )
    parser.add_argument(
        "--class-checkpoint",
        type=Path,
        default=Path("data/class_gnn.pt"),
        help="ClassGNN state dictionary written by train_gnn.py (default: %(default)s)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/evaluation.png"),
        help="Destination PNG (default: %(default)s)",
    )
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--camera-rings", type=int, default=DEFAULT_CAMERA_RINGS)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        metrics = evaluate(
            dataset_root=args.dataset_root,
            energy_checkpoint=args.energy_checkpoint,
            class_checkpoint=args.class_checkpoint,
            output=args.output,
            batch_size=args.batch_size,
            device=args.device,
            camera_rings=args.camera_rings,
        )
    except (FileNotFoundError, OptionalDependencyError, RuntimeError, ValueError) as exc:
        parser.exit(2, f"{parser.prog}: error: {exc}\n")

    energy_text = (
        "unavailable"
        if metrics["energy_rmse_log10_gev"] is None
        else f"{metrics['energy_rmse_log10_gev']:.4f}"
    )
    auc_text = (
        "unavailable" if metrics["roc_auc"] is None else f"{metrics['roc_auc']:.4f}"
    )
    print(
        f"Evaluated {metrics['num_events']} graphs on {metrics['device']}; "
        f"gamma energy RMSE (log10 GeV)={energy_text}, ROC AUC={auc_text}."
    )
    print(f"Saved evaluation plot to {metrics['output']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

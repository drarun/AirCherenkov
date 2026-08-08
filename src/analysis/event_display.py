"""Render one camera graph with current energy and class predictions."""

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


def render_event(
    *,
    dataset_root: Path | str = Path("data/test"),
    event_index: int = 0,
    energy_checkpoint: Path | str = Path("data/energy_gnn.pt"),
    class_checkpoint: Path | str = Path("data/class_gnn.pt"),
    output: Path | str = Path("data/event_display.png"),
    device: str = "auto",
    camera_rings: int = DEFAULT_CAMERA_RINGS,
) -> dict[str, float | int | str]:
    """Run both supported models and save a display for one camera graph."""
    if event_index < 0:
        raise ValueError("event_index must be non-negative")
    if camera_rings < 0:
        raise ValueError("camera_rings must be non-negative")

    runtime = load_runtime(metrics=False)
    torch = runtime.torch
    plt = runtime.pyplot

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
    if event_index >= len(dataset):
        raise ValueError(
            f"event_index {event_index} is outside a dataset containing "
            f"{len(dataset)} camera graphs"
        )

    data = dataset[event_index]
    validate_graph(data, expected_pixels=camera.n_pixels)
    # The current feature contract stores 16 FADC samples first.  Summing those
    # samples displays the integrated trace actually consumed by both models.
    image_amplitudes = data.x[:, :16].detach().sum(dim=1).cpu().numpy()

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
    data = data.to(selected_device)
    batch = torch.zeros(data.x.shape[0], dtype=torch.long, device=selected_device)

    with torch.inference_mode():
        predicted_log_energy = energy_model(
            data.x, data.edge_index, batch
        ).view(-1)[0]
        class_logit = class_model(data.x, data.edge_index, batch).view(-1)[0]

    predicted_gamma_probability = float(torch.sigmoid(class_logit).cpu())
    predicted_energy_gev = float(torch.pow(10.0, predicted_log_energy).cpu())
    true_log_energy = data.y_energy.detach().view(-1)[0]
    true_energy_gev = float(torch.pow(10.0, true_log_energy).cpu())
    true_class = float(data.y_class.detach().view(-1)[0].cpu())
    class_name = "Gamma" if true_class == 1.0 else "Proton"

    figure, axis = plt.subplots(figsize=(10, 8))
    title = (
        "Simulated event display\n"
        f"True class: {class_name} | Predicted gamma probability: "
        f"{predicted_gamma_probability:.1%}\n"
        f"True energy: {true_energy_gev:.2f} GeV | "
        f"Gamma-trained energy estimate: {predicted_energy_gev:.2f} GeV"
    )
    camera.plot_image(image_amplitudes, ax=axis, title=title)

    output = Path(output).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(output, dpi=150)
    plt.close(figure)

    return {
        "event_index": int(event_index),
        "true_class": class_name.lower(),
        "true_energy_gev": true_energy_gev,
        "predicted_energy_gev": predicted_energy_gev,
        "predicted_gamma_probability": predicted_gamma_probability,
        "device": str(selected_device),
        "output": str(output),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="aircherenkov-event-display",
        description=(
            "Render one camera graph using the separate EnergyGNN and ClassGNN "
            "checkpoints written by train_gnn.py."
        ),
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("data/test"),
        help="PyG dataset root containing raw/ or processed/data_v2.pt (default: %(default)s)",
    )
    parser.add_argument("--event-index", type=int, default=0)
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
        default=Path("data/event_display.png"),
        help="Destination PNG (default: %(default)s)",
    )
    parser.add_argument("--camera-rings", type=int, default=DEFAULT_CAMERA_RINGS)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        result = render_event(
            dataset_root=args.dataset_root,
            event_index=args.event_index,
            energy_checkpoint=args.energy_checkpoint,
            class_checkpoint=args.class_checkpoint,
            output=args.output,
            device=args.device,
            camera_rings=args.camera_rings,
        )
    except (FileNotFoundError, OptionalDependencyError, RuntimeError, ValueError) as exc:
        parser.exit(2, f"{parser.prog}: error: {exc}\n")

    print(
        f"Rendered graph {result['event_index']} on {result['device']}: "
        f"P(gamma)={result['predicted_gamma_probability']:.1%}, "
        f"energy estimate={result['predicted_energy_gev']:.2f} GeV."
    )
    print(f"Saved event display to {result['output']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

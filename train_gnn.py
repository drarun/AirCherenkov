"""Train energy-regression and particle-classification camera GNNs."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch


PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


def _group_key(value):
    """Normalize a scalar event identifier without requiring PyG."""
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise ValueError("each graph must contain exactly one event_id")
        return value.detach().cpu().item()
    if hasattr(value, "item"):
        try:
            return value.item()
        except ValueError:
            pass
    return value


def event_grouped_split_indices(event_ids, train_fraction=0.8, seed=0):
    """Split graph indices while keeping every view of a shower together.

    The returned membership is deterministic for a given ordered collection of
    event IDs and seed. At least two distinct shower IDs are required so both
    partitions contain a complete event group.
    """
    if not 0.0 < float(train_fraction) < 1.0:
        raise ValueError("train_fraction must lie strictly between zero and one")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer")

    normalized = [_group_key(event_id) for event_id in event_ids]
    groups = list(dict.fromkeys(normalized))
    if len(groups) < 2:
        raise ValueError(
            "event-grouped train/validation splitting requires at least two showers"
        )

    generator = torch.Generator().manual_seed(seed)
    order = torch.randperm(len(groups), generator=generator).tolist()
    train_group_count = round(float(train_fraction) * len(groups))
    train_group_count = max(1, min(len(groups) - 1, train_group_count))
    train_groups = {groups[index] for index in order[:train_group_count]}

    train_indices = [
        index for index, event_id in enumerate(normalized) if event_id in train_groups
    ]
    validation_indices = [
        index for index, event_id in enumerate(normalized) if event_id not in train_groups
    ]
    return train_indices, validation_indices


def _dataset_event_ids(dataset):
    event_ids = []
    for index in range(len(dataset)):
        sample = dataset[index]
        if not hasattr(sample, "event_id"):
            raise RuntimeError(
                "processed graphs do not contain event_id; remove the old processed "
                "dataset so it can be rebuilt"
            )
        event_ids.append(_group_key(sample.event_id))
    return event_ids


def train_networks(
    *,
    data_root="data/train",
    seed=0,
    train_fraction=0.8,
    batch_size=128,
    epochs=25,
    output_dir="data",
):
    # Keep torch-geometric optional for users of the simulator and visualization
    # packages. Import the ML stack only when training is requested.
    try:
        from torch_geometric.loader import DataLoader
        from torch.utils.data import Subset
        from analysis.dataset import CherenkovDataset
        from recon.gnn import ClassGNN, EnergyGNN
        from sim.camera import Camera
    except ImportError as exc:
        raise RuntimeError(
            "GNN training requires the ML dependencies; install aircherenkov[ml]"
        ) from exc

    if batch_size <= 0:
        raise ValueError("batch_size must be greater than zero")
    if epochs <= 0:
        raise ValueError("epochs must be greater than zero")

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    print("Loading dataset...")
    camera = Camera(n_rings=12)
    position = torch.stack(
        (
            torch.tensor(camera.pixel_x, dtype=torch.float32),
            torch.tensor(camera.pixel_y, dtype=torch.float32),
        ),
        dim=1,
    )
    distance = torch.cdist(position, position)
    adjacency = (distance > 0.01) & (distance < 0.105)
    edge_index = adjacency.nonzero(as_tuple=False).t().contiguous()

    class AddEdgeIndex:
        def __init__(self, edges):
            self.edges = edges

        def __call__(self, data):
            data.edge_index = self.edges
            return data

    dataset = CherenkovDataset(
        root=str(data_root), pre_transform=AddEdgeIndex(edge_index)
    )
    print(f"Dataset loaded with {len(dataset)} telescope graphs.")
    if len(dataset) == 0:
        print("No events found. Please run the training-data generator first.")
        return None

    event_ids = _dataset_event_ids(dataset)
    train_indices, validation_indices = event_grouped_split_indices(
        event_ids, train_fraction=train_fraction, seed=seed
    )
    train_data = Subset(dataset, train_indices)
    validation_data = Subset(dataset, validation_indices)
    print(
        f"Grouped split: {len(set(event_ids[index] for index in train_indices))} "
        f"training showers/{len(train_indices)} views; "
        f"{len(set(event_ids[index] for index in validation_indices))} "
        f"validation showers/{len(validation_indices)} views."
    )

    loader_generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        generator=loader_generator,
    )
    validation_loader = DataLoader(
        validation_data, batch_size=batch_size, shuffle=False
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    energy_model = EnergyGNN().to(device)
    class_model = ClassGNN().to(device)
    optimizer_energy = torch.optim.Adam(energy_model.parameters(), lr=0.001)
    optimizer_class = torch.optim.Adam(class_model.parameters(), lr=0.001)
    criterion_energy = torch.nn.MSELoss()
    criterion_class = torch.nn.BCEWithLogitsLoss()

    print(f"\nTraining on {device} for {epochs} epochs...")
    for epoch in range(epochs):
        energy_model.train()
        class_model.train()
        training_energy_loss = 0.0
        training_class_loss = 0.0
        training_energy_batches = 0

        for batch in train_loader:
            batch = batch.to(device)

            optimizer_class.zero_grad()
            class_logits = class_model(batch.x, batch.edge_index, batch.batch)
            class_loss = criterion_class(
                class_logits.view(-1), batch.y_class.view(-1)
            )
            class_loss.backward()
            torch.nn.utils.clip_grad_norm_(class_model.parameters(), max_norm=2.0)
            optimizer_class.step()
            training_class_loss += class_loss.item()

            gamma_mask = batch.y_class.view(-1) == 1.0
            if gamma_mask.any():
                optimizer_energy.zero_grad()
                energy_prediction = energy_model(
                    batch.x, batch.edge_index, batch.batch
                )
                energy_loss = criterion_energy(
                    energy_prediction.view(-1)[gamma_mask],
                    batch.y_energy.view(-1)[gamma_mask],
                )
                energy_loss.backward()
                torch.nn.utils.clip_grad_norm_(energy_model.parameters(), max_norm=2.0)
                optimizer_energy.step()
                training_energy_loss += energy_loss.item()
                training_energy_batches += 1

        energy_model.eval()
        class_model.eval()
        validation_energy_loss = 0.0
        validation_class_loss = 0.0
        validation_energy_batches = 0
        with torch.no_grad():
            for batch in validation_loader:
                batch = batch.to(device)
                class_logits = class_model(batch.x, batch.edge_index, batch.batch)
                validation_class_loss += criterion_class(
                    class_logits.view(-1), batch.y_class.view(-1)
                ).item()
                gamma_mask = batch.y_class.view(-1) == 1.0
                if gamma_mask.any():
                    energy_prediction = energy_model(
                        batch.x, batch.edge_index, batch.batch
                    )
                    validation_energy_loss += criterion_energy(
                        energy_prediction.view(-1)[gamma_mask],
                        batch.y_energy.view(-1)[gamma_mask],
                    ).item()
                    validation_energy_batches += 1

        train_energy_mean = (
            training_energy_loss / training_energy_batches
            if training_energy_batches else float("nan")
        )
        validation_energy_mean = (
            validation_energy_loss / validation_energy_batches
            if validation_energy_batches else float("nan")
        )
        print(
            f"Epoch {epoch + 1}/{epochs} | "
            f"class train/val: {training_class_loss / len(train_loader):.4f}/"
            f"{validation_class_loss / len(validation_loader):.4f} | "
            f"energy train/val: {train_energy_mean:.4f}/{validation_energy_mean:.4f}"
        )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    energy_path = output_dir / "energy_gnn.pt"
    class_path = output_dir / "class_gnn.pt"
    torch.save(energy_model.state_dict(), energy_path)
    torch.save(class_model.state_dict(), class_path)
    print(f"\nTraining complete. Saved {energy_path} and {class_path}.")
    return energy_path, class_path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="data/train")
    parser.add_argument("--output-dir", default="data")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--train-fraction", type=float, default=0.8)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=25)
    args = parser.parse_args(argv)
    return train_networks(
        data_root=args.data_root,
        output_dir=args.output_dir,
        seed=args.seed,
        train_fraction=args.train_fraction,
        batch_size=args.batch_size,
        epochs=args.epochs,
    )


if __name__ == "__main__":
    main()

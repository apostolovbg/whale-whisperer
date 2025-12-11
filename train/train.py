from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from src.data_pipeline import (
    build_context_windows,
    build_coda_tensor,
    load_dialogue_dataframe,
    normalize_tensor,
)
from src.model import CodaLanguageModel


class CodaDialogueDataset(Dataset):
    """Wraps the dialogue data into context windows for training."""

    def __init__(
        self,
        window_size: int = 4,
        max_icis: int = 20,
    ) -> None:
        df = load_dialogue_dataframe()
        raw_features = build_coda_tensor(df, max_icis=max_icis)
        normalized = normalize_tensor(raw_features)
        windows = build_context_windows(
            df, window_size=window_size, max_icis=max_icis, precomputed_features=normalized
        )
        self.contexts = windows["contexts"]
        self.mask = windows["context_mask"]
        self.targets = windows["targets"]
        self.feature_dim = self.targets.shape[-1]

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        context = torch.from_numpy(self.contexts[idx])
        mask = torch.from_numpy(self.mask[idx])
        target = torch.from_numpy(self.targets[idx])
        return context, mask, target


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a basic coda language model.")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--window", type=int, default=4)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default="cpu")
    return parser.parse_args()


def train(args: argparse.Namespace) -> None:
    dataset = CodaDialogueDataset(window_size=args.window)
    loader = DataLoader(dataset, batch_size=args.batch, shuffle=True)

    model = CodaLanguageModel(dataset.feature_dim, hidden_dim=args.hidden)
    model.to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    loss_fn = nn.MSELoss()

    for epoch in range(1, args.epochs + 1):
        model.train()
        accumulated = 0.0
        for contexts, masks, targets in loader:
            contexts = contexts.to(args.device)
            masks = masks.to(args.device)
            targets = targets.to(args.device)

            optimizer.zero_grad()
            pred, _ = model(contexts, masks)
            loss = loss_fn(pred, targets)
            loss.backward()
            optimizer.step()
            accumulated += loss.item() * targets.size(0)
        avg_loss = accumulated / len(dataset)
        print(f"Epoch {epoch}/{args.epochs} · MSE {avg_loss:.4f}")

    save_path = Path("artifacts") / "coda_model.pt"
    save_path.parent.mkdir(exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print("Saved trained model to", save_path)


if __name__ == "__main__":
    args = parse_args()
    train(args)

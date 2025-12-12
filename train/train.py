from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader

from src.datasets import CodaDialogueDataset
from src.model import CodaLanguageModel


def compute_contrastive_loss(
    context_embeddings: torch.Tensor,
    target_embeddings: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    logits = torch.matmul(context_embeddings, target_embeddings.transpose(0, 1)) / temperature
    labels = torch.arange(logits.size(0), device=logits.device)
    return F.cross_entropy(logits, labels)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a basic coda language model.")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--window", type=int, default=4)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--contrastive-weight", type=float, default=0.0)
    parser.add_argument("--temperature", type=float, default=0.1)
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
        total_mse = 0.0
        total_contrastive = 0.0
        for contexts, masks, targets in loader:
            contexts = contexts.to(args.device)
            masks = masks.to(args.device)
            targets = targets.to(args.device)

            optimizer.zero_grad()
            prediction, context_embedding = model(contexts, masks)
            mse_loss = loss_fn(prediction, targets)
            total_loss = mse_loss
            if args.contrastive_weight > 0.0:
                target_embedding = model.encoder(targets.unsqueeze(1)).squeeze(1)
                contrastive_loss = compute_contrastive_loss(
                    context_embedding, target_embedding, args.temperature
                )
                total_loss = mse_loss + args.contrastive_weight * contrastive_loss
                total_contrastive += contrastive_loss.item() * targets.size(0)

            total_loss.backward()
            optimizer.step()
            total_mse += mse_loss.item() * targets.size(0)

        avg_mse = total_mse / len(dataset)
        message = f"Epoch {epoch}/{args.epochs} · MSE {avg_mse:.4f}"
        if args.contrastive_weight > 0.0:
            avg_contrastive = total_contrastive / len(dataset)
            message += f" · Contrastive {avg_contrastive:.4f}"
        print(message)

    save_path = Path("artifacts") / "coda_model.pt"
    save_path.parent.mkdir(exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print("Saved trained model to", save_path)


if __name__ == "__main__":
    args = parse_args()
    train(args)

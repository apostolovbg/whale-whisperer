from __future__ import annotations

import argparse
from pathlib import Path

import torch

from src.datasets import CodaDialogueDataset
from src.model import CodaLanguageModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Infer the next coda from context.")
    parser.add_argument(
        "--weights",
        type=Path,
        default=Path("artifacts/coda_model.pt"),
        help="Path to the trained model checkpoint.",
    )
    parser.add_argument("--index", type=int, default=-1, help="Dialogue index to inspect.")
    parser.add_argument("--window", type=int, default=4, help="Context window size.")
    parser.add_argument("--hidden", type=int, default=128, help="Model hidden dimension.")
    parser.add_argument("--device", type=str, default="cpu", help="Device for inference.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = CodaDialogueDataset(window_size=args.window)
    model = CodaLanguageModel(dataset.feature_dim, hidden_dim=args.hidden)
    if not args.weights.exists():
        raise FileNotFoundError(f"Model weights not found at {args.weights}")
    model.load_state_dict(torch.load(args.weights, map_location=args.device))
    model.to(args.device)
    model.eval()

    index = args.index
    if index < 0:
        index = len(dataset) + index
    index = max(0, min(len(dataset) - 1, index))

    context, mask, target = dataset[index]
    with torch.no_grad():
        prediction, _ = model(
            context.unsqueeze(0).to(args.device),
            mask.unsqueeze(0).to(args.device),
        )
    prediction = prediction.squeeze(0).cpu().numpy()
    prediction = dataset.denormalize(prediction)
    target = target.numpy()
    target = dataset.denormalize(target)

    print(f"Context index: {index}")
    print("Predicted coda (first 12 values):")
    print(prediction[:12])
    print("Actual target (first 12 values):")
    print(target[:12])
    print("Difference (pred-actual) first 12 values:")
    print(prediction[:12] - target[:12])
    print("Duration difference:", prediction[-2] - target[-2])
    print("Clicks difference:", prediction[-1] - target[-1])


if __name__ == "__main__":
    main()

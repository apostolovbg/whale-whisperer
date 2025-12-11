from __future__ import annotations

import torch
from torch import nn


class CodaEncoder(nn.Module):
    """Encode a single coda feature vector into a dense representation."""

    def __init__(self, feature_dim: int, hidden_dim: int = 128) -> None:
        super().__init__()
        self._proj = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """features: (batch, window, feature_dim) or (batch, feature_dim)"""
        return self._proj(features)


class CodaLanguageModel(nn.Module):
    """Simple context-to-next-coda model with an embedding and regression head."""

    def __init__(self, feature_dim: int, hidden_dim: int = 128) -> None:
        super().__init__()
        self.encoder = CodaEncoder(feature_dim, hidden_dim)
        self.context_key = nn.Linear(hidden_dim, hidden_dim)
        self.pred_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, feature_dim),
        )
        self.embed_head = nn.Linear(hidden_dim, hidden_dim)

    def forward(
        self,
        context_windows: torch.Tensor,
        context_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            context_windows: (batch, window_size, feature_dim)
            context_mask: (batch, window_size) boolean mask with True for real entries
        Returns:
            prediction: (batch, feature_dim)
            embedding: (batch, hidden_dim) normalized context summary
        """
        encoded = self.encoder(context_windows)  # (batch, window, hidden)
        mask = context_mask.unsqueeze(-1).to(encoded.dtype)
        summed = (encoded * mask).sum(dim=1)
        lengths = mask.sum(dim=1).clamp(min=1.0)
        context_repr = summed / lengths
        context_repr = self.context_key(context_repr)
        prediction = self.pred_head(context_repr)
        embedding = self.embed_head(context_repr)
        embedding = nn.functional.normalize(embedding, dim=-1)
        return prediction, embedding

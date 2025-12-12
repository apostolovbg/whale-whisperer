from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset

from src.data_pipeline import (
    NormalizationStats,
    apply_normalization,
    build_context_windows,
    build_coda_tensor,
    compute_normalization_stats,
    denormalize_tensor,
    load_dialogue_dataframe,
)


class CodaDialogueDataset(Dataset):
    """Wrap the published dialogue corpus into context windows."""

    def __init__(
        self,
        window_size: int = 4,
        max_icis: int = 20,
    ) -> None:
        df = load_dialogue_dataframe()
        raw_features = build_coda_tensor(df, max_icis=max_icis)
        self._stats = compute_normalization_stats(raw_features)
        normalized = apply_normalization(raw_features, self._stats)
        windows = build_context_windows(
            df,
            window_size=window_size,
            max_icis=max_icis,
            precomputed_features=normalized,
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

    @property
    def normalization_stats(self) -> NormalizationStats:
        return self._stats

    def denormalize(self, tensor: np.ndarray) -> np.ndarray:
        """Bring normalized vectors back to their original scale."""
        return denormalize_tensor(tensor, self._stats)

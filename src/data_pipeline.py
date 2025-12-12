from dataclasses import dataclass
import pickle
from pathlib import Path
from typing import Any, Dict, Sequence, Tuple

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "sw-combinatoriality" / "data"


def load_dialogue_dataframe(path: Path | str | None = None) -> pd.DataFrame:
    """Return the raw dialogues table with numeric conversions."""
    path = DATA_DIR / "sperm-whale-dialogues.csv" if path is None else Path(path)
    df = pd.read_csv(path)
    df.rename(columns=lambda col: col.strip(), inplace=True)
    for col in ("nClicks", "Duration", "TsTo"):
        if col in df.columns:
            df[col] = df[col].astype(float)
    return df


def load_pickled(name: str, path: Path | str | None = None) -> Any:
    """Load a pickle from the data directory."""
    path = (DATA_DIR / name) if path is None else Path(path)
    with open(path, "rb") as handle:
        return pickle.load(handle)


def load_rhythms() -> np.ndarray:
    """Return the rhythm prototypes as a float32 array."""
    data = load_pickled("rhythms.p")
    return np.asarray(data, dtype=np.float32)


def load_ornaments() -> np.ndarray:
    """Return the ornamentation vectors as a float32 array."""
    data = load_pickled("ornaments.p")
    return np.asarray(data, dtype=np.float32)


def load_tempos() -> Dict[Any, Any]:
    """Return the discrete tempo dictionary."""
    return load_pickled("tempos-dict.p")


def pad_icis(df: pd.DataFrame, max_icis: int = 20) -> np.ndarray:
    """Return zero-padded ICIs through the first `max_icis` columns."""
    ici_cols = [f"ICI{i}" for i in range(1, max_icis + 1)]
    present = [col for col in ici_cols if col in df.columns]
    icis = df[present].fillna(0.0).to_numpy(dtype=np.float32)
    if len(present) < max_icis:
        padding = np.zeros((len(df), max_icis - len(present)), dtype=np.float32)
        icis = np.concatenate([icis, padding], axis=1)
    return icis


def build_coda_tensor(
    df: pd.DataFrame,
    max_icis: int = 20,
    include_duration: bool = True,
    include_n_clicks: bool = True,
) -> np.ndarray:
    """Stack ICIs, duration and click counts into a single tensor."""
    icis = pad_icis(df, max_icis)
    extras: list[np.ndarray] = []
    if include_duration and "Duration" in df.columns:
        extras.append(df["Duration"].to_numpy(dtype=np.float32).reshape(-1, 1))
    if include_n_clicks and "nClicks" in df.columns:
        extras.append(df["nClicks"].to_numpy(dtype=np.float32).reshape(-1, 1))
    if extras:
        matrix = np.concatenate([icis, *extras], axis=1)
    else:
        matrix = icis
    return matrix


def build_context_windows(
    df: pd.DataFrame,
    window_size: int = 4,
    max_icis: int = 20,
    precomputed_features: np.ndarray | None = None,
) -> Dict[str, np.ndarray]:
    """
    Turn the dialogue into fixed-length context windows.

    The returned dict contains `contexts` (N x window_size x features),
    `context_mask` (N x window_size) and `targets` (N x features).
    """
    if precomputed_features is None:
        features = build_coda_tensor(df, max_icis)
    else:
        features = np.asarray(precomputed_features, dtype=np.float32)
    num_rows, feature_dim = features.shape
    contexts = np.zeros((num_rows, window_size, feature_dim), dtype=np.float32)
    mask = np.zeros((num_rows, window_size), dtype=bool)
    for idx in range(num_rows):
        start = max(0, idx - window_size)
        chunk = features[start:idx]
        if chunk.size:
            shift = window_size - chunk.shape[0]
            contexts[idx, shift:] = chunk
            mask[idx, shift:] = True
    whale_ids = df["Whale"].astype(str).to_numpy()
    ts_to_next = df["TsTo"].to_numpy(dtype=np.float32) if "TsTo" in df.columns else np.zeros(num_rows, dtype=np.float32)
    return {
        "contexts": contexts,
        "context_mask": mask,
        "targets": features,
        "whale_ids": whale_ids,
        "ts_to_next": ts_to_next,
    }


@dataclass
class NormalizationStats:
    mean: np.ndarray
    std: np.ndarray


def compute_normalization_stats(tensor: np.ndarray) -> NormalizationStats:
    """Return per-feature mean/std for a tensor."""
    mean = tensor.mean(axis=0, keepdims=True)
    std = tensor.std(axis=0, keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)
    return NormalizationStats(mean=mean.astype(np.float32), std=std.astype(np.float32))


def apply_normalization(
    tensor: np.ndarray, stats: NormalizationStats
) -> np.ndarray:
    """Standardize using pre-computed stats."""
    return ((tensor - stats.mean) / stats.std).astype(np.float32)


def denormalize_tensor(
    tensor: np.ndarray, stats: NormalizationStats
) -> np.ndarray:
    """Invert normalization back to the original scale."""
    return (tensor * stats.std + stats.mean).astype(np.float32)

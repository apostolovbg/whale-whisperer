import sys
from pathlib import Path
import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.data_pipeline import (
    apply_normalization,
    build_coda_tensor,
    build_context_windows,
    compute_normalization_stats,
    denormalize_tensor,
    load_dialogue_dataframe,
    pad_icis,
)


def main() -> None:
    df = load_dialogue_dataframe()
    print("Loaded dialogues:", df.shape)

    coda_tensor = build_coda_tensor(df)
    print("Coda tensor shape:", coda_tensor.shape)

    norm_stats = compute_normalization_stats(coda_tensor)
    normalized = apply_normalization(coda_tensor, norm_stats)
    if not (coda_tensor.shape == normalized.shape):
        raise RuntimeError("Normalized tensor shape mismatch.")
    # Round-trip test
    restored = denormalize_tensor(normalized, norm_stats)
    if not np.allclose(restored, coda_tensor, atol=1e-6):
        raise RuntimeError("Denormalization failed to recover the original tensor.")

    context_structure = build_context_windows(df, window_size=4, precomputed_features=normalized)
    print("Context windows shape:", context_structure["contexts"].shape)
    print("Context mask shape:", context_structure["context_mask"].shape)

    padded = pad_icis(df)
    if padded.dtype != normalized.dtype:
        print("Warning: different dtype for padded ICIs.")

    nan_count = (padded != padded).sum()
    if nan_count:
        raise RuntimeError(f"Found {nan_count} NaNs in padded ICIs.")
    print("All padded ICIs are finite.")

    print("Verification complete.")


if __name__ == "__main__":
    main()

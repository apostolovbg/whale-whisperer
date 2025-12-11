import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.data_pipeline import (
    build_coda_tensor,
    build_context_windows,
    load_dialogue_dataframe,
    pad_icis,
    normalize_tensor,
)


def main() -> None:
    df = load_dialogue_dataframe()
    print("Loaded dialogues:", df.shape)

    coda_tensor = build_coda_tensor(df)
    print("Coda tensor shape:", coda_tensor.shape)

    normalized = normalize_tensor(coda_tensor)
    if not (coda_tensor.shape == normalized.shape):
        raise RuntimeError("Normalized tensor shape mismatch.")

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

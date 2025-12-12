from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path
from statistics import mean

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.data_pipeline import load_dialogue_dataframe


def summarize_rubato(df):
    """Compute average drift for consecutive codas from the same whale."""
    df = df.sort_values(["REC", "TsTo"])
    drifts = []
    for (rec, whale), subset in df.groupby(["REC", "Whale"]):
        durations = subset["Duration"].to_numpy()
        if len(durations) < 2:
            continue
        diffs = abs(durations[1:] - durations[:-1])
        drifts.extend(diffs.tolist())
    return drifts


def random_differences(df, sample=1000):
    durations = df["Duration"].to_list()
    diffs = []
    for _ in range(sample):
        a, b = random.sample(durations, 2)
        diffs.append(abs(a - b))
    return diffs


def main():
    parser = argparse.ArgumentParser(description="Summarize rubato statistics.")
    parser.add_argument("--samples", type=int, default=1000, help="random pair count")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    random.seed(args.seed)
    df = load_dialogue_dataframe()

    observed = summarize_rubato(df)
    random_diffs = random_differences(df, sample=args.samples)

    print(f"Observed same-whale drift (n={len(observed)}): mean {mean(observed):.3f}s")
    print(f"Random two-coda drift  (n={len(random_diffs)}): mean {mean(random_diffs):.3f}s")

    ratio = mean(observed) / mean(random_diffs) if random_diffs else float("nan")
    print(f"Rubato drift is {ratio:.3f}× the random expectation.")


if __name__ == "__main__":
    main()

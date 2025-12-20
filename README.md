# Whale Whisperer

# Whale Whisperer builds on the original *sw-combinatoriality* analysis; the raw dataset, reference notebooks, and the paper’s README now live under `sw-combinatoriality/` so we can treat them as archival material while building the AI components at the top level.

## Layout
- `sw-combinatoriality/`: the original repository (paper README, notebooks, `data/`, `code/`, `whalesbook/`, etc.). Treat this as the read-only historical source.
- `src/`: reusable preprocessing, dataset, and model definitions for the neural workflow.
- `train/`: entry-point (`train.py`) that trains the coda language model with optional contrastive objectives.
- `scripts/`: auxiliary tooling (`verify_data_pipeline.py`, `rubato_summary.py`, `infer_next_coda.py`) for sanity checks and quick analyses.
- `plans/`: the roadmap (`nn-roadmap.md`) outlining data choices, architecture direction, and verification steps.
- `AGENTS.md`: who is responsible for what in this repo.
- `artifacts/`: (created by training) keeps the saved PyTorch checkpoints.

## Getting started
1. **Verify the data pipeline** (makes sure context windows and normalization are correct):
   ```bash
   python3 scripts/verify_data_pipeline.py
   ```
2. **Inspect rubato drift**:
   ```bash
   python3 scripts/rubato_summary.py
   ```
   shows how same-whale tempo changes compare to random pairs using the published CSV.
3. **Install PyTorch** (must match your Python version; the default `python3 -m pip install torch` may fail for Python 3.13—grab the appropriate CPU wheel from https://download.pytorch.org/whl/cpu).
4. **Train the model** (example):
   ```bash
   python3 train/train.py --epochs 10 --batch 32
   ```
   adjust `--contrastive-weight` and `--temperature` once you have the checkpoint at `artifacts/coda_model.pt`.
5. **Run inference** on a trained checkpoint:
   ```bash
   python3 scripts/infer_next_coda.py
   ```
   inspects the last dialogue index by default and prints the predicted vs. true coda vectors.

## Notes
- All dataset code now loads from `sw-combinatoriality/data`. Keep any new processing inside `src/` or `train/` so the top-level project stays clean.
- The notebooks in `sw-combinatoriality/code/` still point to `../data`, so move them into that folder if you reopen them and they expect the original layout.
- Use `AGENTS.md` to coordinate future contributors; the standard change workflow is to edit `src/`, `train/`, or `scripts/` and keep `sw-combinatoriality/` as archival evidence of the published study.

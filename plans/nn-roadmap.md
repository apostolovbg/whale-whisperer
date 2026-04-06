## Neural roadmap for sperm whale vocalisations

### 1. Data representations
- **Derived rhythm/ornament/tempo vectors** (pickles under `data/`): 3,840 rhythm and ornament entries plus ~6 discrete tempo keys. Each entry is already a fixed-length list (4–8 floats) that encodes the combinatorial tweaks that the paper calls rhythm and ornamentation. These are ideal as initial tensors because they require almost no padding and are already aligned with the paper’s combinatorial story.
- **Raw Inter-click Intervals (ICIs)** from `data/DominicaCodas.csv`: each row is a single coda with up to nine inter-click intervals (`ICI1…ICI9`), the number of clicks, duration, clan labels, and metadata. Zero-pad to the same width (9 entries) and normalize per column to convert to a fixed-size tensor while keeping information about tempo via duration/nClicks.
- **Contextual dialogues** from `data/sperm-whale-dialogues.csv`: sequences of codas concatenated with speaker IDs and `TsTo` (pause until the next whale replies). Group these into short windows (e.g., last 4 codas) and mask unused slots to feed into architectures that can capture conversational flow.

### 2. AI objective and architecture
- **Primary objective**: learn a rich embedding space for codas that captures rhythm, ornamentation, and tempo variations while being sensitive to dialogue context. This can be staged as an embedding learning + context prediction task (contrastive loss or autoregressive regression).
- **Architecture sketch**:
  1. Input encoder: 1D CNN (or shallow Transformer) over the padded ICI sequence or a small MLP over the derived feature vectors to produce a per-coda embedding (`d=128`).
  2. Context module: sequence model (bi-LSTM/Transformer encoder) that ingests a window of the last `k` encoded codas (optionally concatenated speaker info) to condition the prediction.
  3. Heads:
     - *Embedding head*: projects into a normalized space for contrastive learning (InfoNCE/MoCo style) so codas from the same dialogue or same coda type stay close.
     - *Prediction head*: regress the next coda’s rhythm+ornament vector (+tempo bin classification), allowing the model to generate the next “click pattern” given recent context.

### 3. Training loop and reuse of notebooks
- Build a deterministic data pipeline inspired by the notebooks:
  - Extract the data-loading cells from e.g., `code/4-rubato.ipynb` into reusable functions (`load_dialogues()`, `load_ornaments()`, `build_tensors()`), placing them in `src/data_pipeline.py`.
  - Use these functions inside both analysis notebooks and the training script so plots and training share the same normalization and pairing logic.
- Training script idea (`train/train.py`):
  - Dataset class wraps the preprocessed tensors, samples windows of codas, and returns (context_embeddings, target_vector/tempo).
  - Use PyTorch Lightning or vanilla PyTorch with `DataLoader`, compute contrastive+regression losses, log via Matplotlib/seaborn plots (similar to how notebooks plot rubato drift).
  - Hooks for evaluation: project embeddings and visualize with UMAP to confirm combinatorial structure.

### 4. Immediate implementation steps
1. Encode the pickled rhythm/ornament/tempo lists as `(N, D)` NumPy arrays, normalize them, and cache them as `.npz` for fast training. Keep the notebook logic that parsed these pickles for reference.
2. Flesh out the model described above in a new `src/model.py` and a `train/train.py` that loops over torch batches and prints loss stats (reusing the notebook plots for validation).
3. Turn one notebook (e.g., `code/4-rubato.ipynb`) into a helper module so future scripts can import `get_dialogue_chunks()` and `plot_rubato_summary()` rather than copy-pasting cells.

### 5. Reuse and validation
- Keep the original notebooks as reference, but have them import from the new modules—this ensures the same normalization/constants power both the research plots and the neural model.
- Validation ingredients: data histograms (nClicks, duration, tempo bin counts), embedding similarity heatmaps, and loss curves saved each epoch, mirroring the notebook visualizations.

### 6. Verification checklist
1. Write a short script that imports `src.data_pipeline`, loads each dataset, prints shapes, and asserts no NaNs remain in the padded ICI columns.
2. During initial training, log training/validation losses and `tsne`/UMAP plots of the learned embeddings so you can compare them with the paper’s combinatorial categories.
3. Keep a `notebooks/` reference that imports the pipeline functions to reproduce one key figure (e.g., rubato drift), ensuring the neural project stays aligned with the published analysis.

# CondOT Codebase Instructions

## Project Overview

CondOT (Conditional Optimal Transport) predicts single-cell responses to drug perturbations via context-conditioned transport maps parameterized by PICNNs (Partially Input Convex Neural Networks). It trains dual networks (f: transport cost, g: transport map) in an adversarial loop, conditioned on drug/dose embeddings.

## Architecture & Key Modules

| Module | Purpose | Entry Points |
|--------|---------|-------------|
| `condot/` | Core model, training, data, inference | `condot/train/train.py`, `condot/models/condot.py` |
| `ccot/` | Evaluation metrics (DEGs-subset only) | `ccot/evaluate/evaluate.py:eval_fxn()` |
| `chemCPA/` | Baseline model (reference only) | — |
| `tmp/ccot_inference_kit/` | Self-contained inference wrapper | `ccot_wrapper.py` |
| `scripts/` | CLI entry points for train/eval | `train.py`, `eval.py` |

**Data flow**: Config YAML merge → AnnData h5ad load → train/test/ood split → control vs treated separation → infinite cycling iterators → alternating f/g optimization → checkpoint to `cache/` → eval via `ccot/evaluate/`.

## Developer Workflows

**Conda 环境**: 所有代码运行、调试、测试必须在 `cell` 环境下进行，激活命令: `conda activate cell`

```bash
# Setup (first time only)
conda create -n cell python=3.9.7 && conda activate cell
pip install -r requirements.txt && python setup.py develop

# Train (compose multiple YAML configs; CLI flags override)
python scripts/train.py \
  --config configs/condot.yaml --config configs/tasks/sciplex3-top1k.yaml \
  --config configs/experiments/ohe.yaml --config configs/projections/pca.yaml \
  --config.data.target givinostat --outdir ./results/my_exp

# Evaluate (same flags as train.py)
python scripts/eval.py --config configs/condot.yaml ... --outdir ./results/my_exp

# Debug: --dry prints resolved config; check {outdir}/config.yaml for actual values
# Monitor: tensorboard --logdir {outdir}/log/
```

**Output structure**: `{outdir}/cache/{best_model,last}.pt`, `{outdir}/eval/DEGs_summary_results.csv`, `{outdir}/log/events.out.tfevents.*`

## Project-Specific Patterns

### Config system (`condot/train/experiment.py`)
- Uses `ml-collections.ConfigDict` with dot-access (e.g., `config.model.kwargs.latent_dim`)
- Multiple `--config` YAML files deep-merged in order; CLI `--config.x.y value` overrides last
- Safe access: `config.get("key", default)` or `"key" in config`
- `--config.data.target` and `--split` are **mutually exclusive** (single drug vs split file)

### Dual-network training (`condot/train/train.py:train_condot()`)
- Inner loop: **g-network trains `config.training.n_inner_iters` steps (default 10), f-network 1 step**
- After each g-step: `g.convexify()` clamps Wz weights non-negative for convexity
- `check_loss()` raises `ValueError` on NaN → caught in `scripts/train.py` → status file set to "bugged"
- Best model selected by **minimum MMD** on validation split
- Async save via `threading.Thread(target=torch.save, ...)` with `.join()` to wait

### Network architecture (`condot/networks/`)
- **PICNN** (`picnn.py`): main network; config selects NPICNN if `model.kwargs.init_type` is set (rarely used)
- `g.transport(x, condition)` = gradient of convex potential w.r.t. x (via `torch.autograd.grad`)
- Condition injected via multiplicative gating: `z * softplus(wzu(u))`
- Activation: `CELU` (hardcoded slope), first layer uses squared activation

### Embedding types (`condot/networks/embeddings.py`)
- `"smiles"` / `"smiles_onehot"`: reads parquet embeddings → L2-normalized → registered as buffer (frozen)
- `"onehot"`: `OneHotEmbeddingSMILES` maps SMILES index to one-hot
- `"value"`: scalar dose embedding
- Embedding dim must match `model.kwargs.cond_dim`; see `load_networks()` in `condot/models/condot.py`

### Data loading (`condot/data/data_chemcpa.py`)
- `AnnDataDataset` wraps AnnData; precomputes all tensors in memory during `__init__` (no dynamic loading)
- Cell types **hardcoded**: A549→0, K562→1, MCF7→2 in `AnnDataDataset.cell_type_map`
- `InfiniteSampler` provides infinite iteration via `itertools.cycle` (no epoch boundaries)
- `align_by_cell_type=True` → uses `CombinedDataLoaderAlignedCellTypes` that creates 3 separate DataLoaders (one per cell type), each with `num_workers` processes
- `align_by_cell_type=False` → uses `CombinedDataLoaderUnalignedCellTypes` with single DataLoader
- **Performance note**: `align_by_cell_type=True` causes **serial** `next()` calls in `__next__()` (line 153-165), which is a CPU bottleneck when GPU is underutilized
- DataLoader params: `num_workers=8`, `pin_memory=True` set in `train.py:308`; `prefetch_factor` and `persistent_workers` not currently used but can be added to DataLoader initialization (line 95, 135)

### Classifier-Free Guidance & Adaptive Mass Transport
- CFG: `config.model.kwargs.cfg_rate` > 0 enables; training randomly drops condition with probability `cfg_rate`
- AMT: `config.model.kwargs.amt_alpha` > 0 enables transform matrix $(I + \alpha W^TW)^{-1}$

## Key Pitfalls

1. **`condot/valid/transport.py` is outdated** — actual transport logic (with CFG, AMT) is inlined in `train.py` and `evaluate.py`
2. **`requirements.txt` is incomplete** — missing runtime deps: `geomloss`, `pot`, `optuna`, `numba`, `rdkit`
3. **Embedding encoder state not saved** — `emb_encoder` save/restore is commented out in training code
4. **Cell type mapping hardcoded** — `data_chemcpa.py` only handles A549/K562/MCF7; new cell types need code changes
5. **Multi-GPU scripts** (`mutil_gpu.py`, `test_mutil_gpu.py`) are standalone tests — training itself is single-GPU
6. **CPU bottleneck in `align_by_cell_type=True` mode** — `CombinedDataLoaderAlignedCellTypes.__next__()` serially calls `next()` on 3 iterators (line 153-165 in `data_chemcpa.py`), causing GPU starvation when CPU usage is high; `torch.cat()` at line 174 performs CPU-side tensor concatenation before GPU transfer
7. **No memory leaks in data pipeline** — `AnnDataDataset` preloads all data once; `custom_collate` and `__next__` use local variables properly cleaned up by GC; safe to use `persistent_workers=True` with `num_workers > 0`

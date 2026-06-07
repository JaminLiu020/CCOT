from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from .metrics import compute_metrics_bundle


CELL_TYPE_TO_CODE = {
    "A549": torch.tensor([0.0]),
    "K562": torch.tensor([1.0]),
    "MCF7": torch.tensor([2.0]),
}

METRIC_COLUMNS = ["mmd", "l2_loss", "fid", "r2", "energy_distance"]


def _bool2idx(x):
    return np.where(x)[0]


def _to_pos_idx(obs_names, group_idx) -> np.ndarray:
    """Convert groupby indices (labels or positions) to positional int indices."""
    arr = np.asarray(group_idx)
    if np.issubdtype(arr.dtype, np.integer):
        return arr.astype(np.int64, copy=False)

    # Pandas groupby over adata.obs may return label indices (obs_names).
    pos = obs_names.get_indexer(arr.astype(str))
    return pos[pos >= 0].astype(np.int64, copy=False)


def _as_tensor(x, device, dtype=None):
    """Convert dataset values to torch tensors robustly.

    `model.get_dataset(...)` may return torch tensors, numpy arrays,
    scipy sparse matrices, or object arrays depending on package versions.
    """
    if torch.is_tensor(x):
        if dtype is not None:
            return x.to(device=device, dtype=dtype)
        return x.to(device)

    if hasattr(x, "toarray"):
        x = x.toarray()

    if isinstance(x, np.ndarray) and x.dtype == np.object_:
        x = np.asarray(x.tolist())

    return torch.as_tensor(x, dtype=dtype, device=device)


def _repeat_n(x, n, device):
    return _as_tensor(x, device).view(1, -1).repeat(n, 1)


def _get_deg_key(cell_type: str, condition: str) -> str:
    return f"{cell_type}_{condition}_1.0"


def _get_deg_index(adata, deg_key: str, degs_key_priority=("lincs_DEGs", "all_DEGs")):
    for key in degs_key_priority:
        if key in adata.uns and deg_key in adata.uns[key]:
            bool_de = adata.var_names.isin(np.array(adata.uns[key][deg_key]))
            return np.where(bool_de)[0]
    return np.array([], dtype=int)


def _build_control_subset(dataset_control: dict, cell_type: str):
    code = CELL_TYPE_TO_CODE[cell_type]
    idx = _bool2idx(dataset_control["cell_type"] == code)
    subset = {}
    layer = "X" if "X" in dataset_control else "layers"
    subset[layer] = dataset_control[layer][idx]
    subset["ind_x"] = dataset_control["ind_x"][idx]
    for k, v in dataset_control.items():
        if k in [layer, "ind_x"]:
            continue
        subset[k] = v[idx]
    return subset


def _predict_for_group(
    model,
    dataset_treated,
    group_idx: np.ndarray,
    control_subset: dict,
    sample_from_generative: bool = True,
    generator: torch.Generator | None = None,
):
    """Run biolord counterfactual prediction for a single (pert, cell_type) group.

    biolord's unknown-attribute embeddings collapse to ~0 during training (due
    to the L2 penalty + large injection noise), so `get_expression(...)` returns
    a `mus` tensor whose rows are effectively identical for all control cells
    that share the same known attributes. Using only `mus` would make the
    predicted "distribution" degenerate to a single point, penalizing biolord
    unfairly on MMD / FID / energy-distance.

    biolord is trained under a Gaussian likelihood (`gene_likelihood='normal'`),
    so `get_expression(...)` also returns a per-gene variance tensor. When
    `sample_from_generative=True`, we draw one sample per control cell from
    `Normal(mus, sqrt(variances))` — this is exactly what the model's generative
    distribution predicts the per-cell expression population to look like, and
    recovers per-cell variation for distribution-level metrics.
    """
    layer = "X" if "X" in dataset_treated else "layers"
    y_true = _as_tensor(dataset_treated[layer][group_idx, :], model.device, dtype=torch.float32)
    n_obs = control_subset[layer].shape[0]

    idx_ref = group_idx[0]
    feed = {
        layer: _as_tensor(control_subset[layer], model.device, dtype=torch.float32),
        "ind_x": _as_tensor(control_subset["ind_x"], model.device, dtype=torch.long),
    }
    for k in control_subset:
        if k in [layer, "ind_x"]:
            continue
        feed[k] = _repeat_n(dataset_treated[k][idx_ref, :], n_obs, model.device)

    mus, variances = model.module.get_expression(feed)
    mus = mus.detach()
    variances = variances.detach()
    if sample_from_generative:
        stds = torch.sqrt(torch.clamp(variances, min=0.0))
        if generator is not None:
            noise = torch.randn(mus.shape, generator=generator, device="cpu").to(mus.device)
        else:
            noise = torch.randn_like(mus)
        y_pred = mus + stds * noise
    else:
        y_pred = mus
    source = _as_tensor(control_subset[layer], model.device, dtype=torch.float32)
    return y_true, y_pred, source


def _condot_align_sizes(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    source: torch.Tensor,
    cap: int = 5000,
    generator: torch.Generator | None = None,
):
    """Replicate condot's size-alignment logic.

    1. If both source and target exceed `cap`, random-subsample both to `cap`
       (same indices on source/y_pred so the pairing is preserved).
    2. Then align source-size and target-size by random-subsampling the larger
       one down to the smaller.
    """
    device = y_true.device
    n_src = source.shape[0]
    n_tgt = y_true.shape[0]

    def _randperm(n: int) -> torch.Tensor:
        return torch.randperm(n, generator=generator, device="cpu").to(device)

    if n_src > cap and n_tgt > cap:
        src_idx = _randperm(n_src)[:cap]
        tgt_idx = _randperm(n_tgt)[:cap]
        source = source[src_idx]
        y_pred = y_pred[src_idx]
        y_true = y_true[tgt_idx]
        n_src = source.shape[0]
        n_tgt = y_true.shape[0]

    if n_src > n_tgt:
        idx = _randperm(n_src)[:n_tgt]
        source = source[idx]
        y_pred = y_pred[idx]
    elif n_tgt > n_src:
        idx = _randperm(n_tgt)[:n_src]
        y_true = y_true[idx]

    return y_true, y_pred, source


def evaluate_biolord_condot_aligned(
    model,
    adata,
    split_key: str,
    outdir: str,
    condition_col: str = "condition",
    subsample_cap: int = 5000,
    divide_by_constant_num_cell_types: bool = True,
    num_cell_types_for_divisor: int = 3,
    sample_from_generative: bool = True,
    seed: int = 42,
) -> Dict[str, pd.DataFrame]:
    """Evaluate a trained biolord model using condot-aligned logic.

    Parameters
    ----------
    subsample_cap : int
        Matches condot's hardcoded `5000` cap. Set to `0` to disable.
    divide_by_constant_num_cell_types : bool
        If True (default), per-pert metrics are divided by
        `num_cell_types_for_divisor` regardless of how many cell types actually
        contributed, matching condot exactly. If False, divides by the number
        of cell types with enough cells (biolord's original behaviour).
    sample_from_generative : bool
        If True (default), draw one sample per control cell from biolord's
        learned Gaussian output distribution `Normal(mus, sqrt(variances))`.
        This avoids the degenerate "single point" prediction caused by
        biolord's unknown-attribute collapse, and is what enables fair
        MMD/FID/ED comparison. Set to False to revert to the legacy
        attribute-only point prediction (you probably don't want this).
    seed : int
        Seed used for both the random subsampling and the generative sampling
        so that eval results are reproducible.
    """
    outdir = Path(outdir)
    eval_dir = outdir / "eval"
    data_dir = eval_dir / "data"
    eval_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))

    idx_test_control = np.where((adata.obs[split_key] == "test") & (adata.obs["control"] == 1))[0]
    adata_test_control = adata[idx_test_control].copy()
    dataset_control = model.get_dataset(adata_test_control)

    stage_defs = {
        "test": adata[(adata.obs[split_key] == "test") & (adata.obs["control"] == 0)].copy(),
        "ood": adata[(adata.obs[split_key] == "ood")].copy(),
    }

    per_stage = {}
    summary_rows = {}

    for stage, adata_stage in stage_defs.items():
        dataset_stage = model.get_dataset(adata_stage)
        groupby = adata_stage.obs.groupby([condition_col, "cell_type"], observed=True).groups
        pert_list = adata_stage.obs[condition_col].astype(str).unique().tolist()

        metrics_pert = pd.DataFrame(columns=METRIC_COLUMNS)

        for pert in tqdm(pert_list, desc=f"evaluating {stage}", ncols=100):
            agg = {k: 0.0 for k in METRIC_COLUMNS}
            valid_ct = 0

            for cell_type in ["A549", "K562", "MCF7"]:
                if (pert, cell_type) not in groupby:
                    continue

                group_idx = _to_pos_idx(adata_stage.obs_names, groupby[(pert, cell_type)])
                if len(group_idx) <= 5:
                    continue

                control_subset = _build_control_subset(dataset_control, cell_type)
                if control_subset[("X" if "X" in control_subset else "layers")].shape[0] == 0:
                    continue

                y_true, y_pred, source = _predict_for_group(
                    model,
                    dataset_stage,
                    group_idx,
                    control_subset,
                    sample_from_generative=sample_from_generative,
                    generator=generator,
                )

                deg_key = _get_deg_key(cell_type=cell_type, condition=str(pert))
                idx_de = _get_deg_index(adata, deg_key)
                if len(idx_de) < 2:
                    continue

                if subsample_cap and subsample_cap > 0:
                    y_true, y_pred, source = _condot_align_sizes(
                        y_true, y_pred, source,
                        cap=int(subsample_cap),
                        generator=generator,
                    )
                else:
                    min_len = min(y_true.shape[0], y_pred.shape[0], source.shape[0])
                    y_true = y_true[:min_len, :]
                    y_pred = y_pred[:min_len, :]
                    source = source[:min_len, :]

                y_true_de = y_true[:, idx_de]
                y_pred_de = y_pred[:, idx_de]
                source_de = source[:, idx_de]

                stage_deg_dir = data_dir / f"{stage}_DEGs" / str(pert)
                stage_deg_dir.mkdir(parents=True, exist_ok=True)
                np.savez(
                    stage_deg_dir / f"{cell_type}.npz",
                    pred=y_pred_de.detach().cpu().numpy(),
                    real=y_true_de.detach().cpu().numpy(),
                    source=source_de.detach().cpu().numpy(),
                )

                vals = compute_metrics_bundle(y_true_de, y_pred_de)
                for k in agg:
                    agg[k] += vals[k]
                valid_ct += 1

            if valid_ct > 0:
                divisor = (
                    int(num_cell_types_for_divisor)
                    if divide_by_constant_num_cell_types
                    else valid_ct
                )
                metrics_pert.loc[str(pert)] = {k: agg[k] / divisor for k in agg}

        per_stage[stage] = metrics_pert
        metrics_pert.to_csv(eval_dir / f"{stage}_DEGs_results.csv")
        summary_rows[f"{stage}_mean"] = metrics_pert.mean(numeric_only=True)
        summary_rows[f"{stage}_std"] = metrics_pert.std(numeric_only=True)

    summary = pd.DataFrame(summary_rows).T
    summary.to_csv(eval_dir / "DEGs_summary_results.csv")
    per_stage["summary"] = summary
    return per_stage

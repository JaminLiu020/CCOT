from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd
from anndata import AnnData
from sklearn.model_selection import train_test_split


def _to_dense(adata: AnnData) -> None:
    if hasattr(adata.X, "toarray"):
        adata.X = adata.X.toarray()


def build_or_validate_condot_split(
    adata: AnnData,
    *,
    split_key: str,
    condition_key: str,
    control_key: str,
    ood_drugs,
    random_seed: int,
    test_size: float,
) -> AnnData:
    """Ensure split column exists and follows CondOT train/test/ood convention.

    If `split_key` exists, keep it unchanged but validate required labels.
    Otherwise create a deterministic split aligned with CondOT logic:
    - OOD drugs -> "ood"
    - Remaining drugs -> train/test per-drug split with fixed seed.
    """
    adata = adata.copy()

    if split_key in adata.obs.columns:
        observed_labels = set(adata.obs[split_key].astype(str).unique())
        required = {"train", "test", "ood"}
        missing = required - observed_labels
        if missing:
            raise ValueError(
                f"Existing split column '{split_key}' is missing labels: {sorted(missing)}"
            )
        return adata

    split = pd.Series(index=adata.obs_names, dtype=object)

    is_ood = adata.obs[condition_key].astype(str).isin(set(ood_drugs))
    split.loc[is_ood] = "ood"

    remaining = adata.obs.loc[~is_ood]
    for cond, index in remaining.groupby(condition_key).groups.items():
        train_idx, test_idx = train_test_split(
            list(index),
            random_state=random_seed,
            test_size=test_size,
        )
        split.loc[train_idx] = "train"
        split.loc[test_idx] = "test"

    if split.isna().any():
        raise RuntimeError("Split assignment failed: some samples were not assigned.")

    adata.obs[split_key] = split.astype("category")

    # Ensure control cells are never OOD in the generated split.
    if control_key in adata.obs.columns:
        control_mask = adata.obs[control_key].astype(int) == 1
        adata.obs.loc[control_mask & (adata.obs[split_key] == "ood"), split_key] = "test"

    return adata


def build_split_views(
    adata: AnnData,
    *,
    split_key: str,
    control_key: str,
) -> Dict[str, AnnData]:
    """Create train/test/ood slices compatible with CondOT evaluation layout."""
    _to_dense(adata)

    views = {
        "train_control": adata[(adata.obs[split_key] == "train") & (adata.obs[control_key] == 1)].copy(),
        "train_treated": adata[(adata.obs[split_key] == "train") & (adata.obs[control_key] == 0)].copy(),
        "test_control": adata[(adata.obs[split_key] == "test") & (adata.obs[control_key] == 1)].copy(),
        "test_treated": adata[(adata.obs[split_key] == "test") & (adata.obs[control_key] == 0)].copy(),
        "ood": adata[(adata.obs[split_key] == "ood")].copy(),
    }
    return views


def summarize_split(
    adata: AnnData,
    *,
    split_key: str,
    condition_key: str,
    cell_type_key: str,
) -> pd.DataFrame:
    grouped = (
        adata.obs.groupby([split_key, condition_key, cell_type_key], observed=True)
        .size()
        .rename("n_cells")
        .reset_index()
    )
    return grouped

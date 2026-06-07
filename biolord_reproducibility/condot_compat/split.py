from __future__ import annotations

from typing import Iterable, Optional

import numpy as np
import pandas as pd


def _sample_index(index: pd.Index, frac: float, seed: int) -> pd.Index:
    if len(index) == 0:
        return index
    n = int(np.floor(len(index) * frac))
    if n <= 0:
        return index[:0]
    return pd.Series(index).sample(n=n, random_state=seed, replace=False).values


def _fallback_rebuild_split(
    adata,
    output_split_key: str,
    seed: int,
    condition_col: str,
    dose_col: str,
    control_col: str,
    ood_drugs: Iterable[str],
    validation_drugs: Iterable[str],
) -> None:
    split = pd.Series("train", index=adata.obs.index, dtype=object)
    cond = adata.obs[condition_col].astype(str)
    dose = adata.obs[dose_col].astype(float)
    control = adata.obs[control_col]

    split.loc[cond.isin(list(ood_drugs))] = "ood"

    mask = cond.isin(list(validation_drugs)) & dose.isin([1e3, 1e4]) & (split == "train")
    idx = _sample_index(adata.obs.index[mask], frac=0.4, seed=seed + 1)
    split.loc[idx] = "test"

    mask = cond.isin(list(validation_drugs)) & dose.isin([1e1, 1e2]) & (split == "train")
    idx = _sample_index(adata.obs.index[mask], frac=0.2, seed=seed + 2)
    split.loc[idx] = "test"

    mask = split == "train"
    idx = _sample_index(adata.obs.index[mask], frac=0.04, seed=seed + 3)
    split.loc[idx] = "test"

    mask = (split == "train") & (control == 1)
    idx = _sample_index(adata.obs.index[mask], frac=0.05, seed=seed + 4)
    split.loc[idx] = "test"

    adata.obs[output_split_key] = pd.Categorical(split, categories=["train", "test", "ood"])


def apply_condot_aligned_split(
    adata,
    output_split_key: str = "split_eval",
    preferred_split_key: str = "split_ood_finetuning",
    fallback_split_key: str = "split_ood",
    seed: int = 42,
    condition_col: str = "condition",
    dose_col: str = "dose",
    control_col: str = "control",
    ood_drugs: Optional[Iterable[str]] = None,
    validation_drugs: Optional[Iterable[str]] = None,
    strict: bool = True,
) -> str:
    if preferred_split_key in adata.obs.columns:
        adata.obs[output_split_key] = adata.obs[preferred_split_key].astype("category")
        return output_split_key

    if fallback_split_key in adata.obs.columns:
        adata.obs[output_split_key] = adata.obs[fallback_split_key].astype("category")
        return output_split_key

    if strict:
        raise KeyError(
            f"Neither '{preferred_split_key}' nor '{fallback_split_key}' is present in "
            f"adata.obs. Condot-aligned evaluation requires the h5ad file to already carry "
            f"the condot split column (typically 'split_ood_finetuning'). "
            f"Re-run the condot preprocessing to produce it, or pass strict=False to fall "
            f"back to a biolord-style rebuild (NOT condot-aligned)."
        )

    if ood_drugs is None:
        from utils.parameters_sciplex3 import ood_drugs as _ood_drugs

        ood_drugs = _ood_drugs
    if validation_drugs is None:
        from utils.parameters_sciplex3 import validation_drugs as _validation_drugs

        validation_drugs = _validation_drugs

    _fallback_rebuild_split(
        adata=adata,
        output_split_key=output_split_key,
        seed=seed,
        condition_col=condition_col,
        dose_col=dose_col,
        control_col=control_col,
        ood_drugs=ood_drugs,
        validation_drugs=validation_drugs,
    )
    return output_split_key


def summarize_split(adata, split_key: str, condition_col: str = "condition", cell_type_col: str = "cell_type"):
    split_counts = adata.obs[split_key].value_counts(dropna=False).rename("count").to_frame()
    by_condition = (
        adata.obs.groupby([split_key, condition_col], observed=True).size().rename("count").reset_index()
    )
    by_cell_type = (
        adata.obs.groupby([split_key, cell_type_col], observed=True).size().rename("count").reset_index()
    )
    return {
        "split_counts": split_counts,
        "by_condition": by_condition,
        "by_cell_type": by_cell_type,
    }

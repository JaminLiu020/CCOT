from __future__ import annotations

from typing import List, Sequence

import numpy as np
from anndata import AnnData


def _safe_var_index(adata: AnnData, genes: Sequence[str]) -> np.ndarray:
    keep = [g for g in genes if g in adata.var_names]
    return np.where(adata.var_names.isin(keep))[0]


def get_deg_gene_names(
    adata: AnnData,
    *,
    condition: str,
    cell_type: str,
    dose_value: float,
    n_deg: int = 50,
) -> List[str]:
    """Resolve DEG list using CondOT-style lincs_DEGs or rank_genes_groups fallback."""
    if "lincs_DEGs" in adata.uns:
        key = f"{cell_type}_{condition}_{dose_value}"
        if key in adata.uns["lincs_DEGs"]:
            return list(adata.uns["lincs_DEGs"][key])[:n_deg]

    if "rank_genes_groups" in adata.uns:
        names = adata.uns["rank_genes_groups"]["names"]
        if condition in names.dtype.names:
            return list(names[condition])[:n_deg]

    raise KeyError(
        "Cannot find DEG definitions. Expect adata.uns['lincs_DEGs'] or adata.uns['rank_genes_groups']."
    )


def get_deg_indices(
    adata: AnnData,
    *,
    condition: str,
    cell_type: str,
    dose_value: float,
    n_deg: int = 50,
) -> np.ndarray:
    genes = get_deg_gene_names(
        adata,
        condition=condition,
        cell_type=cell_type,
        dose_value=dose_value,
        n_deg=n_deg,
    )
    idx = _safe_var_index(adata, genes)
    if idx.size == 0:
        raise ValueError(
            f"No DEG genes found in adata.var_names for condition={condition}, cell_type={cell_type}."
        )
    return idx

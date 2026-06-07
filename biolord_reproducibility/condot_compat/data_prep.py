from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc


def load_adata(path: str) -> "sc.AnnData":
    return sc.read_h5ad(path)


def save_adata(adata: "sc.AnnData", path: str) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    adata.write_h5ad(out)


def _require_obs_columns(adata: "sc.AnnData", columns: list[str]) -> None:
    missing = [c for c in columns if c not in adata.obs.columns]
    if missing:
        raise KeyError(f"Missing required adata.obs columns: {missing}")


def compute_rdkit2d_features(
    adata: "sc.AnnData",
    smiles_col: str = "SMILES",
    dose_col: str = "dose",
    variance_threshold: float = 0.001,
) -> "sc.AnnData":
    _require_obs_columns(adata, [smiles_col, dose_col])

    try:
        from chemprop.features.features_generators import (
            rdkit_2d_normalized_features_generator,
        )
    except ImportError as exc:
        raise ImportError(
            "chemprop is required for RDKit2D feature generation. Install chemprop first."
        ) from exc

    smiles_series = adata.obs[smiles_col]
    if pd.api.types.is_categorical_dtype(smiles_series):
        unique_smiles = pd.Index(smiles_series.cat.categories)
    else:
        unique_smiles = pd.Index(pd.Series(smiles_series).dropna().astype(str).unique())

    features = {}
    for mol in unique_smiles:
        features[mol] = rdkit_2d_normalized_features_generator(mol)

    features_df = pd.DataFrame.from_dict(features).T
    features_df = features_df.fillna(0)

    std = features_df.std()
    keep = np.where(std > variance_threshold)[0]
    if len(keep) == 0:
        raise ValueError("No RDKit2D features left after variance filtering.")

    features_df = features_df.iloc[:, keep]
    normalized_df = (features_df - features_df.mean()) / features_df.std()
    normalized_df = normalized_df.fillna(0)

    features_cells = np.zeros((adata.shape[0], normalized_df.shape[1] + 1))
    smiles_array = adata.obs[smiles_col].values
    for mol, rdkit_2d in normalized_df.iterrows():
        mask = pd.Series(smiles_array).isin([mol]).values
        features_cells[mask, :-1] = rdkit_2d.values

    dose = adata.obs[dose_col].astype(float).values
    dose_max = float(np.max(dose)) if len(dose) else 1.0
    dose_norm = dose / dose_max if dose_max > 0 else dose
    features_cells[:, -1] = dose_norm

    adata.obsm["rdkit2d"] = features_cells[:, :-1]
    adata.obsm["rdkit2d_dose"] = features_cells
    return adata

from __future__ import annotations

import sys
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import pandas as pd
from rdkit import Chem

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "datasets" / "reference" / "smiles"
LABEL_IN = DATA_DIR / "label.csv"
SCIPLEX_IN = DATA_DIR / "sciplex3_drugs.csv"
LABEL_OUT = DATA_DIR / "label_canonical_unique.csv"
SCIPLEX_OUT = DATA_DIR / "sciplex3_drugs_canonical.csv"


def canonicalize_smiles(smiles: str) -> Optional[str]:
    """Return canonical SMILES or None if parsing fails."""
    if pd.isna(smiles):
        return None
    try:
        mol = Chem.MolFromSmiles(str(smiles))
    except Exception:
        return None
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True)


def convert_frame(df: pd.DataFrame, column: str) -> Tuple[pd.DataFrame, List[Tuple[int, str]]]:
    """Canonicalize a SMILES column, record failures."""
    failures: List[Tuple[int, str]] = []
    canonical: List[Optional[str]] = []
    for idx, smi in df[column].items():
        canon = canonicalize_smiles(smi)
        if canon is None:
            failures.append((idx, str(smi)))
        canonical.append(canon)
    df = df.copy()
    df["canonical_smiles"] = canonical
    df = df.dropna(subset=["canonical_smiles"])
    return df, failures


def process_label() -> None:
    df = pd.read_csv(LABEL_IN)
    df_conv, failures = convert_frame(df, "canonical_smiles")
    df_conv = df_conv.drop_duplicates(subset=["canonical_smiles"], keep="first")
    df_conv.to_csv(LABEL_OUT, index=False)
    _log_failures("label.csv", failures)


def process_sciplex() -> None:
    df = pd.read_csv(SCIPLEX_IN)
    df_conv, failures = convert_frame(df, "SMILES")
    df_conv = df_conv[["index", "canonical_smiles"]]
    df_conv.to_csv(SCIPLEX_OUT, index=False)
    _log_failures("sciplex3_drugs.csv", failures)


def _log_failures(name: str, failures: Iterable[Tuple[int, str]]) -> None:
    failures = list(failures)
    if not failures:
        print(f"{name}: all SMILES parsed", file=sys.stderr)
        return
    print(f"{name}: {len(failures)} SMILES failed to parse", file=sys.stderr)
    for idx, smi in failures:
        print(f"  row {idx}: {smi}", file=sys.stderr)


def main() -> None:
    process_label()
    process_sciplex()
    print(f"Wrote {LABEL_OUT.name} and {SCIPLEX_OUT.name} to {DATA_DIR}")


if __name__ == "__main__":
    main()

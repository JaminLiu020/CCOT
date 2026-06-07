from dataclasses import dataclass
from typing import Sequence


DEFAULT_OOD_DRUGS = [
    "Dacinostat",
    "Givinostat",
    "Belinostat",
    "Hesperadin",
    "Quisinostat",
    "Alvespimycin",
    "Tanespimycin",
    "TAK-901",
    "Flavopiridol",
]


@dataclass
class EvalConfig:
    data_path: str = "/data/jamin_datasets/condot/scrna-sciplex3/hvg.h5ad"
    output_root: str = "results/perturbnet_condot_sciplex"
    experiment_name: str = "default"
    split_key: str = "split_ood_finetuning"
    condition_key: str = "condition"
    control_key: str = "control"
    cell_type_key: str = "cell_type"
    dose_key: str = "dose"
    smiles_key: str = "SMILES"
    random_seed: int = 42
    test_size: float = 0.2
    ood_drugs: Sequence[str] = tuple(DEFAULT_OOD_DRUGS)
    n_deg: int = 50
